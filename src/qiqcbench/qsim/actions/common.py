from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.actions.final_answer_shape import (
    answer_shape_problems,
    answer_shape_refusal,
)
from qiqcbench.qsim.core.wire import FinalAnswer
from qiqcbench.qsim.jobs import is_internal_job_failure

# Any completed result whose serialized JobResult exceeds this bound is published
# under /qsim_logs (which `main` already mounts read-only) and the inline payload is
# replaced by a pointer to it. Invariant 5 is preserved -- the agent still receives
# every raw bit, it just reads them from a file.
#
# The bound is universal rather than opt-in per payload kind. The opt-in list this
# replaced was extended four times, each after a live incident, and it still left 40
# of the 48 registered result kinds returning unbounded payloads: 1,238 stored
# `get_job_result` deliveries across 240 trials and 32 tasks were destructively
# clipped mid-document by the agent harness, burning ~14.9M tokens of context on
# fragments that were no longer valid JSON.
#
# 32 KiB, not the historical 64 KiB, because both harnesses clip BELOW 64 KiB and a
# 64 KiB trigger would leave the entire observed failure band untouched. Measured
# ceilings over 931 stored runs / 1,455 trials: codex elides the middle of a tool
# result at 48,026 B, and the smallest oversized result Claude Code diverted to a
# file was 53,862 B. The largest JobResult ever delivered intact into model context
# was 49,210 B. 32 KiB clears both ceilings with headroom for the transport envelope
# that wraps this payload, and still sits far above the ~13 KB typical single
# shot-block result, so ordinary payloads stay inline.
MAX_INLINE_RESULT_BYTES = 32 * 1024
MAX_INLINE_IQ_SAMPLES = 1_000
# Result kinds whose worker-private raw_results/<job_id>.json record is published
# into the agent-visible public_job_results/ tree at the first terminal
# get_job_result delivery, with the delivered data.raw_data_file repointed at the
# published copy. Separate-mode bundles blind main to raw_results/,
# so a delivered pointer into it would dangle; publication at the poll keeps the
# documented file-delivery contract while preserving poll-gating (an unpolled
# record never becomes agent-visible). Kinds absent from this set keep their
# historical private-path delivery, which their shipped task editions were
# scored under; repointing one would silently move that edition. No released
# bundle exposes raw_results/ to the agent, and a new raw-record kind must be
# added here before its public pointer is enabled.
POLL_PUBLISHED_RAW_RECORD_KINDS = frozenset(
    {
        "atom_dm_sweep_record",
        "drift_stream",
        "located_erasure_control_episode",
        "rydberg_stroboscopic_trajectory",
    }
)
PUBLISHED_RAW_RECORD_SUBDIR = "raw_results"

# Delivery artifacts publish into this subdirectory of ``public_job_results/``
# rather than into its root. The root is a nested mount point and cannot be
# unlinked -- ``rmdir`` on it fails EBUSY. ext4 never shrinks a directory inode
# when its entries are unlinked, so
# a root that had held one file per job kept the ``st_size``/``st_blocks`` those
# files caused (measured: 4096 B / 8 blocks empty, 8192 / 16 after 123
# job-shaped names, persisting after removal), and a retried agent could
# ``stat`` it through its read-only mount and recover roughly how many results
# its predecessor published.
#
# Publishing one level down makes the growing directory an ORDINARY one, which
# the entrypoint's boot reset can remove: its ``find -depth -type d -exec rmdir``
# pass now empties and unlinks this subdirectory, and only the mount root
# survives -- holding at most the two subdirectory entries, so its inode stays
# at the single-block minimum no matter how many jobs ran.
#
# The machine-readable agent contract remains the exact ``relative_path`` in
# each offload pointer. Public instructions must direct agents to that pointer,
# rather than restating a concrete filename below the mount root.
PUBLISHED_RESULT_SUBDIR = "results"

# Retention bound for the terminal-delivery cache (state._job_result_delivery_cache).
# Entries are canonical JSON strings of already-bounded deliveries (inline payloads
# are <= MAX_INLINE_RESULT_BYTES; the variable part is the per-point counts summary),
# so the cap is a hard byte budget rather than an entry count. 64 MiB holds every
# stored trial's full completed-job set (the largest observed, 123 heavy harper
# sweeps, needs ~49 MiB) while bounding worst-case retention far below the raw
# results it replaces (~540 MB of heap per wedge-shaped result before eviction).
# A poll for an LRU-evicted delivery is rebuilt from the published artifact.
JOB_RESULT_DELIVERY_CACHE_MAX_BYTES = 64 * 1024 * 1024

# Qsim-private spool for materialized-but-not-yet-polled delivery artifacts.
# The worker thread persists a completed result's exact artifact bytes HERE at
# completion time; the file is renamed into ``public_job_results/`` only when
# the first terminal ``get_job_result`` poll delivers the pointer. Poll-gated
# publication is a load-bearing isolation property: in separate verifier mode
# ``public_job_results/`` is the only ``/qsim_logs`` tree the agent container
# sees, and a result the agent never polled must never appear there. A
# job that is never polled therefore leaves its artifact in the spool at trial
# end -- fine: verifiers consume ``raw_results/`` and the ferried evidence
# tree, and no ``task.toml`` ``[[artifacts]]`` source names ``/qsim_logs``
# itself, so this dot-prefixed directory is never collected as an artifact.
_DELIVERY_SPOOL_DIRNAME = ".delivery_spool"
_SAFE_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
JOB_RESULT_FAILURE_ACTION = "get_job_result_failure"
QSIM_INTERNAL_FAILURE_KIND = "qsim_internal"
RESULT_ARTIFACT_PERSISTENCE_STAGE = "result_artifact_persistence"
RESULT_POLL_LOG_PUBLICATION_STAGE = "result_poll_log_publication"
RESULT_POLL_BUDGET_STAGE = "result_poll_budget_reservation"
RESULT_ARTIFACT_PERSISTENCE_ERROR = "qsim result artifact publication failed"
RESULT_POLL_LOG_PUBLICATION_ERROR = "qsim result poll publication failed"
RESULT_POLL_BUDGET_ERROR = "qsim result poll budget enforcement failed"
FINAL_ANSWER_FAILURE_ACTION = "submit_final_answer_failure"
FINAL_ANSWER_PERSISTENCE_STAGE = "final_answer_persistence"
FINAL_ANSWER_LOG_PUBLICATION_STAGE = "final_answer_log_publication"
FINAL_ANSWER_PERSISTENCE_ERROR = "qsim final-answer persistence failed"
FINAL_ANSWER_LOG_PUBLICATION_ERROR = "qsim final-answer publication failed"

# --- File-based final-answer submission --------------------------------------
# A task may provide the sanctioned agent->qsim ``submission`` volume: compose
# mounts it read-write in ``main`` at /submission and READ-ONLY in qsim at
# /submission_artifacts (harbor enforces that qsim -- which holds hidden truth
# -- never gets a writable handle). QSIM_SUBMISSIONS_DIR names qsim's read-only
# view. The agent writes its answer dict as UTF-8 JSON to a single-segment file
# there and calls ``submit_final_answer`` with ``answer_file`` + ``answer_sha256``
# instead of the inline dict, so a large answer never passes through the model's
# context.
#
# THE INVARIANT: a file submission of answer dict A leaves evidence
# byte-identical to an inline submission of A. After the file is loaded and
# hash-verified, ``answer`` is the parsed dict and every downstream step --
# string/nesting resource checks, the compact-envelope byte cap, the
# submission-count budget, the single logged ``submit_final_answer`` event, and
# ``final_answer.json`` -- runs on that dict exactly as the inline path does.
# The transport writes NOTHING extra to the evidence log: verifiers that
# enumerate every logged action against an allowlist (e.g. the surrogate
# scorer's disposition check) would reject any new event, and the "do not
# change the verifier side" contract means the evidence must be
# indistinguishable from inline. Transport provenance is therefore returned to
# the agent in the tool result only (ephemeral, never persisted), so an
# allowlist and a filter verifier alike see an ordinary submission.
#
# TOCTOU / oracle safety. The agent owns /submission read-write and could race
# qsim by swapping the file for a symlink to hidden truth (qsim can read
# /app/configs). So the read never re-traverses an agent-controlled pathname
# after validating it: ``answer_file`` is restricted to ONE path segment under
# the mount root, and the file is opened once with ``O_NOFOLLOW`` (a symlink at
# that name is rejected, not followed) and then only the resulting fd is used
# (fstat for type/size, read from the fd). ``answer_sha256`` is required on top:
# an agent can only submit bytes whose exact hash it already knows, so even a
# race that somehow won would not turn this into a read oracle.
SUBMISSIONS_DIR_ENV = "QSIM_SUBMISSIONS_DIR"
# Hard transport bound on the on-disk file. Pretty-printed JSON is allowed, so
# this sits far above every task's compact-envelope cap; the task budget still
# governs the compact envelope after parsing.
MAX_ANSWER_FILE_BYTES = 64 * 1024 * 1024
# ``answer_file`` is a single filename directly under the mount root: no "/", so
# no intermediate agent-controlled directory component can be a symlink.
_SAFE_ANSWER_FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_ANSWER_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")

METADATA_CALL_FAILURE_ACTION = "metadata_call_failure"
METADATA_LOG_PUBLICATION_STAGE = "metadata_log_publication"
METADATA_LOG_PUBLICATION_ERROR = "qsim metadata-call publication failed"

# Actions that exist only to record a qsim-owned fault. They always carry the
# stamp above, so these names are redundant with it -- but a marker whose own
# ``ctx.state.log`` call is what broke can reach a verifier truncated, and the
# ``*_infrastructure_failure`` suffix is the convention several capability
# action layers already publish under.
_QSIM_FAULT_ACTIONS = frozenset(
    {
        JOB_RESULT_FAILURE_ACTION,
        FINAL_ANSWER_FAILURE_ACTION,
        METADATA_CALL_FAILURE_ACTION,
    }
)


def qsim_internal_fault(events: Iterable[Any]) -> str | None:
    """Return a description of the first qsim-owned fault in ``events``, else ``None``.

    This is the single ownership rule for the whole suite. qsim
    stamps ``failure_kind == QSIM_INTERNAL_FAILURE_KIND`` onto its own faults --
    a job that failed or was abandoned inside qsim (the terminal
    ``get_job_result`` event), a result-artifact or poll-log publication
    failure, a final-answer persistence failure, a metadata-call publication
    failure. Every writer of that stamp is a qsim-owned code path: a request the
    model got wrong keeps an untyped, model-visible error and never reaches
    here (``qsim/jobs.py::is_internal_job_failure``). The stamp is
    therefore exactly the ownership signal, and the agent cannot provoke it.

    A verifier that finds a fault here has observed a trial qsim broke, and must
    take its reserved infrastructure exit with no reward file rather than record
    a scientific failure the model did not commit. That holds even when the
    model went on to complete the science: a fault it worked around still voids
    the trial, which costs a rerun and never mis-attributes.

    Non-mapping entries are skipped so a torn or partially parsed log can still
    be scanned for ownership; deciding what to do about the tear stays with the
    caller's own evidence recovery.
    """

    for event in events:
        if not isinstance(event, Mapping):
            continue
        action = event.get("action")
        if event.get("failure_kind") == QSIM_INTERNAL_FAILURE_KIND:
            stage = event.get("failure_stage")
            where = f"{action}" + (f"/{stage}" if stage else "")
            return f"qsim recorded an internal failure ({where})"
        if action in _QSIM_FAULT_ACTIONS or (
            isinstance(action, str) and action.endswith("_infrastructure_failure")
        ):
            return f"qsim recorded an internal failure ({action})"
    return None


def qsim_internal_fault_in_log(log_path: str | Path) -> str | None:
    """``qsim_internal_fault`` read straight from an experiment log.

    A missing log is not a fault here and an unparseable line is skipped: this
    scan decides ownership only, and each verifier keeps its own torn-log and
    missing-log policy where it already lives.
    """

    path = Path(log_path)
    if not path.is_file():
        return None
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return qsim_internal_fault(events)


def _best_effort_log_result_failure(
    ctx: ActionContext,
    *,
    job_id: str,
    status: str,
    failure_stage: str,
) -> None:
    """Publish a sanitized poll failure without masking the triggering error."""

    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": JOB_RESULT_FAILURE_ACTION,
                "tool": "get_job_result",
                "job_id": job_id,
                "status": status,
                "failure_kind": QSIM_INTERNAL_FAILURE_KIND,
                "failure_stage": failure_stage,
            }
        )
    except Exception:
        # A persistent log failure cannot be represented in that same log.
        # Keep the original public failure disposition deterministic.
        return


def _best_effort_log_final_answer_failure(
    ctx: ActionContext,
    *,
    task_id: str,
    failure_stage: str,
) -> None:
    """Publish a sanitized final-answer failure without masking its cause."""

    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": FINAL_ANSWER_FAILURE_ACTION,
                "tool": "submit_final_answer",
                "task_id": task_id,
                "failure_kind": QSIM_INTERNAL_FAILURE_KIND,
                "failure_stage": failure_stage,
            }
        )
    except Exception:
        # A persistent log failure cannot be represented in that same log.
        # Keep the original public failure disposition deterministic.
        return


def _best_effort_log_metadata_failure(
    ctx: ActionContext,
    *,
    tool: str,
    failed_action: str,
) -> None:
    """Publish a bounded marker when an accepted metadata call cannot be logged."""

    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": METADATA_CALL_FAILURE_ACTION,
                "tool": tool,
                "failed_action": failed_action,
                "failure_kind": QSIM_INTERNAL_FAILURE_KIND,
                "failure_stage": METADATA_LOG_PUBLICATION_STAGE,
            }
        )
    except Exception:
        return


def _public_final_answer_budget(ctx: ActionContext) -> object | None:
    """Return an attribute-based public final-answer policy when one is declared."""

    public = getattr(ctx.state, "public", None)
    budget = getattr(public, "budget", None)
    if budget is None:
        # Some qtypes name the aggregate experiment policy ``budgets``. Accept
        # that shape only when it declares the complete final-answer policy
        # below; unrelated experiment budgets remain unaffected.
        budget = getattr(public, "budgets", None)
    if budget is None or not hasattr(budget, "max_final_answer_serialized_bytes"):
        return None
    required = (
        "max_final_answer_serialized_bytes",
        "max_final_answer_submissions",
        "max_answer_string_characters",
    )
    for name in required:
        value = getattr(budget, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"public final-answer resource policy has invalid {name}")
    return budget


def _public_job_result_poll_limit(ctx: ActionContext) -> int | None:
    """Return the public cumulative poll cap when the qtype declares one."""

    public = getattr(ctx.state, "public", None)
    budget = getattr(public, "budget", None)
    if budget is None or not hasattr(budget, "max_job_result_polls"):
        return None
    limit = budget.max_job_result_polls
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise RuntimeError("public job-result poll policy has an invalid limit")
    return limit


def _active_vqe_evaluation_budget(ctx: ActionContext, *, task_id: str):
    """Load this task's public resource policy when digital VQE is active."""

    if not task_id or "digital_vqe" not in getattr(ctx.state, "active_capabilities", ()):
        return None
    # Deferred to keep common action imports independent of task-material and
    # qtype registry initialization.
    from qiqcbench.qsim.qtypes.digital_gate_model.capabilities.vqe.materials import (
        load_vqe_public_materials,
    )
    from qiqcbench.qsim.task_materials import public_task_material_dir

    return load_vqe_public_materials(public_task_material_dir(task_id)).ansatz.evaluation_budget


def _task_final_answer_precheck(
    ctx: ActionContext, *, task_id: str, answer: dict[str, Any]
) -> dict[str, Any] | None:
    """Run a task's public structural precheck of the final answer, if it has one.

    A precheck sees only the validated answer dict and qsim-owned run facts. It
    either returns public diagnostics for the accepted response or raises
    ``ValueError`` to refuse the submission before any submission budget is
    reserved, so the refusal consumes nothing and leaves no evidence. It never
    scores, never reads hidden truth, and is never persisted.

    The task-agnostic answer-shape registry
    (``final_answer_shape``) refuses an answer that breaks the shape the task
    instructions publish, using the same pure function the task's verifier
    applies as its own contract gate.
    """

    shape_problems = answer_shape_problems(task_id, answer)
    if shape_problems:
        raise ValueError(answer_shape_refusal(shape_problems))
    return None


def _validate_answer_string_resources(
    answer: dict[str, Any],
    *,
    max_characters: int,
    max_job_id_characters: int | None = None,
    max_depth: int | None = None,
) -> None:
    """Bound nested strings without deciding task-answer scientific validity."""

    pending: list[tuple[object, str, str | None, int]] = [(answer, "answer", None, 0)]
    while pending:
        value, path, key_name, depth = pending.pop()
        if max_depth is not None and depth > max_depth:
            raise ValueError(f"{path} exceeds the public final-answer nesting depth of {max_depth}")
        if isinstance(value, str):
            job_id_bounded = key_name == "job_id" and max_job_id_characters is not None
            limit = max_job_id_characters if job_id_bounded else max_characters
            if len(value) > limit:
                label = "citation job_id" if job_id_bounded else "answer string"
                raise ValueError(f"{path} exceeds the public {label} limit of {limit} characters")
        elif isinstance(value, dict):
            for key, nested in value.items():
                if len(str(key)) > max_characters:
                    raise ValueError(
                        f"{path} contains an object key exceeding the public answer "
                        f"string limit of {max_characters} characters"
                    )
                pending.append((nested, f"{path}.{key}", str(key), depth + 1))
        elif isinstance(value, (list, tuple)):
            pending.extend(
                (nested, f"{path}[{index}]", None, depth + 1) for index, nested in enumerate(value)
            )


def compact_final_answer_bytes(payload: dict[str, Any]) -> bytes:
    """Serialize the complete envelope under the public compact-JSON rule."""

    try:
        serialized = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ValueError("final answer cannot be serialized as finite compact JSON") from exc
    return serialized.encode("utf-8")


def _public_iq_result_path(log_dir: Path | None, *, job_id: str) -> tuple[Path, str]:
    """Return a confined public artifact path for a validated qsim job ID."""
    if not isinstance(job_id, str) or _SAFE_JOB_ID.fullmatch(job_id) is None:
        raise ValueError("job_id must be a safe filename component")
    if log_dir is None:
        raise RuntimeError("large IQ results require a qsim log_dir")

    root = log_dir.resolve()
    mount_root = root / "public_job_results"
    if mount_root.is_symlink() or (mount_root.exists() and not mount_root.is_dir()):
        raise ValueError("public IQ result directory must be a real directory")
    mount_root.mkdir(parents=True, exist_ok=True)
    resolved_mount_root = mount_root.resolve()
    if resolved_mount_root.parent != root:
        raise ValueError("public IQ result directory escapes the qsim log directory")
    mount_root.chmod(0o755)

    # One level down, so the directory that grows is one the boot reset can
    # remove. Each component is validated separately: the mount
    # root cannot be unlinked, but this subdirectory can, so a hostile or
    # corrupted replacement of it must not be followed.
    directory = mount_root / PUBLISHED_RESULT_SUBDIR
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise ValueError("public IQ result directory must be a real directory")
    directory.mkdir(parents=True, exist_ok=True)
    resolved_directory = directory.resolve()
    if resolved_directory.parent != resolved_mount_root:
        raise ValueError("public IQ result directory escapes the qsim log directory")
    directory.chmod(0o755)

    filename = f"{job_id}.json"
    target = directory / filename
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise ValueError("public IQ result target must be a real file")
    if target.resolve().parent != resolved_directory:
        raise ValueError("public IQ result path escapes the qsim log directory")
    return target, filename


def _delivery_spool_path(log_dir: Path | None, *, job_id: str) -> Path:
    """Return the confined qsim-private spool slot for a validated job ID."""
    if not isinstance(job_id, str) or _SAFE_JOB_ID.fullmatch(job_id) is None:
        raise ValueError("job_id must be a safe filename component")
    if log_dir is None:
        raise RuntimeError("large IQ results require a qsim log_dir")
    directory = log_dir.resolve() / _DELIVERY_SPOOL_DIRNAME
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise ValueError("delivery spool directory must be a real directory")
    target = directory / f"{job_id}.json"
    # One atomic lstat snapshot: a concurrent first poll renames spool entries
    # away, and a separate exists()/is_file() pair can straddle that rename
    # and misread a vanishing regular file as a non-file.
    try:
        mode = target.lstat().st_mode
    except FileNotFoundError:
        mode = None
    if mode is not None and not stat.S_ISREG(mode):
        raise ValueError("delivery spool target must be a real file")
    return target


def _publish_spooled_job_result(state: Any, *, job_id: str) -> None:
    """Publish a spooled delivery artifact into ``public_job_results/``.

    Called only from poll context, so publication stays poll-gated. The spool
    lives on the ``qsim_logs`` filesystem while ``public_job_results/`` is a
    distinct nested volume in separate-mode bundles, so promotion must never rename or link across
    that boundary -- POSIX rename across mount points fails ``EXDEV``.
    Instead it copies the bounded artifact bytes into a private temporary
    INSIDE the destination directory, fsyncs, and hard-links it into place:
    atomic on the destination filesystem and never overwriting published
    evidence. This is the same copy-through-the-gate pattern as
    ``_publish_poll_gated_raw_record``. A missing spool file means a previous
    poll already promoted this job's artifact -- a no-op.
    """
    log_dir = getattr(state, "log_dir", None)
    if log_dir is None:
        return
    spool = _delivery_spool_path(log_dir, job_id=job_id)
    try:
        artifact_bytes = spool.read_bytes()
    except FileNotFoundError:
        # Already promoted by an earlier or concurrent poll.
        return
    target, _ = _public_iq_result_path(log_dir, job_id=job_id)
    if target.exists():
        # An idempotent re-materialization can leave a fresh spool copy behind
        # an already-published artifact; equal bytes collapse to cleanup, and
        # anything else must never silently replace published evidence.
        if target.is_symlink() or not target.is_file() or target.read_bytes() != artifact_bytes:
            raise RuntimeError(f"conflicting immutable JobResult artifact for {job_id!r}")
        spool.unlink(missing_ok=True)
        return
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "wb",
            dir=target.parent,
            prefix=".public_job_result.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temporary = Path(f.name)
            f.write(artifact_bytes)
            f.flush()
            # Durable before visible: the link below makes the artifact the
            # agent-facing contract, so its bytes must already be on disk.
            os.fsync(f.fileno())
        temporary.chmod(0o644)
        try:
            os.link(temporary, target)
        except FileExistsError:
            # A concurrent poll placed these exact bytes first.
            if _artifact_bytes_match(target, artifact_bytes) is not True:
                raise RuntimeError(
                    f"conflicting immutable JobResult artifact for {job_id!r}"
                ) from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    spool.unlink(missing_ok=True)


def _iq_sample_count(iq: object) -> int:
    """Count raw IQ pairs without changing their representation."""
    if not isinstance(iq, list):
        return 0
    if len(iq) == 2 and all(isinstance(value, (int, float)) for value in iq):
        return 1
    return sum(_iq_sample_count(item) for item in iq)


def _artifact_bytes_match(path: Path, artifact_bytes: bytes) -> bool | None:
    """Compare an existing artifact's bytes, tolerating a concurrent rename.

    Returns True/False for a readable regular file, and ``None`` when the file
    vanished before it could be read (the promotion rename moved it). A
    symlink or non-file is a genuine conflict, never a race artifact.
    """
    if path.is_symlink():
        return False
    try:
        return path.read_bytes() == artifact_bytes
    except (FileNotFoundError, NotADirectoryError):
        return None


def _persist_conflict_converges(
    log_dir: Path | None, *, job_id: str, target: Path, artifact_bytes: bytes, spool: bool
) -> bool:
    """Decide whether an existing-artifact link conflict is a true conflict.

    A first terminal poll may rename the spool entry into
    ``public_job_results/`` between another builder's ``os.link`` conflict and
    its byte comparison; the identical bytes then live at the public
    destination instead. Before this check, that window produced false
    ``qsim result artifact publication failed`` polls under concurrent
    identical rebuilds (measured 1426 false failures across 500 six-thread
    iterations). Fail only when bytes that can actually be read
    genuinely differ.
    """
    matched = _artifact_bytes_match(target, artifact_bytes)
    if matched is not None:
        return matched
    if not spool or log_dir is None:
        return False
    # Comparison-only path construction: never mkdir the public tree from a
    # completion-time (worker thread) builder -- publication stays poll-gated.
    promoted = log_dir.resolve() / "public_job_results" / PUBLISHED_RESULT_SUBDIR / f"{job_id}.json"
    return _artifact_bytes_match(promoted, artifact_bytes) is True


def _persist_large_iq_result(
    log_dir: Path | None, *, job_id: str, serialized_payload: str, spool: bool = False
) -> dict[str, Any]:
    """Atomically persist a JobResult and return its exact confined pointer.

    Serves every result kind, not only IQ -- the ``iq`` in these helper names is
    historical, from when only oversized IQ payloads were published.

    With ``spool=True`` the bytes land in the qsim-private delivery spool
    instead of the agent-visible ``public_job_results/`` tree; publication is
    then the O(1) poll-time rename in ``_publish_spooled_job_result``. The
    returned pointer always names the public path: it is the delivery
    contract, and the first terminal poll promotes the artifact before any
    agent or verifier can read the pointer.
    """
    if spool:
        target = _delivery_spool_path(log_dir, job_id=job_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        filename = target.name
    else:
        target, filename = _public_iq_result_path(log_dir, job_id=job_id)
    artifact_bytes = (serialized_payload + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "wb",
            dir=target.parent,
            prefix=".public_job_result.",
            suffix=".tmp",
            delete=False,
        ) as f:
            temporary = Path(f.name)
            f.write(artifact_bytes)
        temporary.chmod(0o644)
        try:
            # A hard link publishes the fully written inode without replacing
            # any prior evidence under the same job ID.
            os.link(temporary, target)
        except FileExistsError:
            if not _persist_conflict_converges(
                log_dir,
                job_id=job_id,
                target=target,
                artifact_bytes=artifact_bytes,
                spool=spool,
            ):
                raise RuntimeError(
                    f"conflicting immutable JobResult artifact for {job_id!r}"
                ) from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "schema_version": 1,
        "relative_path": f"public_job_results/{PUBLISHED_RESULT_SUBDIR}/{filename}",
        "size_bytes": len(artifact_bytes),
        "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
    }


def _build_bounded_iq_result_payload(
    state: Any, *, job_id: str, result_payload: dict[str, Any], spool: bool = False
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Persist selected raw results and bound oversized responses to a pointer."""
    data = result_payload.get("data")
    if result_payload.get("status") != "complete" or not isinstance(data, dict):
        return result_payload, None
    kind = data.get("kind")

    serialized_payload = json.dumps(result_payload, indent=2, allow_nan=False)
    oversized = len(serialized_payload.encode("utf-8")) > MAX_INLINE_RESULT_BYTES
    if kind == "iq":
        sample_count = sum(_iq_sample_count(samples) for samples in data.get("iq", {}).values())
        oversized = oversized or sample_count > MAX_INLINE_IQ_SAMPLES
    log_dir = getattr(state, "log_dir", None)
    # Every completed raw-evidence result is scientific evidence, not only the
    # oversized ones. Persist it and log its pointer even when it is small
    # enough to stay inline, so the exact raw bytes live in the run's artifact
    # closure instead of only in the agent's transcript. A small result that a
    # verifier cannot reread is evidence that exists once and then is gone.
    #
    # This condition carries no kind test, so retiring the kind gate above
    # widened artifact closure from the 8 admitted kinds to all 48 registered
    # ones in the same step that widened the size bound. That is deliberate:
    # bounding a payload out of the response without closing it as an artifact
    # would leave the agent pointed at a file qsim never wrote.
    #
    # Scored/runtime sidecars always provide a log directory and therefore get
    # a closed raw-evidence artifact for every completed result. Preserve the
    # historical inline-only local CLI path when no evidence directory was
    # requested.
    persist_for_evidence = log_dir is not None
    if not oversized and not persist_for_evidence:
        return result_payload, None
    if log_dir is None and kind != "iq":
        # Local dev `serve-mcp` without `--log-dir` has nowhere to publish the
        # artifact. Return the oversized payload inline rather than failing the
        # poll: this path cannot occur in a scored run, because Harbor's qsim
        # entrypoint always passes `--log-dir /qsim_logs`, and every consumer of
        # the offload pointer (both hidden verifiers) already depends on that.
        # The IQ path keeps its historical raise -- narrowing an existing
        # fail-closed guarantee is not part of this change.
        return result_payload, None

    offloaded_result = _persist_large_iq_result(
        log_dir,
        job_id=job_id,
        serialized_payload=serialized_payload,
        spool=spool,
    )
    if not oversized:
        return result_payload, offloaded_result

    # Derive the agent-visible absolute path from the pointer itself rather than
    # rebuilding it from the bare filename: the pointer is the single authority
    # on where the artifact was published, so this cannot drift out of step with
    # the publish layout if a path is reassembled independently.
    public_path = f"/qsim_logs/{offloaded_result['relative_path']}"
    metadata = dict(result_payload["metadata"])
    warnings = list(metadata.get("public_warnings", []))
    label = "raw IQ data" if kind == "iq" else f"raw {kind} data"
    warnings.append(
        f"This result was too large to return inline, so `data` is null here. The complete "
        f"unmodified result, with `data` populated with the full {label}, is the JSON file "
        f"{public_path}. Read that file with a script instead of printing it; nothing was "
        f"truncated, aggregated, or omitted."
    )
    metadata["public_warnings"] = warnings
    # A machine-readable locator beside the prose one. The prose line stays: it is
    # what artifact-reading trials actually acted on. ``result_kind`` is carried
    # HERE and deliberately not in ``offloaded_result`` -- that pointer is the
    # verifier-facing contract and two hidden verifiers pin its exact field set.
    metadata["result_artifact"] = {**offloaded_result, "result_kind": kind}
    return {**result_payload, "data": None, "metadata": metadata}, offloaded_result


def _bitstring_counts_summary(result_payload: dict[str, Any]) -> dict[str, Any] | None:
    """Summarize a completed digital job's per-shot bitstrings as per-outcome counts.

    The experiment log stores *counts* (not every shot) so task verifiers can recompute
    estimators from the agent's actual completed evidence rather than trusting a reported
    scalar. Returns ``None`` for non-complete jobs or non-bitstring result data (e.g. IQ
    shots), so it is a no-op for pulse qtypes and backward-compatible for existing tasks.
    """
    if result_payload.get("status") != "complete":
        return None
    data = result_payload.get("data")
    if not isinstance(data, dict) or data.get("kind") != "bitstring":
        return None
    bitstrings = data.get("bitstrings")
    if not isinstance(bitstrings, list):
        return None
    counts: list[dict[str, Any]] = []
    for index, shots in enumerate(bitstrings):
        if not isinstance(shots, list):
            return None
        counts.append(
            {
                "index": index,
                "shots": len(shots),
                "counts": dict(sorted(Counter(str(shot) for shot in shots).items())),
            }
        )
    return {
        "kind": "bitstring_counts",
        "measured_qubits": data.get("measured_qubits"),
        "counts": counts,
    }


def _require_regular_file(path: Path, *, description: str) -> None:
    """lstat-based gate: ``path`` must be a regular non-symlink file."""
    file_stat = path.lstat()
    if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"{description} must be a regular non-symlink file")


def _publish_poll_gated_raw_record(
    state: Any, *, job_id: str, result_payload: dict[str, Any]
) -> None:
    """Publish the polled job's own private raw record at its terminal poll.

    The worker persists exactly ``raw_results/<job_id>.json`` when the job
    completes, which is before any poll; that tree is qsim-private in
    separate-mode bundles. For the kinds in
    ``POLL_PUBLISHED_RAW_RECORD_KINDS``, copy the exact record bytes into
    ``public_job_results/raw_results/<job_id>.json`` while the first terminal
    ``get_job_result`` delivery is built and repoint the delivered
    ``data.raw_data_file`` at the published copy, so the raw record becomes
    agent-visible only through a completed poll.

    The source is bound to the POLLED job, never taken from the pointer
    (a qsim-owned pointer defect naming another
    job's record or another private artifact such as the hidden-truth snapshot
    must fail the poll BEFORE anything is read or copied into the
    agent-visible volume — refusing the response after publication would not
    undo a leak). The delivered pointer may hold only the worker-canonical
    private path or the already-published canonical path; anything else is
    refused with zero filesystem access. The single admissible source is the
    exact regular non-symlink file ``<log_dir>/raw_results/<job_id>.json``
    (lstat-checked before the first read), and the public target name is
    derived from the validated ``job_id``. The published copy is immutable: a
    matching existing target is reused byte-for-byte, and a mismatched one
    fails the poll as qsim-owned evidence corruption without being rewritten.
    """
    if result_payload.get("status") != "complete":
        return
    data = result_payload.get("data")
    if not isinstance(data, dict) or data.get("kind") not in POLL_PUBLISHED_RAW_RECORD_KINDS:
        return
    raw_relative = data.get("raw_data_file")
    if not isinstance(raw_relative, str) or not raw_relative:
        return
    log_dir = getattr(state, "log_dir", None)
    if log_dir is None:
        return
    expected_source_relative = f"{PUBLISHED_RAW_RECORD_SUBDIR}/{job_id}.json"
    published_relative = f"public_job_results/{PUBLISHED_RAW_RECORD_SUBDIR}/{job_id}.json"
    if raw_relative == published_relative:
        return
    if raw_relative != expected_source_relative:
        raise ValueError("raw record pointer does not name the polled job's own record")
    root = Path(log_dir).resolve()
    source = root / PUBLISHED_RAW_RECORD_SUBDIR / f"{job_id}.json"
    _require_regular_file(source, description="raw record source")
    payload = source.read_bytes()
    target = root / Path(published_relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(target):
        _require_regular_file(target, description="published raw record")
        if target.read_bytes() != payload:
            raise ValueError("published raw record no longer matches its private original")
    else:
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, target)
    data["raw_data_file"] = published_relative


def _verify_published_raw_record(
    state: Any, *, job_id: str, response_payload: dict[str, Any]
) -> None:
    """Re-verify a poll-published raw record on EVERY delivery, cache hits included.

    ``_publish_poll_gated_raw_record`` validates the copy only while a delivery
    is built, and the delivery cache returns without rebuilding, so a published
    file corrupted after the first poll would otherwise be served silently.
    Both paths are derived from the POLLED
    ``job_id``, never from the delivered pointer: the pointer must name
    exactly the polled job's published record, and the published bytes'
    SHA-256 must match the immutable qsim-private original the publication
    copied. Any mismatch, missing file, unreadable copy, or non-regular file
    fails the poll through the RESULT_ARTIFACT_PERSISTENCE qsim-owned
    disposition.
    """
    if response_payload.get("status") != "complete":
        return
    data = response_payload.get("data")
    if not isinstance(data, dict) or data.get("kind") not in POLL_PUBLISHED_RAW_RECORD_KINDS:
        return
    log_dir = getattr(state, "log_dir", None)
    if log_dir is None:
        return
    delivered_pointer = data.get("raw_data_file")
    if delivered_pointer is None:
        return
    published_relative = f"public_job_results/{PUBLISHED_RAW_RECORD_SUBDIR}/{job_id}.json"
    if delivered_pointer != published_relative:
        raise ValueError(
            "delivered raw record pointer does not name the polled job's published record"
        )
    root = Path(log_dir).resolve()
    published = root / Path(published_relative)
    original = root / PUBLISHED_RAW_RECORD_SUBDIR / f"{job_id}.json"
    _require_regular_file(published, description="published raw record")
    _require_regular_file(original, description="raw record source")
    published_digest = hashlib.sha256(published.read_bytes()).hexdigest()
    original_digest = hashlib.sha256(original.read_bytes()).hexdigest()
    if published_digest != original_digest:
        raise ValueError("published raw record no longer matches its private original")
    declared_digest = data.get("raw_data_sha256")
    if declared_digest is not None and declared_digest != original_digest:
        raise ValueError("raw record digest does not match its qsim-private original")


def _raw_result_data_evicted(state: Any, job_id: str) -> bool:
    evicted = getattr(getattr(state, "jobs", None), "result_data_evicted", None)
    return callable(evicted) and evicted(job_id)


def _load_persisted_result_payload(state: Any, *, job_id: str) -> dict[str, Any]:
    """Reread the exact persisted JobResult artifact for a data-evicted job.

    The spool is checked first: a job whose delivery was precomputed but never
    polled (or LRU-evicted before its first poll) still holds its artifact in
    the qsim-private spool, not in ``public_job_results/``.
    """
    log_dir = getattr(state, "log_dir", None)
    spool = _delivery_spool_path(log_dir, job_id=job_id)
    if spool.is_file():
        try:
            with spool.open("r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            # A concurrent poll promoted the spool entry between the check and
            # the open; the same bytes now live at the public destination.
            pass
    target, _ = _public_iq_result_path(log_dir, job_id=job_id)
    with target.open("r", encoding="utf-8") as f:
        return json.load(f)


def _materialize_job_result_delivery(
    state: Any, *, job_id: str, result
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Build or reuse one immutable terminal delivery and evidence pointer.

    Cached deliveries are canonical JSON strings: immutable by construction, so
    repeat polls parse a private copy instead of deep-copying shared objects,
    and their exact byte length bounds the cache. The build itself runs outside
    the lock -- concurrent builders (the worker-thread precompute and a first
    poll) converge on identical bytes because spool persistence is idempotent
    for equal content.

    Persistence lands in the qsim-private delivery spool, never directly in
    ``public_job_results/``: publication is the separate poll-gated
    ``_publish_spooled_job_result`` rename, so a completed-but-unpolled result
    is never agent-visible.
    """

    cache = getattr(state, "_job_result_delivery_cache", None)
    lock = getattr(state, "_job_result_delivery_lock", None)
    usable_cache = isinstance(cache, dict) and lock is not None

    if result.status == "complete" and usable_cache:
        with lock:
            cached = cache.pop(job_id, None)
            if cached is not None:
                # Reinsert to refresh LRU recency (dict preserves insert order).
                cache[job_id] = cached
        if cached is not None:
            response_payload, pointer, summary = json.loads(cached)
            return response_payload, pointer, summary

    if result.status == "complete" and _raw_result_data_evicted(state, job_id):
        # The raw bytes were dropped after a previous delivery was persisted
        # and cached, and the cache entry has since been LRU-evicted. The
        # published artifact holds the complete original payload; rebuilding
        # from it keeps repeat polls byte-identical with the first delivery.
        result_payload = _load_persisted_result_payload(state, job_id=job_id)
    else:
        result_payload = result.model_dump()
        # Repoint before the bounded build so the delivered response and the
        # persisted artifact carry the same published pointer; the retained
        # JobResult, the completion event, and the private evidence artifact
        # keep the original raw_results/ path. Raw-record kinds never
        # enter the worker-thread precompute (it skips them), so this
        # publication only ever runs while a terminal poll is being delivered.
        # The evicted-rebuild branch above needs no call: its persisted
        # artifact already carries the published pointer.
        _publish_poll_gated_raw_record(state, job_id=job_id, result_payload=result_payload)
    response_payload, pointer = _build_bounded_iq_result_payload(
        state, job_id=job_id, result_payload=result_payload, spool=True
    )
    summary = _bitstring_counts_summary(result_payload)
    delivery = (response_payload, pointer, summary)

    if result.status == "complete" and usable_cache and pointer is not None:
        entry = json.dumps(delivery, allow_nan=False, separators=(",", ":"))
        with lock:
            cache[job_id] = entry
            # Byte-budget LRU: evict oldest entries, always retaining the newest.
            while (
                len(cache) > 1
                and sum(map(len, cache.values())) > JOB_RESULT_DELIVERY_CACHE_MAX_BYTES
            ):
                cache.pop(next(iter(cache)))
        # Only after the artifact bytes are durably persisted (pointer: spool
        # or already-published public file) AND the delivery is cached is the
        # raw result data safe to drop from the retained job store.
        evict = getattr(getattr(state, "jobs", None), "evict_result_data", None)
        if callable(evict):
            evict(job_id)
    return delivery


def precompute_job_result_delivery(state: Any, *, job_id: str, result) -> bool:
    """Materialize, spool, and cache one completed delivery off the event loop.

    Invoked by ``JobManager`` from the worker thread BEFORE the terminal result
    is published, so serializing/persisting a large payload never runs on the
    transport event loop and a poll that observes ``complete`` always finds the
    warm cache. The artifact bytes land only in the qsim-private
    delivery spool here; publication into ``public_job_results/`` remains
    poll-gated (``_publish_spooled_job_result``), so a completed-but-unpolled
    result is never agent-visible. Returns True when the delivery is cached,
    telling the caller the raw data is safe to evict once stored. Exceptions
    propagate to the caller, which swallows them: the poll path rebuilds
    inline and surfaces persistence failures with its existing typed
    qsim-internal disposition.
    """

    if result.status != "complete":
        return False
    if getattr(getattr(result, "data", None), "kind", None) in POLL_PUBLISHED_RAW_RECORD_KINDS:
        # Poll-published raw-record kinds must not be materialized off
        # the poll path: their delivery build copies the private raw record
        # into the agent-visible tree and repoints the delivered payload,
        # which is admissible only while a terminal poll is being delivered.
        # Building here — at completion time, on the worker thread — would
        # publish an unpolled record. These deliveries build inline at poll.
        return False
    _materialize_job_result_delivery(state, job_id=job_id, result=result)
    cache = getattr(state, "_job_result_delivery_cache", None)
    lock = getattr(state, "_job_result_delivery_lock", None)
    if not isinstance(cache, dict) or lock is None:
        return False
    with lock:
        return job_id in cache


def reserve_job_admission(state: Any, job_id: str) -> None:
    """Atomically claim a queue slot before any budget or evidence side effect.

    Unlike the retired best-effort capacity pre-check, the reservation counts
    toward the admission cap immediately, so a concurrent submit can never
    fill the queue between this call and ``jobs.submit`` -- the typed
    ``JobQueueFullError`` fires here, before anything is logged or reserved,
    and ``submit`` consumes the slot without re-checking. Pair with
    :func:`release_job_admission` on every later pre-submit failure path.
    Test doubles without ``reserve_admission`` skip this seam; the binding
    enforcement remains ``JobManager.submit``.
    """

    reserve = getattr(getattr(state, "jobs", None), "reserve_admission", None)
    if callable(reserve):
        reserve(job_id)


def release_job_admission(state: Any, job_id: str) -> None:
    """Roll back an unconsumed admission reservation (idempotent, guard-safe)."""

    release = getattr(getattr(state, "jobs", None), "release_admission", None)
    if callable(release):
        release(job_id)


def _log_metadata_call(ctx: ActionContext, event: dict[str, Any]) -> None:
    """Reserve and publish one bounded public-metadata call when configured."""

    budget = getattr(getattr(ctx.state, "public", None), "budget", None)
    limit = getattr(budget, "max_metadata_calls", None)
    if limit is None:
        ctx.state.log(event)
        return
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise RuntimeError("public metadata-call resource policy is invalid")
    reserve = getattr(ctx.state, "reserve_action_budgets", None)
    if not callable(reserve):
        raise RuntimeError("active qsim state cannot enforce the public metadata-call budget")
    _, used = reserve(reservations={"metadata_calls": (1, limit)})["metadata_calls"]
    try:
        ctx.state.log(
            {
                **event,
                "metadata_calls_used": used,
                "max_metadata_calls": limit,
            }
        )
    except Exception:
        _best_effort_log_metadata_failure(
            ctx,
            tool=str(event.get("tool", "")),
            failed_action=str(event.get("action", "")),
        )
        raise RuntimeError(METADATA_LOG_PUBLICATION_ERROR) from None


def list_devices(ctx: ActionContext) -> list[str]:
    """Return device IDs hosted by this qsim state."""
    _log_metadata_call(
        ctx,
        {"surface": ctx.surface, "action": "list_devices", "tool": "list_devices"},
    )
    return [ctx.state.public.device_id]


def get_device_spec(ctx: ActionContext, *, device_id: str) -> dict[str, Any]:
    """Return the public hardware spec for ``device_id``."""
    if device_id != ctx.state.public.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    _log_metadata_call(
        ctx,
        {
            "surface": ctx.surface,
            "action": "get_device_spec",
            "tool": "get_device_spec",
            "device_id": device_id,
        },
    )
    return ctx.state.public.model_dump()


def get_lab_notebook(ctx: ActionContext, *, device_id: str) -> dict[str, Any]:
    """Return the lab notebook for ``device_id``."""
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    _log_metadata_call(
        ctx,
        {
            "surface": ctx.surface,
            "action": "get_lab_notebook",
            "tool": "get_lab_notebook",
            "device_id": device_id,
        },
    )
    return ctx.state.notebook.model_dump()


def get_job_result(ctx: ActionContext, *, job_id: str) -> dict[str, Any]:
    """Poll a qsim job and return its wire-model payload."""
    if not isinstance(job_id, str) or _SAFE_JOB_ID.fullmatch(job_id) is None:
        raise ValueError("job_id must be a safe filename component")
    public_poll_limit = _public_job_result_poll_limit(ctx)
    poll_budget_evidence: tuple[int, int, int] | None = None
    reserve_poll = getattr(ctx.state, "reserve_job_result_poll", None)
    if public_poll_limit is not None:
        peek = getattr(ctx.state.jobs, "peek", None)
        if not callable(reserve_poll) or not callable(peek):
            _best_effort_log_result_failure(
                ctx,
                job_id=job_id,
                status="unknown",
                failure_stage=RESULT_POLL_BUDGET_STAGE,
            )
            raise RuntimeError(RESULT_POLL_BUDGET_ERROR)
        current = peek(job_id)
        if current is None:
            raise ValueError(f"Unknown job_id {job_id!r}")
    try:
        if callable(reserve_poll):
            poll_budget_evidence = reserve_poll(job_id)
    except ValueError:
        # Public budget exhaustion is a synchronous model-owned rejection.
        raise
    except Exception:
        _best_effort_log_result_failure(
            ctx,
            job_id=job_id,
            status="unknown",
            failure_stage=RESULT_POLL_BUDGET_STAGE,
        )
        raise RuntimeError(RESULT_POLL_BUDGET_ERROR) from None
    if public_poll_limit is not None and (
        not isinstance(poll_budget_evidence, tuple)
        or len(poll_budget_evidence) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in poll_budget_evidence
        )
        or poll_budget_evidence[0] < 0
        or poll_budget_evidence[1] != poll_budget_evidence[0] + 1
        or poll_budget_evidence[2] != public_poll_limit
        or poll_budget_evidence[1] > poll_budget_evidence[2]
    ):
        _best_effort_log_result_failure(
            ctx,
            job_id=job_id,
            status="unknown",
            failure_stage=RESULT_POLL_BUDGET_STAGE,
        )
        raise RuntimeError(RESULT_POLL_BUDGET_ERROR)
    result = ctx.state.jobs.get(job_id)
    if result is None:
        if poll_budget_evidence is not None:
            _best_effort_log_result_failure(
                ctx,
                job_id=job_id,
                status="unknown",
                failure_stage=RESULT_POLL_BUDGET_STAGE,
            )
            raise RuntimeError(RESULT_POLL_BUDGET_ERROR)
        raise ValueError(f"Unknown job_id {job_id!r}")
    if result.status in {"queued", "running"} and hasattr(ctx.state.backend, "poll_job_result"):
        updated = ctx.state.backend.poll_job_result(job_id, result)
        if updated is not None:
            ctx.state.jobs.set_result(job_id, updated)
            result = updated
    try:
        response_payload, offloaded_result, summary = _materialize_job_result_delivery(
            ctx.state, job_id=job_id, result=result
        )
        if offloaded_result is not None:
            # First terminal poll: copy the spooled artifact bytes into the
            # agent-visible destination volume and link them into place.
            # Publication is gated on this poll, never on completion (see
            # _DELIVERY_SPOOL_DIRNAME); the copy is the bounded artifact
            # bytes, once per job, and crosses the volume boundary that a
            # rename cannot (EXDEV).
            _publish_spooled_job_result(ctx.state, job_id=job_id)
        _verify_published_raw_record(ctx.state, job_id=job_id, response_payload=response_payload)
    except Exception:
        # Artifact publication is part of a completed poll's qsim-owned
        # evidence graph. Preserve a sanitized, machine-readable marker before
        # surfacing the error so a verifier cannot mistake the missing poll for
        # model behavior. Exception text may contain private filesystem paths.
        _best_effort_log_result_failure(
            ctx,
            job_id=job_id,
            status=result.status,
            failure_stage=RESULT_ARTIFACT_PERSISTENCE_STAGE,
        )
        raise RuntimeError(RESULT_ARTIFACT_PERSISTENCE_ERROR) from None
    event = {
        "surface": ctx.surface,
        "action": "get_job_result",
        "tool": "get_job_result",
        "job_id": job_id,
        "status": result.status,
    }
    if poll_budget_evidence is not None:
        _, used_after, limit = poll_budget_evidence
        event["job_result_polls_used"] = used_after
        event["max_job_result_polls"] = limit
    if offloaded_result is not None:
        event["offloaded_result"] = offloaded_result
    if result.status == "failed" and result.error:
        # The agent already receives this public error in the JobResult.  Retain
        # it in qsim-owned evidence so verifiers can distinguish an explicit
        # model rejection from a typed runtime/evidence failure.
        event["error"] = result.error
        if is_internal_job_failure(result.error):
            event["failure_kind"] = QSIM_INTERNAL_FAILURE_KIND
    if summary is not None:
        event["data_summary"] = summary
    try:
        ctx.state.log(event)
    except Exception:
        _best_effort_log_result_failure(
            ctx,
            job_id=job_id,
            status=result.status,
            failure_stage=RESULT_POLL_LOG_PUBLICATION_STAGE,
        )
        raise RuntimeError(RESULT_POLL_LOG_PUBLICATION_ERROR) from None
    return response_payload


def file_submission_enabled() -> bool:
    """Whether this deployment gives the agent the submission directory.

    ``QSIM_SUBMISSIONS_DIR`` is the only runtime determinant. qsim cannot read
    the compose file, and a mounted ``submission`` volume alone is not the file
    transport: three of the six bundles that mount it use it purely as a static
    code-submission surface and never point qsim at it.

    MCP tool registration reads this so a deployment without the
    directory never advertises ``answer_file``/``answer_sha256`` at all, rather
    than advertising a route it would refuse -- and refuse in words that would
    tell the agent it is running inside a configured harness. The action-layer
    guard below therefore stops being agent-reachable and stays purely as
    fail-closed defense in depth for non-MCP callers.
    """
    return bool(os.environ.get(SUBMISSIONS_DIR_ENV))


def _load_final_answer_file(
    answer_file: str, answer_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load an agent-written answer file from the submission directory.

    Fail-closed ingress: every rejection is a plain ValueError (an atomic tool
    rejection consuming no budget). Acceptance returns the parsed answer dict
    plus the transport evidence, which the caller returns to the agent but never
    persists (the evidence log stays byte-identical to an inline submission).
    """
    submissions_root = os.environ.get(SUBMISSIONS_DIR_ENV)
    if not submissions_root:
        # Unreachable from MCP once registration is gated on
        # ``file_submission_enabled``. Says nothing about tasks, mounts, or
        # configuration: a rejection is not an occasion to describe the rig.
        raise ValueError("answer_file is not accepted; pass the answer inline instead")
    if not isinstance(answer_file, str) or _SAFE_ANSWER_FILE_NAME.fullmatch(answer_file) is None:
        raise ValueError(
            "answer_file must be a single filename directly in the submission "
            "directory (letters, digits, '_', '-', and non-leading '.'; no '/')"
        )
    if not isinstance(answer_sha256, str) or _ANSWER_SHA256_HEX.fullmatch(answer_sha256) is None:
        raise ValueError("answer_sha256 must be 64 lowercase hex characters")
    root = Path(submissions_root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError("the submission directory is unavailable")

    # Open once, then use only the fd. O_NOFOLLOW rejects a symlink at this name
    # rather than following it (the agent owns this volume and could point it at
    # hidden truth), and using the fd for stat + read means no agent-controlled
    # pathname is ever re-traversed after this point -- no TOCTOU window. The
    # mount root itself is a validated non-symlink directory and, being a volume
    # mount point, is not agent-replaceable.
    try:
        # O_NONBLOCK so a non-regular name (a FIFO the agent created, a device
        # node) never blocks the open waiting for a writer -- it returns at once
        # and is then rejected by the S_ISREG check below. O_NONBLOCK is a no-op
        # for reads on the regular file this must be.
        fd = os.open(
            os.path.join(str(root), answer_file),
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0),
        )
    except OSError:
        # ENOENT (missing), ELOOP (a symlink at the name), or similar.
        raise ValueError(
            f"answer_file {answer_file!r} is not a readable regular file in the "
            "submission directory"
        ) from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ValueError("answer_file must name a regular file")
        if st.st_size > MAX_ANSWER_FILE_BYTES:
            raise ValueError(
                f"answer_file exceeds the transport limit of {MAX_ANSWER_FILE_BYTES} bytes"
            )
        chunks: list[bytes] = []
        remaining = MAX_ANSWER_FILE_BYTES + 1
        while remaining > 0:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    if len(raw) > MAX_ANSWER_FILE_BYTES:
        raise ValueError(
            f"answer_file exceeds the transport limit of {MAX_ANSWER_FILE_BYTES} bytes"
        )
    digest = hashlib.sha256(raw).hexdigest()
    if digest != answer_sha256:
        raise ValueError(
            "answer_sha256 does not match the file content; finish writing the "
            "file, recompute the hash over its exact bytes, and retry"
        )
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, RecursionError, json.JSONDecodeError):
        raise ValueError("answer_file must contain valid UTF-8 JSON") from None
    if not isinstance(parsed, dict):
        raise ValueError("answer_file must contain a single JSON object (the answer dict)")
    transport_evidence = {
        "answer_transport": "file",
        "answer_file": answer_file,
        "answer_file_bytes": len(raw),
        "answer_file_sha256": digest,
    }
    return parsed, transport_evidence


def submit_final_answer(
    ctx: ActionContext,
    *,
    task_id: str,
    answer: dict[str, Any] | None = None,
    answer_file: str | None = None,
    answer_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate and persist the task final answer."""
    if ctx.state.task_id is not None and task_id != ctx.state.task_id:
        raise ValueError(
            f"Final answer task_id {task_id!r} does not match active task {ctx.state.task_id!r}"
        )
    transport_evidence: dict[str, Any] | None = None
    if answer_file is not None:
        if answer is not None:
            raise ValueError("pass exactly one of answer or answer_file, not both")
        if answer_sha256 is None:
            raise ValueError(
                "answer_file requires answer_sha256 (hex SHA-256 of the exact file bytes)"
            )
        answer, transport_evidence = _load_final_answer_file(answer_file, answer_sha256)
    elif answer is None:
        raise ValueError("pass exactly one of answer or answer_file")
    elif answer_sha256 is not None:
        raise ValueError("answer_sha256 is only meaningful together with answer_file")
    resource_budget = _public_final_answer_budget(ctx)
    if resource_budget is not None:
        # Bound iterative string walking and finite compact serialization before
        # Pydantic's recursive FinalAnswer validator sees model-controlled
        # nesting. This turns an oversized/deep request into a normal tool
        # rejection instead of a recursion or memory failure.
        _validate_answer_string_resources(
            answer,
            max_characters=resource_budget.max_answer_string_characters,
            max_depth=getattr(resource_budget, "max_final_answer_nesting_depth", None),
        )
        prevalidated_payload = compact_final_answer_bytes(
            {"schema_version": 2, "task_id": task_id, "answer": answer}
        )
        if len(prevalidated_payload) > resource_budget.max_final_answer_serialized_bytes:
            raise ValueError(
                "compact UTF-8 FinalAnswer envelope uses "
                f"{len(prevalidated_payload)} bytes; public limit is "
                f"{resource_budget.max_final_answer_serialized_bytes}"
            )
    final_answer = FinalAnswer.model_validate({"task_id": task_id, "answer": answer})
    payload = final_answer.model_dump()
    evaluation_budget = _active_vqe_evaluation_budget(ctx, task_id=task_id)
    serialized_payload: bytes | None = None
    if resource_budget is not None:
        _validate_answer_string_resources(
            payload["answer"],
            max_characters=resource_budget.max_answer_string_characters,
            max_depth=getattr(resource_budget, "max_final_answer_nesting_depth", None),
        )
        serialized_payload = compact_final_answer_bytes(payload)
        if len(serialized_payload) > resource_budget.max_final_answer_serialized_bytes:
            raise ValueError(
                "compact UTF-8 FinalAnswer envelope uses "
                f"{len(serialized_payload)} bytes; public limit is "
                f"{resource_budget.max_final_answer_serialized_bytes}"
            )
    # Task-owned public structural diagnostics run here, after the answer is a
    # bounded validated dict and before any submission budget is reserved, so a
    # refusal is an ordinary atomic tool rejection.
    task_precheck = _task_final_answer_precheck(ctx, task_id=task_id, answer=payload["answer"])
    if evaluation_budget is not None:
        _validate_answer_string_resources(
            answer,
            max_characters=evaluation_budget.max_answer_string_characters,
            max_job_id_characters=evaluation_budget.max_citation_job_id_characters,
        )
        serialized_payload = compact_final_answer_bytes(payload)
        if len(serialized_payload) > evaluation_budget.max_final_answer_serialized_bytes:
            raise ValueError(
                "compact UTF-8 FinalAnswer envelope uses "
                f"{len(serialized_payload)} bytes; public limit is "
                f"{evaluation_budget.max_final_answer_serialized_bytes}"
            )
        ctx.state.reserve_action_budgets(
            reservations={
                f"digital_vqe:{task_id}:final_answer_submissions": (
                    1,
                    evaluation_budget.max_final_answer_submissions,
                )
            }
        )
    event = {
        "surface": ctx.surface,
        "action": "submit_final_answer",
        "tool": "submit_final_answer",
        "task_id": task_id,
        "answer": payload["answer"],
    }
    with ctx.state._final_answer_lock:
        if resource_budget is not None:
            submission_budget_evidence = ctx.state.reserve_action_budgets(
                reservations={
                    "final_answer_submissions": (
                        1,
                        resource_budget.max_final_answer_submissions,
                    )
                }
            )["final_answer_submissions"]
            event.update(
                {
                    "final_answer_serialized_bytes": len(serialized_payload),
                    "final_answer_submissions_used": submission_budget_evidence[1],
                    "max_final_answer_submissions": resource_budget.max_final_answer_submissions,
                }
            )
        try:
            if ctx.state.log_dir is not None:
                path = ctx.state.log_dir
                path.mkdir(parents=True, exist_ok=True)
                target = path / "final_answer.json"
                temporary: Path | None = None
                try:
                    with NamedTemporaryFile(
                        "wb" if serialized_payload is not None else "w",
                        encoding=None if serialized_payload is not None else "utf-8",
                        dir=path,
                        prefix=".final_answer.",
                        suffix=".tmp",
                        delete=False,
                    ) as f:
                        temporary = Path(f.name)
                        if serialized_payload is not None:
                            f.write(serialized_payload)
                        else:
                            json.dump(payload, f, indent=2, allow_nan=False)
                            f.write("\n")
                    # ``NamedTemporaryFile`` is owner-only by default.  The final
                    # answer is a verifier-facing run artifact, so retain the
                    # historical cross-container readability of the old direct
                    # writer before atomically publishing it.
                    temporary.chmod(0o644)
                    temporary.replace(target)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
        except Exception:
            _best_effort_log_final_answer_failure(
                ctx,
                task_id=task_id,
                failure_stage=FINAL_ANSWER_PERSISTENCE_STAGE,
            )
            raise RuntimeError(FINAL_ANSWER_PERSISTENCE_ERROR) from None
        try:
            ctx.state.log(event)
        except Exception:
            _best_effort_log_final_answer_failure(
                ctx,
                task_id=task_id,
                failure_stage=FINAL_ANSWER_LOG_PUBLICATION_STAGE,
            )
            raise RuntimeError(FINAL_ANSWER_LOG_PUBLICATION_ERROR) from None
    # The evidence log and final_answer.json above are byte-identical to an
    # inline submission of the same dict. Transport provenance and the task
    # precheck's diagnostics are returned to the agent only -- never persisted,
    # so no verifier (allowlist or filter) can tell a file submission from an
    # inline one, or a prechecked submission from a plain one.
    response: dict[str, Any] = {"accepted": True, "task_id": task_id}
    if transport_evidence is not None:
        response.update(transport_evidence)
    if task_precheck is not None:
        response["manifest_structure"] = task_precheck
    return response
