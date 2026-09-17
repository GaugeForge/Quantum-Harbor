"""Action seam for the ``neutral_atom_logical_processor`` qtype.

Routes ``run_atom_program`` into the active backend (async job model) and serves the
synchronous, free ``validate_schedule`` legality/timing helper. Capability-gated and
fail-closed at this layer (defense in depth alongside MCP registration). The per-call log
records the program shape (op count + digest + shots + budget) so the verifier can bind the
agent's declared schedule to qsim-owned evidence. The bounded JSONL event references a
content-addressed qsim artifact containing the exact validated request and lossless raw
shot result; the same raw result still goes to the agent through the normal job response.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.evidence import persist_public_raw_record
from qiqcbench.qsim.hidden_commitment import (
    public_device_model_commitment as task_bound_public_device_model_commitment,
)
from qiqcbench.qsim.qtypes.neutral_atom_logical_processor.capabilities.located_erasure_repair.runtime import (
    controller_digest,
    episode_request_digest,
)
from qiqcbench.qsim.qtypes.neutral_atom_logical_processor.capabilities.located_erasure_repair.wire import (
    ErasureRepairEpisodeRequest,
    JobErasureRepairEpisodeData,
)
from qiqcbench.qsim.qtypes.neutral_atom_logical_processor.evidence import (
    MAX_NEUTRAL_ATOM_RESULT_ARTIFACT_BYTES,
    NEUTRAL_ATOM_EVIDENCE_CONTRACT,
    NEUTRAL_ATOM_EVIDENCE_SCHEMA_VERSION,
    NEUTRAL_ATOM_INFRASTRUCTURE_FAILURE,
    NeutralAtomResultArtifact,
    canonical_json_bytes,
    hidden_device_model_commitment,
    persist_result_artifact,
    program_digest,
    program_measurement_labels,
    public_device_model_commitment,
)
from qiqcbench.qsim.qtypes.neutral_atom_logical_processor.wire import (
    AtomProgramRequest,
    JobAtomShotData,
)

_CAP = "rydberg_logical_control"
_REPAIR_CAP = "located_erasure_repair"
_GATES = {"h", "s", "sdg", "x", "y", "z", "rx", "ry", "rz"}


def _require(ctx: ActionContext, device_id: str) -> None:
    if _CAP not in ctx.state.active_capabilities:
        raise ValueError(f"{_CAP} capability is not active for this run")
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")


def _evidence_identity(ctx: ActionContext) -> dict[str, Any]:
    execution_context_id = getattr(ctx.state, "execution_context_id", None)
    commitment_secret = getattr(ctx.state, "hidden_commitment_secret", None)
    task_id = getattr(ctx.state, "task_id", None) or ctx.state.public.task_id
    identity = {
        "evidence_contract": NEUTRAL_ATOM_EVIDENCE_CONTRACT,
        "evidence_schema_version": NEUTRAL_ATOM_EVIDENCE_SCHEMA_VERSION,
        "public_device_model_commitment": public_device_model_commitment(ctx.state.public),
    }
    if execution_context_id is not None:
        if commitment_secret is None:
            raise RuntimeError("execution-bound atom evidence requires a hidden commitment secret")
        if not task_id:
            raise RuntimeError("execution-bound atom evidence requires an active task ID")
        identity["hidden_device_model_commitment"] = hidden_device_model_commitment(
            ctx.state.hidden,
            task_id=task_id,
            commitment_secret=commitment_secret,
        )
    return identity


def _validate_result_capacity(ctx: ActionContext, request: AtomProgramRequest) -> None:
    if len(request.ops) > ctx.state.public.budgets.max_ops:
        raise ValueError(
            f"program has {len(request.ops)} ops > max_ops {ctx.state.public.budgets.max_ops}"
        )
    if request.shots > ctx.state.public.budgets.max_shots_per_call:
        raise ValueError(
            f"shots {request.shots} exceeds max_shots_per_call "
            f"{ctx.state.public.budgets.max_shots_per_call}"
        )
    labels = program_measurement_labels(request.ops)
    recorded_bits = request.shots * len(labels)
    public_limit = ctx.state.public.budgets.max_recorded_bits_per_call
    if recorded_bits > public_limit:
        raise ValueError(
            "raw result exceeds max_recorded_bits_per_call: "
            f"{recorded_bits} > {public_limit}; split the experiment into smaller jobs"
        )
    packed_bytes = (recorded_bits + 7) // 8
    encoded_bytes_per_field = 4 * ((packed_bytes + 2) // 3)
    metadata_bytes = len(canonical_json_bytes(request.model_dump(mode="json"))) + len(
        canonical_json_bytes(labels)
    )
    conservative_artifact_bytes = 2 * encoded_bytes_per_field + metadata_bytes + 4096
    if conservative_artifact_bytes > MAX_NEUTRAL_ATOM_RESULT_ARTIFACT_BYTES:
        raise ValueError(
            "raw result would exceed the neutral-atom evidence artifact limit; "
            "split the experiment into smaller jobs"
        )


def _infrastructure_failure_result(
    ctx: ActionContext,
    *,
    job_id: str,
    device_id: str,
    program_digest_value: str,
    identity: dict[str, Any],
    stage: str,
) -> JobResult:
    """Return and best-effort record one non-model qsim terminal failure."""

    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": "atom_program_infrastructure_failure",
                "tool": "run_atom_program",
                "device_id": device_id,
                "job_id": job_id,
                "program_digest": program_digest_value,
                "failure_class": NEUTRAL_ATOM_INFRASTRUCTURE_FAILURE,
                "failure_stage": stage,
                **identity,
            }
        )
    except Exception:
        # The failed JobResult carries the same stable public code. A later poll
        # can still persist that classification if this particular log write was
        # the failing operation.
        pass
    return JobResult(
        job_id=job_id,
        device_id=device_id,
        status="failed",
        error=NEUTRAL_ATOM_INFRASTRUCTURE_FAILURE,
    )


def submit_atom_program(
    ctx: ActionContext, *, device_id: str, ops: list[dict], shots: int, layout: list[int] | None
) -> dict[str, Any]:
    _require(ctx, device_id)
    request = AtomProgramRequest.model_validate({"ops": ops, "shots": shots, "layout": layout})
    _validate_result_capacity(ctx, request)
    identity = _evidence_identity(ctx)
    reserve = getattr(ctx.state.backend, "reserve_atom_program", None)
    reservation_token = reserve(request) if callable(reserve) else None
    salt = ctx.state.next_salt()
    dig = program_digest(request.ops)
    submission_logged = threading.Event()

    def runner(job_id: str) -> JobResult:
        submission_logged.wait()
        try:
            if reservation_token is None:
                result = ctx.state.backend.run_atom_program(request, job_id, salt)
            else:
                result = ctx.state.backend.run_atom_program(
                    request,
                    job_id,
                    salt,
                    reservation_token=reservation_token,
                )
        except Exception:
            return _infrastructure_failure_result(
                ctx,
                job_id=job_id,
                device_id=device_id,
                program_digest_value=dig,
                identity=identity,
                stage="backend_execution",
            )
        data = result.data
        if isinstance(data, JobAtomShotData):
            try:
                event = {
                    "surface": ctx.surface,
                    "action": "atom_program_result",
                    "tool": "run_atom_program",
                    "device_id": device_id,
                    "job_id": job_id,
                    "status": result.status,
                    "n_ops": len(ops),
                    "program_digest": dig,
                    "layout": request.layout,
                    "shots": data.shots,
                    "n_recorded": data.n_recorded,
                    "measure_labels": data.measure_labels,
                    **identity,
                }
                log_dir = getattr(ctx.state, "log_dir", None)
                if log_dir is not None:
                    artifact = NeutralAtomResultArtifact(
                        job_id=job_id,
                        device_id=device_id,
                        program_digest=dig,
                        public_device_model_commitment=identity["public_device_model_commitment"],
                        hidden_device_model_commitment=identity.get(
                            "hidden_device_model_commitment"
                        ),
                        request=request,
                        result_status=result.status,
                        result=data,
                    )
                    event["result_artifact"] = persist_result_artifact(
                        log_dir,
                        artifact,
                    ).model_dump(mode="json")
                ctx.state.log(event)
            except Exception:
                return _infrastructure_failure_result(
                    ctx,
                    job_id=job_id,
                    device_id=device_id,
                    program_digest_value=dig,
                    identity=identity,
                    stage="evidence_recording",
                )
        return result

    try:
        job_id = ctx.state.jobs.submit(runner)
    except Exception:
        discard = getattr(ctx.state.backend, "discard_atom_program_reservation", None)
        if reservation_token is not None and callable(discard):
            discard(reservation_token)
        raise
    try:
        ctx.state.log(
            {
                "action": "submit_atom_program",
                "job_id": job_id,
                "surface": ctx.surface,
                "device_id": device_id,
                "n_ops": len(ops),
                "program_digest": dig,
                "layout": request.layout,
                "shots": shots,
                **identity,
            }
        )
    finally:
        submission_logged.set()
    return {"job_id": job_id, "status": "queued"}


def validate_schedule(
    ctx: ActionContext, *, device_id: str, ops: list[dict], layout: list[int] | None
) -> dict[str, Any]:
    """SYNC + FREE: legality verdict + nominal serial schedule duration (no shots, no fidelity)."""
    _require(ctx, device_id)
    pub = ctx.state.public
    n = pub.n_atoms
    dur = pub.durations_us
    reasons: list[str] = []
    if len(ops) > pub.budgets.max_ops:
        reasons.append(f"{len(ops)} ops > max_ops {pub.budgets.max_ops}")
    total_us = 0.0

    def _check_op(op: dict, label: str) -> float:
        """Append legality reasons for one op (recursing into feedforward `then`).

        Returns the op's nominal serial duration contribution; a conditioned op
        contributes 0 (its execution is data-dependent) but is fully legality-checked
        so the nested `then` cannot bypass the same rules the engine enforces.
        """
        t = op.get("type")
        if t == "gate":
            g = op.get("gate")
            if g not in _GATES:
                reasons.append(f"op {label}: unknown gate {g!r}")
            for a in op.get("atoms") or []:
                if not (0 <= a < n):
                    reasons.append(f"op {label}: atom {a} out of range [0,{n})")
            return dur.rotation
        if t == "cz":
            pair = op.get("pair") or []
            if len(pair) != 2 or pair[0] == pair[1] or not all(0 <= a < n for a in pair):
                reasons.append(f"op {label}: bad cz pair {pair}")
            return dur.cz
        if t == "init":
            # `init` prepares a single `atom` or, as a convenience, a batch `atoms`. Check each
            # target in range. NB: use an explicit None test, not `atom or -1` — atom 0 is a
            # legal (and falsy) index, so the `or` fallback would spuriously flag it out of range.
            a = op.get("atom")
            targets = [a] if a is not None else list(op.get("atoms") or [])
            if not targets:
                reasons.append(f"op {label}: init op requires 'atom' or 'atoms'")
            for k in targets:
                if not (isinstance(k, int) and not isinstance(k, bool) and 0 <= k < n):
                    reasons.append(f"op {label}: init atom {k} out of range [0,{n})")
            return 0.0
        if t == "measure":
            for a in op.get("atoms") or []:
                if not (0 <= a < n):
                    reasons.append(f"op {label}: measure atom {a} out of range")
            return dur.readout
        if t == "move":
            a = op.get("atom")
            if not (isinstance(a, int) and not isinstance(a, bool) and 0 <= a < n):
                reasons.append(f"op {label}: move atom {a} out of range [0,{n})")
            return dur.move_per_um * 10.0
        if t == "feedforward":
            then = op.get("then")
            if op.get("on_bit") is None or op.get("value") is None or then is None:
                reasons.append(f"op {label}: feedforward requires 'on_bit', 'value', and 'then'")
            elif isinstance(then, dict):
                _check_op(then, f"{label}.then")
            else:
                reasons.append(f"op {label}: feedforward 'then' must be an op object")
            return 0.0
        reasons.append(f"op {label}: unknown op type {t!r}")
        return 0.0

    for i, op in enumerate(ops):
        total_us += _check_op(op, str(i))
    return {
        "legal": not reasons,
        "reasons": reasons,
        "schedule_duration_us": round(total_us, 3),
        "n_ops": len(ops),
        "note": "duration is a serial nominal estimate; it drives decoherence/loss, not a score",
    }


def submit_located_erasure_control_episodes(
    ctx: ActionContext,
    *,
    device_id: str,
    horizon: int,
    episodes: int,
    controller: dict,
) -> dict[str, Any]:
    """Submit a bounded causal controller for raw role-level episodes."""
    if _REPAIR_CAP not in ctx.state.active_capabilities:
        raise ValueError(f"{_REPAIR_CAP} capability is not active for this run")
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    request = ErasureRepairEpisodeRequest.model_validate(
        {
            "horizon": horizon,
            "episodes": episodes,
            "controller": controller,
        }
    )
    repair = ctx.state.public.erasure_repair
    if repair is None:
        raise ValueError("located-erasure control is not configured for this device")
    if request.episodes > repair.max_episodes_per_call:
        raise ValueError(
            f"episodes exceeds public max_episodes_per_call ({repair.max_episodes_per_call})"
        )
    digest = controller_digest(request.controller)
    request_digest = episode_request_digest(request)
    salt = ctx.state.next_salt()
    reserve = getattr(ctx.state.backend, "reserve_erasure_repair_episodes", None)
    reservation_token = reserve(request) if callable(reserve) else None
    task_id = getattr(ctx.state, "task_id", None) or ctx.state.public.task_id
    if not task_id:
        raise RuntimeError("located-erasure evidence requires an active task ID")
    public_commitment = task_bound_public_device_model_commitment(
        ctx.state.public,
        task_id=task_id,
    )
    execution_context_id = getattr(ctx.state, "execution_context_id", None)
    submission_logged = threading.Event()

    def infrastructure_failure(job_id: str, *, stage: str) -> JobResult:
        try:
            ctx.state.log(
                {
                    "surface": ctx.surface,
                    "action": "located_erasure_control_infrastructure_failure",
                    "tool": "run_located_erasure_control_episodes",
                    "device_id": device_id,
                    "job_id": job_id,
                    "request_digest": request_digest,
                    "failure_class": NEUTRAL_ATOM_INFRASTRUCTURE_FAILURE,
                    "failure_stage": stage,
                    "public_device_model_commitment": public_commitment,
                }
            )
        except Exception:
            pass
        return JobResult(
            job_id=job_id,
            device_id=device_id,
            status="failed",
            error=NEUTRAL_ATOM_INFRASTRUCTURE_FAILURE,
        )

    def runner(job_id: str) -> JobResult:
        submission_logged.wait()
        try:
            result = ctx.state.backend.run_located_erasure_control_episodes(
                request,
                job_id,
                salt,
                reservation_token=reservation_token,
            )
        except Exception:
            return infrastructure_failure(job_id, stage="backend_execution")
        if isinstance(result.data, JobErasureRepairEpisodeData):
            try:
                if result.data.request_digest != request_digest:
                    raise RuntimeError("backend result is not bound to the accepted request")
                raw_record = {
                    "artifact_kind": "located_erasure_control_episode_record_v1",
                    "schema_version": 1,
                    "complete": True,
                    "kind": result.data.kind,
                    "task_id": task_id,
                    "device_id": device_id,
                    "job_id": job_id,
                    "tool": "run_located_erasure_control_episodes",
                    "execution_context_id": execution_context_id,
                    "public_device_model_commitment": public_commitment,
                    "request_digest": request_digest,
                    "request": request.model_dump(mode="json"),
                    **result.data.model_dump(
                        mode="json",
                        include={
                            "horizon",
                            "episodes",
                            "controller_digest",
                            "raw_schema_version",
                            "trace_encoding",
                            "trace_payload_b64",
                        },
                    ),
                }
                raw_data_file = persist_public_raw_record(
                    ctx.state.log_dir,
                    job_id=job_id,
                    record=raw_record,
                )
                if raw_data_file is None:
                    raise RuntimeError("scored qsim run has no raw evidence directory")
                raw_path = ctx.state.log_dir / raw_data_file
                raw_bytes = raw_path.read_bytes()
                result.data = result.data.model_copy(
                    update={
                        "trace_payload_b64": None,
                        "raw_data_file": raw_data_file,
                        "raw_data_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                        "raw_data_size_bytes": len(raw_bytes),
                    }
                )
                ctx.state.log(
                    {
                        "surface": ctx.surface,
                        "action": "located_erasure_control_episode_result",
                        "tool": "run_located_erasure_control_episodes",
                        "device_id": device_id,
                        "job_id": job_id,
                        "request_digest": request_digest,
                        "controller_digest": digest,
                        "public_device_model_commitment": public_commitment,
                        "data": result.data.model_dump(mode="json"),
                    }
                )
            except Exception:
                return infrastructure_failure(job_id, stage="evidence_recording")
        return result

    try:
        job_id = ctx.state.jobs.submit(runner)
    except Exception:
        discard = getattr(ctx.state.backend, "discard_erasure_repair_reservation", None)
        if reservation_token is not None and callable(discard):
            discard(reservation_token, request)
        raise
    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": "submit_located_erasure_control_episodes",
                "tool": "run_located_erasure_control_episodes",
                "device_id": device_id,
                "job_id": job_id,
                "horizon": horizon,
                "episodes": episodes,
                "controller": request.controller.model_dump(mode="json"),
                "controller_digest": digest,
                "request_digest": request_digest,
                "public_device_model_commitment": public_commitment,
            }
        )
    finally:
        submission_logged.set()
    return {
        "job_id": job_id,
        "status": "queued",
        "controller_digest": digest,
        "request_digest": request_digest,
    }


__all__ = [
    "submit_atom_program",
    "submit_located_erasure_control_episodes",
    "validate_schedule",
]
