#!/usr/bin/env python3
"""Verifier driver for ``adaptive_clustered_clbcs_h2o`` (rubric mode).

Usage: ``score_clbcs.py <experiment_log.jsonl> <final_answer.json>
<artifacts_dir> [reward_path] [trial_root]``

Recomputes every scored quantity from completed qsim evidence — Track-A V_Haar
from the submitted scheme, Track-B sigma_private from the LOCKED scheme + control
variates + the reconstructed hidden state, and the production energy / cluster SE
from the returned raw counts. The driver first validates qsim-owned lifecycle and
locked-request evidence; the scorer then evaluates only model-owned scientific
claims. It writes ``score_report.json`` and exits 0 iff the five named admission
checks clear. Runs in the separate qsim-based verifier image, so it imports the
authoritative scorer directly (single implementation, no parity drift).
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

# Make src/ importable for local smoke tests; the separate verifier image bakes
# the package + configs instead.
_THIS = Path(__file__).resolve()
for ancestor in [_THIS, *_THIS.parents]:
    if (ancestor / "src" / "qiqcbench" / "__init__.py").is_file():
        sys.path.insert(0, str(ancestor / "src"))
        break

from qiqcbench.eval.contracts.outcome import OutcomeDisposition  # noqa: E402
from qiqcbench.eval.verifier.legacy_rubric import emit_legacy_rubric_report  # noqa: E402
from qiqcbench.qsim.actions.common import (  # noqa: E402
    PUBLISHED_RESULT_SUBDIR,
    qsim_internal_fault,
)
from qiqcbench.qsim.devices import configs_root  # noqa: E402
from qiqcbench.qsim.hidden_dynamics.adaptive_clustered_clbcs_h2o import (  # noqa: E402
    construction as C,
)
from qiqcbench.qsim.hidden_dynamics.adaptive_clustered_clbcs_h2o import scorer as S  # noqa: E402

TASK_ID = "adaptive_clustered_clbcs_h2o"
DEVICE_ID = "clbcs_h2o_14q_v0"
_FINAL_ACTION = "submit_final_answer"
_PRODUCTION_SUBMIT_ACTION = "submit_locked_composite_scheme"
_PRODUCTION_RESULT_ACTION = "locked_production_result"
_RESULT_ARTIFACT_FIELDS = {"schema_version", "relative_path", "size_bytes", "sha256"}


class VerifierError(Exception):
    """Model-owned rejection: emit reward 0."""


class InfrastructureError(Exception):
    """Verifier/qsim-owned failure: reserved exit and no reward."""


class EvidenceIntegrityError(InfrastructureError):
    """qsim-owned evidence is missing, malformed, or internally inconsistent."""


def _reject_duplicate_keys(pairs):
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate key {key!r}")
        payload[key] = value
    return payload


def _reject_constant(name: str):
    raise ValueError(f"non-standard JSON constant {name}")


def _strict_json_loads(text: str):
    return json.loads(
        text,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
    )


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise InfrastructureError(f"missing experiment_log.jsonl at {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise EvidenceIntegrityError(f"experiment_log.jsonl is unreadable: {exc}") from None
    events: list[dict] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = _strict_json_loads(line)
        except (json.JSONDecodeError, ValueError):
            raise EvidenceIntegrityError(
                f"experiment_log.jsonl line {line_number} is not valid JSON"
            ) from None
        if not isinstance(event, dict):
            raise EvidenceIntegrityError(
                f"experiment_log.jsonl line {line_number} is not an object"
            )
        events.append(event)
    return events


def _load_answer(path: Path) -> dict:
    try:
        payload = _strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise EvidenceIntegrityError("final_answer.json is unreadable") from None
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "task_id", "answer"}
        or payload.get("schema_version") != 2
        or payload.get("task_id") != TASK_ID
        or not isinstance(payload.get("answer"), dict)
    ):
        raise EvidenceIntegrityError("final_answer.json has an invalid qsim-owned envelope")
    return payload["answer"]


def preflight_qsim_faults(events: list[dict]) -> None:
    fault = qsim_internal_fault(events)
    if fault is not None:
        raise InfrastructureError(fault)


def resolve_final_submission(events: list[dict], answer_path: Path) -> tuple[dict, int]:
    """Bind the qsim-owned final file to the latest accepted task final event."""

    submissions = [
        (index, event) for index, event in enumerate(events) if event.get("action") == _FINAL_ACTION
    ]
    if not submissions:
        if answer_path.is_file():
            raise EvidenceIntegrityError(
                "final_answer.json exists but no submit_final_answer event was logged"
            )
        raise VerifierError("no final answer was submitted")
    if not answer_path.is_file():
        raise EvidenceIntegrityError(
            "submit_final_answer was logged but final_answer.json is missing"
        )

    final_index, final_event = submissions[-1]
    if final_event.get("task_id") != TASK_ID:
        raise EvidenceIntegrityError("latest submit_final_answer event has the wrong task_id")
    logged_answer = final_event.get("answer")
    if not isinstance(logged_answer, dict):
        raise EvidenceIntegrityError("latest submit_final_answer event lacks its answer payload")
    answer = _load_answer(answer_path)
    if answer != logged_answer:
        raise EvidenceIntegrityError(
            "final_answer.json does not match the latest logged submit_final_answer event"
        )
    return answer, final_index


def _strict_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _read_result_artifact(pointer: object, job_id: str, artifacts_dir: Path) -> dict:
    if not isinstance(pointer, dict) or set(pointer) != _RESULT_ARTIFACT_FIELDS:
        raise EvidenceIntegrityError(f"job {job_id}: invalid result artifact pointer")
    if pointer.get("schema_version") != 1:
        raise EvidenceIntegrityError(f"job {job_id}: unsupported result artifact schema")
    # Derived, not restated: this scorer runs in shared mode and already imports
    # from `actions.common`, so there is no import barrier to excuse a literal.
    # Publication is one level below the mount root; an older reader was
    # left at the old level -- it validated the new pointer's shape and then
    # rejected it, so correct evidence scored as an infrastructure failure.
    expected_relative = f"public_job_results/{PUBLISHED_RESULT_SUBDIR}/{job_id}.json"
    if pointer.get("relative_path") != expected_relative:
        raise EvidenceIntegrityError(f"job {job_id}: unsafe result artifact path")

    # Guard BOTH directory levels the pointer traverses. Guarding only the
    # innermost one silently stops covering the mount root, which is the
    # narrowing this sweep already had to repair twice elsewhere.
    mount_root = artifacts_dir / "public_job_results"
    directory = mount_root / PUBLISHED_RESULT_SUBDIR
    path = directory / f"{job_id}.json"
    if (
        mount_root.is_symlink()
        or not mount_root.is_dir()
        or not directory.is_dir()
        or directory.is_symlink()
        or not path.is_file()
        or path.is_symlink()
    ):
        raise EvidenceIntegrityError(f"job {job_id}: result artifact is missing or invalid")
    try:
        raw = path.read_bytes()
    except OSError:
        raise EvidenceIntegrityError(f"job {job_id}: result artifact is unreadable") from None
    size = pointer.get("size_bytes")
    if not _strict_nonnegative_int(size) or size == 0 or len(raw) != size:
        raise EvidenceIntegrityError(f"job {job_id}: result artifact size mismatch")
    digest = pointer.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        or hashlib.sha256(raw).hexdigest() != digest
    ):
        raise EvidenceIntegrityError(f"job {job_id}: result artifact digest mismatch")
    try:
        payload = _strict_json_loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise EvidenceIntegrityError(f"job {job_id}: result artifact is not valid JSON") from None
    if not isinstance(payload, dict):
        raise EvidenceIntegrityError(f"job {job_id}: result artifact is not an object")
    return payload


def _validate_production_rows(
    rows: object,
    *,
    job_id: str,
    scorer: S.AdaptiveClbcsHiddenScorer,
) -> list[dict]:
    if not isinstance(rows, list) or len(rows) != scorer.production_settings:
        raise EvidenceIntegrityError(f"job {job_id}: production row count is invalid")
    for index, row in enumerate(rows):
        label = f"job {job_id} production row {index}"
        if not isinstance(row, dict) or set(row) != {"basis", "counts"}:
            raise EvidenceIntegrityError(f"{label}: row shape is invalid")
        basis, counts = row.get("basis"), row.get("counts")
        if (
            not isinstance(basis, str)
            or len(basis) != C.NQ
            or any(axis not in "XYZ" for axis in basis)
            or not isinstance(counts, dict)
            or any(
                not isinstance(bits, str)
                or len(bits) != C.NQ
                or any(bit not in "01" for bit in bits)
                or not _strict_nonnegative_int(count)
                for bits, count in counts.items()
            )
            or sum(counts.values()) != scorer.production_shots
        ):
            raise EvidenceIntegrityError(f"{label}: raw counts are invalid")
    return rows


def _verify_production_artifact(
    pointer: object,
    production_event: dict,
    artifacts_dir: Path,
    scorer: S.AdaptiveClbcsHiddenScorer,
) -> list[dict]:
    job_id = production_event["job_id"]
    payload = _read_result_artifact(pointer, job_id, artifacts_dir)
    data = payload.get("data")
    if (
        payload.get("schema_version") != 3
        or payload.get("job_id") != job_id
        or payload.get("device_id") != production_event.get("device_id")
        or payload.get("status") != "complete"
        or not isinstance(data, dict)
        or data.get("kind") != "cme_production_counts"
        or data.get("request_digest") != production_event.get("request_digest")
        or data.get("production_settings") != scorer.production_settings
        or data.get("production_shots_per_setting") != scorer.production_shots
    ):
        raise EvidenceIntegrityError(f"job {job_id}: result artifact envelope mismatch")
    artifact_rows = _validate_production_rows(data.get("rows"), job_id=job_id, scorer=scorer)
    event_rows = _validate_production_rows(
        production_event.get("production_rows"), job_id=job_id, scorer=scorer
    )
    if artifact_rows != event_rows:
        raise EvidenceIntegrityError(
            f"job {job_id}: result artifact does not match production evidence"
        )
    return artifact_rows


def _validate_locked_request(
    production: dict,
    *,
    job_id: str,
    scorer: S.AdaptiveClbcsHiddenScorer,
) -> tuple[dict, list[dict], str]:
    """Return the canonical locked object after validating qsim-owned evidence."""

    request_digest = production.get("request_digest")
    if (
        not isinstance(request_digest, str)
        or len(request_digest) != 64
        or any(character not in "0123456789abcdef" for character in request_digest)
    ):
        raise EvidenceIntegrityError(f"job {job_id}: request_digest is malformed")

    canonical = production.get("canonical")
    if not isinstance(canonical, dict) or set(canonical) != {
        "task_id",
        "hamiltonian_sha256",
        "components",
        "control_variates",
    }:
        raise EvidenceIntegrityError(f"job {job_id}: canonical locked request is malformed")
    if (
        canonical.get("task_id") != scorer.task_id
        or canonical.get("hamiltonian_sha256") != scorer.public_hamiltonian_sha256
    ):
        raise EvidenceIntegrityError(f"job {job_id}: canonical locked request has wrong identity")
    entries = canonical.get("control_variates")
    if not isinstance(entries, list):
        raise EvidenceIntegrityError(f"job {job_id}: canonical control variates are malformed")
    if len(entries) > scorer.max_control_variates:
        raise EvidenceIntegrityError(f"job {job_id}: qsim accepted too many control variates")

    _, _, term_indices, _ = C.load_hamiltonian(scorer.task_id)
    valid_indices = set(term_indices)
    seen_indices: set[int] = set()
    for index, entry in enumerate(entries):
        label = f"job {job_id} control variate {index}"
        if not isinstance(entry, dict) or set(entry) != {"term_index", "mean"}:
            raise EvidenceIntegrityError(f"{label}: entry shape is invalid")
        term_index, mean = entry["term_index"], entry["mean"]
        if (
            isinstance(term_index, bool)
            or not isinstance(term_index, int)
            or term_index not in valid_indices
            or term_index in seen_indices
        ):
            raise EvidenceIntegrityError(f"{label}: term_index is invalid")
        if (
            isinstance(mean, bool)
            or not isinstance(mean, int | float)
            or not math.isfinite(float(mean))
            or abs(float(mean)) > 1.0
        ):
            raise EvidenceIntegrityError(f"{label}: mean is invalid")
        seen_indices.add(term_index)

    try:
        canonical_digest = C.request_digest(canonical)
        if canonical_digest != request_digest:
            raise EvidenceIntegrityError(
                f"job {job_id}: locked request does not reproduce request_digest"
            )
        canonical_weights, canonical_beta, canonical_entries = C.locked_request_from_canonical(
            canonical,
            num_components=scorer.num_components,
        )
        rebuilt = C.canonical_request(
            canonical_weights,
            canonical_beta,
            canonical_entries,
            scorer.task_id,
            scorer.public_hamiltonian_sha256,
        )
        terms, _, _, _ = C.load_hamiltonian(scorer.task_id)
        _, coverage_min = C.coverage_floor(terms, canonical_weights, canonical_beta)
    except EvidenceIntegrityError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
        raise EvidenceIntegrityError(f"job {job_id}: locked request is invalid: {exc}") from None
    if rebuilt != canonical:
        raise EvidenceIntegrityError(f"job {job_id}: locked request is not canonical")
    if coverage_min < scorer.coverage_floor:
        raise EvidenceIntegrityError(
            f"job {job_id}: qsim accepted a scheme below its coverage floor"
        )
    return (
        {
            "mixture_weights": canonical_weights.tolist(),
            "local_basis_probabilities_xyz": canonical_beta.tolist(),
        },
        canonical_entries,
        request_digest,
    )


def recover_scored_evidence(
    events: list[dict],
    final_index: int,
    artifacts_dir: Path,
    scorer: S.AdaptiveClbcsHiddenScorer,
) -> dict:
    """Recover the one completed, retrieved production result available before final."""

    pre_final = events[:final_index]
    production_submits: dict[str, tuple[int, dict]] = {}
    for index, event in enumerate(pre_final):
        if event.get("action") != _PRODUCTION_SUBMIT_ACTION:
            continue
        job_id = event.get("job_id")
        if not isinstance(job_id, str) or not job_id or job_id in production_submits:
            raise EvidenceIntegrityError("production submission has an invalid or duplicate job_id")
        production_submits[job_id] = (index, event)

    production_results = [
        (index, event)
        for index, event in enumerate(pre_final)
        if event.get("action") == _PRODUCTION_RESULT_ACTION
    ]
    if not production_results:
        raise VerifierError("no completed production evidence was available before final answer")
    if len(production_results) != 1:
        raise EvidenceIntegrityError("qsim recorded more than one successful production result")
    result_index, production = production_results[0]
    job_id = production.get("job_id")
    if not isinstance(job_id, str) or job_id not in production_submits:
        raise EvidenceIntegrityError("production result has no matching accepted submission")
    submit_index, submit = production_submits[job_id]
    if submit_index >= result_index:
        raise EvidenceIntegrityError(f"job {job_id}: production result precedes its submission")
    if (
        submit.get("device_id") != DEVICE_ID
        or production.get("device_id") != DEVICE_ID
        or production.get("production_settings") != scorer.production_settings
        or production.get("production_shots_per_setting") != scorer.production_shots
    ):
        raise EvidenceIntegrityError(f"job {job_id}: production evidence is malformed")

    locked_scheme, locked_control_entries, request_digest = _validate_locked_request(
        production,
        job_id=job_id,
        scorer=scorer,
    )

    completed_pointer: dict | None = None
    saw_failed = False
    for index, event in enumerate(pre_final):
        if event.get("action") != "get_job_result" or event.get("job_id") != job_id:
            continue
        status = event.get("status")
        if status not in {"queued", "running", "complete", "failed"}:
            raise EvidenceIntegrityError(f"job {job_id}: invalid poll status")
        if status == "failed":
            saw_failed = True
        elif status == "complete":
            if index <= result_index:
                raise EvidenceIntegrityError(f"job {job_id}: completed poll precedes its result")
            pointer = event.get("offloaded_result")
            if not isinstance(pointer, dict):
                raise EvidenceIntegrityError(f"job {job_id}: completed poll lacks artifact pointer")
            if completed_pointer is not None and completed_pointer != pointer:
                raise EvidenceIntegrityError(f"job {job_id}: completed polls disagree on artifact")
            completed_pointer = pointer
    if saw_failed and completed_pointer is not None:
        raise EvidenceIntegrityError(f"job {job_id}: job was polled as both failed and complete")
    if completed_pointer is None:
        raise VerifierError(
            "production result was not retrieved through a completed poll before final answer"
        )

    rows = _verify_production_artifact(completed_pointer, production, artifacts_dir, scorer)
    pilot_events = [event for event in pre_final if event.get("action") == "pilot_result"]
    if pilot_events:
        budget = pilot_events[-1].get("budget")
        if not isinstance(budget, dict):
            raise EvidenceIntegrityError("latest pilot result has no budget view")
        pilot_settings = budget.get("pilot_settings_used")
        pilot_shots = budget.get("pilot_shots_used")
        if not _strict_nonnegative_int(pilot_settings) or not _strict_nonnegative_int(pilot_shots):
            raise EvidenceIntegrityError("latest pilot result has a malformed budget view")
        if pilot_settings > scorer.pilot_max_settings or pilot_shots > scorer.pilot_max_shots:
            raise EvidenceIntegrityError("qsim accepted pilot evidence beyond its public budget")

    return {
        "locked_scheme": locked_scheme,
        "locked_control_entries": locked_control_entries,
        "production_rows": rows,
        "request_digest": request_digest,
    }


def _resolve_public_dir() -> Path:
    return configs_root() / "task_materials" / TASK_ID / "public"


def _resolve_hidden_scorer() -> Path:
    return configs_root() / "task_materials" / TASK_ID / "hidden" / "hidden_scorer.yaml"


def run(
    log_path: Path,
    ans_path: Path,
    artifacts_dir: Path,
    reward_path: Path | None = None,
    trial_root: Path | None = None,
) -> int:
    events = _load_jsonl(log_path)
    preflight_qsim_faults(events)
    scorer = S.load_hidden_scorer(_resolve_hidden_scorer())
    try:
        answer, final_index = resolve_final_submission(events, ans_path)
        evidence = recover_scored_evidence(events, final_index, artifacts_dir, scorer)
    except VerifierError as exc:
        _write(
            {
                "task_id": TASK_ID,
                "reward": 0,
                "penalties": [str(exc)],
            },
            artifacts_dir,
            log_path=log_path,
            ans_path=ans_path,
            reward_path=reward_path,
            trial_root=trial_root,
            disposition="model_failure",
        )
        return 1

    try:
        report = S.score_submission(
            answer=answer,
            locked_scheme=evidence["locked_scheme"],
            locked_control_entries=evidence["locked_control_entries"],
            production_rows=evidence["production_rows"],
            request_digest=evidence["request_digest"],
            scorer=scorer,
            public_dir=_resolve_public_dir(),
        )
    except S.SubmissionValidationError as exc:
        _write(
            {
                "task_id": TASK_ID,
                "reward": 0,
                "penalties": [f"invalid final answer: {exc}"],
            },
            artifacts_dir,
            log_path=log_path,
            ans_path=ans_path,
            reward_path=reward_path,
            trial_root=trial_root,
            disposition="model_failure",
        )
        return 1
    _write(
        report,
        artifacts_dir,
        log_path=log_path,
        ans_path=ans_path,
        reward_path=reward_path,
        trial_root=trial_root,
        disposition="scored",
    )
    return 0 if report["reward"] == 1 else 1


def _write(
    report: dict,
    artifacts_dir: Path,
    *,
    log_path: Path,
    ans_path: Path,
    reward_path: Path | None,
    trial_root: Path | None,
    disposition: OutcomeDisposition,
) -> None:
    resolved_reward = reward_path or artifacts_dir.parent / "reward.txt"
    resolved_root = trial_root or artifacts_dir.parent.parent
    evidence_paths = {
        role: path
        for role, path in (("experiment_log", log_path), ("final_answer", ans_path))
        if path.is_file()
    }
    emit_legacy_rubric_report(
        json.dumps(report, indent=2).encode("utf-8"),
        reward_binary=int(report["reward"]),
        report_path=artifacts_dir / "score_report.json",
        reward_path=resolved_reward,
        trial_root=resolved_root,
        evidence_paths=evidence_paths,
        disposition=disposition,
    )
    print(json.dumps({k: report.get(k) for k in ("reward", "total_score", "penalties")}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(
            "usage: score_clbcs.py <experiment_log.jsonl> <final_answer.json> "
            "<artifacts_dir> [reward_path] [trial_root]"
        )
    log_file = Path(sys.argv[1])
    ans_file = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    reward_file = Path(sys.argv[4]) if len(sys.argv) > 4 else out_dir.parent / "reward.txt"
    root_dir = Path(sys.argv[5]) if len(sys.argv) > 5 else out_dir.parent.parent
    try:
        rc = run(log_file, ans_file, out_dir, reward_file, root_dir)
    except Exception:  # noqa: BLE001 - every unexpected verifier failure is non-model
        print("verifier failure; private details withheld", file=sys.stderr)
        rc = 3
    sys.exit(rc)
