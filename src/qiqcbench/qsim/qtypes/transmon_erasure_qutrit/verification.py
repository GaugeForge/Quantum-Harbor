"""Task-neutral verification of round-resolved erasure-qutrit evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qiqcbench.eval.verifier.context import load_verifier_run_context
from qiqcbench.eval.verifier.hidden_commitment import derive_hidden_commitment_secret
from qiqcbench.qsim.actions.common import PUBLISHED_RESULT_SUBDIR, qsim_internal_fault
from qiqcbench.qsim.execution_context import EXECUTION_CONTEXT_SCHEMA_VERSION
from qiqcbench.qsim.hidden_commitment import (
    hidden_device_model_commitment,
    public_device_model_commitment,
)
from qiqcbench.qsim.qtypes.transmon_erasure_qutrit.device import (
    HiddenErasureQutritConfig,
    PublicErasureQutritSpec,
)

_POINTER_KEYS = frozenset({"schema_version", "relative_path", "size_bytes", "sha256"})
_ENVELOPE_KEYS = frozenset(
    {"schema_version", "job_id", "device_id", "status", "shots", "data", "error", "metadata"}
)
_ERASURE_DATA_KEYS = frozenset(
    {
        "kind",
        "points",
        "data_qubit",
        "prep_state",
        "prep_basis",
        "measure_basis",
        "cycle_time_us",
        "dd",
    }
)
_ERASURE_POINT_KEYS = frozenset(
    {
        "n_rounds",
        "total_evolution_us",
        "mid_circuit_ancilla_post_readout_bitstrings",
        "final_qutrit_post_readout_assignments",
    }
)
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_BOOTSTRAP_FIELDS = {
    "ts",
    "action",
    "context_schema_version",
    "execution_context_id",
    "public_device_model_commitment",
    "hidden_device_model_commitment",
}
_SUBMISSION_ACTIONS = {"submit_logical_memory", "submit_logical_memory_sweep"}
_EVIDENCE_ACTION = "logical_memory_evidence"


class ModelSubmissionError(ValueError):
    """A well-formed execution failed the public answer/evidence contract."""


class VerifierInfrastructureError(RuntimeError):
    """qsim evidence or verifier-side binding material is not trustworthy."""


@dataclass(frozen=True)
class VerifiedErasureJob:
    submission_index: int
    result_index: int
    poll_index: int
    submission: dict[str, Any]
    result_event: dict[str, Any]
    payload: dict[str, Any]
    artifact_manifest: dict[str, Any]


def read_experiment_log(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise VerifierInfrastructureError("experiment_log.jsonl is missing or invalid")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise VerifierInfrastructureError(f"experiment log cannot be read: {exc}") from exc
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise VerifierInfrastructureError(
                f"experiment log line {line_number} is malformed JSON"
            ) from exc
        if not isinstance(event, dict):
            raise VerifierInfrastructureError(f"experiment log line {line_number} is not an object")
        events.append(event)
    return events


def validate_qsim_run_binding(
    events: list[dict[str, Any]],
    *,
    task_id: str,
    public: PublicErasureQutritSpec,
    hidden: HiddenErasureQutritConfig,
) -> None:
    try:
        context = load_verifier_run_context()
    except (TypeError, ValueError) as exc:
        raise VerifierInfrastructureError(f"verifier execution context is invalid: {exc}") from exc

    bootstraps = [
        (index, event)
        for index, event in enumerate(events)
        if event.get("action") == "qsim_bootstrap_context"
    ]
    has_context_stamp = any("execution_context_id" in event for event in events)
    if context is None:
        if bootstraps or has_context_stamp:
            raise VerifierInfrastructureError(
                "execution-bound qsim evidence is missing its private verifier context"
            )
        return
    if context.task_id != task_id or context.evidence_binding_mode != "qsim_context_v1":
        raise VerifierInfrastructureError("verifier execution context targets another contract")
    if len(bootstraps) != 1 or bootstraps[0][0] != 0:
        raise VerifierInfrastructureError("official qsim bootstrap must be the unique first event")
    bootstrap = bootstraps[0][1]
    if set(bootstrap) != _BOOTSTRAP_FIELDS:
        raise VerifierInfrastructureError("official qsim bootstrap has an invalid field set")
    if bootstrap.get("context_schema_version") != EXECUTION_CONTEXT_SCHEMA_VERSION:
        raise VerifierInfrastructureError("official qsim bootstrap schema version is unsupported")
    expected_context_id = context.execution.qsim_evidence_nonce
    if any(event.get("execution_context_id") != expected_context_id for event in events):
        raise VerifierInfrastructureError("qsim event execution binding is inconsistent")
    try:
        secret = derive_hidden_commitment_secret(context)
        expected_public = public_device_model_commitment(public, task_id=task_id)
        expected_hidden = hidden_device_model_commitment(
            hidden,
            task_id=task_id,
            commitment_secret=secret,
        )
    except (TypeError, ValueError) as exc:
        raise VerifierInfrastructureError(f"device commitment cannot be recomputed: {exc}") from exc
    if bootstrap.get("public_device_model_commitment") != expected_public:
        raise VerifierInfrastructureError("qsim public-device commitment does not match")
    if bootstrap.get("hidden_device_model_commitment") != expected_hidden:
        raise VerifierInfrastructureError("qsim hidden-device commitment does not match")


def preflight_qsim_failures(events: list[dict[str, Any]]) -> None:
    """Reject any qsim-owned fault before the model's answer is judged.

    This used to look only for ``submit_final_answer_failure``. The
    per-citation terminal-poll check below honours the ``get_job_result`` stamp,
    but only for a job the model actually cited -- a fault on an uncited job, or
    one that stopped the model citing anything, still reached a model-owned
    reward 0. The shared rule covers every marker qsim writes.
    """

    fault = qsim_internal_fault(events)
    if fault is not None:
        raise VerifierInfrastructureError(fault)


def read_final_answer(path: Path, *, task_id: str) -> dict[str, Any] | None:
    if path.is_symlink():
        raise VerifierInfrastructureError("final_answer.json path is invalid")
    if not path.exists():
        return None
    if not path.is_file():
        raise VerifierInfrastructureError("final_answer.json path is invalid")
    # Every check below tests the ENVELOPE, which qsim authors alone:
    # ``submit_final_answer`` builds it through ``FinalAnswer.model_validate``
    # (``schema_version`` is ``Literal[2]``, ``task_id`` is refused unless it
    # equals the bound task, ``answer`` is typed ``dict``) and publishes it
    # atomically. No agent action can produce a file that fails one of these, so
    # a file that does is a torn transfer or a foreign file -- infrastructure,
    # exactly like the missing-artifact branch of ``bind_latest_final``. The
    # model-owned dispositions live elsewhere and are deliberately unmoved: an
    # absent file with no logged submission is "no final answer was submitted",
    # and an inner object that fails the task's own answer schema is rejected
    # during scoring.
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerifierInfrastructureError(f"final_answer.json is malformed: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "task_id", "answer"}:
        raise VerifierInfrastructureError("final_answer.json is not the FinalAnswer v2 envelope")
    if payload.get("schema_version") != 2 or payload.get("task_id") != task_id:
        raise VerifierInfrastructureError("final answer schema_version/task_id is incorrect")
    if not isinstance(payload.get("answer"), dict):
        raise VerifierInfrastructureError("final_answer.answer must be an object")
    return payload


def bind_latest_final(
    events: list[dict[str, Any]],
    artifact: dict[str, Any] | None,
    *,
    task_id: str,
) -> tuple[dict[str, Any], int]:
    finals = [
        (index, event)
        for index, event in enumerate(events)
        if event.get("action") == "submit_final_answer" and event.get("task_id") == task_id
    ]
    if artifact is None:
        if finals:
            raise VerifierInfrastructureError("logged final submission is missing its artifact")
        raise ModelSubmissionError("no final answer was submitted")
    if not finals:
        raise VerifierInfrastructureError("final answer artifact lacks a logged submission")
    final_index, final_event = finals[-1]
    if final_event.get("answer") != artifact["answer"]:
        raise VerifierInfrastructureError(
            "final answer artifact does not match the latest logged submission"
        )
    return artifact["answer"], final_index


def strict_float(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ModelSubmissionError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ModelSubmissionError(f"{label} must be a finite number")
    return result


def same_float(left: Any, right: Any, tolerance: float) -> bool:
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return False
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tolerance


def _verify_artifact(
    *,
    job_id: str,
    pointer: dict[str, Any],
    artifacts_dir: Path,
    device_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if set(pointer) != _POINTER_KEYS or pointer.get("schema_version") != 1:
        raise VerifierInfrastructureError(f"job {job_id}: result pointer contract is invalid")
    size = pointer.get("size_bytes")
    digest = pointer.get("sha256")
    relative = pointer.get("relative_path")
    if type(size) is not int or size < 0:
        raise VerifierInfrastructureError(f"job {job_id}: result pointer size is invalid")
    if not isinstance(digest, str) or _SHA256_HEX.fullmatch(digest) is None:
        raise VerifierInfrastructureError(f"job {job_id}: result pointer digest is invalid")
    expected_relative = f"public_job_results/{PUBLISHED_RESULT_SUBDIR}/{job_id}.json"
    if relative != expected_relative:
        raise VerifierInfrastructureError(f"job {job_id}: result pointer path is not confined")
    # Read exactly the relative path just validated, so the read level cannot
    # drift from the level qsim publishes into.
    path = artifacts_dir / expected_relative
    directory = path.parent
    # Guard BOTH directory levels the pointer traverses. Checking `path.parent`
    # alone would silently stop covering
    # the mount root -- a narrower guard that still passes every test.
    if directory.parent.is_symlink() or directory.is_symlink() or path.is_symlink():
        raise VerifierInfrastructureError(f"job {job_id}: result artifact path is a symlink")
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise VerifierInfrastructureError(f"job {job_id}: result artifact is unavailable") from exc
    with os.fdopen(fd, "rb") as handle:
        raw = handle.read(size + 1)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        raise VerifierInfrastructureError(f"job {job_id}: result artifact bytes do not match")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerifierInfrastructureError(f"job {job_id}: result artifact is not JSON") from exc
    if not isinstance(payload, dict) or set(payload) != _ENVELOPE_KEYS:
        raise VerifierInfrastructureError(f"job {job_id}: result artifact envelope is invalid")
    if (
        payload.get("schema_version") != 3
        or payload.get("job_id") != job_id
        or payload.get("device_id") != device_id
        or payload.get("status") != "complete"
        or payload.get("error") is not None
    ):
        raise VerifierInfrastructureError(f"job {job_id}: completed artifact binding is invalid")
    shots = payload.get("shots")
    data = payload.get("data")
    if type(shots) is not int or shots <= 0:
        raise VerifierInfrastructureError(f"job {job_id}: raw shot count is invalid")
    if (
        not isinstance(data, dict)
        or set(data) != _ERASURE_DATA_KEYS
        or data.get("kind") != "erasure_memory_round_resolved"
    ):
        raise VerifierInfrastructureError(
            f"job {job_id}: result does not contain round-resolved raw erasure records"
        )
    points = data.get("points")
    if not isinstance(points, list) or not points:
        raise VerifierInfrastructureError(f"job {job_id}: raw point list is invalid")
    for point in points:
        if not isinstance(point, dict) or set(point) != _ERASURE_POINT_KEYS:
            raise VerifierInfrastructureError(f"job {job_id}: raw point contract is invalid")
        rounds = point.get("n_rounds")
        total = point.get("total_evolution_us")
        syndromes = point.get("mid_circuit_ancilla_post_readout_bitstrings")
        assignments = point.get("final_qutrit_post_readout_assignments")
        if (
            type(rounds) is not int
            or rounds < 0
            or not isinstance(total, (int, float))
            or isinstance(total, bool)
            or not math.isfinite(float(total))
            or float(total) < 0.0
            or not isinstance(syndromes, list)
            or not isinstance(assignments, list)
            or len(syndromes) != shots
            or len(assignments) != shots
        ):
            raise VerifierInfrastructureError(f"job {job_id}: raw point shape is invalid")
        if any(
            not isinstance(bits, str) or len(bits) != rounds or bool(set(bits) - {"0", "1"})
            for bits in syndromes
        ):
            raise VerifierInfrastructureError(f"job {job_id}: ancilla syndrome bits are malformed")
        if any(type(value) is not int or value not in {0, 1, 2} for value in assignments):
            raise VerifierInfrastructureError(
                f"job {job_id}: final qutrit assignments are malformed"
            )
    metadata = payload.get("metadata")
    coords = metadata.get("sweep_coords") if isinstance(metadata, dict) else None
    round_coords = coords.get("n_rounds") if isinstance(coords, dict) else None
    time_coords = coords.get("total_evolution_us") if isinstance(coords, dict) else None
    if (
        not isinstance(round_coords, list)
        or not isinstance(time_coords, list)
        or len(round_coords) != len(points)
        or len(time_coords) != len(points)
    ):
        raise VerifierInfrastructureError(f"job {job_id}: sweep coordinates are invalid")
    for index, point in enumerate(points):
        if not same_float(round_coords[index], point["n_rounds"], 1e-12) or not same_float(
            time_coords[index], point["total_evolution_us"], 1e-12
        ):
            raise VerifierInfrastructureError(f"job {job_id}: sweep coordinates differ")
    return payload, {
        "role": "cited_public_job_result",
        "job_id": job_id,
        "relative_path": relative,
        "size_bytes": size,
        "sha256": digest,
    }


def _submission_rounds(submission: dict[str, Any]) -> list[int]:
    if submission.get("action") == "submit_logical_memory":
        values = [submission.get("n_rounds")]
    elif submission.get("action") == "submit_logical_memory_sweep":
        values = submission.get("n_rounds_grid")
    else:
        raise VerifierInfrastructureError("cited job has an unknown submission action")
    if not isinstance(values, list) or any(type(value) is not int or value < 0 for value in values):
        raise VerifierInfrastructureError("cited job has malformed round-count metadata")
    return values


def _raw_point_aggregate(point: dict[str, Any], expected: int) -> dict[str, Any]:
    syndromes = point["mid_circuit_ancilla_post_readout_bitstrings"]
    assignments = point["final_qutrit_post_readout_assignments"]
    flags = ["1" in bits for bits in syndromes]
    no_flag = [not flag for flag in flags]
    return {
        "n_rounds": point["n_rounds"],
        "total_evolution_us": point["total_evolution_us"],
        "n_shots": len(syndromes),
        "n_flag": sum(flags),
        "n_no_flag": sum(no_flag),
        "n_no_flag_codespace": sum(
            keep and assignment != 1 for keep, assignment in zip(no_flag, assignments, strict=True)
        ),
        "n_no_flag_correct": sum(
            keep and assignment == expected
            for keep, assignment in zip(no_flag, assignments, strict=True)
        ),
        "n_leak_outcome": sum(assignment == 1 for assignment in assignments),
    }


def _verify_result_event(
    *,
    job_id: str,
    submission: dict[str, Any],
    result_event: dict[str, Any],
    payload: dict[str, Any],
    data_qubit: str,
) -> None:
    data = payload["data"]
    rounds = _submission_rounds(submission)
    cycle = submission.get("cycle_time_us")
    if (
        not isinstance(cycle, (int, float))
        or isinstance(cycle, bool)
        or not math.isfinite(float(cycle))
    ):
        raise VerifierInfrastructureError(f"job {job_id}: submitted cadence is malformed")
    expected_times = [value * float(cycle) for value in rounds]
    artifact_rounds = [point["n_rounds"] for point in data["points"]]
    artifact_times = [float(point["total_evolution_us"]) for point in data["points"]]
    if artifact_rounds != rounds or any(
        not same_float(actual, expected, 1e-9)
        for actual, expected in zip(artifact_times, expected_times, strict=True)
    ):
        raise VerifierInfrastructureError(f"job {job_id}: request/result coordinates differ")
    if payload["shots"] != submission.get("shots"):
        raise VerifierInfrastructureError(f"job {job_id}: request/result shots differ")
    scalar_fields = ("prep_state", "prep_basis", "measure_basis", "cycle_time_us", "dd")
    if any(result_event.get(name) != submission.get(name) for name in scalar_fields):
        raise VerifierInfrastructureError(f"job {job_id}: result event request binding differs")
    if any(data.get(name) != submission.get(name) for name in scalar_fields):
        raise VerifierInfrastructureError(f"job {job_id}: artifact request binding differs")
    if data.get("data_qubit") != data_qubit:
        raise VerifierInfrastructureError(f"job {job_id}: artifact data-qubit binding differs")
    if result_event.get("device_id") != submission.get("device_id"):
        raise VerifierInfrastructureError(f"job {job_id}: result event device binding differs")
    if result_event.get("result_kind") != "erasure_memory_round_resolved":
        raise VerifierInfrastructureError(f"job {job_id}: result event kind differs")
    records = result_event.get("points")
    if not isinstance(records, list) or len(records) != len(data["points"]):
        raise VerifierInfrastructureError(f"job {job_id}: result event point count differs")
    expected_assignment = {"0L": 0, "1L": 2, "+X": 0, "-X": 2}.get(
        str(submission.get("prep_state")), -1
    )
    for index, (record, point) in enumerate(zip(records, data["points"], strict=True)):
        if record != _raw_point_aggregate(point, expected_assignment):
            raise VerifierInfrastructureError(
                f"job {job_id}: result summary does not match raw point {index}"
            )


def verify_cited_job(
    *,
    job_id: str,
    events: list[dict[str, Any]],
    final_index: int,
    artifacts_dir: Path,
    device_id: str,
    data_qubit: str,
) -> VerifiedErasureJob:
    submissions = [
        (index, event)
        for index, event in enumerate(events[:final_index])
        if event.get("job_id") == job_id and event.get("action") in _SUBMISSION_ACTIONS
    ]
    if not submissions:
        raise ModelSubmissionError(f"citation {job_id} does not name a submitted memory job")
    if len(submissions) != 1:
        raise VerifierInfrastructureError(f"job {job_id}: duplicate submission records")
    submission_index, submission = submissions[0]
    if any(
        event.get("job_id") == job_id and event.get("action") == "get_job_result_failure"
        for event in events[:final_index]
    ):
        raise VerifierInfrastructureError(f"job {job_id}: qsim recorded result-delivery failure")
    terminal = [
        (index, event)
        for index, event in enumerate(events[:final_index])
        if event.get("job_id") == job_id
        and event.get("action") == "get_job_result"
        and event.get("status") in {"complete", "failed"}
    ]
    if not terminal:
        raise ModelSubmissionError(f"citation {job_id} was not terminally polled before final")
    statuses = {event.get("status") for _, event in terminal}
    if statuses != {"complete"}:
        if statuses == {"failed"}:
            if any(event.get("failure_kind") == "qsim_internal" for _, event in terminal):
                raise VerifierInfrastructureError(
                    f"job {job_id}: qsim recorded an internal execution failure"
                )
            raise ModelSubmissionError(f"citation {job_id} names a failed job")
        raise VerifierInfrastructureError(f"job {job_id}: conflicting terminal poll statuses")
    results = [
        (index, event)
        for index, event in enumerate(events[:final_index])
        if event.get("job_id") == job_id and event.get("action") == _EVIDENCE_ACTION
    ]
    if len(results) != 1:
        raise VerifierInfrastructureError(f"job {job_id}: missing or duplicate result evidence")
    result_index, result_event = results[0]
    pointers = [event.get("offloaded_result") for _, event in terminal]
    if any(not isinstance(pointer, dict) for pointer in pointers):
        raise VerifierInfrastructureError(
            f"job {job_id}: completed poll lacks its artifact pointer"
        )
    pointer = pointers[0]
    if any(candidate != pointer for candidate in pointers[1:]):
        raise VerifierInfrastructureError(f"job {job_id}: completed polls disagree on artifact")
    poll_index = terminal[0][0]
    if not submission_index < result_index < poll_index < final_index:
        raise VerifierInfrastructureError(
            f"job {job_id}: expected submission -> result -> terminal poll -> final ordering"
        )
    if submission.get("device_id") != device_id:
        raise VerifierInfrastructureError(f"job {job_id}: submission device binding differs")
    payload, manifest = _verify_artifact(
        job_id=job_id,
        pointer=pointer,
        artifacts_dir=artifacts_dir,
        device_id=device_id,
    )
    _verify_result_event(
        job_id=job_id,
        submission=submission,
        result_event=result_event,
        payload=payload,
        data_qubit=data_qubit,
    )
    return VerifiedErasureJob(
        submission_index=submission_index,
        result_index=result_index,
        poll_index=poll_index,
        submission=submission,
        result_event=result_event,
        payload=payload,
        artifact_manifest=manifest,
    )


__all__ = [
    "ModelSubmissionError",
    "VerifiedErasureJob",
    "VerifierInfrastructureError",
    "bind_latest_final",
    "preflight_qsim_failures",
    "read_experiment_log",
    "read_final_answer",
    "same_float",
    "strict_float",
    "validate_qsim_run_binding",
    "verify_cited_job",
]
