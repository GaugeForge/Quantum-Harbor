"""Hidden verifier scorer for time_budgeted_shadow_surrogate_60q.

Binds the submitted surrogate predictions to the closed-form hidden truth and
the qsim-owned evidence log. All three gates are disclosed verbatim in the
public instruction; no undisclosed numerical convention enters the verdict:

  G1 accuracy     overall hidden RMSE <= RMSE_THRESHOLD AND worst per-target
                  RMSE <= WORST_TARGET_RMSE_THRESHOLD (an error burst on one
                  target cannot hide in the 40-target average).
  G2 tail         max absolute cell error <= MAX_CELL_ERROR_THRESHOLD (a large
                  localized error cannot hide in the 70,800-cell averages).
  G3 validity     schema/shape/band checks; completed measurement evidence and
                  an agent-visible completed get_job_result poll (referencing
                  a real measured job) BEFORE the final answer; an
                  irreversible measurement seal whose revealed targets and
                  commitment opening equal hidden truth; and exact receipt
                  citations. Self-reported shots/jobs are informational and
                  never compared with the sealed counters.

Evidence-channel integrity is separate from the gates: the artifact must be
the exact FinalAnswer payload qsim logged (matching task_id, schema, and
payload; every result event chained to a logged submission). Violations
RAISE — they can only arise from infrastructure faults or tampering, since
the qsim action seam writes both sides under one lock — and the verifier
shell converts the raise into the reserved infra disposition (no reward).

The numerical prediction guard is asymmetric around the physical interval:
C = (XX+YY+ZZ)/3 = (2*SWAP-I)/3 has spectrum {-1, +1/3}, so expectations live
in [-1, 1/3], while the accepted guard band is [-1.05, 0.35].

Threshold provenance: calibration runs recorded in the track reference doc;
reproduce with `python -m ...time_budgeted_shadow_surrogate_60q.reference`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from qiqcbench.qsim.actions.common import (
    PUBLISHED_RAW_RECORD_SUBDIR,
    PUBLISHED_RESULT_SUBDIR,
)
from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.device import (
    SEALED_TARGET_COMMITMENT_SCHEME,
    canonical_target_inputs_sha256,
    sealed_target_commitment_sha256,
)

TASK_ID = "time_budgeted_shadow_surrogate_60q"
FINAL_ANSWER_SCHEMA_VERSION = 2

# The public-class reference reaches RMSE
# 0.0016-0.0017 at about 1,000 inputs on the hard roots; the even-X/Y/Z
# allocation baseline sits at 0.0029-0.0031.  These gates give the reference
# about 1.5x margin and fail that baseline decisively.
RMSE_THRESHOLD = 0.0025
# G1(b): worst per-target RMSE (over the 1770 pairs of one target). Catches an
# error burst concentrated on one or a few targets that the overall RMSE
# averages away. The certificate applies its frozen accuracy headroom to this
# extreme statistic on every design stream.
WORST_TARGET_RMSE_THRESHOLD = 0.0035
# G2: max absolute error over all 70,800 cells. Caps localized damage that
# every averaged statistic dilutes. The reference's maximum over 70,800 cells
# ranges 0.020-0.033 across shot streams; the certificate applies only a small
# tail margin to this extreme statistic (construction.CERT_MIN_TAIL_MARGIN).
MAX_CELL_ERROR_THRESHOLD = 0.040
N_TARGETS = 40
N_PAIRS = 1770
N_QUBITS = 60
INPUT_DIM = 6
DEGREE_BOUND = 3
DEVICE_ID = "boundedgate_60q_v0"
TOTAL_SHOT_BUDGET = 315_000
MAX_JOBS = 40
MAX_BLOCKS_PER_JOB = 64
MAX_SETTINGS_PER_JOB = 192
MAX_SHOTS_PER_SETTING = 4_000
MAX_PUBLIC_JOB_RESULT_BYTES = 64 * 1024 * 1024
MAX_PUBLIC_JOB_RESULTS_TOTAL_BYTES = 256 * 1024 * 1024
MAX_METADATA_CALLS = 256
MAX_JOB_RESULT_POLLS = 4_096
MAX_SEALED_HOLDOUT_REVEALS = 4
MAX_FINAL_ANSWER_SERIALIZED_BYTES = 4 * 1024 * 1024
MAX_FINAL_ANSWER_SUBMISSIONS = 4
MAX_ANSWER_STRING_CHARACTERS = 16_384
# Spectrum of (XX+YY+ZZ)/3 = (2*SWAP - I)/3 is {-1, +1/3}. The slightly wider
# asymmetric interval is a numerical guard band, not the physical spectrum.
PREDICTION_BAND_LOW = -1.05
PREDICTION_BAND_HIGH = 0.35

_RESULT_FAILURE_STAGES = frozenset({"result_artifact_persistence", "result_poll_log_publication"})
_FINAL_FAILURE_STAGES = frozenset({"final_answer_persistence", "final_answer_log_publication"})
_SUBMISSION_FAILURE_STAGES = frozenset(
    {"poll_budget_registration", "submission_log_publication", "job_enqueue"}
)
_BASIS_RESULT_FAILURE_STAGES = frozenset({"result_log_publication"})
_REVEAL_FAILURE_STAGES = frozenset({"reveal_execution", "reveal_log_publication"})
_METADATA_ACTIONS = frozenset({"list_devices", "get_device_spec", "get_lab_notebook"})
_KNOWN_TASK_QSIM_ACTIONS = _METADATA_ACTIONS | frozenset(
    {
        "qsim_bootstrap_context",
        "metadata_call_failure",
        "submit_basis_shots",
        "submit_basis_shots_failure",
        "basis_shots_result",
        "basis_shots_result_failure",
        "get_job_result",
        "get_job_result_failure",
        "sealed_holdout_reveal",
        "sealed_holdout_reveal_failure",
        "submit_final_answer",
        "submit_final_answer_failure",
    }
)
_FAILURE_EVENT_ENVELOPE_FIELDS = frozenset({"ts", "execution_context_id", "surface"})

_QSIM_EVENT_REQUIRED_ENVELOPE_FIELDS = frozenset({"ts", "surface"})
_QSIM_EVENT_OPTIONAL_ENVELOPE_FIELDS = frozenset({"execution_context_id"})

_PUBLIC_JOB_RESULT_NAME = re.compile(r"([A-Za-z0-9][A-Za-z0-9_.-]{0,127})\.json\Z")
_OFFLOADED_RESULT_FIELDS = frozenset({"schema_version", "relative_path", "size_bytes", "sha256"})

SEALED_REVEAL_ACTION = "sealed_holdout_reveal"
_REVEAL_REQUIRED_FIELDS = frozenset(
    {
        "action",
        "schema_version",
        "task_id",
        "device_id",
        "challenge_id",
        "commitment_scheme",
        "target_inputs",
        "target_inputs_sha256",
        "target_commitment_sha256",
        "commitment_nonce",
        "shots_used",
        "jobs_used",
        "last_admission_sequence",
    }
)
# The device clock is instrument state echoed by
# the engine with every budget view and frozen into the sealed receipt. It is
# informational evidence, never a gate, and optional so evidence recorded by
# engines that predate the clock still validates.
_DEVICE_CLOCK_OPTIONAL_FIELDS = frozenset({"elapsed_wall_clock_s"})
_REVEAL_REQUIRED_EVENT_FIELDS = frozenset(
    {
        "action",
        "tool",
        "schema_version",
        "task_id",
        "device_id",
        "challenge_id",
        "commitment_scheme",
        "target_inputs",
        "target_inputs_sha256",
        "target_commitment_sha256",
        "commitment_nonce",
        "shots_used",
        "jobs_used",
        "last_admission_sequence",
        "is_repeat",
        "sealed_holdout_reveals_used",
        "max_sealed_holdout_reveals",
    }
)
_FINAL_SUBMISSION_REQUIRED_EVENT_FIELDS = frozenset(
    {
        "action",
        "tool",
        "task_id",
        "answer",
        "final_answer_serialized_bytes",
        "final_answer_submissions_used",
        "max_final_answer_submissions",
    }
)

# ``protocol`` and ``surrogate_model`` are free-form informational strings
# (owner decision, 2026-08-27): naming method families in the answer schema
# is a method hint, and the verdict never depends on them. They are recorded
# verbatim when they are non-empty strings, truncated to a bounded length.
INFORMATIONAL_STRING_MAX_CHARS = 200
# Scored fields: the prediction matrix and the sealed-holdout citation that
# fixes its row order. Everything else in the public answer schema is
# informational (the human-readable layer and self-reported bookkeeping):
# it is recorded when present and well-typed, but it never gates the verdict
# and never enters G1-G2. Unknown extra fields are tolerated for the same
# reason. Earlier revisions rejected exact predictions for a one-off
# shots_used, a validation_rmse outside [0, 1], or a non-literal protocol
# string.
SCORED_ANSWER_FIELDS = frozenset(
    {
        "predictions_as_measured",
        "challenge_id",
        "target_inputs_sha256",
    }
)
INFORMATIONAL_ANSWER_FIELDS = frozenset(
    {
        "protocol",
        "surrogate_model",
        "shots_used",
        "jobs_used",
        "validation_rmse",
        "method",
    }
)
FINAL_ANSWER_FIELDS = SCORED_ANSWER_FIELDS | INFORMATIONAL_ANSWER_FIELDS


def scorer_policy() -> dict[str, float]:
    """The complete gate-threshold policy, in one canonical dict.

    This is the object a certificate must record verbatim, the object whose
    SHA-256 the emitted hidden device config binds (``edition_binding``), and
    the object the Harbor wrapper recomputes to prove the scorer revision in
    force matches the edition it is scoring.
    """
    return {
        "rmse": RMSE_THRESHOLD,
        "worst_target_rmse": WORST_TARGET_RMSE_THRESHOLD,
        "max_cell_error": MAX_CELL_ERROR_THRESHOLD,
        "prediction_band_low": PREDICTION_BAND_LOW,
        "prediction_band_high": PREDICTION_BAND_HIGH,
    }


def scorer_policy_sha256() -> str:
    import hashlib

    return hashlib.sha256(
        (json.dumps(scorer_policy(), indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()


@dataclass
class ScoreResult:
    passed: bool
    gates: dict[str, bool]
    metrics: dict[str, Any]
    reject_reasons: list[str] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "gates": self.gates,
            "metrics": self.metrics,
            "reject_reasons": self.reject_reasons,
        }


def _fail(reasons: list[str], message: str) -> None:
    reasons.append(message)


def _as_matrix(value: Any, name: str, reasons: list[str]) -> np.ndarray | None:
    if (
        not isinstance(value, list)
        or len(value) != N_TARGETS
        or any(not isinstance(row, list) or len(row) != N_PAIRS for row in value)
    ):
        try:
            shape = np.asarray(value, dtype=object).shape
        except (TypeError, ValueError):
            shape = "invalid"
        _fail(reasons, f"{name} has shape {shape}; expected ({N_TARGETS}, {N_PAIRS})")
        return None
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        for row in value
        for item in row
    ):
        _fail(reasons, f"{name} is not a numeric array")
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        _fail(reasons, f"{name} is not a numeric array")
        return None
    if not np.all(np.isfinite(arr)):
        _fail(reasons, f"{name} contains non-finite values")
        return None
    return arr


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _compact_final_answer_bytes(*, task_id: str, answer: dict[str, Any]) -> bytes:
    """Mirror the qsim action's finite canonical serialization for its byte meter."""
    try:
        payload = json.dumps(
            {
                "schema_version": FINAL_ANSWER_SCHEMA_VERSION,
                "task_id": task_id,
                "answer": answer,
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ValueError("logged final answer is not finite compact JSON") from exc
    return payload.encode("utf-8")


def _validate_answer_string_resources(answer: dict[str, Any]) -> None:
    """Recheck the task's recursive qsim string/key cap without coercion."""
    pending: list[Any] = [answer]
    while pending:
        value = pending.pop()
        if isinstance(value, str):
            if len(value) > MAX_ANSWER_STRING_CHARACTERS:
                raise ValueError("logged final answer contains a string exceeding the public limit")
        elif isinstance(value, dict):
            for key, nested in value.items():
                if type(key) is not str:
                    raise ValueError("logged final answer object keys must be strings")
                if len(key) > MAX_ANSWER_STRING_CHARACTERS:
                    raise ValueError(
                        "logged final answer contains an object key exceeding the public "
                        "string limit"
                    )
                pending.append(nested)
        elif isinstance(value, list):
            pending.extend(value)


def _validate_reveal_event_envelope(event: dict[str, Any]) -> None:
    _require_qsim_event_shape(
        event,
        action=SEALED_REVEAL_ACTION,
        required_fields=_REVEAL_REQUIRED_EVENT_FIELDS,
        optional_fields=_DEVICE_CLOCK_OPTIONAL_FIELDS,
    )
    _require_device_clock(event)
    if event["surface"] != "mcp":
        raise ValueError("sealed holdout reveal has the wrong surface")
    if event["tool"] != "lock_measurements_and_reveal_challenge":
        raise ValueError("sealed holdout reveal has the wrong tool")
    if type(event["schema_version"]) is not int or event["schema_version"] != 1:
        raise ValueError("sealed holdout reveal schema_version must be 1")
    for field_name in (
        "task_id",
        "device_id",
        "challenge_id",
        "commitment_scheme",
        "commitment_nonce",
    ):
        if type(event[field_name]) is not str or not event[field_name]:
            raise ValueError(f"sealed holdout reveal {field_name} must be a non-empty string")
    if event["task_id"] != TASK_ID or event["device_id"] != DEVICE_ID:
        raise ValueError("sealed holdout reveal has the wrong task/device binding")
    if event["commitment_scheme"] != SEALED_TARGET_COMMITMENT_SCHEME:
        raise ValueError("sealed holdout reveal commitment_scheme is unsupported")
    if not _is_sha256(event["commitment_nonce"]):
        raise ValueError("sealed holdout reveal commitment_nonce must be 256-bit hex")
    for field_name in ("target_inputs_sha256", "target_commitment_sha256"):
        if not _is_sha256(event[field_name]):
            raise ValueError(f"sealed holdout reveal {field_name} must be a lowercase SHA-256")
    for field_name in ("shots_used", "jobs_used", "last_admission_sequence"):
        value = event[field_name]
        if type(value) is not int or value < 0:
            raise ValueError(f"sealed holdout reveal {field_name} must be a non-negative integer")
    reveals_used = event["sealed_holdout_reveals_used"]
    reveal_limit = event["max_sealed_holdout_reveals"]
    if type(reveals_used) is not int or not 1 <= reveals_used <= MAX_SEALED_HOLDOUT_REVEALS:
        raise ValueError("sealed holdout reveal counter violates the task contract")
    if type(reveal_limit) is not int or reveal_limit != MAX_SEALED_HOLDOUT_REVEALS:
        raise ValueError("sealed holdout reveal max disagrees with the task contract")
    if type(event["is_repeat"]) is not bool:
        raise ValueError("sealed holdout reveal is_repeat must be boolean")

    targets = _validated_target_inputs(event["target_inputs"], owner="sealed reveal")
    target_digest = canonical_target_inputs_sha256(targets)
    if event["target_inputs_sha256"] != target_digest:
        raise ValueError("sealed reveal target_inputs_sha256 does not match its target_inputs")
    commitment = sealed_target_commitment_sha256(
        challenge_id=event["challenge_id"],
        commitment_nonce=event["commitment_nonce"],
        target_inputs_sha256=target_digest,
    )
    if event["target_commitment_sha256"] != commitment:
        raise ValueError("sealed reveal commitment does not match its opening")


def _validate_final_submission_event(event: dict[str, Any]) -> None:
    _require_qsim_event_shape(
        event,
        action="submit_final_answer",
        required_fields=_FINAL_SUBMISSION_REQUIRED_EVENT_FIELDS,
    )
    if event["surface"] != "mcp":
        raise ValueError("logged submit_final_answer has the wrong surface")
    if event["tool"] != "submit_final_answer":
        raise ValueError("logged submit_final_answer has the wrong tool")
    if event["task_id"] != TASK_ID:
        raise ValueError(f"logged submission task_id {event['task_id']!r} != {TASK_ID!r}")
    answer = event["answer"]
    if not isinstance(answer, dict):
        raise ValueError("logged submit_final_answer event carries no answer object")
    _validate_answer_string_resources(answer)

    used = event["final_answer_submissions_used"]
    maximum = event["max_final_answer_submissions"]
    serialized_bytes = event["final_answer_serialized_bytes"]
    if type(used) is not int or not 1 <= used <= MAX_FINAL_ANSWER_SUBMISSIONS:
        raise ValueError("logged final-answer counter violates the task contract")
    if type(maximum) is not int or maximum != MAX_FINAL_ANSWER_SUBMISSIONS:
        raise ValueError("logged final-answer max disagrees with the task contract")
    expected_bytes = len(_compact_final_answer_bytes(task_id=TASK_ID, answer=answer))
    if type(serialized_bytes) is not int or serialized_bytes != expected_bytes:
        raise ValueError("logged final-answer byte count disagrees with its payload")
    if serialized_bytes > MAX_FINAL_ANSWER_SERIALIZED_BYTES:
        raise ValueError("logged final-answer bytes exceed the task contract")


def _validated_target_inputs(value: Any, *, owner: str) -> list[list[float]]:
    """Return a JSON-native target matrix or raise on corrupt trusted evidence."""
    if not isinstance(value, list) or len(value) != N_TARGETS:
        raise ValueError(f"{owner} target_inputs must contain exactly {N_TARGETS} rows")
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != INPUT_DIM:
            raise ValueError(
                f"{owner} target_inputs row {row_index} must contain exactly {INPUT_DIM} values"
            )
        for value_index, item in enumerate(row):
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(item)
            ):
                raise ValueError(
                    f"{owner} target_inputs[{row_index}][{value_index}] is not a finite number"
                )
            if not -math.pi <= item <= math.pi:
                raise ValueError(
                    f"{owner} target_inputs[{row_index}][{value_index}] lies outside [-pi, pi]"
                )
    return value


def validate_hidden_truth(truth: dict[str, Any]) -> None:
    """Fail closed on malformed trusted labels before assigning model blame."""
    if not isinstance(truth, dict):
        raise ValueError("hidden truth must be an object")
    if truth.get("task_id") != TASK_ID or truth.get("device_id") != DEVICE_ID:
        raise ValueError("hidden truth task/device binding is invalid")
    if truth.get("input_dimension") != INPUT_DIM or truth.get("n_qubits") != N_QUBITS:
        raise ValueError("hidden truth dimensions do not match the scorer contract")
    try:
        labels = np.asarray(truth.get("labels"), dtype=float)
    except (TypeError, ValueError):
        raise ValueError("hidden truth labels are not numeric") from None
    if labels.shape != (N_TARGETS, N_PAIRS):
        raise ValueError(f"hidden truth labels have unexpected shape {labels.shape}")
    if not np.all(np.isfinite(labels)):
        raise ValueError("hidden truth labels contain non-finite values")
    tolerance = 1e-10
    if np.any(labels < -1.0 - tolerance) or np.any(labels > (1.0 / 3.0) + tolerance):
        raise ValueError("hidden truth labels violate the observable spectrum [-1, 1/3]")


@dataclass(frozen=True)
class _SealedLifecycle:
    receipt: dict[str, Any] | None
    reveal_index: int | None


def _trusted_target_challenge(truth: dict[str, Any]) -> dict[str, Any]:
    """Validate the private challenge record bound into hidden truth.

    The commitment opening is part of the hidden truth record served to this
    private verifier. The temporary flat aliases are cross-checked against
    the canonical nested ``target_challenge`` during the schema transition.
    """
    nested = truth.get("target_challenge")
    if not isinstance(nested, dict):
        raise ValueError("hidden truth target_challenge record is not an object")
    challenge = nested
    if truth.get("task_id") != TASK_ID:
        raise ValueError("hidden truth task_id does not match this scorer")

    expected_keys = {
        "schema_version",
        "challenge_id",
        "commitment_scheme",
        "target_inputs",
        "target_inputs_sha256",
        "target_commitment_sha256",
        "commitment_nonce",
    }
    missing = expected_keys - set(challenge)
    if missing:
        raise ValueError(f"hidden target_challenge lacks fields {sorted(missing)}")
    if challenge["schema_version"] != 1:
        raise ValueError("hidden target_challenge schema_version must be 1")
    for field_name in (
        "challenge_id",
        "target_inputs_sha256",
        "target_commitment_sha256",
        "commitment_nonce",
    ):
        if field_name in truth and truth[field_name] != challenge[field_name]:
            raise ValueError(f"hidden truth {field_name} diverges from target_challenge")

    device_id = challenge.get("device_id", truth.get("device_id"))
    if not isinstance(device_id, str) or not device_id:
        raise ValueError("hidden target_challenge has no device_id binding")
    if truth.get("device_id") != device_id:
        raise ValueError("hidden target_challenge device_id diverges from hidden truth")
    challenge_id = challenge.get("challenge_id")
    commitment_scheme = challenge["commitment_scheme"]
    nonce = challenge.get("commitment_nonce")
    if not isinstance(challenge_id, str) or not challenge_id:
        raise ValueError("hidden target_challenge challenge_id must be a non-empty string")
    if commitment_scheme != SEALED_TARGET_COMMITMENT_SCHEME:
        raise ValueError("hidden target_challenge commitment_scheme is unsupported")
    if not _is_sha256(nonce):
        raise ValueError("hidden target_challenge commitment_nonce must be 256-bit hex")

    top_level_targets = _validated_target_inputs(truth.get("target_inputs"), owner="hidden truth")
    nested_targets = _validated_target_inputs(
        challenge["target_inputs"], owner="hidden target_challenge"
    )
    if nested_targets != top_level_targets:
        raise ValueError(
            "hidden target_challenge targets diverge from the hidden truth label order"
        )

    target_digest = canonical_target_inputs_sha256(top_level_targets)
    if challenge.get("target_inputs_sha256") != target_digest:
        raise ValueError("hidden target_inputs_sha256 does not match hidden truth targets")
    commitment = sealed_target_commitment_sha256(
        challenge_id=challenge_id,
        commitment_nonce=nonce,
        target_inputs_sha256=target_digest,
    )
    if challenge.get("target_commitment_sha256") != commitment:
        raise ValueError("hidden target commitment does not match its opening")
    return {
        "task_id": TASK_ID,
        "device_id": device_id,
        "challenge_id": challenge_id,
        "commitment_scheme": commitment_scheme,
        "target_inputs": top_level_targets,
        "target_inputs_sha256": target_digest,
        "target_commitment_sha256": commitment,
        "commitment_nonce": nonce,
    }


def _validate_reveal_receipt(event: dict[str, Any], *, trusted: dict[str, Any]) -> dict[str, Any]:
    _validate_reveal_event_envelope(event)
    for field_name, expected in trusted.items():
        if event.get(field_name) != expected:
            raise ValueError(f"sealed reveal {field_name} diverges from the hidden truth challenge")
    return {field_name: event[field_name] for field_name in _REVEAL_REQUIRED_FIELDS}


def logged_budget(events: list[dict[str, Any]]) -> tuple[int, int]:
    """Authoritative cumulative (shots_used, jobs_used) from qsim evidence.

    Scoped to events BEFORE the last logged ``submit_final_answer`` when one
    exists: the scored answer can only bind to evidence that preceded it, and
    post-final spending must not enter the budget the agent is held to.
    """
    final_idx = None
    for i, event in enumerate(events):
        if event.get("action") == "submit_final_answer":
            final_idx = i
    scope = events if final_idx is None else events[:final_idx]
    # Once measurements are sealed, the receipt is the authoritative snapshot
    # even if an already-admitted worker completes later. This is the only
    # budget that the final answer may cite.
    receipts = [event for event in scope if event.get("action") == SEALED_REVEAL_ACTION]
    if receipts:
        return int(receipts[-1]["shots_used"]), int(receipts[-1]["jobs_used"])

    shots, jobs = 0, 0
    for event in scope:
        if event.get("action") == "basis_shots_result" and "shots_used" in event:
            shots = max(shots, int(event["shots_used"]))
            jobs = max(jobs, int(event["jobs_used"]))
    return shots, jobs


def _require_device_clock(event: dict[str, Any]) -> None:
    """When the device clock is present it must be a finite non-negative number."""
    value = event.get("elapsed_wall_clock_s")
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("qsim event elapsed_wall_clock_s must be a finite number")
    if value < 0:
        raise ValueError("qsim event elapsed_wall_clock_s must be non-negative")


def _require_qsim_event_shape(
    event: dict[str, Any],
    *,
    action: str,
    required_fields: frozenset[str],
    optional_fields: frozenset[str] = frozenset(),
) -> None:
    required = required_fields | _QSIM_EVENT_REQUIRED_ENVELOPE_FIELDS
    optional = optional_fields | _QSIM_EVENT_OPTIONAL_ENVELOPE_FIELDS
    missing = required - set(event)
    unknown = set(event) - required - optional
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unknown:
            details.append(f"unknown {sorted(unknown)}")
        raise ValueError(f"corrupt {action} qsim event ({'; '.join(details)})")
    if event.get("action") != action:
        raise ValueError(f"corrupt {action} qsim event action")
    for field_name in ("ts", "surface"):
        if not isinstance(event.get(field_name), str) or not event[field_name]:
            raise ValueError(f"corrupt {action} qsim event {field_name}")
    try:
        timestamp = datetime.fromisoformat(event["ts"])
    except ValueError:
        raise ValueError(f"corrupt {action} qsim event ts") from None
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"corrupt {action} qsim event ts")
    if "execution_context_id" in event and (
        not isinstance(event["execution_context_id"], str) or not event["execution_context_id"]
    ):
        raise ValueError(f"corrupt {action} qsim event execution_context_id")


def _valid_job_id(value: Any) -> bool:
    return isinstance(value, str) and _PUBLIC_JOB_RESULT_NAME.fullmatch(f"{value}.json") is not None


def _require_positive_int(value: Any, *, owner: str, maximum: int | None = None) -> int:
    if type(value) is not int or value <= 0 or (maximum is not None and value > maximum):
        suffix = "" if maximum is None else f" no greater than {maximum}"
        raise ValueError(f"{owner} must be a positive integer{suffix}")
    return value


def _validate_result_blocks(value: Any, *, owner: str) -> tuple[int, int]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_BLOCKS_PER_JOB:
        raise ValueError(f"{owner} blocks violate the public per-job block bound")
    total_settings = 0
    total_shots = 0
    for block_index, block in enumerate(value):
        if not isinstance(block, dict) or set(block) != {"x", "settings"}:
            raise ValueError(f"{owner} block {block_index} has a corrupt envelope")
        x = block["x"]
        if not isinstance(x, list) or len(x) != INPUT_DIM:
            raise ValueError(f"{owner} block {block_index} has an invalid input dimension")
        for value_index, item in enumerate(x):
            if (
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(item)
                or not -math.pi <= item <= math.pi
            ):
                raise ValueError(
                    f"{owner} block {block_index} input {value_index} is not finite in [-pi, pi]"
                )
        settings = block["settings"]
        if not isinstance(settings, list) or not settings:
            raise ValueError(f"{owner} block {block_index} has no settings")
        total_settings += len(settings)
        if total_settings > MAX_SETTINGS_PER_JOB:
            raise ValueError(f"{owner} exceeds the public per-job setting bound")
        for setting_index, setting in enumerate(settings):
            if not isinstance(setting, dict) or set(setting) != {"basis", "shots", "digest"}:
                raise ValueError(
                    f"{owner} block {block_index} setting {setting_index} has a corrupt envelope"
                )
            basis = setting["basis"]
            if not isinstance(basis, str) or (
                basis != "random_pauli"
                and (len(basis) != N_QUBITS or any(char not in "xyz" for char in basis))
            ):
                raise ValueError(
                    f"{owner} block {block_index} setting {setting_index} has invalid basis"
                )
            total_shots += _require_positive_int(
                setting["shots"],
                owner=f"{owner} block {block_index} setting {setting_index} shots",
                maximum=MAX_SHOTS_PER_SETTING,
            )
            if not _is_sha256(setting["digest"]):
                raise ValueError(
                    f"{owner} block {block_index} setting {setting_index} has invalid digest"
                )
    if total_shots > TOTAL_SHOT_BUDGET:
        raise ValueError(f"{owner} exceeds the immutable shot budget")
    return len(value), total_shots


def _validate_offloaded_result_pointer(pointer: Any, *, job_id: str) -> dict[str, Any]:
    if not isinstance(pointer, dict) or set(pointer) != _OFFLOADED_RESULT_FIELDS:
        raise ValueError("offloaded_result must have the exact schema-1 field set")
    if type(pointer.get("schema_version")) is not int or pointer["schema_version"] != 1:
        raise ValueError("offloaded_result schema_version must be integer 1")

    relative_path = pointer.get("relative_path")
    expected_path = f"public_job_results/{PUBLISHED_RESULT_SUBDIR}/{job_id}.json"
    if (
        not isinstance(relative_path, str)
        or relative_path != expected_path
        or PurePosixPath(relative_path).is_absolute()
        or "\\" in relative_path
    ):
        raise ValueError(
            f"offloaded_result for job {job_id!r} has an invalid confined relative_path"
        )
    size = pointer.get("size_bytes")
    if type(size) is not int or size <= 0 or size > MAX_PUBLIC_JOB_RESULT_BYTES:
        raise ValueError(f"offloaded_result for job {job_id!r} has invalid size_bytes")
    if not _is_sha256(pointer.get("sha256")):
        raise ValueError(f"offloaded_result for job {job_id!r} has invalid sha256")
    return {
        "schema_version": 1,
        "relative_path": relative_path,
        "size_bytes": size,
        "sha256": pointer["sha256"],
    }


def _validate_task_qsim_event(event: dict[str, Any]) -> None:
    """Validate exact task-owned qsim event envelopes before attribution."""
    action = event.get("action")
    if action in _METADATA_ACTIONS:
        required = {"action", "tool", "metadata_calls_used", "max_metadata_calls"}
        if action != "list_devices":
            required.add("device_id")
        _require_qsim_event_shape(event, action=action, required_fields=frozenset(required))
        if event["tool"] != action:
            raise ValueError(f"corrupt {action} qsim event tool binding")
        if action != "list_devices" and event["device_id"] != DEVICE_ID:
            raise ValueError(f"corrupt {action} qsim event device binding")
        _require_positive_int(
            event["metadata_calls_used"],
            owner=f"{action} metadata call counter",
            maximum=MAX_METADATA_CALLS,
        )
        if (
            type(event["max_metadata_calls"]) is not int
            or event["max_metadata_calls"] != MAX_METADATA_CALLS
        ):
            raise ValueError(f"{action} max metadata calls violates the task contract")
        return

    if action == "submit_basis_shots":
        required = frozenset(
            {
                "action",
                "tool",
                "device_id",
                "n_blocks",
                "requested_shots",
                "job_id",
                "salt",
                "admission_sequence",
            }
        )
        _require_qsim_event_shape(event, action=action, required_fields=required)
        if event["tool"] != "run_basis_shots" or event["device_id"] != DEVICE_ID:
            raise ValueError("corrupt submit_basis_shots qsim event tool/device binding")
        if not _valid_job_id(event["job_id"]):
            raise ValueError("corrupt submit_basis_shots qsim event job_id")
        _require_positive_int(
            event["n_blocks"], owner="submit_basis_shots n_blocks", maximum=MAX_BLOCKS_PER_JOB
        )
        _require_positive_int(
            event["requested_shots"],
            owner="submit_basis_shots requested_shots",
            maximum=TOTAL_SHOT_BUDGET,
        )
        _require_positive_int(event["salt"], owner="submit_basis_shots salt")
        _require_positive_int(
            event["admission_sequence"],
            owner="submit_basis_shots admission_sequence",
            maximum=MAX_JOBS,
        )
        return

    if action == "basis_shots_result":
        failed = event.get("status") == "failed"
        required = frozenset(
            {
                "action",
                "tool",
                "device_id",
                "job_id",
                "admission_sequence",
                "status",
                "error",
            }
            if failed
            else {
                "action",
                "tool",
                "device_id",
                "job_id",
                "admission_sequence",
                "blocks",
                "shots_used",
                "total_shot_budget",
                "jobs_used",
                "max_jobs",
                "salt",
            }
        )
        _require_qsim_event_shape(
            event,
            action=action,
            required_fields=required,
            optional_fields=_DEVICE_CLOCK_OPTIONAL_FIELDS,
        )
        _require_device_clock(event)
        if event["tool"] != "run_basis_shots" or event["device_id"] != DEVICE_ID:
            raise ValueError("corrupt basis_shots_result qsim event tool/device binding")
        if not _valid_job_id(event["job_id"]):
            raise ValueError("corrupt basis_shots_result qsim event job_id")
        _require_positive_int(
            event["admission_sequence"],
            owner="basis_shots_result admission_sequence",
            maximum=MAX_JOBS,
        )
        if failed:
            if not isinstance(event["error"], str) or not event["error"]:
                raise ValueError("corrupt failed basis_shots_result qsim event error")
            return
        if "status" in event:
            raise ValueError("successful basis_shots_result qsim event must omit status")
        _validate_result_blocks(event["blocks"], owner="basis_shots_result")
        _require_positive_int(
            event["shots_used"],
            owner="basis_shots_result shots_used",
            maximum=TOTAL_SHOT_BUDGET,
        )
        if (
            event["total_shot_budget"] != TOTAL_SHOT_BUDGET
            or type(event["total_shot_budget"]) is not int
        ):
            raise ValueError("basis_shots_result total_shot_budget violates the task contract")
        _require_positive_int(
            event["jobs_used"], owner="basis_shots_result jobs_used", maximum=MAX_JOBS
        )
        if event["max_jobs"] != MAX_JOBS or type(event["max_jobs"]) is not int:
            raise ValueError("basis_shots_result max_jobs violates the task contract")
        _require_positive_int(event["salt"], owner="basis_shots_result salt")
        return

    if action == "get_job_result":
        status = event.get("status")
        required = {
            "action",
            "tool",
            "job_id",
            "status",
            "job_result_polls_used",
            "max_job_result_polls",
        }
        if status == "complete":
            required.add("offloaded_result")
        elif status == "failed":
            required.add("error")
        _require_qsim_event_shape(event, action=action, required_fields=frozenset(required))
        if event["tool"] != "get_job_result" or not _valid_job_id(event["job_id"]):
            raise ValueError("corrupt get_job_result qsim event tool/job binding")
        if status not in {"queued", "running", "complete", "failed"}:
            raise ValueError("corrupt get_job_result qsim event status")
        _require_positive_int(event["job_result_polls_used"], owner="get_job_result poll counter")
        _require_positive_int(event["max_job_result_polls"], owner="get_job_result max poll count")
        if status == "complete":
            _validate_offloaded_result_pointer(event["offloaded_result"], job_id=event["job_id"])
        elif status == "failed" and (not isinstance(event["error"], str) or not event["error"]):
            raise ValueError("corrupt failed get_job_result qsim event error")


def verify_backend_result_disposition(log_events: list[dict[str, Any]]) -> None:
    """Validate task events and raise on qsim failures, including no-answer runs."""
    final_submission_counters: list[int] = []
    for event in log_events:
        if not isinstance(event, dict):
            raise ValueError("qsim evidence event must be an object")
        action = event.get("action")
        if type(action) is not str or action not in _KNOWN_TASK_QSIM_ACTIONS:
            raise ValueError(f"unknown qsim evidence action {action!r}")
        if action in _METADATA_ACTIONS | {
            "submit_basis_shots",
            "basis_shots_result",
            "get_job_result",
        }:
            _validate_task_qsim_event(event)
        if action == SEALED_REVEAL_ACTION or (
            event.get("tool") == "lock_measurements_and_reveal_challenge"
            and action != "sealed_holdout_reveal_failure"
        ):
            _validate_reveal_event_envelope(event)
        if action == "submit_final_answer" or (
            event.get("tool") == "submit_final_answer" and action != "submit_final_answer_failure"
        ):
            _validate_final_submission_event(event)
            final_submission_counters.append(event["final_answer_submissions_used"])
        if action == "metadata_call_failure":
            required = frozenset(
                {
                    "action",
                    "tool",
                    "failed_action",
                    "failure_kind",
                    "failure_stage",
                }
            )
            try:
                _require_qsim_event_shape(event, action=action, required_fields=required)
            except ValueError:
                raise ValueError("malformed qsim metadata-call failure marker") from None
            if (
                event.get("tool") not in _METADATA_ACTIONS
                or event.get("failed_action") != event.get("tool")
                or event.get("failure_kind") != "qsim_internal"
                or event.get("failure_stage") != "metadata_log_publication"
            ):
                raise ValueError("malformed qsim metadata-call failure marker")
            raise ValueError(
                "qsim could not publish metadata-call evidence: infrastructure disposition"
            )
        if action == "submit_basis_shots_failure":
            required = frozenset(
                {
                    "action",
                    "tool",
                    "device_id",
                    "job_id",
                    "admission_sequence",
                    "failure_kind",
                    "failure_stage",
                }
            )
            try:
                _require_qsim_event_shape(event, action=action, required_fields=required)
            except ValueError:
                raise ValueError("malformed qsim basis-shot submission failure marker") from None
            if (
                event.get("tool") != "run_basis_shots"
                or event.get("device_id") != DEVICE_ID
                or not _valid_job_id(event.get("job_id"))
                or type(event.get("admission_sequence")) is not int
                or event["admission_sequence"] <= 0
                or event.get("failure_kind") != "qsim_internal"
                or event.get("failure_stage") not in _SUBMISSION_FAILURE_STAGES
            ):
                raise ValueError("malformed qsim basis-shot submission failure marker")
            raise ValueError(
                "qsim could not publish an admitted basis-shot job: infrastructure disposition"
            )
        if action == "basis_shots_result_failure":
            required = frozenset(
                {
                    "action",
                    "tool",
                    "device_id",
                    "job_id",
                    "admission_sequence",
                    "failure_kind",
                    "failure_stage",
                }
            )
            try:
                _require_qsim_event_shape(event, action=action, required_fields=required)
            except ValueError:
                raise ValueError("malformed qsim basis-shot result failure marker") from None
            if (
                event.get("tool") != "run_basis_shots"
                or event.get("device_id") != DEVICE_ID
                or not _valid_job_id(event.get("job_id"))
                or type(event.get("admission_sequence")) is not int
                or not 1 <= event["admission_sequence"] <= MAX_JOBS
                or event.get("failure_kind") != "qsim_internal"
                or event.get("failure_stage") not in _BASIS_RESULT_FAILURE_STAGES
            ):
                raise ValueError("malformed qsim basis-shot result failure marker")
            raise ValueError(
                "qsim could not publish basis-shot result evidence: infrastructure disposition"
            )
        if action == "sealed_holdout_reveal_failure":
            required = frozenset(
                {
                    "action",
                    "tool",
                    "task_id",
                    "device_id",
                    "failure_kind",
                    "failure_stage",
                }
            )
            try:
                _require_qsim_event_shape(event, action=action, required_fields=required)
            except ValueError:
                raise ValueError("malformed qsim sealed-reveal failure marker") from None
            if (
                event.get("tool") != "lock_measurements_and_reveal_challenge"
                or event.get("task_id") != TASK_ID
                or event.get("device_id") != DEVICE_ID
                or event.get("failure_kind") != "qsim_internal"
                or event.get("failure_stage") not in _REVEAL_FAILURE_STAGES
            ):
                raise ValueError("malformed qsim sealed-reveal failure marker")
            raise ValueError(
                "qsim could not publish the sealed holdout reveal: infrastructure disposition"
            )
        if action == "get_job_result_failure":
            required = frozenset(
                {"action", "tool", "job_id", "status", "failure_kind", "failure_stage"}
            )
            try:
                _require_qsim_event_shape(event, action=action, required_fields=required)
            except ValueError:
                raise ValueError("malformed qsim result-publication failure marker") from None
            if (
                event.get("tool") != "get_job_result"
                or event.get("failure_kind") != "qsim_internal"
                or event.get("failure_stage") not in _RESULT_FAILURE_STAGES
                or not _valid_job_id(event.get("job_id"))
                or event.get("status") not in {"queued", "running", "complete", "failed"}
            ):
                raise ValueError("malformed qsim result-publication failure marker")
            raise ValueError(
                f"qsim could not publish result evidence for job {event['job_id']!r}: "
                "infrastructure disposition"
            )
        if action == "submit_final_answer_failure":
            required = frozenset({"action", "tool", "task_id", "failure_kind", "failure_stage"})
            try:
                _require_qsim_event_shape(event, action=action, required_fields=required)
            except ValueError:
                raise ValueError("malformed qsim final-publication failure marker") from None
            if (
                event.get("tool") != "submit_final_answer"
                or event.get("task_id") != TASK_ID
                or event.get("failure_kind") != "qsim_internal"
                or event.get("failure_stage") not in _FINAL_FAILURE_STAGES
            ):
                raise ValueError("malformed qsim final-publication failure marker")
            raise ValueError(
                "qsim could not publish final-answer evidence: infrastructure disposition"
            )
        if event.get("action") == "basis_shots_result" and event.get("status") == "failed":
            raise ValueError(
                f"qsim backend failed basis-shot job {event.get('job_id')!r}: "
                "infrastructure disposition"
            )
        if event.get("action") == "get_job_result" and event.get("status") == "failed":
            raise ValueError(
                f"qsim returned failed job {event.get('job_id')!r}: infrastructure disposition"
            )
    if final_submission_counters != list(range(1, len(final_submission_counters) + 1)):
        raise ValueError("logged final-answer counters do not exactly recount submissions")
    if len(final_submission_counters) > MAX_FINAL_ANSWER_SUBMISSIONS:
        raise ValueError("logged final-answer submissions exceed the task contract")


def _raw_setting_digest(bitstrings: list[str], random_bases: list[str] | None) -> str:
    """Mirror the qsim action's digest over one raw basis-shot setting."""
    hasher = hashlib.sha256()
    for row in bitstrings:
        hasher.update(row.encode("ascii"))
        hasher.update(b"\n")
    if random_bases is not None:
        for row in random_bases:
            hasher.update(row.encode("ascii"))
            hasher.update(b"\n")
    return hasher.hexdigest()


def _validated_offload_pointers(
    log_events: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return the unique exact runtime pointer for each offloaded completed job."""
    basis_result_job_ids = {
        event.get("job_id")
        for event in log_events
        if event.get("action") == "basis_shots_result"
        and event.get("status") != "failed"
        and isinstance(event.get("job_id"), str)
    }
    pointers: dict[str, dict[str, Any]] = {}
    complete_polls: dict[str, list[bool]] = {}
    for event in log_events:
        has_pointer = "offloaded_result" in event
        pointer = event.get("offloaded_result")
        action = event.get("action")
        status = event.get("status")
        job_id = event.get("job_id")
        if action == "get_job_result" and status == "complete" and isinstance(job_id, str):
            complete_polls.setdefault(job_id, []).append(has_pointer)
        if not has_pointer:
            continue
        if action != "get_job_result" or status != "complete":
            raise ValueError("offloaded_result may appear only on a completed get_job_result event")
        if (
            not isinstance(job_id, str)
            or _PUBLIC_JOB_RESULT_NAME.fullmatch(f"{job_id}.json") is None
        ):
            raise ValueError("offloaded_result is attached to an unsafe job_id")
        if not isinstance(pointer, dict) or set(pointer) != _OFFLOADED_RESULT_FIELDS:
            raise ValueError("offloaded_result must have the exact schema-1 field set")
        if pointer.get("schema_version") != 1 or isinstance(pointer.get("schema_version"), bool):
            raise ValueError("offloaded_result schema_version must be integer 1")

        relative_path = pointer.get("relative_path")
        expected_path = f"public_job_results/{PUBLISHED_RESULT_SUBDIR}/{job_id}.json"
        if (
            not isinstance(relative_path, str)
            or relative_path != expected_path
            or PurePosixPath(relative_path).is_absolute()
            or "\\" in relative_path
        ):
            raise ValueError(
                f"offloaded_result for job {job_id!r} has an invalid confined relative_path"
            )
        size = pointer.get("size_bytes")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or size > MAX_PUBLIC_JOB_RESULT_BYTES
        ):
            raise ValueError(f"offloaded_result for job {job_id!r} has invalid size_bytes")
        if not _is_sha256(pointer.get("sha256")):
            raise ValueError(f"offloaded_result for job {job_id!r} has invalid sha256")

        normalized = {
            "schema_version": 1,
            "relative_path": relative_path,
            "size_bytes": size,
            "sha256": pointer["sha256"],
        }
        prior = pointers.get(job_id)
        if prior is not None and prior != normalized:
            raise ValueError(f"completed polls for job {job_id!r} carry conflicting pointers")
        pointers[job_id] = normalized

    for job_id, pointer_presence in complete_polls.items():
        if job_id in basis_result_job_ids and not all(pointer_presence):
            raise ValueError(
                f"every completed basis_shots poll for job {job_id!r} must carry "
                "an offloaded_result pointer"
            )
        if any(pointer_presence) and not all(pointer_presence):
            raise ValueError(
                f"completed polls for job {job_id!r} inconsistently record offload evidence"
            )
    return pointers


def verify_public_job_result_artifacts(
    log_events: list[dict[str, Any]], artifact_dir: Path
) -> dict[str, dict[str, Any]]:
    """Validate the closed, flat set of copied raw basis-shot artifacts.

    ``artifact_dir`` is the ``public_job_results`` tree root (the qsim mount
    root, or the verifier's copy of it). Delivery artifacts are published one
    level below it, under ``PUBLISHED_RESULT_SUBDIR``.

    The structured pointer on the completed poll is the authoritative
    declaration that an artifact exists. Each declared file is bound to that
    pointer, the qsim-owned submission/result chain, per-setting raw-data
    digests, and the task's baked public device contract. The returned manifest
    lets the Harbor wrapper compare source and copied trees byte-for-byte.
    """
    from qiqcbench.qsim.core.wire import SCHEMA_VERSION, JobResult
    from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.wire import (
        RANDOM_PAULI,
        JobBasisShotData,
    )

    pointers = _validated_offload_pointers(log_events)
    mount_root = Path(artifact_dir)
    if mount_root.is_symlink():
        raise ValueError(f"public_job_results must not be a symlink: {mount_root}")
    if mount_root.exists() and not mount_root.is_dir():
        raise ValueError(f"public_job_results is not a directory: {mount_root}")
    # The level this inventory LISTS must track the level the producer WRITES.
    # Delivery artifacts live one directory down, so listing the mount
    # root would find only subdirectory entries and report an empty-but-closed
    # inventory -- a silent evidence weakening that fails nothing. Each
    # component is validated separately, exactly as the producer validates them:
    # the mount root cannot be unlinked, but this subdirectory can, so a hostile
    # or corrupted replacement of it must not be followed.
    # Close the MOUNT ROOT too, not only the directory below it. Validating the
    # root's type and then enumerating one level down leaves an illegal entry
    # sitting directly in the root -- an artifact from a former layout, or corrupt
    # transfer debris -- invisible to the closure claim, so it can neither be
    # rejected nor carried into the separate verifier's evidence. Only the two
    # contractual publication subdirectories may appear here; both are
    # directories, and nothing is ever published into the root itself.
    if mount_root.is_dir():
        allowed_at_root = {PUBLISHED_RESULT_SUBDIR, PUBLISHED_RAW_RECORD_SUBDIR}
        undeclared_at_root = sorted(
            entry.name for entry in mount_root.iterdir() if entry.name not in allowed_at_root
        )
        if undeclared_at_root:
            raise ValueError(
                "public_job_results mount root holds undeclared entries: "
                f"{undeclared_at_root}"
            )
        # An allowed NAME is not an allowed thing. Both publication targets are
        # directories; a regular file or a symlink wearing one of these names is
        # transfer debris or a redirect, and admitting it on the name alone
        # degrades a verifier-owned fault into an ordinary model rejection.
        # Absent is still fine -- neither subdirectory is mandatory.
        for name in sorted(allowed_at_root):
            entry = mount_root / name
            if entry.is_symlink() or (entry.exists() and not entry.is_dir()):
                raise ValueError(
                    f"public_job_results/{name} must be a real directory: {entry}"
                )

    root = mount_root / PUBLISHED_RESULT_SUBDIR
    if root.is_symlink():
        raise ValueError(f"public_job_results must not be a symlink: {root}")
    if not root.exists():
        if not pointers:
            return {}
        raise ValueError(f"public_job_results is missing despite declared offload pointers: {root}")
    if not root.is_dir():
        raise ValueError(f"public_job_results is not a directory: {root}")
    resolved_root = root.resolve(strict=True)

    submissions = {
        event.get("job_id"): event
        for event in log_events
        if event.get("action") == "submit_basis_shots" and isinstance(event.get("job_id"), str)
    }
    results = {
        event.get("job_id"): event
        for event in log_events
        if event.get("action") == "basis_shots_result"
        and event.get("status") != "failed"
        and isinstance(event.get("job_id"), str)
    }
    entries = sorted(root.iterdir(), key=lambda path: path.name)
    if not pointers and entries:
        raise ValueError("public_job_results contains files not declared by offload pointers")
    expected_names = {f"{job_id}.json" for job_id in pointers}
    actual_names = {entry.name for entry in entries}
    if actual_names != expected_names:
        raise ValueError(
            "public_job_results inventory does not equal the authoritative offload pointers; "
            f"missing={sorted(expected_names - actual_names)}, "
            f"undeclared={sorted(actual_names - expected_names)}"
        )

    manifest: dict[str, dict[str, Any]] = {}
    total_size = 0
    for entry in entries:
        match = _PUBLIC_JOB_RESULT_NAME.fullmatch(entry.name)
        if (
            match is None
            or entry.is_symlink()
            or not entry.is_file()
            or entry.resolve(strict=True).parent != resolved_root
        ):
            raise ValueError(
                "public_job_results inventory is not a flat confined set of job JSON "
                f"files: {entry.name!r}"
            )
        job_id = match.group(1)
        pointer = pointers[job_id]
        submission = submissions.get(job_id)
        result_event = results.get(job_id)
        if submission is None or result_event is None:
            raise ValueError(
                f"public job-result artifact {entry.name!r} has no declared "
                "submit -> result -> completed-poll evidence chain"
            )

        size = entry.stat().st_size
        if size <= 0 or size > MAX_PUBLIC_JOB_RESULT_BYTES:
            raise ValueError(
                f"public job-result artifact {entry.name!r} has invalid byte size {size}"
            )
        total_size += size
        if total_size > MAX_PUBLIC_JOB_RESULTS_TOTAL_BYTES:
            raise ValueError("public_job_results exceeds the verifier's bounded total byte size")
        raw_bytes = entry.read_bytes()
        if len(raw_bytes) != size:
            raise ValueError(f"public job-result artifact {entry.name!r} changed while reading")
        if size != pointer["size_bytes"]:
            raise ValueError(
                f"public job-result artifact {entry.name!r} does not match pointer size_bytes"
            )
        artifact_digest = hashlib.sha256(raw_bytes).hexdigest()
        if artifact_digest != pointer["sha256"]:
            raise ValueError(
                f"public job-result artifact {entry.name!r} does not match pointer sha256"
            )
        try:
            payload = JobResult.model_validate(json.loads(raw_bytes))
        except Exception:
            raise ValueError(
                f"public job-result artifact {entry.name!r} is not a valid JobResult"
            ) from None
        if (
            payload.schema_version != SCHEMA_VERSION
            or payload.job_id != job_id
            or payload.device_id != DEVICE_ID
            or submission.get("device_id") != DEVICE_ID
            or result_event.get("device_id") != DEVICE_ID
            or payload.status != "complete"
            or not isinstance(payload.data, JobBasisShotData)
        ):
            raise ValueError(
                f"public job-result artifact {entry.name!r} does not match its logged job envelope"
            )
        if payload.shots != submission.get("requested_shots"):
            raise ValueError(
                f"public job-result artifact {entry.name!r} shot count disagrees with submission"
            )

        if not 1 <= len(payload.data.blocks) <= MAX_BLOCKS_PER_JOB:
            raise ValueError(
                f"public job-result artifact {entry.name!r} violates the public block cap"
            )
        setting_count = sum(len(block.settings) for block in payload.data.blocks)
        if not 1 <= setting_count <= MAX_SETTINGS_PER_JOB:
            raise ValueError(
                f"public job-result artifact {entry.name!r} violates the public setting cap"
            )
        artifact_shots = 0

        logged_blocks = result_event.get("blocks")
        if not isinstance(logged_blocks, list) or len(payload.data.blocks) != len(logged_blocks):
            raise ValueError(
                f"public job-result artifact {entry.name!r} block inventory disagrees with log"
            )
        for block_index, (block, logged_block) in enumerate(
            zip(payload.data.blocks, logged_blocks, strict=True)
        ):
            if len(block.x) != INPUT_DIM or any(
                not math.isfinite(value) or not -math.pi <= value <= math.pi for value in block.x
            ):
                raise ValueError(
                    f"public job-result artifact {entry.name!r} block {block_index} "
                    "violates the public input dimension/domain contract"
                )
            if not isinstance(logged_block, dict) or block.x != logged_block.get("x"):
                raise ValueError(
                    f"public job-result artifact {entry.name!r} block {block_index} "
                    "input disagrees with log"
                )
            logged_settings = logged_block.get("settings")
            if not isinstance(logged_settings, list) or len(block.settings) != len(logged_settings):
                raise ValueError(
                    f"public job-result artifact {entry.name!r} block {block_index} "
                    "setting inventory disagrees with log"
                )
            for setting_index, (setting, logged_setting) in enumerate(
                zip(block.settings, logged_settings, strict=True)
            ):
                fixed_basis = len(setting.basis) == N_QUBITS and set(setting.basis).issubset(
                    {"x", "y", "z"}
                )
                if (
                    not isinstance(logged_setting, dict)
                    or setting.basis != logged_setting.get("basis")
                    or setting.shots != logged_setting.get("shots")
                    or not 1 <= setting.shots <= MAX_SHOTS_PER_SETTING
                    or (setting.basis != RANDOM_PAULI and not fixed_basis)
                    or len(setting.bitstrings) != setting.shots
                    or any(
                        len(bitstring) != N_QUBITS or not set(bitstring).issubset({"0", "1"})
                        for bitstring in setting.bitstrings
                    )
                    or (
                        setting.basis == RANDOM_PAULI
                        and (
                            setting.random_bases is None
                            or len(setting.random_bases) != setting.shots
                            or any(
                                len(random_basis) != N_QUBITS
                                or not set(random_basis).issubset({"x", "y", "z"})
                                for random_basis in setting.random_bases
                            )
                        )
                    )
                    or (setting.basis != RANDOM_PAULI and setting.random_bases is not None)
                ):
                    raise ValueError(
                        f"public job-result artifact {entry.name!r} block {block_index} "
                        f"setting {setting_index} shape disagrees with log"
                    )
                artifact_shots += setting.shots
                try:
                    setting_digest = _raw_setting_digest(setting.bitstrings, setting.random_bases)
                except UnicodeEncodeError:
                    raise ValueError(
                        f"public job-result artifact {entry.name!r} contains non-ASCII raw data"
                    ) from None
                if setting_digest != logged_setting.get("digest"):
                    raise ValueError(
                        f"public job-result artifact {entry.name!r} block {block_index} "
                        f"setting {setting_index} raw-data digest disagrees with log"
                    )

        if artifact_shots != payload.shots:
            raise ValueError(
                f"public job-result artifact {entry.name!r} shot total violates the public contract"
            )

        for field_name in ("shots_used", "total_shot_budget", "jobs_used", "max_jobs"):
            if getattr(payload.data.budget, field_name) != result_event.get(field_name):
                raise ValueError(
                    f"public job-result artifact {entry.name!r} budget {field_name} "
                    "disagrees with log"
                )
        budget = payload.data.budget
        if (
            budget.total_shot_budget != TOTAL_SHOT_BUDGET
            or budget.max_jobs != MAX_JOBS
            or not 1 <= budget.jobs_used <= MAX_JOBS
            or not payload.shots <= budget.shots_used <= TOTAL_SHOT_BUDGET
        ):
            raise ValueError(
                f"public job-result artifact {entry.name!r} violates the public budget contract"
            )
        manifest[entry.name] = {
            "relative_path": pointer["relative_path"],
            "size_bytes": size,
            "sha256": artifact_digest,
        }

    return manifest


def verify_lifecycle_evidence(
    log_events: list[dict[str, Any]], truth: dict[str, Any]
) -> _SealedLifecycle:
    """Validate task experiment/reveal evidence independently of a final answer.

    This is exported for the verifier's no-answer path: a missing final answer
    must not hide corrupt backend, admission, or sealed-challenge evidence.
    """
    validate_hidden_truth(truth)
    verify_backend_result_disposition(log_events)

    trusted_challenge = _trusted_target_challenge(truth)

    submissions: dict[str, tuple[int, int, int, int, int]] = {}
    sequences: dict[int, str] = {}
    results: set[str] = set()
    last_sequence = 0
    reveals: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    for event_index, event in enumerate(log_events):
        action = event.get("action")
        if action == "submit_basis_shots":
            job_id = event.get("job_id")
            sequence = event.get("admission_sequence")
            requested_shots = event.get("requested_shots")
            salt = event["salt"]
            n_blocks = event["n_blocks"]
            if job_id in submissions or sequence in sequences:
                raise ValueError("duplicate basis-shot job_id or admission_sequence")
            if sequence != last_sequence + 1:
                raise ValueError(
                    f"inconsistent admission_sequence {sequence}; expected {last_sequence + 1}"
                )
            submissions[job_id] = (event_index, sequence, requested_shots, salt, n_blocks)
            sequences[sequence] = job_id
            last_sequence = sequence
            if sum(submission[2] for submission in submissions.values()) > TOTAL_SHOT_BUDGET:
                raise ValueError("logged basis-shot admissions exceed the immutable shot budget")
        elif action == "basis_shots_result":
            job_id = event.get("job_id")
            if job_id not in submissions:
                raise ValueError(
                    f"basis_shots_result for job {job_id!r} has no logged submission event"
                )
            sequence = event.get("admission_sequence")
            if sequence != submissions[job_id][1]:
                raise ValueError(
                    f"basis_shots_result for job {job_id!r} has an inconsistent admission_sequence"
                )
            block_count, result_shots = _validate_result_blocks(
                event["blocks"], owner=f"basis_shots_result for job {job_id!r}"
            )
            _, _, requested_shots, salt, n_blocks = submissions[job_id]
            if block_count != n_blocks or result_shots != requested_shots:
                raise ValueError(
                    f"basis_shots_result for job {job_id!r} disagrees with its submission cost"
                )
            if event["salt"] != salt:
                raise ValueError(f"basis_shots_result for job {job_id!r} has an inconsistent salt")
            # The worker snapshots counters under the admission lock, then
            # releases that lock before it publishes this result. Later
            # submissions may therefore be visible above this line even though
            # they are absent from the snapshot. The reported job count still
            # names an exact monotone admission prefix containing this job.
            reported_jobs_used = event["jobs_used"]
            expected_shots_used = sum(
                submission[2]
                for submission in submissions.values()
                if submission[1] <= reported_jobs_used
            )
            if (
                not sequence <= reported_jobs_used <= len(submissions)
                or event["shots_used"] != expected_shots_used
            ):
                raise ValueError(
                    f"basis_shots_result for job {job_id!r} has inexact cumulative counters"
                )
            if job_id in results:
                raise ValueError(f"duplicate basis_shots_result for job {job_id!r}")
            results.add(job_id)
        elif action == "get_job_result":
            job_id = event["job_id"]
            if job_id not in submissions:
                raise ValueError(
                    f"get_job_result for job {job_id!r} has no logged submission event"
                )
            if event["status"] == "complete" and job_id not in results:
                raise ValueError(
                    f"completed get_job_result for job {job_id!r} precedes its result event"
                )
            # A poll can snapshot ``queued`` or ``running``, then lose the log
            # append race to the worker's terminal event. The resulting event
            # order is not a device-state regression and cannot be distinguished
            # from one without an action-owned snapshot sequence. A completed
            # poll is still required before a submission can pass G3.
        elif action == SEALED_REVEAL_ACTION:
            receipt = _validate_reveal_receipt(event, trusted=trusted_challenge)
            reveals.append((event_index, event, receipt))

    if not reveals:
        return _SealedLifecycle(receipt=None, reveal_index=None)

    if [event["sealed_holdout_reveals_used"] for _, event, _ in reveals] != list(
        range(1, len(reveals) + 1)
    ):
        raise ValueError("sealed holdout reveal counters do not exactly recount accepted reveals")
    if len(reveals) > MAX_SEALED_HOLDOUT_REVEALS:
        raise ValueError("sealed holdout reveal count exceeds the task contract")

    first_index, _, first_receipt = reveals[0]
    for _, _, receipt in reveals[1:]:
        if receipt != first_receipt:
            raise ValueError("duplicate sealed holdout reveals carry differing receipts")

    repeat_markers = [event.get("is_repeat") for _, event, _ in reveals]
    if any(marker is not None for marker in repeat_markers):
        if any(marker is None for marker in repeat_markers):
            raise ValueError("sealed reveal repetition markers are only partially recorded")
        if repeat_markers[0] is not False or any(
            marker is not True for marker in repeat_markers[1:]
        ):
            raise ValueError("sealed reveal log must contain exactly one first transition")

    pre_seal = [
        (job_id, sequence, shots)
        for job_id, (event_index, sequence, shots, _, _) in submissions.items()
        if event_index < first_index
    ]
    expected_cutoff = max((sequence for _, sequence, _ in pre_seal), default=0)
    expected_shots = sum(shots for _, _, shots in pre_seal)
    expected_jobs = len(pre_seal)
    if first_receipt["last_admission_sequence"] != expected_cutoff:
        raise ValueError("sealed reveal last_admission_sequence is inconsistent with submissions")
    if first_receipt["shots_used"] != expected_shots or first_receipt["jobs_used"] != expected_jobs:
        raise ValueError("sealed reveal budget counters are inconsistent with admitted jobs")
    if first_receipt["jobs_used"] != first_receipt["last_admission_sequence"]:
        raise ValueError("sealed reveal job counter and admission cutoff disagree")
    if first_receipt["shots_used"] > TOTAL_SHOT_BUDGET or first_receipt["jobs_used"] > MAX_JOBS:
        raise ValueError("sealed reveal budget exceeds the immutable task contract")

    for job_id, (event_index, sequence, _, _, _) in submissions.items():
        if event_index <= first_index:
            continue
        if sequence <= first_receipt["last_admission_sequence"]:
            raise ValueError("a post-seal submission carries a pre-seal admission_sequence")
        raise ValueError(f"basis-shot job {job_id!r} was admitted after the irreversible seal")
    return _SealedLifecycle(
        receipt=first_receipt,
        reveal_index=first_index,
    )


def verify_evidence_channel(
    final_answer: dict[str, Any],
    log_events: list[dict[str, Any]],
    truth: dict[str, Any],
) -> _SealedLifecycle:
    """Raise on any final-artifact or lifecycle binding violation (infra)."""
    if final_answer.get("task_id") != TASK_ID:
        raise ValueError(
            f"artifact task_id {final_answer.get('task_id')!r} != {TASK_ID!r}: "
            "scoring materials are not this task's submission"
        )
    schema_version = final_answer.get("schema_version")
    if schema_version != FINAL_ANSWER_SCHEMA_VERSION:
        raise ValueError(f"artifact schema_version {schema_version!r} is not supported")
    unknown = set(final_answer) - {"task_id", "answer", "schema_version"}
    if unknown:
        raise ValueError(f"artifact carries unknown top-level fields {sorted(unknown)}")

    finals = [event for event in log_events if event.get("action") == "submit_final_answer"]
    if not finals:
        raise ValueError("final answer artifact has no logged submit_final_answer event")
    last_final = finals[-1]
    if last_final.get("task_id") != TASK_ID:
        raise ValueError(f"logged submission task_id {last_final.get('task_id')!r} != {TASK_ID!r}")
    if "answer" not in last_final:
        raise ValueError("logged submit_final_answer event carries no answer payload")
    if last_final["answer"] != final_answer.get("answer"):
        raise ValueError(
            "final answer artifact does not equal the latest logged submission payload"
        )
    return verify_lifecycle_evidence(log_events, truth)


def _informational_fields(
    answer: dict[str, Any], shots_logged: int, jobs_logged: int
) -> dict[str, Any]:
    """Record the human-readable / self-reported layer without gating on it."""
    info: dict[str, Any] = {}
    for key in ("protocol", "surrogate_model"):
        value = answer.get(key)
        info[key] = (
            value.strip()[:INFORMATIONAL_STRING_MAX_CHARS]
            if isinstance(value, str) and value.strip()
            else None
        )
    val_rmse = answer.get("validation_rmse")
    info["validation_rmse"] = (
        float(val_rmse)
        if isinstance(val_rmse, (int, float))
        and not isinstance(val_rmse, bool)
        and math.isfinite(val_rmse)
        else None
    )
    for key, logged in (("shots_used", shots_logged), ("jobs_used", jobs_logged)):
        reported = answer.get(key)
        ok = isinstance(reported, int) and not isinstance(reported, bool)
        info[key] = reported if ok else None
        info[f"{key}_matches_device"] = bool(ok and reported == logged)
    method = answer.get("method")
    info["method_present"] = isinstance(method, str) and bool(method.strip())
    info["unknown_fields"] = sorted(set(answer) - FINAL_ANSWER_FIELDS)
    return info


def score_submission(
    final_answer: dict[str, Any],
    log_events: list[dict[str, Any]],
    truth: dict[str, Any],
) -> ScoreResult:
    reasons: list[str] = []
    # Establish that this is the exact qsim-owned submission before inspecting
    # the task payload.  Otherwise a corrupt artifact with a malformed
    # ``answer`` can take the early model-failure path and hide a missing or
    # mismatched submission event that must be classified as infrastructure.
    lifecycle = verify_evidence_channel(final_answer, log_events, truth)
    answer = final_answer.get("answer")
    if not isinstance(answer, dict):
        return ScoreResult(
            passed=False,
            gates={
                "g1_accuracy": False,
                "g2_tail": False,
                "g3_validity": False,
            },
            metrics={},
            reject_reasons=["final answer has no 'answer' object"],
        )

    # ----- validity: shape / band / sealed-holdout citation -----
    # Physics binds before bookkeeping: validity reasons are collected here but
    # appended to ``reasons`` only AFTER the scientific gates below, so a wrong
    # answer is always reported for a scientific reason first.
    validity: list[str] = []
    predictions = _as_matrix(
        answer.get("predictions_as_measured"), "predictions_as_measured", validity
    )
    if predictions is not None and (
        np.any(predictions < PREDICTION_BAND_LOW) or np.any(predictions > PREDICTION_BAND_HIGH)
    ):
        _fail(
            validity,
            f"predictions outside the numerical guard band [{PREDICTION_BAND_LOW}, "
            f"{PREDICTION_BAND_HIGH}] around the physical spectrum {{-1, +1/3}}",
        )
    last_final_idx = max(
        (i for i, e in enumerate(log_events) if e.get("action") == "submit_final_answer"),
        default=len(log_events),
    )
    pre_final = log_events[:last_final_idx]
    if lifecycle.receipt is None:
        _fail(
            validity,
            "measurements were not sealed and the holdout targets were not revealed "
            "before submission",
        )
    else:
        if lifecycle.reveal_index is None or lifecycle.reveal_index >= last_final_idx:
            _fail(validity, "the sealed holdout reveal must precede the final answer")
        for field_name in ("challenge_id", "target_inputs_sha256"):
            if answer.get(field_name) != lifecycle.receipt[field_name]:
                _fail(
                    validity,
                    f"final answer {field_name} does not cite the sealed holdout receipt",
                )
    measured_ids = {
        e.get("job_id")
        for e in pre_final
        if e.get("action") == "basis_shots_result" and "shots_used" in e
    }
    if not any(
        e.get("action") == "get_job_result"
        and e.get("status") == "complete"
        and e.get("job_id") in measured_ids
        for e in pre_final
    ):
        _fail(
            validity,
            "no agent-visible completed get_job_result poll of a measured job "
            "precedes the final answer",
        )
    shots_logged, jobs_logged = logged_budget(log_events)
    if shots_logged <= 0:
        _fail(
            validity,
            "no completed basis_shots evidence precedes the scored final answer",
        )

    g3 = not validity
    metrics: dict[str, Any] = {
        "shots_used_logged": shots_logged,
        "jobs_used_logged": jobs_logged,
        "thresholds": scorer_policy(),
        "informational": _informational_fields(answer, shots_logged, jobs_logged),
    }

    if predictions is None:
        reasons.extend(validity)
        return ScoreResult(
            passed=False,
            gates={
                "g1_accuracy": False,
                "g2_tail": False,
                "g3_validity": g3,
            },
            metrics=metrics,
            reject_reasons=reasons,
        )

    # ----- G1-G2 vs hidden truth -----
    # The receipt is verified byte-for-byte against the hidden target table;
    # predictions are therefore interpreted in the row order qsim revealed.
    labels = np.asarray(truth["labels"], dtype=float)
    if labels.shape != (N_TARGETS, N_PAIRS):
        raise ValueError(f"hidden truth labels have unexpected shape {labels.shape}")
    residual = predictions - labels
    rmse = float(np.sqrt(np.mean(residual**2)))
    worst_target_rmse = float(np.max(np.sqrt(np.mean(residual**2, axis=1))))
    pair_rmse = np.sqrt(np.mean(residual**2, axis=0))
    max_pair = float(np.max(pair_rmse))
    max_cell_error = float(np.max(np.abs(residual)))
    g1 = rmse <= RMSE_THRESHOLD and worst_target_rmse <= WORST_TARGET_RMSE_THRESHOLD
    g2 = max_cell_error <= MAX_CELL_ERROR_THRESHOLD
    if not g1:
        _fail(
            reasons,
            f"G1 accuracy: hidden RMSE {rmse:.4f} must be <= {RMSE_THRESHOLD} and worst "
            f"per-target RMSE {worst_target_rmse:.4f} must be <= {WORST_TARGET_RMSE_THRESHOLD}",
        )
    if not g2:
        _fail(
            reasons,
            f"G2 tail: max absolute cell error {max_cell_error:.4f} must be <= "
            f"{MAX_CELL_ERROR_THRESHOLD}",
        )
    reasons.extend(validity)

    metrics.update(
        rmse=rmse,
        worst_target_rmse=worst_target_rmse,
        max_pair_rmse=max_pair,
        max_cell_error=max_cell_error,
        p99_pair_rmse=float(np.quantile(pair_rmse, 0.99, method="linear")),
    )
    return ScoreResult(
        passed=g1 and g2 and g3,
        gates={"g1_accuracy": g1, "g2_tail": g2, "g3_validity": g3},
        metrics=metrics,
        reject_reasons=reasons,
    )


# ---------- file-oriented entry point for the Harbor verifier ----------


def _json_object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def load_log_events(log_path: Path) -> list[dict[str, Any]]:
    """Parse the evidence log, failing CLOSED on any malformed line.

    Including the trailing line: qsim writes the final-answer artifact before
    appending its (large) log event, so a torn tail can hide the very event
    that binds the artifact — silently dropping it would turn an interrupted
    append into a model-attributed G3 failure. The wrapper converts the raise
    into the reserved infrastructure disposition instead.
    """
    lines = log_path.read_text(encoding="utf-8").splitlines()
    events: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        if not line.strip():
            raise ValueError(
                f"blank evidence log line {idx + 1} of {len(lines)}: evidence channel corrupt"
            )
        try:
            event = json.loads(
                line,
                object_pairs_hook=_json_object_without_duplicate_keys,
                parse_constant=_reject_nonfinite_json_constant,
            )
        except (json.JSONDecodeError, ValueError):
            raise ValueError(
                f"malformed evidence log line {idx + 1} of {len(lines)}: "
                "evidence channel corrupt or append interrupted"
            ) from None
        if not isinstance(event, dict):
            raise ValueError(f"evidence log line {idx + 1} of {len(lines)} is not a JSON object")
        events.append(event)
    return events


def score_files(final_answer_path: Path, log_path: Path, truth_path: Path) -> ScoreResult:
    final_answer = json.loads(final_answer_path.read_text())
    events = load_log_events(log_path)
    truth = json.loads(truth_path.read_text())
    return score_submission(final_answer, events, truth)
