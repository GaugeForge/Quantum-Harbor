"""Observable helpers for the digital VQE capability."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qiqcbench.qsim.core.wire import (
    CircuitOp,
    GateOp,
    JobObservableBitstringData,
    ObservableBatchPoint,
    ObservableBatchRequest,
    ObservableSettingBitstrings,
)

__all__ = [
    "OBSERVABLE_BATCH_EVIDENCE_CONTRACT",
    "OBSERVABLE_BATCH_INTERNAL_FAILURE_ERROR",
    "OBSERVABLE_BATCH_INTERNAL_FAILURE_KIND",
    "OBSERVABLE_BATCH_JOB_ENQUEUE_FAILURE_STAGE",
    "OBSERVABLE_BATCH_SUBMISSION_FAILURE_ACTION",
    "OBSERVABLE_BATCH_SUBMISSION_FAILURE_ERROR",
    "OBSERVABLE_BATCH_SUBMISSION_LOG_FAILURE_STAGE",
    "OBSERVABLE_BATCH_TERMINAL_LOG_FAILURE_STAGE",
    "OBSERVABLE_BATCH_EVIDENCE_SCHEMA_VERSION",
    "build_point_value_digest_rows",
    "compute_energy_and_standard_error",
    "compute_observable_batch_request_digest",
    "compute_point_value_digest",
    "compute_public_material_digests",
    "measurement_basis_ops",
    "summarize_observable_batch_result",
]


OBSERVABLE_BATCH_EVIDENCE_CONTRACT = "digital_vqe_observable_batch_v2"
OBSERVABLE_BATCH_EVIDENCE_SCHEMA_VERSION = 2
OBSERVABLE_BATCH_INTERNAL_FAILURE_KIND = "qsim_internal"
OBSERVABLE_BATCH_INTERNAL_FAILURE_ERROR = "observable-batch execution failed"
OBSERVABLE_BATCH_TERMINAL_LOG_FAILURE_STAGE = "terminal_result_log_publication"
OBSERVABLE_BATCH_SUBMISSION_FAILURE_ACTION = "submit_observable_batch_failure"
OBSERVABLE_BATCH_SUBMISSION_FAILURE_ERROR = "observable-batch submission failed"
OBSERVABLE_BATCH_SUBMISSION_LOG_FAILURE_STAGE = "submission_log_publication"
OBSERVABLE_BATCH_JOB_ENQUEUE_FAILURE_STAGE = "job_enqueue"
_PUBLIC_MATERIAL_FILENAMES = (
    "hamiltonian.json",
    "ansatz_spec.yaml",
    "public_noise_model.json",
)


def _canonical_json_digest(payload: object) -> str:
    serialized = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_point_value_digest(values: list[float]) -> str:
    """Canonical sha256 digest of a parameter point's values.

    Binds submitted parameter points (logged at observable-batch submit time)
    to the final-answer parameter vector at scoring time. The canonical
    representation casts every entry to ``float`` and JSON-serializes with
    default separators so the digest is independent of insignificant
    formatting differences between producer and consumer.
    """
    normalized = [float(v) for v in values]
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("observable-batch point values must all be finite")
    # IEEE-754 signed zero has no physical effect on a rotation angle.  Bind
    # numerically identical parameter vectors to the same evidence digest even
    # when one JSON producer serializes a zero as ``-0.0``.
    normalized = [0.0 if value == 0.0 else value for value in normalized]
    return _canonical_json_digest(normalized)


def build_point_value_digest_rows(
    points: Sequence[ObservableBatchPoint],
) -> list[dict[str, str]]:
    """Return the ordered, unique point IDs and their canonical value digests."""
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for point in points:
        if not point.point_id:
            raise ValueError("observable-batch point_id must be non-empty")
        if point.point_id in seen:
            raise ValueError(f"observable-batch point_id {point.point_id!r} appears more than once")
        seen.add(point.point_id)
        rows.append(
            {
                "point_id": point.point_id,
                "value_digest": compute_point_value_digest(point.values),
            }
        )
    return rows


def compute_observable_batch_request_digest(request: ObservableBatchRequest) -> str:
    """Digest the complete validated observable-batch wire request."""
    return _canonical_json_digest(request.model_dump(mode="json"))


def compute_public_material_digests(public_dir: Path) -> dict[str, str]:
    """Digest the exact public VQE material bytes used to validate a request."""
    return {
        filename: hashlib.sha256((public_dir / filename).read_bytes()).hexdigest()
        for filename in _PUBLIC_MATERIAL_FILENAMES
    }


def _bitstrings_digest(bitstrings: list[str]) -> str:
    return _canonical_json_digest(bitstrings)


def _pauli_outcome(bitstring: str, row: ObservableSettingBitstrings) -> int:
    measured_qubits = row.measured_qubits
    if len(measured_qubits) != len(set(measured_qubits)):
        raise ValueError(f"setting {row.setting_id!r} measured_qubits contains duplicates")
    if len(bitstring) != len(measured_qubits) or set(bitstring) - {"0", "1"}:
        raise ValueError(f"setting {row.setting_id!r} contains a malformed measured bitstring")

    measured_positions = {qubit: index for index, qubit in enumerate(measured_qubits)}
    parity = 0
    for qubit, letter in enumerate(row.pauli):
        if letter == "I":
            continue
        if letter not in "XYZ":
            raise ValueError(f"setting {row.setting_id!r} has unsupported Pauli letter {letter!r}")
        try:
            measured_index = measured_positions[qubit]
        except KeyError as exc:
            raise ValueError(
                f"setting {row.setting_id!r} does not measure Pauli support qubit {qubit}"
            ) from exc
        parity ^= int(bitstring[measured_index])
    return 1 if parity == 0 else -1


def summarize_observable_batch_result(
    data: JobObservableBitstringData,
    point_value_digest_rows: Sequence[Mapping[str, str]],
    *,
    expected_shots_per_setting: int,
) -> list[dict[str, Any]]:
    """Reduce raw bitstrings to exact verifier-owned Pauli sufficient statistics.

    Raw bitstrings remain in the agent-facing ``JobResult``. The returned rows
    are the bounded qsim-owned evidence needed to recompute each ungrouped
    Hamiltonian energy and its finite-shot standard error.
    """
    summaries: list[dict[str, Any]] = []
    by_point_id: dict[str, dict[str, Any]] = {}
    for point in point_value_digest_rows:
        if set(point) != {"point_id", "value_digest"}:
            raise ValueError("point digest rows must contain point_id and value_digest")
        point_id = point["point_id"]
        if not point_id or point_id in by_point_id:
            raise ValueError("point digest rows must have unique non-empty point_id values")
        summary: dict[str, Any] = {
            "point_id": point_id,
            "value_digest": point["value_digest"],
            "settings": [],
        }
        summaries.append(summary)
        by_point_id[point_id] = summary

    seen_settings: set[tuple[str, str]] = set()
    for row in data.results:
        try:
            point_summary = by_point_id[row.point_id]
        except KeyError as exc:
            raise ValueError(
                f"observable result references unknown point_id {row.point_id!r}"
            ) from exc
        setting_key = (row.point_id, row.setting_id)
        if not row.setting_id or setting_key in seen_settings:
            raise ValueError(
                f"observable result has duplicate or empty setting_id {row.setting_id!r} "
                f"for point {row.point_id!r}"
            )
        seen_settings.add(setting_key)
        if not math.isfinite(row.coefficient):
            raise ValueError(f"setting {row.setting_id!r} coefficient must be finite")
        if len(row.bitstrings) != expected_shots_per_setting:
            raise ValueError(
                f"setting {row.setting_id!r} has {len(row.bitstrings)} shots; "
                f"expected {expected_shots_per_setting}"
            )

        n_plus = 0
        n_minus = 0
        for bitstring in row.bitstrings:
            if _pauli_outcome(bitstring, row) == 1:
                n_plus += 1
            else:
                n_minus += 1
        point_summary["settings"].append(
            {
                "setting_id": row.setting_id,
                "pauli": row.pauli,
                "coefficient": row.coefficient,
                "measured_qubits": list(row.measured_qubits),
                "shots": len(row.bitstrings),
                "n_plus": n_plus,
                "n_minus": n_minus,
                "bitstrings_digest": _bitstrings_digest(row.bitstrings),
            }
        )

    for point_summary in summaries:
        if not point_summary["settings"]:
            raise ValueError(
                f"observable result has no settings for point {point_summary['point_id']!r}"
            )
    return summaries


def compute_energy_and_standard_error(
    settings: Sequence[Mapping[str, Any]],
) -> tuple[float, float]:
    """Compute an ungrouped Pauli energy and its independent-setting 1-sigma SE.

    For setting ``j``, ``m_j = (n_plus - n_minus) / N_j`` and the unbiased
    estimated variance of the sample mean is ``(1 - m_j**2) / (N_j - 1)``.
    The runtime samples every Pauli setting independently, so variances add as
    ``sum_j coefficient_j**2 * variance_j``.
    """
    if not settings:
        raise ValueError("at least one observable setting is required")

    energy = 0.0
    variance = 0.0
    seen_setting_ids: set[str] = set()
    for setting in settings:
        setting_id = setting.get("setting_id")
        coefficient = setting.get("coefficient")
        shots = setting.get("shots")
        n_plus = setting.get("n_plus")
        n_minus = setting.get("n_minus")
        if not isinstance(setting_id, str) or not setting_id:
            raise ValueError("setting_id must be a non-empty string")
        if setting_id in seen_setting_ids:
            raise ValueError(f"duplicate setting_id {setting_id!r}")
        seen_setting_ids.add(setting_id)
        if (
            isinstance(coefficient, bool)
            or not isinstance(coefficient, (int, float))
            or not math.isfinite(float(coefficient))
        ):
            raise ValueError(f"setting {setting_id!r} coefficient must be finite")
        for name, value in (
            ("shots", shots),
            ("n_plus", n_plus),
            ("n_minus", n_minus),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"setting {setting_id!r} {name} must be a nonnegative integer")
        if shots < 2:
            raise ValueError(
                f"setting {setting_id!r} needs at least two shots for a 1-sigma estimate"
            )
        if n_plus + n_minus != shots:
            raise ValueError(f"setting {setting_id!r} parity counts do not sum to shots")

        mean = (n_plus - n_minus) / shots
        coefficient_float = float(coefficient)
        energy += coefficient_float * mean
        variance += coefficient_float**2 * (1.0 - mean**2) / (shots - 1)

    return energy, math.sqrt(max(variance, 0.0))


def measurement_basis_ops(pauli: str) -> list[CircuitOp]:
    """Pre-measurement basis rotations: H for X; Sdg then H for Y.

    ``pauli[q]`` is read as the Pauli letter acting on qubit ``q``.
    """
    ops: list[CircuitOp] = []
    for q, letter in enumerate(pauli):
        if letter == "X":
            ops.append(GateOp(name="h", qubits=[q]))
        elif letter == "Y":
            ops.append(GateOp(name="sdg", qubits=[q]))
            ops.append(GateOp(name="h", qubits=[q]))
        elif letter in ("I", "Z"):
            continue
        else:  # pragma: no cover - guarded by PauliTerm validation.
            raise ValueError(f"Unsupported Pauli letter {letter!r}")
    return ops
