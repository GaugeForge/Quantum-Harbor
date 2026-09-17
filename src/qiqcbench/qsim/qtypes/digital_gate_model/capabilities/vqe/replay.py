"""Provider-replay support for digital VQE observable batches."""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import ValidationError

from qiqcbench.qsim.backends.provider_artifacts import canonical_request_hash
from qiqcbench.qsim.core.wire import (
    JobObservableBitstringData,
    JobResult,
    JobResultMetadata,
    ObservableBatchRequest,
)
from qiqcbench.qsim.qtypes.digital_gate_model.capabilities.vqe.materials import (
    VQETaskPublicMaterials,
    validate_observable_batch_request,
)
from qiqcbench.qsim.qtypes.digital_gate_model.device import HiddenDigitalConfig

__all__ = ["replay_observable_batch_request", "observable_batch_replay_fixture_path"]


def _fixture_request_mismatch(
    data: JobObservableBitstringData,
    request: ObservableBatchRequest,
    materials: VQETaskPublicMaterials,
) -> str | None:
    expected_count = len(request.points) * len(materials.hamiltonian.pauli_terms)
    if len(data.results) != expected_count:
        return (
            f"expected {expected_count} result row(s) for "
            f"{len(request.points)} point(s) x "
            f"{len(materials.hamiltonian.pauli_terms)} Pauli term(s); "
            f"fixture has {len(data.results)}"
        )

    measured = list(range(materials.ansatz.n_qubits))
    row_idx = 0
    for point in request.points:
        for setting_idx, term in enumerate(materials.hamiltonian.pauli_terms):
            row = data.results[row_idx]
            expected_setting_id = f"s{setting_idx}"
            if row.point_id != point.point_id:
                return (
                    f"row {row_idx} point_id {row.point_id!r} does not match "
                    f"request point_id {point.point_id!r}"
                )
            if row.setting_id != expected_setting_id:
                return (
                    f"row {row_idx} setting_id {row.setting_id!r} does not match "
                    f"expected {expected_setting_id!r}"
                )
            if row.pauli != term.pauli:
                return (
                    f"row {row_idx} pauli {row.pauli!r} does not match "
                    f"public Hamiltonian term {term.pauli!r}"
                )
            if row.coefficient != term.coefficient:
                return (
                    f"row {row_idx} coefficient {row.coefficient!r} does not match "
                    f"public Hamiltonian coefficient {term.coefficient!r}"
                )
            if row.measured_qubits != measured:
                return (
                    f"row {row_idx} measured_qubits {row.measured_qubits!r} "
                    f"does not match expected {measured!r}"
                )
            if len(row.bitstrings) != request.shots_per_setting:
                return (
                    f"row {row_idx} has {len(row.bitstrings)} bitstring(s); "
                    f"expected shots_per_setting {request.shots_per_setting}"
                )
            for shot_idx, bitstring in enumerate(row.bitstrings):
                if len(bitstring) != len(measured):
                    return (
                        f"row {row_idx} shot {shot_idx} bitstring length "
                        f"{len(bitstring)} does not match measured qubit count "
                        f"{len(measured)}"
                    )
            row_idx += 1

    return None


def observable_batch_replay_fixture_path(
    replay_root: str | Path,
    task_id: str,
    device_id: str,
    request: ObservableBatchRequest,
    *,
    must_exist: bool = False,
) -> Path:
    """Return the whole observable-batch replay fixture path for ``request``."""

    path = (
        Path(replay_root)
        / task_id
        / device_id
        / f"{canonical_request_hash(request)}-shots{request.shots_per_setting}.json"
    )
    if must_exist and not path.exists():
        raise FileNotFoundError(f"replay fixture not found: {path}")
    return path


def replay_observable_batch_request(
    request: ObservableBatchRequest,
    hidden: HiddenDigitalConfig,
    materials: VQETaskPublicMaterials,
    *,
    task_id: str,
    replay_root: str | Path,
    job_id: str,
) -> JobResult:
    """Load a provider-replay observable-batch fixture as raw bitstrings."""

    t_start = time.perf_counter()
    try:
        validate_observable_batch_request(request, materials)
    except ValueError as exc:
        return JobResult(
            job_id=job_id,
            device_id=hidden.device_id,
            status="failed",
            shots=request.shots_per_setting,
            error=str(exc),
        )

    try:
        path = observable_batch_replay_fixture_path(
            replay_root,
            task_id,
            hidden.device_id,
            request,
            must_exist=True,
        )
    except FileNotFoundError as exc:
        return JobResult(
            job_id=job_id,
            device_id=hidden.device_id,
            status="failed",
            shots=request.shots_per_setting,
            error=str(exc),
        )

    try:
        data = JobObservableBitstringData.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        return JobResult(
            job_id=job_id,
            device_id=hidden.device_id,
            status="failed",
            shots=request.shots_per_setting,
            error=f"invalid observable-batch replay fixture {path}: {exc}",
        )

    mismatch = _fixture_request_mismatch(data, request, materials)
    if mismatch is not None:
        return JobResult(
            job_id=job_id,
            device_id=hidden.device_id,
            status="failed",
            shots=request.shots_per_setting,
            error=f"observable-batch replay fixture/request mismatch {path}: {mismatch}",
        )

    return JobResult(
        job_id=job_id,
        device_id=hidden.device_id,
        status="complete",
        shots=request.shots_per_setting,
        data=data,
        metadata=JobResultMetadata(
            wallclock_ms=int((time.perf_counter() - t_start) * 1000),
        ),
    )
