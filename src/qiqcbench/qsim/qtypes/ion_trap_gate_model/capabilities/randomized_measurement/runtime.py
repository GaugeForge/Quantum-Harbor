"""Capability runtime for randomized_measurement.

Validates a request against the public materials, builds a per-job engine from
the backend, executes, and wraps raw bitstrings in a JobResult. Returns a failed
JobResult (never raises) on validation/runtime errors so the async job model
surfaces a clean status to the agent.
"""

from __future__ import annotations

import time
from typing import Any

from qiqcbench.qsim.core.wire import JobResult, JobResultMetadata
from qiqcbench.qsim.qtypes.ion_trap_gate_model.capabilities.randomized_measurement.materials import (
    RandomizedMeasurementMaterials,
    validate_ion_circuit_request,
    validate_rm_batch_request,
)
from qiqcbench.qsim.qtypes.ion_trap_gate_model.wire import (
    IonCircuitRequest,
    JobIonBitstringData,
    JobRandomizedMeasurementDataV2,
    RandomizedMeasurementBatchRequestV2,
    RmSettingBitstrings,
)

__all__ = ["run_ion_circuit_request", "run_rm_batch_request"]


def run_ion_circuit_request(
    backend: Any,
    request: IonCircuitRequest,
    *,
    materials: RandomizedMeasurementMaterials,
    job_id: str,
    device_id: str,
    salt: int = 0,
) -> JobResult:
    t0 = time.perf_counter()
    try:
        if backend.public is None:
            raise ValueError("ion-trap backend lacks its public admission contract")
        validate_ion_circuit_request(request, backend.public)
        engine = backend.new_engine(job_salt=salt)
        bits = engine.run_circuit_shots(
            list(request.circuit), list(request.measured_qubits), request.shots
        )
    except (ValueError, NotImplementedError) as exc:
        return JobResult(
            job_id=job_id,
            device_id=device_id,
            status="failed",
            shots=request.shots,
            error=str(exc),
        )
    return JobResult(
        job_id=job_id,
        device_id=device_id,
        status="complete",
        shots=request.shots,
        data=JobIonBitstringData(
            bitstrings=bits,
            measured_qubits=list(request.measured_qubits),
        ),
        metadata=JobResultMetadata(wallclock_ms=int((time.perf_counter() - t0) * 1000)),
    )


def run_rm_batch_request(
    backend: Any,
    request: RandomizedMeasurementBatchRequestV2,
    *,
    materials: RandomizedMeasurementMaterials,
    job_id: str,
    device_id: str,
    salt: int = 0,
) -> JobResult:
    t0 = time.perf_counter()
    total_shots = request.n_unitaries * request.shots_per_unitary
    try:
        if backend.public is None:
            raise ValueError("ion-trap backend lacks its public admission contract")
        validate_rm_batch_request(request, backend.public)
        engine = backend.new_engine(job_salt=salt, stream=0)
        settings = engine.run_rm_batch(
            list(request.prepare_circuit),
            n_unitaries=request.n_unitaries,
            shots_per_unitary=request.shots_per_unitary,
            subsystem_qubits=list(request.subsystem_qubits),
            basis_rng=backend.new_rng(job_salt=salt, stream=1),
        )
    except (ValueError, NotImplementedError) as exc:
        return JobResult(
            job_id=job_id,
            device_id=device_id,
            status="failed",
            shots=total_shots,
            error=str(exc),
        )
    return JobResult(
        job_id=job_id,
        device_id=device_id,
        status="complete",
        shots=total_shots,
        data=JobRandomizedMeasurementDataV2(
            settings=[
                RmSettingBitstrings(setting_index=s, bitstrings=bits) for (s, bits) in settings
            ],
            subsystem_qubits=list(request.subsystem_qubits),
            experiment_tag=request.experiment_tag,
            n_unitaries=request.n_unitaries,
            shots_per_unitary=request.shots_per_unitary,
        ),
        metadata=JobResultMetadata(wallclock_ms=int((time.perf_counter() - t0) * 1000)),
    )
