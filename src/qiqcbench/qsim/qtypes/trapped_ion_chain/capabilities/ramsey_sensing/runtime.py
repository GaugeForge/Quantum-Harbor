"""Capability runtime: glue between an incoming request and the engine.

The action layer calls these functions; they validate, instantiate the engine,
call the qtype runner, and return a JobResult.
"""

from __future__ import annotations

from typing import Any

from qiqcbench.qsim.core.wire import (
    JobResult,
    RamseyExperimentRequest,
    RamseySweepRequest,
)
from qiqcbench.qsim.qtypes.trapped_ion_chain.capabilities.ramsey_sensing.materials import (
    RamseySensingPublicMaterials,
    validate_ramsey_experiment_request,
    validate_ramsey_sweep_request,
)
from qiqcbench.qsim.qtypes.trapped_ion_chain.runner import (
    run_ramsey_experiment,
    run_ramsey_sweep,
)

__all__ = ["run_ramsey_experiment_request", "run_ramsey_sweep_request"]


def _backend_n_ions(backend: Any) -> int:
    hidden = getattr(backend, "hidden", None)
    ions = getattr(hidden, "ions", None)
    if ions is None:
        raise ValueError("Ramsey runtime backend must expose hidden.ions for validation")
    return len(ions)


def run_ramsey_experiment_request(
    backend: Any,
    request: RamseyExperimentRequest,
    *,
    materials: RamseySensingPublicMaterials,
    job_id: str,
    device_id: str,
    salt: int = 0,
) -> JobResult:
    try:
        validate_ramsey_experiment_request(
            request, materials, n_ions_in_device=_backend_n_ions(backend)
        )
    except ValueError as exc:
        return JobResult(
            job_id=job_id,
            device_id=device_id,
            status="failed",
            shots=request.shots,
            error=str(exc),
        )

    engine = backend.new_engine(job_salt=salt)
    return run_ramsey_experiment(request, engine, job_id=job_id, device_id=device_id)


def run_ramsey_sweep_request(
    backend: Any,
    request: RamseySweepRequest,
    *,
    materials: RamseySensingPublicMaterials,
    job_id: str,
    device_id: str,
    salt: int = 0,
) -> JobResult:
    try:
        validate_ramsey_sweep_request(request, materials, n_ions_in_device=_backend_n_ions(backend))
    except ValueError as exc:
        return JobResult(
            job_id=job_id,
            device_id=device_id,
            status="failed",
            shots=request.shots,
            error=str(exc),
        )

    engine = backend.new_engine(job_salt=salt)
    return run_ramsey_sweep(request, engine, job_id=job_id, device_id=device_id)
