"""Per-job runner functions for the trapped_ion_chain qtype.

These are invoked by the capability action layer (actions/ramsey_sensing.py)
to execute a validated request against the engine. The functions return a
JobResult populated with JobRamseyBitstringData.
"""

from __future__ import annotations

import time

from qiqcbench.qsim.core.wire import (
    JobRamseyBitstringData,
    JobRamseyPointBitstrings,
    JobResult,
    JobResultMetadata,
    RamseyExperimentRequest,
    RamseySweepRequest,
)
from qiqcbench.qsim.qtypes.trapped_ion_chain.engine import IonChainEngine


def run_ramsey_experiment(
    request: RamseyExperimentRequest,
    engine: IonChainEngine,
    *,
    job_id: str,
    device_id: str,
) -> JobResult:
    t_start = time.perf_counter()
    try:
        prepared = engine.prepare_state(circuit=list(request.prepare_circuit))
        bits = engine.sample_prepared_state(
            prepared,
            free_evolution_us=request.free_evolution_us,
            phase_source=request.phase_source,
            reference_detuning_hz=request.reference_detuning_hz,
            analysis_circuit=list(request.analysis_circuit),
            measured_ions=list(request.measured_ions),
            shots=request.shots,
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
        data=JobRamseyBitstringData(
            points=[
                JobRamseyPointBitstrings(
                    free_evolution_us=request.free_evolution_us,
                    bitstrings=bits,
                    measured_ions=list(request.measured_ions),
                )
            ],
            phase_source=request.phase_source,
            reference_detuning_hz=request.reference_detuning_hz,
        ),
        metadata=JobResultMetadata(
            wallclock_ms=int((time.perf_counter() - t_start) * 1000),
        ),
    )


def run_ramsey_sweep(
    request: RamseySweepRequest,
    engine: IonChainEngine,
    *,
    job_id: str,
    device_id: str,
) -> JobResult:
    t_start = time.perf_counter()
    try:
        prepared = engine.prepare_state(circuit=list(request.prepare_circuit))
        points = []
        for t_us in request.free_evolution_us_values:
            bits = engine.sample_prepared_state(
                prepared,
                free_evolution_us=t_us,
                phase_source=request.phase_source,
                reference_detuning_hz=request.reference_detuning_hz,
                analysis_circuit=list(request.analysis_circuit),
                measured_ions=list(request.measured_ions),
                shots=request.shots,
            )
            points.append(
                JobRamseyPointBitstrings(
                    free_evolution_us=t_us,
                    bitstrings=bits,
                    measured_ions=list(request.measured_ions),
                )
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
        data=JobRamseyBitstringData(
            points=points,
            phase_source=request.phase_source,
            reference_detuning_hz=request.reference_detuning_hz,
        ),
        metadata=JobResultMetadata(
            wallclock_ms=int((time.perf_counter() - t_start) * 1000),
        ),
    )


__all__ = ["run_ramsey_experiment", "run_ramsey_sweep"]
