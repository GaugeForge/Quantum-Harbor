"""Logical-memory runners for the g-f erasure-qutrit qtype.

Bridges wire schemas through ``ErasureQutritEngine`` into a ``JobResult``. The
primitive routes via the qtype backend + actions layer (not the qtype-agnostic
``runner`` facade), so these take the hidden config (rates + readout) directly.

No blanket except around ``run_memory``: the wire model and the erasure backend
check every agent-visible bound (prep/measure basis pairing, cycle time, round
count, shots, raw-record budgets) before the runner is reached, so anything
raised below is a qsim-owned execution fault and must reach ``JobManager`` to be
typed as ``JOB_EXECUTION_ERROR``. Catching it and returning ``error=str(exc)``
destroyed that marker -- empty for ``MemoryError`` -- so the memory and
phase-flip verifiers, which route an unstamped failed job to a model rejection,
scored a qsim fault as a model science failure.
"""

from __future__ import annotations

import secrets
import time

import numpy as np

from qiqcbench.qsim.core.wire import JobResult, JobResultMetadata
from qiqcbench.qsim.qtypes.transmon_erasure_qutrit.device import (
    HiddenErasureQutritConfig,
)
from qiqcbench.qsim.qtypes.transmon_erasure_qutrit.engine import (
    ErasureQutritEngine,
)
from qiqcbench.qsim.qtypes.transmon_erasure_qutrit.model import (
    erasure_rates_from_hidden_config,
)
from qiqcbench.qsim.qtypes.transmon_erasure_qutrit.wire import (
    JobErasureMemoryRoundResolvedData,
    JobErasureMemoryRoundResolvedPoint,
    LogicalMemoryRequest,
    LogicalMemorySweepRequest,
)


def fresh_run_entropy() -> int:
    """Draw private per-attempt shot entropy from the OS CSPRNG."""

    return secrets.randbits(64)


def _make_rng(
    hidden: HiddenErasureQutritConfig,
    salt: int,
    run_entropy: int | None = None,
) -> np.random.Generator:
    """Domain-separate committed device, attempt, and per-call randomness."""

    if run_entropy is None:
        run_entropy = fresh_run_entropy()
    return np.random.default_rng([hidden.seed, run_entropy, salt])


def build_engine(
    hidden: HiddenErasureQutritConfig, rng: np.random.Generator
) -> ErasureQutritEngine:
    rates = erasure_rates_from_hidden_config(hidden)
    confusion = np.asarray(hidden.readout_confusion, dtype=float)
    return ErasureQutritEngine(rates=rates, readout_confusion=confusion, rng=rng)


def _run(
    *,
    hidden: HiddenErasureQutritConfig,
    prep_state: str,
    prep_basis: str,
    measure_basis: str,
    n_rounds_grid: list[int],
    cycle_time_us: float,
    dd: str,
    shots: int,
    data_qubit: str,
    job_id: str,
    salt: int,
    run_entropy: int | None,
) -> JobResult:
    t_start = time.perf_counter()
    rng = _make_rng(hidden, salt, run_entropy)
    engine = build_engine(hidden, rng)
    points = engine.run_memory(
        prep_state=prep_state,
        measure_basis=measure_basis,
        n_rounds_grid=n_rounds_grid,
        cycle_time_us=cycle_time_us,
        dd=dd,
        shots=shots,
    )
    data = JobErasureMemoryRoundResolvedData(
        points=[
            JobErasureMemoryRoundResolvedPoint(
                n_rounds=p["n_rounds"],
                total_evolution_us=p["total_evolution_us"],
                mid_circuit_ancilla_post_readout_bitstrings=p[
                    "mid_circuit_ancilla_post_readout_bitstrings"
                ],
                final_qutrit_post_readout_assignments=p["final_qutrit_post_readout_assignments"],
            )
            for p in points
        ],
        data_qubit=data_qubit,
        prep_state=prep_state,
        prep_basis=prep_basis,  # type: ignore[arg-type]
        measure_basis=measure_basis,  # type: ignore[arg-type]
        cycle_time_us=cycle_time_us,
        dd=dd,  # type: ignore[arg-type]
    )
    return JobResult(
        job_id=job_id,
        device_id=hidden.device_id,
        status="complete",
        shots=shots,
        data=data,
        metadata=JobResultMetadata(
            wallclock_ms=int((time.perf_counter() - t_start) * 1000),
            sweep_coords={
                "total_evolution_us": [p["total_evolution_us"] for p in points],
                "n_rounds": [float(p["n_rounds"]) for p in points],
            },
        ),
    )


def run_logical_memory(
    request: LogicalMemoryRequest,
    hidden: HiddenErasureQutritConfig,
    job_id: str,
    salt: int,
    *,
    data_qubit: str,
    run_entropy: int | None = None,
) -> JobResult:
    return _run(
        hidden=hidden,
        prep_state=request.prep_state,
        prep_basis=request.prep_basis,
        measure_basis=request.measure_basis,
        n_rounds_grid=[request.n_rounds],
        cycle_time_us=request.cycle_time_us,
        dd=request.dd,
        shots=request.shots,
        data_qubit=data_qubit,
        job_id=job_id,
        salt=salt,
        run_entropy=run_entropy,
    )


def run_logical_memory_sweep(
    request: LogicalMemorySweepRequest,
    hidden: HiddenErasureQutritConfig,
    job_id: str,
    salt: int,
    *,
    data_qubit: str,
    run_entropy: int | None = None,
) -> JobResult:
    return _run(
        hidden=hidden,
        prep_state=request.prep_state,
        prep_basis=request.prep_basis,
        measure_basis=request.measure_basis,
        n_rounds_grid=[int(n) for n in request.n_rounds_grid],
        cycle_time_us=request.cycle_time_us,
        dd=request.dd,
        shots=request.shots,
        data_qubit=data_qubit,
        job_id=job_id,
        salt=salt,
        run_entropy=run_entropy,
    )


__all__ = [
    "build_engine",
    "fresh_run_entropy",
    "run_logical_memory",
    "run_logical_memory_sweep",
]
