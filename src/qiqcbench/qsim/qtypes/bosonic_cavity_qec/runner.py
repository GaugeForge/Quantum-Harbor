"""Program runners for the bosonic_cavity_qec qtype.

Bridge wire schemas through ``CavityQecEngine`` into a ``JobResult``. The engine
gets a fresh RNG per run (injected, derived from ``hidden.seed`` + a per-submit
salt) so independent submissions are deterministic but distinct.

No blanket except around the engine calls: ``CavityQecSimulatorBackend`` builds
and validates every op (and every materialized sweep point) before delegating
here, so anything raised below is a qsim-owned execution fault and must reach
``JobManager`` to be typed as ``JOB_EXECUTION_ERROR``. Catching it and returning
``error=str(exc)`` destroyed that marker -- empty for ``MemoryError`` -- so a
qsim fault was attributed to the model.
"""

from __future__ import annotations

import time

import numpy as np

from qiqcbench.qsim.core.wire import JobResult, JobResultMetadata
from qiqcbench.qsim.qtypes.bosonic_cavity_qec.device import HiddenBosonicCavityConfig
from qiqcbench.qsim.qtypes.bosonic_cavity_qec.engine import CavityQecEngine, program_digest
from qiqcbench.qsim.qtypes.bosonic_cavity_qec.wire import (
    BosonicOp,
    BosonicProgramPoint,
    BosonicProgramRequest,
    BosonicProgramSweepRequest,
    JobBosonicProgramData,
)


def _make_rng(hidden: HiddenBosonicCavityConfig, salt: int) -> np.random.Generator:
    return np.random.default_rng((hidden.seed * 1_000_003) ^ salt)


def build_engine(hidden: HiddenBosonicCavityConfig, rng: np.random.Generator) -> CavityQecEngine:
    return CavityQecEngine(hidden, rng)


def _result(
    hidden: HiddenBosonicCavityConfig,
    job_id: str,
    shots: int,
    points: list[dict],
    digest: str,
    sweep_coords: dict | None,
    t_start: float,
) -> JobResult:
    data = JobBosonicProgramData(
        points=[BosonicProgramPoint(**p) for p in points],
        n_max=int(hidden.n_max),
        program_digest=digest,
    )
    return JobResult(
        job_id=job_id,
        device_id=hidden.device_id,
        status="complete",
        shots=shots,
        data=data,
        metadata=JobResultMetadata(
            wallclock_ms=int((time.perf_counter() - t_start) * 1000),
            sweep_coords=sweep_coords,
        ),
    )


def run_bosonic_program(
    request: BosonicProgramRequest,
    hidden: HiddenBosonicCavityConfig,
    job_id: str,
    salt: int,
) -> JobResult:
    t_start = time.perf_counter()
    engine = build_engine(hidden, _make_rng(hidden, salt))
    point = engine.run_program(list(request.ops), shots=request.shots)
    return _result(
        hidden, job_id, request.shots, [point], program_digest(list(request.ops)), None, t_start
    )


def run_bosonic_program_sweep(
    request: BosonicProgramSweepRequest,
    hidden: HiddenBosonicCavityConfig,
    job_id: str,
    salt: int,
    *,
    materialized_programs: list[list[BosonicOp]] | None = None,
) -> JobResult:
    t_start = time.perf_counter()
    engine = build_engine(hidden, _make_rng(hidden, salt))
    ops = list(request.ops)
    idx = request.sweep_op_index
    field = request.sweep_field
    points: list[dict] = []
    programs = materialized_programs
    if programs is None:
        programs = []
        selected = ops[idx]
        for value in request.sweep_values:
            payload = selected.model_dump(mode="python")
            payload[field] = value
            swept = type(selected).model_validate(payload)
            programs.append([*ops[:idx], swept, *ops[idx + 1 :]])
    for value, ops_point in zip(request.sweep_values, programs, strict=True):
        points.append(engine.run_program(ops_point, shots=request.shots, sweep_value=float(value)))
    coords = {"sweep_values": [float(v) for v in request.sweep_values]}
    return _result(hidden, job_id, request.shots, points, program_digest(ops), coords, t_start)


__all__ = ["build_engine", "run_bosonic_program", "run_bosonic_program_sweep"]
