"""Pulse-train runner for the dipolar-spin-ensemble qtype.

Bridges the dipolar wire schema through ``DipolarEnsembleEngine`` into a
``JobResult`` carrying per-cycle, per-shot collective-magnetization bitstrings
(``+1 -> '1'``, ``-1 -> '0'``). One outer record per stroboscopic cycle
(``0..n_cycles``); each shot string has one char per measured axis.

No blanket except around ``sample_sequence``: the wire model bounds every op
field and ``DipolarSimulatorBackend`` checks shots, returned-payload size,
simulation work units, total sequence length, and pulse axes before the runner is
reached, so anything raised below is a qsim-owned execution fault and must reach
``JobManager`` to be typed as ``JOB_EXECUTION_ERROR``. Catching it and returning
``error=str(exc)`` destroyed that marker -- empty for ``MemoryError`` -- so a
qsim fault was attributed to the model.
"""

from __future__ import annotations

import time

import numpy as np

from qiqcbench.qsim.core.wire import JobBitstringData, JobResult, JobResultMetadata
from qiqcbench.qsim.qtypes.dipolar_spin_ensemble.device import HiddenDipolarConfig
from qiqcbench.qsim.qtypes.dipolar_spin_ensemble.engine import (
    DipolarEnsembleEngine,
    DipolarPhysics,
)
from qiqcbench.qsim.qtypes.dipolar_spin_ensemble.wire import DipolarSequenceRequest

_AXIS_CHANNEL = {"x": 0, "y": 1, "z": 2}


def _make_rng(hidden: HiddenDipolarConfig, salt: int) -> np.random.Generator:
    return np.random.default_rng((hidden.seed * 1_000_003) ^ salt)


def build_physics(hidden: HiddenDipolarConfig) -> DipolarPhysics:
    return DipolarPhysics(
        n_spins=hidden.n_spins,
        n_realizations=hidden.n_realizations,
        w_mhz=hidden.disorder_w_mhz,
        j_mhz=hidden.interaction_j_khz / 1000.0,
        coeffs=tuple(hidden.dipolar_coeffs),
        rabi_mhz=hidden.rabi_mhz,
        rotation_error=hidden.rotation_error,
        readout_p_pm=hidden.readout_p_plus_to_minus,
        readout_p_mp=hidden.readout_p_minus_to_plus,
    )


def build_engine(hidden: HiddenDipolarConfig, rng: np.random.Generator) -> DipolarEnsembleEngine:
    return DipolarEnsembleEngine(build_physics(hidden), rng)


def run_pulse_train(
    request: DipolarSequenceRequest,
    hidden: HiddenDipolarConfig,
    job_id: str,
    salt: int,
) -> JobResult:
    t_start = time.perf_counter()
    rng = _make_rng(hidden, salt)
    engine = build_engine(hidden, rng)
    per_cycle = engine.sample_sequence(
        init_axis=request.init_axis,
        sequence=[op.model_dump() for op in request.sequence],
        n_cycles=request.n_cycles,
        measure_axes=list(request.measure_axes),
        inject_error=request.inject_rotation_error,
        shots=request.shots,
    )
    t_cycle = sum(op.duration_ns for op in request.sequence if op.kind == "free")
    return JobResult(
        job_id=job_id,
        device_id=hidden.device_id,
        status="complete",
        shots=request.shots,
        data=JobBitstringData(
            bitstrings=per_cycle,
            measured_qubits=[_AXIS_CHANNEL[a] for a in request.measure_axes],
        ),
        metadata=JobResultMetadata(
            sequence_duration_ns=t_cycle * request.n_cycles,
            wallclock_ms=int((time.perf_counter() - t_start) * 1000),
            sweep_coords={"cycle_time_ns": [t_cycle * k for k in range(request.n_cycles + 1)]},
        ),
    )


__all__ = ["build_engine", "build_physics", "run_pulse_train"]
