"""Stabilize-and-measure runners for the driven-dissipative array qtype.

Bridges the qtype-local wire schemas through ``DdtaEngine`` into a ``JobResult``.
The engine carries the realized Hamiltonian + noise model (all hidden), so the
runner only needs the hidden config. The single and sweep entry points differ
only in the list of stabilization durations handed to the engine.
"""

from __future__ import annotations

import numpy as np

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.driven_dissipative_transmon_array.device import HiddenDdtaConfig
from qiqcbench.qsim.qtypes.driven_dissipative_transmon_array.engine import DdtaEngine
from qiqcbench.qsim.qtypes.driven_dissipative_transmon_array.wire import (
    StabilizeRequest,
    StabilizeSweepRequest,
)


def _make_rng(hidden: HiddenDdtaConfig, salt: int) -> np.random.Generator:
    """Combine the hidden seed with a per-call salt so repeated calls differ."""
    return np.random.default_rng((hidden.seed * 1_000_003) ^ salt)


def build_engine(hidden: HiddenDdtaConfig, rng: np.random.Generator) -> DdtaEngine:
    return DdtaEngine(hidden, rng)


def run_stabilization(
    request: StabilizeRequest,
    hidden: HiddenDdtaConfig,
    job_id: str,
    salt: int,
) -> JobResult:
    engine = build_engine(hidden, _make_rng(hidden, salt))
    return engine.run(
        pair_label=request.pair,
        initial_state=request.initial_state,
        delta_s_mhz=request.delta_s_mhz,
        delta_d_mhz=request.delta_d_mhz,
        g_s_mhz=request.g_s_mhz,
        g_d_mhz=request.g_d_mhz,
        durations_us=[request.duration_us],
        measure=[(m.site, m.basis) for m in request.measure],
        shots=request.shots,
        device_id=hidden.device_id,
        job_id=job_id,
    )


def run_stabilization_sweep(
    request: StabilizeSweepRequest,
    hidden: HiddenDdtaConfig,
    job_id: str,
    salt: int,
) -> JobResult:
    engine = build_engine(hidden, _make_rng(hidden, salt))
    return engine.run(
        pair_label=request.pair,
        initial_state=request.initial_state,
        delta_s_mhz=request.delta_s_mhz,
        delta_d_mhz=request.delta_d_mhz,
        g_s_mhz=request.g_s_mhz,
        g_d_mhz=request.g_d_mhz,
        durations_us=list(request.duration_grid_us),
        measure=[(m.site, m.basis) for m in request.measure],
        shots=request.shots,
        device_id=hidden.device_id,
        job_id=job_id,
    )


__all__ = ["build_engine", "run_stabilization", "run_stabilization_sweep"]
