"""Runner for the tunable-coupler CZ qtype."""

from __future__ import annotations

import numpy as np

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.transmon_pair_tunable_coupler.device import (
    HiddenTunableCouplerConfig,
)
from qiqcbench.qsim.qtypes.transmon_pair_tunable_coupler.engine import TunableCouplerEngine
from qiqcbench.qsim.qtypes.transmon_pair_tunable_coupler.wire import FluxPulseRequest


def _make_rng(hidden: HiddenTunableCouplerConfig, salt: int) -> np.random.Generator:
    return np.random.default_rng((hidden.seed * 1_000_003) ^ salt)


def run_flux_pulse(
    request: FluxPulseRequest, hidden: HiddenTunableCouplerConfig, job_id: str, salt: int
) -> JobResult:
    engine = TunableCouplerEngine(hidden, _make_rng(hidden, salt))
    return engine.run(request, device_id=hidden.device_id, job_id=job_id)


__all__ = ["run_flux_pulse"]
