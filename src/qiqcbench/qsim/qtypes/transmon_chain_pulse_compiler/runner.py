"""Circuit runner for the chain pulse-compiler qtype.

Bridges the chain wire schema through the density-matrix engine into a
``JobResult``. Uses a qtype-named entry point (``run_chain_circuit``) since the
compilation control surface is distinct from the pulse/circuit runner slots
(mirrors gmon).
"""

from __future__ import annotations

import numpy as np

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.transmon_chain_pulse_compiler.device import (
    HiddenChainCompilerConfig,
)
from qiqcbench.qsim.qtypes.transmon_chain_pulse_compiler.engine import ChainCompilerEngine
from qiqcbench.qsim.qtypes.transmon_chain_pulse_compiler.wire import ChainCircuitRequest


def _make_rng(hidden: HiddenChainCompilerConfig, salt: int) -> np.random.Generator:
    """Combine the hidden seed with a per-call salt so repeated calls differ."""
    return np.random.default_rng((hidden.seed * 1_000_003) ^ salt)


def run_chain_circuit(
    request: ChainCircuitRequest,
    hidden: HiddenChainCompilerConfig,
    job_id: str,
    salt: int,
) -> JobResult:
    rng = _make_rng(hidden, salt)
    engine = ChainCompilerEngine(hidden, rng)
    return engine.run_circuit(request, device_id=hidden.device_id, job_id=job_id)


__all__ = ["run_chain_circuit"]
