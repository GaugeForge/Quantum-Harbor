"""Local-simulator backend façade.

Concrete adapters now live under the qtype packages
(``qsim.qtypes.transmon_pulse.backend`` and
``qsim.qtypes.digital_gate_model.backend``). This module re-exports them so
existing call sites (``backends.factory``, downstream tests) can stay on
the stable ``qiqcbench.qsim.backends.simulator`` import path while the
qtype-specific construction lives next to the qtype's runner and engine.
"""

from __future__ import annotations

from qiqcbench.qsim.qtypes.digital_gate_model.backend import (
    DigitalSimulatorBackend,
    build_digital_simulator_backend,
)
from qiqcbench.qsim.qtypes.transmon_pulse.backend import (
    TransmonSimulatorBackend,
    build_transmon_simulator_backend,
)

__all__ = [
    "DigitalSimulatorBackend",
    "TransmonSimulatorBackend",
    "build_digital_simulator_backend",
    "build_transmon_simulator_backend",
]
