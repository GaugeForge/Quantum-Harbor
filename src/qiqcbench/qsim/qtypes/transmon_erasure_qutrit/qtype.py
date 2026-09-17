"""transmon_erasure_qutrit qtype descriptor (discovered by qtypes/registry.py).

A transmon qutrit (g/e/f) operated as a g-f erasure qubit (|0_L>=|g>, |1_L>=|f>,
|e>=erasure) plus an ancilla for mid-circuit erasure detection. Simulator-only,
MCP-only. Supports code-space memory and phase-flip measurements.
"""

from __future__ import annotations

from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.transmon_erasure_qutrit.device import (
    HiddenErasureQutritConfig,
    PublicErasureQutritSpec,
)
from qiqcbench.qsim.qtypes.transmon_erasure_qutrit.lab_notebook import (
    ErasureQutritLabNotebook,
    build_erasure_qutrit_lab_notebook,
)
from qiqcbench.qsim.qtypes.transmon_erasure_qutrit.wire import (
    JobErasureMemoryData,
    JobErasureMemoryRoundResolvedData,
)

_PKG = "qiqcbench.qsim.qtypes.transmon_erasure_qutrit"

DESCRIPTOR = QtypeDescriptor(
    qtype="transmon_erasure_qutrit",
    public_model=PublicErasureQutritSpec,
    hidden_model=HiddenErasureQutritConfig,
    notebook_model=ErasureQutritLabNotebook,
    build_lab_notebook=build_erasure_qutrit_lab_notebook,
    # Keep the sticky v1 kind parseable for historical replay while new
    # simulator jobs publish only the round-resolved kind.
    result_data_models=(JobErasureMemoryData, JobErasureMemoryRoundResolvedData),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"basic_measurement", "mid_circuit_erasure_detection"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_erasure_qutrit_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
