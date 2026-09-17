"""neutral_atom_quantum_network qtype descriptor (discovered by qtypes/registry.py).

A two-node neutral-atom quantum network at the protocol/algorithm level (no atomic-physics
pulse design). One async experiment tool (run_distributed_estimation), gated on the
``vfl_distributed_estimation`` capability. Simulator-only, MCP-only; the engine is run-long
stateful (the communication budget accumulates across calls). No standard run_pulse/run_circuit
runners — those slots stay fail-closed.
"""

from __future__ import annotations

from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.neutral_atom_quantum_network.device import (
    HiddenNetworkConfig,
    PublicNetworkSpec,
)
from qiqcbench.qsim.qtypes.neutral_atom_quantum_network.lab_notebook import (
    NetworkLabNotebook,
    build_neutral_atom_quantum_network_lab_notebook,
)
from qiqcbench.qsim.qtypes.neutral_atom_quantum_network.wire import JobEstimationData

_PKG = "qiqcbench.qsim.qtypes.neutral_atom_quantum_network"

DESCRIPTOR = QtypeDescriptor(
    qtype="neutral_atom_quantum_network",
    public_model=PublicNetworkSpec,
    hidden_model=HiddenNetworkConfig,
    notebook_model=NetworkLabNotebook,
    build_lab_notebook=build_neutral_atom_quantum_network_lab_notebook,
    result_data_models=(JobEstimationData,),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"vfl_distributed_estimation"}),
    build_simulator_backend=lazy(
        f"{_PKG}.backend", "build_neutral_atom_quantum_network_simulator_backend"
    ),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
