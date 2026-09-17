"""spin_chain_control descriptor (auto-discovered by the registry)."""

from __future__ import annotations

from qiqcbench.qsim.core.wire import JobBitstringData
from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.spin_chain_control.device import (
    HiddenSpinChainControlConfig,
    PublicSpinChainControlSpec,
)
from qiqcbench.qsim.qtypes.spin_chain_control.lab_notebook import (
    SpinChainControlLabNotebook,
    build_spin_chain_control_lab_notebook,
)

_PKG = "qiqcbench.qsim.qtypes.spin_chain_control"

DESCRIPTOR = QtypeDescriptor(
    qtype="spin_chain_control",
    public_model=PublicSpinChainControlSpec,
    hidden_model=HiddenSpinChainControlConfig,
    notebook_model=SpinChainControlLabNotebook,
    build_lab_notebook=build_spin_chain_control_lab_notebook,
    result_data_models=(JobBitstringData,),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"structured_control_probe"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_spin_chain_control_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
