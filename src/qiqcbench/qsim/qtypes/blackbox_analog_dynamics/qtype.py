"""blackbox_analog_dynamics qtype descriptor (discovered by qtypes/registry.py)."""

from __future__ import annotations

from qiqcbench.qsim.core.wire import JobProbeOutcomeData
from qiqcbench.qsim.qtypes.blackbox_analog_dynamics.device import (
    HiddenAnalogConfig,
    PublicAnalogSpec,
)
from qiqcbench.qsim.qtypes.blackbox_analog_dynamics.lab_notebook import (
    AnalogLabNotebook,
    build_analog_lab_notebook,
)
from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions

_PKG = "qiqcbench.qsim.qtypes.blackbox_analog_dynamics"

DESCRIPTOR = QtypeDescriptor(
    qtype="blackbox_analog_dynamics",
    public_model=PublicAnalogSpec,
    hidden_model=HiddenAnalogConfig,
    notebook_model=AnalogLabNotebook,
    build_lab_notebook=build_analog_lab_notebook,
    result_data_models=(JobProbeOutcomeData,),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"hamiltonian_probe"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_analog_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
