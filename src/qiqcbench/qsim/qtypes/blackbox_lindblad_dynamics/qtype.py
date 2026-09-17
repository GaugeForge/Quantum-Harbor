"""blackbox_lindblad_dynamics qtype descriptor (discovered by qtypes/registry.py)."""

from __future__ import annotations

from qiqcbench.qsim.core.wire import JobProbeOutcomeData
from qiqcbench.qsim.qtypes.blackbox_lindblad_dynamics.device import (
    HiddenLindbladConfig,
    PublicLindbladSpec,
)
from qiqcbench.qsim.qtypes.blackbox_lindblad_dynamics.lab_notebook import (
    LindbladLabNotebook,
    build_lindblad_lab_notebook,
)
from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions

_PKG = "qiqcbench.qsim.qtypes.blackbox_lindblad_dynamics"

DESCRIPTOR = QtypeDescriptor(
    qtype="blackbox_lindblad_dynamics",
    public_model=PublicLindbladSpec,
    hidden_model=HiddenLindbladConfig,
    notebook_model=LindbladLabNotebook,
    build_lab_notebook=build_lindblad_lab_notebook,
    result_data_models=(JobProbeOutcomeData,),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"lindblad_probe"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_lindblad_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
