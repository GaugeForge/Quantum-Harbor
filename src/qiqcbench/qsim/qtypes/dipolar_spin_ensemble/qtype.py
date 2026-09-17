"""dipolar_spin_ensemble qtype descriptor (discovered by qtypes/registry.py)."""

from __future__ import annotations

from qiqcbench.qsim.core.wire import JobBitstringData
from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.dipolar_spin_ensemble.device import (
    HiddenDipolarConfig,
    PublicDipolarSpec,
)
from qiqcbench.qsim.qtypes.dipolar_spin_ensemble.lab_notebook import (
    DipolarLabNotebook,
    build_dipolar_lab_notebook,
)

_PKG = "qiqcbench.qsim.qtypes.dipolar_spin_ensemble"

DESCRIPTOR = QtypeDescriptor(
    qtype="dipolar_spin_ensemble",
    public_model=PublicDipolarSpec,
    hidden_model=HiddenDipolarConfig,
    notebook_model=DipolarLabNotebook,
    build_lab_notebook=build_dipolar_lab_notebook,
    result_data_models=(JobBitstringData,),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"basic_measurement", "toggling_frame_control"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_dipolar_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
