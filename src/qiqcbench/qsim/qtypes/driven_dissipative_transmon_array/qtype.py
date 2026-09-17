"""driven_dissipative_transmon_array qtype descriptor (discovered by qtypes/registry.py)."""

from __future__ import annotations

from qiqcbench.qsim.core.wire import JobBitstringData
from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.driven_dissipative_transmon_array.device import (
    HiddenDdtaConfig,
    PublicDdtaSpec,
)
from qiqcbench.qsim.qtypes.driven_dissipative_transmon_array.lab_notebook import (
    DdtaLabNotebook,
    build_ddta_lab_notebook,
)

_PKG = "qiqcbench.qsim.qtypes.driven_dissipative_transmon_array"

DESCRIPTOR = QtypeDescriptor(
    qtype="driven_dissipative_transmon_array",
    public_model=PublicDdtaSpec,
    hidden_model=HiddenDdtaConfig,
    notebook_model=DdtaLabNotebook,
    build_lab_notebook=build_ddta_lab_notebook,
    result_data_models=(JobBitstringData,),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"basic_measurement", "local_reservoir_stabilization"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_ddta_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
