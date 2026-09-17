"""transmon_multilevel_pulse qtype descriptor (discovered by qtypes/registry.py)."""

from __future__ import annotations

from qiqcbench.qsim.core.wire import JobLevelOutcomeData
from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.transmon_multilevel_pulse.device import (
    HiddenTransmonMultilevelConfig,
    PublicTransmonMultilevelSpec,
)
from qiqcbench.qsim.qtypes.transmon_multilevel_pulse.lab_notebook import (
    TransmonMultilevelLabNotebook,
    build_transmon_multilevel_lab_notebook,
)

_PKG = "qiqcbench.qsim.qtypes.transmon_multilevel_pulse"

DESCRIPTOR = QtypeDescriptor(
    qtype="transmon_multilevel_pulse",
    public_model=PublicTransmonMultilevelSpec,
    hidden_model=HiddenTransmonMultilevelConfig,
    notebook_model=TransmonMultilevelLabNotebook,
    build_lab_notebook=build_transmon_multilevel_lab_notebook,
    result_data_models=(JobLevelOutcomeData,),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"basic_measurement", "multilevel_pulse_control"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_transmon_multilevel_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
