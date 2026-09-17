"""scheduled_transmon_gate_model descriptor (auto-discovered by the registry)."""

from __future__ import annotations

from qiqcbench.qsim.core.wire import JobBitstringData
from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.scheduled_transmon_gate_model.device import (
    HiddenScheduledTransmonConfig,
    PublicScheduledTransmonSpec,
)
from qiqcbench.qsim.qtypes.scheduled_transmon_gate_model.lab_notebook import (
    ScheduledTransmonLabNotebook,
    build_scheduled_transmon_lab_notebook,
)

_PKG = "qiqcbench.qsim.qtypes.scheduled_transmon_gate_model"

DESCRIPTOR = QtypeDescriptor(
    qtype="scheduled_transmon_gate_model",
    public_model=PublicScheduledTransmonSpec,
    hidden_model=HiddenScheduledTransmonConfig,
    notebook_model=ScheduledTransmonLabNotebook,
    build_lab_notebook=build_scheduled_transmon_lab_notebook,
    result_data_models=(JobBitstringData,),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"compiled_candidate_evaluation"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_scheduled_transmon_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
