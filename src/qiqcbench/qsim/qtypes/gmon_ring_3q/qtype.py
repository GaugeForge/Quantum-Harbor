"""gmon_ring_3q qtype descriptor (discovered by qtypes/registry.py)."""

from __future__ import annotations

from qiqcbench.qsim.core.wire import JobIQData
from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.gmon_ring_3q.device import (
    HiddenGmonRingConfig,
    PublicGmonRingSpec,
)
from qiqcbench.qsim.qtypes.gmon_ring_3q.lab_notebook import (
    GmonRingLabNotebook,
    build_gmon_ring_3q_lab_notebook,
)

_PKG = "qiqcbench.qsim.qtypes.gmon_ring_3q"

DESCRIPTOR = QtypeDescriptor(
    qtype="gmon_ring_3q",
    public_model=PublicGmonRingSpec,
    hidden_model=HiddenGmonRingConfig,
    notebook_model=GmonRingLabNotebook,
    build_lab_notebook=build_gmon_ring_3q_lab_notebook,
    result_data_models=(JobIQData,),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"basic_measurement", "ring_modulation"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_gmon_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
