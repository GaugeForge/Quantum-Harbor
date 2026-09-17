"""Self-registering DESCRIPTOR for the ``transmon_repeater_array`` qtype.

Auto-discovered by ``qtypes/registry.py``; the device / lab-notebook / capability /
JobData unions are derived from it. Schema imports are eager; backend / mcp /
instructions are ``lazy(...)`` so discovery stays import-light and cycle-free.
"""

from __future__ import annotations

from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.transmon_repeater_array.device import (
    HiddenRepeaterConfig,
    PublicRepeaterSpec,
)
from qiqcbench.qsim.qtypes.transmon_repeater_array.lab_notebook import (
    RepeaterLabNotebook,
    build_transmon_repeater_array_lab_notebook,
)
from qiqcbench.qsim.qtypes.transmon_repeater_array.wire import JobPurificationData

_PKG = "qiqcbench.qsim.qtypes.transmon_repeater_array"

DESCRIPTOR = QtypeDescriptor(
    qtype="transmon_repeater_array",
    public_model=PublicRepeaterSpec,
    hidden_model=HiddenRepeaterConfig,
    notebook_model=RepeaterLabNotebook,
    build_lab_notebook=build_transmon_repeater_array_lab_notebook,
    result_data_models=(JobPurificationData,),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"entanglement_purification"}),
    build_simulator_backend=lazy(
        f"{_PKG}.backend", "build_transmon_repeater_array_simulator_backend"
    ),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
