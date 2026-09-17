"""transmon_pair_tunable_coupler qtype descriptor (discovered by qtypes/registry.py).

Schema imports eager; backend/MCP/instructions lazy. New result kind
``JobPairLevelOutcomeData`` defined in this qtype's ``wire.py`` and listed in
``result_data_models`` so the shared ``JobData`` union picks it up (``core/wire.py``
is never edited).
"""

from __future__ import annotations

from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.transmon_pair_tunable_coupler.device import (
    HiddenTunableCouplerConfig,
    PublicTunableCouplerSpec,
)
from qiqcbench.qsim.qtypes.transmon_pair_tunable_coupler.lab_notebook import (
    TunableCouplerLabNotebook,
    build_tunable_coupler_lab_notebook,
)
from qiqcbench.qsim.qtypes.transmon_pair_tunable_coupler.wire import JobPairLevelOutcomeData

_PKG = "qiqcbench.qsim.qtypes.transmon_pair_tunable_coupler"

DESCRIPTOR = QtypeDescriptor(
    qtype="transmon_pair_tunable_coupler",
    public_model=PublicTunableCouplerSpec,
    hidden_model=HiddenTunableCouplerConfig,
    notebook_model=TunableCouplerLabNotebook,
    build_lab_notebook=build_tunable_coupler_lab_notebook,
    result_data_models=(JobPairLevelOutcomeData,),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"basic_measurement", "netzero_flux_control"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_tunable_coupler_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
