"""transmon_pulse qtype descriptor (discovered by qtypes/registry.py).

Schema models are imported eagerly (lightweight); backend/mcp/runner/replay/live
are :func:`lazy` so importing this module at discovery time does not pull in
engines or ``core.wire`` consumers.
"""

from __future__ import annotations

from qiqcbench.qsim.core.wire import JobIQData
from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.transmon_pulse.device import (
    HiddenTransmonConfig,
    PublicTransmonSpec,
)
from qiqcbench.qsim.qtypes.transmon_pulse.lab_notebook import (
    LabNotebook,
    build_transmon_lab_notebook,
)

_PKG = "qiqcbench.qsim.qtypes.transmon_pulse"

DESCRIPTOR = QtypeDescriptor(
    qtype="transmon_pulse",
    public_model=PublicTransmonSpec,
    hidden_model=HiddenTransmonConfig,
    notebook_model=LabNotebook,
    build_lab_notebook=build_transmon_lab_notebook,
    result_data_models=(JobIQData,),
    supported_backend_modes=frozenset({"simulator", "provider_replay", "live_provider"}),
    supported_surfaces=frozenset({"mcp", "qcodes"}),
    supported_capabilities=frozenset({"basic_measurement", "pulse_sweep"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_transmon_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
    build_replay_backend=lazy(f"{_PKG}.replay", "build_transmon_replay_backend"),
    build_live_provider_backend=lazy(f"{_PKG}.live_ibm", "build_transmon_live_provider_backend"),
    run_pulse_sequence=lazy(f"{_PKG}.runner", "run_pulse_sequence"),
    run_sweep=lazy(f"{_PKG}.runner", "run_sweep"),
)
