"""``composite_measurement_estimation`` qtype descriptor (discovered by registry.py).

A fixed 14-qubit Pauli observable + one hidden prepared state. Two synchronous
calculators (get_observable_spec, evaluate_composite_scheme) and two async
experiments (run_pilot_measurements, execute_locked_composite_scheme), gated on
the ``composite_measurement`` capability. Simulator-only, MCP-only; the engine is
run-long stateful (pilot budget accumulates; production runs once). No standard
run_pulse/run_circuit runners — those slots stay fail-closed by default.
"""

from __future__ import annotations

from qiqcbench.qsim.qtypes.composite_measurement_estimation.device import (
    HiddenCmeConfig,
    PublicCmeSpec,
)
from qiqcbench.qsim.qtypes.composite_measurement_estimation.lab_notebook import (
    CmeLabNotebook,
    build_composite_measurement_lab_notebook,
)
from qiqcbench.qsim.qtypes.composite_measurement_estimation.wire import (
    JobPilotCountsData,
    JobProductionCountsData,
)
from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions

_PKG = "qiqcbench.qsim.qtypes.composite_measurement_estimation"

DESCRIPTOR = QtypeDescriptor(
    qtype="composite_measurement_estimation",
    public_model=PublicCmeSpec,
    hidden_model=HiddenCmeConfig,
    notebook_model=CmeLabNotebook,
    build_lab_notebook=build_composite_measurement_lab_notebook,
    result_data_models=(JobPilotCountsData, JobProductionCountsData),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"composite_measurement"}),
    build_simulator_backend=lazy(
        f"{_PKG}.backend", "build_composite_measurement_simulator_backend"
    ),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
