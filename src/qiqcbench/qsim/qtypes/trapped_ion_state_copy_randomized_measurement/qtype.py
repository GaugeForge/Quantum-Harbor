"""``trapped_ion_state_copy_randomized_measurement`` qtype descriptor (discovered by registry.py).

A six-ion trapped-ion state-copy simulator for nonlinear randomized-measurement estimation of
virtual-cooling ladders. Three async experiment tools (run_readout_calibration,
run_local_pauli_batch, run_copy_block_batch), gated on the ``nonlinear_randomized_measurement``
capability. Simulator-only, MCP-only; the engine is run-long stateful (the state-copy and
randomized-basis budgets accumulate across calls). No standard run_pulse/run_circuit runners.
"""

from __future__ import annotations

from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.trapped_ion_state_copy_randomized_measurement.device import (
    HiddenStateCopyConfig,
    PublicStateCopySpec,
)
from qiqcbench.qsim.qtypes.trapped_ion_state_copy_randomized_measurement.lab_notebook import (
    StateCopyLabNotebook,
    build_trapped_ion_state_copy_randomized_measurement_lab_notebook,
)
from qiqcbench.qsim.qtypes.trapped_ion_state_copy_randomized_measurement.wire import (
    JobCopyBlockData,
    JobJointCopyBlockData,
    JobLocalPauliData,
    JobOrmData,
    JobReadoutCalData,
)

_PKG = "qiqcbench.qsim.qtypes.trapped_ion_state_copy_randomized_measurement"

DESCRIPTOR = QtypeDescriptor(
    qtype="trapped_ion_state_copy_randomized_measurement",
    public_model=PublicStateCopySpec,
    hidden_model=HiddenStateCopyConfig,
    notebook_model=StateCopyLabNotebook,
    build_lab_notebook=build_trapped_ion_state_copy_randomized_measurement_lab_notebook,
    result_data_models=(
        JobReadoutCalData,
        JobOrmData,
        JobLocalPauliData,
        JobCopyBlockData,
        JobJointCopyBlockData,
    ),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"nonlinear_randomized_measurement"}),
    build_simulator_backend=lazy(
        f"{_PKG}.backend",
        "build_trapped_ion_state_copy_randomized_measurement_simulator_backend",
    ),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
