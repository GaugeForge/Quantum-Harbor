"""cycle_error_recon qtype descriptor (discovered by qtypes/registry.py).

A 5-qubit CZ-cycle device for contextual folded cycle-error reconstruction. Two batched
experiment tools (run_readout_calibration_batch, run_folded_cer_batch) on the async job
model, gated on the ``folded_cer`` capability. Simulator-only, MCP-only; the engine is
run-long stateful (budgets accumulate across batches). No standard run_pulse/run_circuit
runners — the descriptor leaves those slots fail-closed by default.
"""

from __future__ import annotations

from qiqcbench.qsim.qtypes.cycle_error_recon.device import HiddenCerConfig, PublicCerSpec
from qiqcbench.qsim.qtypes.cycle_error_recon.lab_notebook import (
    CerLabNotebook,
    build_cycle_error_recon_lab_notebook,
)
from qiqcbench.qsim.qtypes.cycle_error_recon.wire import JobCerCountsData, JobReadoutCalibData
from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions

_PKG = "qiqcbench.qsim.qtypes.cycle_error_recon"

DESCRIPTOR = QtypeDescriptor(
    qtype="cycle_error_recon",
    public_model=PublicCerSpec,
    hidden_model=HiddenCerConfig,
    notebook_model=CerLabNotebook,
    build_lab_notebook=build_cycle_error_recon_lab_notebook,
    result_data_models=(JobReadoutCalibData, JobCerCountsData),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"folded_cer"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_cycle_error_recon_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
