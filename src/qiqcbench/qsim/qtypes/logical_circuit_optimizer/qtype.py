"""``logical_circuit_optimizer`` qtype descriptor (discovered by qtypes/registry.py).

A **device-less / static-instance** qtype (sibling of ``ftqc_resource_estimation``):
no quantum dynamics, no shots, no async job. Its one MCP tool
``get_optimization_instance`` is a synchronous calculator. Simulator + MCP only; no
replay/live backend and no ``run_*`` runners (the descriptor defaults fail closed).
"""

from __future__ import annotations

from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.logical_circuit_optimizer.device import (
    HiddenLogicalOptConfig,
    PublicLogicalOptSpec,
)
from qiqcbench.qsim.qtypes.logical_circuit_optimizer.lab_notebook import (
    LogicalOptLabNotebook,
    build_logical_opt_lab_notebook,
)
from qiqcbench.qsim.qtypes.logical_circuit_optimizer.wire import LogicalOptInstanceData

_PKG = "qiqcbench.qsim.qtypes.logical_circuit_optimizer"

DESCRIPTOR = QtypeDescriptor(
    qtype="logical_circuit_optimizer",
    public_model=PublicLogicalOptSpec,
    hidden_model=HiddenLogicalOptConfig,
    notebook_model=LogicalOptLabNotebook,
    build_lab_notebook=build_logical_opt_lab_notebook,
    result_data_models=(LogicalOptInstanceData,),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"clifford_t_optimization"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_logical_opt_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
