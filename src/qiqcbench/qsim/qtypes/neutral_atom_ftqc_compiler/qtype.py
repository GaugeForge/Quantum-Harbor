"""neutral_atom_ftqc_compiler qtype descriptor (discovered by qtypes/registry.py).

A **device-less / static-instance** qtype (sibling to
``ftqc_resource_estimation``): no quantum dynamics, no shots, no async job model.
Its MCP tool (``get_target_circuit``) is a synchronous calculator. The descriptor
declares no ``run_*`` runners (the defaults fail closed) and no replay/live
backend — simulator + MCP only.
"""

from __future__ import annotations

from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.neutral_atom_ftqc_compiler.device import (
    HiddenNeutralAtomConfig,
    PublicNeutralAtomSpec,
)
from qiqcbench.qsim.qtypes.neutral_atom_ftqc_compiler.lab_notebook import (
    NeutralAtomLabNotebook,
    build_neutral_atom_lab_notebook,
)
from qiqcbench.qsim.qtypes.neutral_atom_ftqc_compiler.wire import NeutralAtomTargetData

_PKG = "qiqcbench.qsim.qtypes.neutral_atom_ftqc_compiler"

DESCRIPTOR = QtypeDescriptor(
    qtype="neutral_atom_ftqc_compiler",
    public_model=PublicNeutralAtomSpec,
    hidden_model=HiddenNeutralAtomConfig,
    notebook_model=NeutralAtomLabNotebook,
    build_lab_notebook=build_neutral_atom_lab_notebook,
    result_data_models=(NeutralAtomTargetData,),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"neutral_atom_ftqc_compilation"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_neutral_atom_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
