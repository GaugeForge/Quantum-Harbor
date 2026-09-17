"""qccd_ion_compiler qtype descriptor (discovered by qtypes/registry.py).

Device-less / synchronous-calculator qtype (sibling in form to
``ftqc_resource_estimation``): no quantum dynamics, no shots, no async job model.
Its MCP tools (``get_compilation_instance`` + ``evaluate_schedule``) are
synchronous calculators gated on the ``qccd_compilation`` capability. Schema
imports are eager; backend/mcp/instructions are lazy so registry discovery stays
import-light and cycle-free.
"""

from __future__ import annotations

from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.qccd_ion_compiler.device import (
    HiddenQccdConfig,
    PublicQccdSpec,
)
from qiqcbench.qsim.qtypes.qccd_ion_compiler.lab_notebook import (
    QccdLabNotebook,
    build_qccd_lab_notebook,
)
from qiqcbench.qsim.qtypes.qccd_ion_compiler.wire import QccdEvaluationData

_PKG = "qiqcbench.qsim.qtypes.qccd_ion_compiler"

DESCRIPTOR = QtypeDescriptor(
    qtype="qccd_ion_compiler",
    public_model=PublicQccdSpec,
    hidden_model=HiddenQccdConfig,
    notebook_model=QccdLabNotebook,
    build_lab_notebook=build_qccd_lab_notebook,
    result_data_models=(QccdEvaluationData,),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"qccd_compilation"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_qccd_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
