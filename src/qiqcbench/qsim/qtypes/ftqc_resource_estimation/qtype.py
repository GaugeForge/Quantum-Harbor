"""ftqc_resource_estimation qtype descriptor (discovered by qtypes/registry.py).

A **device-less / static-instance** qtype: no quantum dynamics, no
shots, no async job model. Its MCP tools are synchronous calculators. The
descriptor declares no `run_*` runners (the defaults fail closed) and no
replay/live backend — simulator + MCP only.
"""

from __future__ import annotations

from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.ftqc_resource_estimation.device import (
    HiddenFtqcConfig,
    PublicFtqcSpec,
)
from qiqcbench.qsim.qtypes.ftqc_resource_estimation.lab_notebook import (
    FtqcLabNotebook,
    build_ftqc_lab_notebook,
)
from qiqcbench.qsim.qtypes.ftqc_resource_estimation.wire import (
    FtqcEvaluationData,
    FtqcPhysicalEvaluationData,
)

_PKG = "qiqcbench.qsim.qtypes.ftqc_resource_estimation"

DESCRIPTOR = QtypeDescriptor(
    qtype="ftqc_resource_estimation",
    public_model=PublicFtqcSpec,
    hidden_model=HiddenFtqcConfig,
    notebook_model=FtqcLabNotebook,
    build_lab_notebook=build_ftqc_lab_notebook,
    result_data_models=(FtqcEvaluationData, FtqcPhysicalEvaluationData),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset(
        {
            "ftqc_physical_estimation",
            "mps_qpe_resource_planning",
        }
    ),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_ftqc_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
