"""blackbox_boundedgate_circuit qtype descriptor (discovered by qtypes/registry.py)."""

from __future__ import annotations

from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.device import (
    HiddenBoundedgateConfig,
    PublicBoundedgateSpec,
)
from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.lab_notebook import (
    BoundedgateLabNotebook,
    build_boundedgate_lab_notebook,
)
from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.wire import JobBasisShotData
from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions

_PKG = "qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit"

DESCRIPTOR = QtypeDescriptor(
    qtype="blackbox_boundedgate_circuit",
    public_model=PublicBoundedgateSpec,
    hidden_model=HiddenBoundedgateConfig,
    notebook_model=BoundedgateLabNotebook,
    build_lab_notebook=build_boundedgate_lab_notebook,
    result_data_models=(JobBasisShotData,),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"basis_shot_sampling", "sealed_holdout_reveal"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_boundedgate_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
