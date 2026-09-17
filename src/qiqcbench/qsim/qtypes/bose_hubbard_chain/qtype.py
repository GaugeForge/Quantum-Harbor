"""bose_hubbard_chain qtype descriptor (discovered by qtypes/registry.py)."""

from __future__ import annotations

from qiqcbench.qsim.core.wire import JobBitstringData
from qiqcbench.qsim.qtypes.bose_hubbard_chain.device import (
    HiddenBoseHubbardConfig,
    PublicBoseHubbardSpec,
)
from qiqcbench.qsim.qtypes.bose_hubbard_chain.lab_notebook import (
    BoseHubbardLabNotebook,
    build_bose_hubbard_lab_notebook,
)
from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions

_PKG = "qiqcbench.qsim.qtypes.bose_hubbard_chain"

DESCRIPTOR = QtypeDescriptor(
    qtype="bose_hubbard_chain",
    public_model=PublicBoseHubbardSpec,
    hidden_model=HiddenBoseHubbardConfig,
    notebook_model=BoseHubbardLabNotebook,
    build_lab_notebook=build_bose_hubbard_lab_notebook,
    result_data_models=(JobBitstringData,),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"basic_measurement", "bose_hubbard_spectroscopy"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_bose_hubbard_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
