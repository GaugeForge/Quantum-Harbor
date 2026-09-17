"""transmon_chain_pulse_compiler qtype descriptor (discovered by qtypes/registry.py).

Schema imports are eager (safe at discovery time); the backend/MCP/instructions
are ``lazy(...)`` so engines are imported only on first use. Defines a new result
kind ``JobChainLevelOutcomeData`` in the qtype's own ``wire.py`` and lists it in
``result_data_models`` so the shared ``JobData`` union picks it up — ``core/wire.py``
is never edited.
"""

from __future__ import annotations

from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.transmon_chain_pulse_compiler.device import (
    HiddenChainCompilerConfig,
    PublicChainCompilerSpec,
)
from qiqcbench.qsim.qtypes.transmon_chain_pulse_compiler.lab_notebook import (
    ChainCompilerLabNotebook,
    build_chain_compiler_lab_notebook,
)
from qiqcbench.qsim.qtypes.transmon_chain_pulse_compiler.wire import JobChainLevelOutcomeData

_PKG = "qiqcbench.qsim.qtypes.transmon_chain_pulse_compiler"

DESCRIPTOR = QtypeDescriptor(
    qtype="transmon_chain_pulse_compiler",
    public_model=PublicChainCompilerSpec,
    hidden_model=HiddenChainCompilerConfig,
    notebook_model=ChainCompilerLabNotebook,
    build_lab_notebook=build_chain_compiler_lab_notebook,
    result_data_models=(JobChainLevelOutcomeData,),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"basic_measurement", "chain_pulse_compilation"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_chain_compiler_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
