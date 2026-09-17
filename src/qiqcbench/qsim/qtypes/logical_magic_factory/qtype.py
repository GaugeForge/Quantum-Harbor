"""logical_magic_factory qtype descriptor (discovered by qtypes/registry.py).

A black-box fault-tolerant Steane-code ([[7,1,3]]) magic-state factory (Goto,
Sci. Rep. 6, 19578, 2016). Simulator-only, MCP-only. Hosts the
logical_magic_bell_benchmarking task (estimate the factory's logical
infidelity epsilon to multiplicative precision under a hard magic-state
budget via a twirl + two-copy Bell-measurement / SWAP-test protocol).

This qtype models the factory at the logical-effective level, not the physical
atom-circuit schedule used to prepare |H_L>.
"""

from __future__ import annotations

from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.logical_magic_factory.device import (
    HiddenMagicFactoryConfig,
    PublicMagicFactorySpec,
)
from qiqcbench.qsim.qtypes.logical_magic_factory.lab_notebook import (
    MagicFactoryLabNotebook,
    build_magic_factory_lab_notebook,
)
from qiqcbench.qsim.qtypes.logical_magic_factory.wire import (
    JobCultivationData,
    JobMagicBenchmarkData,
)

_PKG = "qiqcbench.qsim.qtypes.logical_magic_factory"

DESCRIPTOR = QtypeDescriptor(
    qtype="logical_magic_factory",
    public_model=PublicMagicFactorySpec,
    hidden_model=HiddenMagicFactoryConfig,
    notebook_model=MagicFactoryLabNotebook,
    build_lab_notebook=build_magic_factory_lab_notebook,
    result_data_models=(JobMagicBenchmarkData, JobCultivationData),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"logical_magic_benchmarking", "magic_state_cultivation"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_magic_factory_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
