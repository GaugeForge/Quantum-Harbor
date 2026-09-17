"""trapped_ion_chain qtype descriptor (discovered by qtypes/registry.py)."""

from __future__ import annotations

from qiqcbench.qsim.core.wire import JobRamseyBitstringData
from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.trapped_ion_chain.device import (
    HiddenTrappedIonChainConfig,
    PublicTrappedIonChainSpec,
)
from qiqcbench.qsim.qtypes.trapped_ion_chain.lab_notebook import (
    IonChainLabNotebook,
    build_ion_chain_lab_notebook,
)

_PKG = "qiqcbench.qsim.qtypes.trapped_ion_chain"

DESCRIPTOR = QtypeDescriptor(
    qtype="trapped_ion_chain",
    public_model=PublicTrappedIonChainSpec,
    hidden_model=HiddenTrappedIonChainConfig,
    notebook_model=IonChainLabNotebook,
    build_lab_notebook=build_ion_chain_lab_notebook,
    result_data_models=(JobRamseyBitstringData,),
    supported_backend_modes=frozenset({"simulator", "provider_replay"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"basic_measurement", "ramsey_sensing"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_ion_chain_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
    build_replay_backend=lazy(f"{_PKG}.replay", "build_ion_chain_replay_backend"),
)
