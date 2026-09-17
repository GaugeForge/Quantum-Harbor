"""kitaev_chain qtype descriptor (discovered by qtypes/registry.py).

A semiconductor Kitaev chain (gate-defined quantum dots coupled through
superconductor hybrid segments) operated through programmable gates and charge
readout. Charge-readout characterization uses exact free-fermion / matchgate
physics; pulse control uses an explicitly effective three-level model.

Simulator-only, MCP-only. Three charge-readout tools on the async job model,
gated on the ``kitaev_charge_readout`` capability (the defect-localization task).
A second capability ``majorana_pulse_control`` supports calibration plus
qsim-owned single-qubit Clifford RB. No standard run_pulse/run_circuit runners
are registered; those slots stay fail-closed by default.
"""

from __future__ import annotations

from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.kitaev_chain.device import (
    HiddenKitaevChainConfig,
    PublicKitaevChainSpec,
)
from qiqcbench.qsim.qtypes.kitaev_chain.lab_notebook import (
    KitaevChainLabNotebook,
    build_kitaev_chain_lab_notebook,
)
from qiqcbench.qsim.qtypes.kitaev_chain.wire import (
    KitaevChargeStabilityData,
    KitaevCliffordRBData,
    KitaevPolarizationData,
    KitaevPulseBatchData,
    KitaevSubchainGapData,
)

_PKG = "qiqcbench.qsim.qtypes.kitaev_chain"

DESCRIPTOR = QtypeDescriptor(
    qtype="kitaev_chain",
    public_model=PublicKitaevChainSpec,
    hidden_model=HiddenKitaevChainConfig,
    notebook_model=KitaevChainLabNotebook,
    build_lab_notebook=build_kitaev_chain_lab_notebook,
    result_data_models=(
        KitaevChargeStabilityData,
        KitaevPolarizationData,
        KitaevSubchainGapData,
        KitaevPulseBatchData,
        KitaevCliffordRBData,
    ),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"kitaev_charge_readout", "majorana_pulse_control"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_kitaev_chain_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
