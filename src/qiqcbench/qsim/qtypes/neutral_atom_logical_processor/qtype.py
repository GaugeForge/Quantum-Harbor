"""neutral_atom_logical_processor qtype descriptor (discovered by qtypes/registry.py).

A zoned reconfigurable neutral-atom (Rydberg) array operated at the gate + tweezer-schedule
level: single-atom gates, Rydberg-blockade CZ, AOD transport, mid-circuit fluorescence
readout, classical feedforward. Per-shot statevector trajectory simulation with honest atom
loss. One async experiment tool (run_atom_program) + one sync free helper (validate_schedule),
gated on ``rydberg_logical_control``. Run-long stateful (the shot budget accumulates).
Simulator-only, MCP-only.
"""

from __future__ import annotations

from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.neutral_atom_logical_processor.capabilities.located_erasure_repair.wire import (
    JobErasureRepairEpisodeData,
)
from qiqcbench.qsim.qtypes.neutral_atom_logical_processor.device import (
    HiddenNeutralAtomConfig,
    PublicNeutralAtomSpec,
)
from qiqcbench.qsim.qtypes.neutral_atom_logical_processor.lab_notebook import (
    NeutralAtomLabNotebook,
    build_neutral_atom_lab_notebook,
)
from qiqcbench.qsim.qtypes.neutral_atom_logical_processor.wire import (
    JobAtomShotData,
)

_PKG = "qiqcbench.qsim.qtypes.neutral_atom_logical_processor"

DESCRIPTOR = QtypeDescriptor(
    qtype="neutral_atom_logical_processor",
    public_model=PublicNeutralAtomSpec,
    hidden_model=HiddenNeutralAtomConfig,
    notebook_model=NeutralAtomLabNotebook,
    build_lab_notebook=build_neutral_atom_lab_notebook,
    result_data_models=(JobAtomShotData, JobErasureRepairEpisodeData),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"rydberg_logical_control", "located_erasure_repair"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_neutral_atom_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
