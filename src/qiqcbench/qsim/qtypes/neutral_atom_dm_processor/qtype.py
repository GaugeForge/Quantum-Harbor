"""neutral_atom_dm_processor qtype descriptor (discovered by qtypes/registry.py).

A zoned reconfigurable neutral-atom (Rydberg) array operated at the gate +
tweezer-schedule level, backed by an EXACT density-matrix branch-tree engine:
unitaries act as rho -> U rho U^dagger, every noise channel is an exact Kraus
map (non-Pauli amplitude damping included), mid-circuit measurements split
deterministic outcome branches (feedforward conditions on recorded bits), and
terminal readout is sampled classically from confusion-transformed diagonals.
Public run parameters ``idle_scale`` (multiplies durations) and ``noise_scale``
(multiplies all error rates together) are part of the wire contract, plus a
sweep tool over either. Async ``run_atom_program`` / ``run_atom_program_sweep``
+ sync free ``validate_schedule``, gated on ``rydberg_dm_control``. Run-long
stateful (shot budget accumulates). Simulator-only, MCP-only.

Sibling of ``neutral_atom_logical_processor`` (per-shot statevector
trajectories); the two stay separate qtypes because this engine's contract is
exact deterministic expectations, which the verifier's noise-scale slope gate
requires.
"""

from __future__ import annotations

from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.neutral_atom_dm_processor.device import (
    HiddenNeutralAtomDmConfig,
    PublicNeutralAtomDmSpec,
)
from qiqcbench.qsim.qtypes.neutral_atom_dm_processor.lab_notebook import (
    NeutralAtomDmLabNotebook,
    build_neutral_atom_dm_lab_notebook,
)
from qiqcbench.qsim.qtypes.neutral_atom_dm_processor.wire import (
    JobAtomDmShotData,
    JobAtomDmSweepData,
)

_PKG = "qiqcbench.qsim.qtypes.neutral_atom_dm_processor"

DESCRIPTOR = QtypeDescriptor(
    qtype="neutral_atom_dm_processor",
    public_model=PublicNeutralAtomDmSpec,
    hidden_model=HiddenNeutralAtomDmConfig,
    notebook_model=NeutralAtomDmLabNotebook,
    build_lab_notebook=build_neutral_atom_dm_lab_notebook,
    result_data_models=(JobAtomDmShotData, JobAtomDmSweepData),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"rydberg_dm_control"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_neutral_atom_dm_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
