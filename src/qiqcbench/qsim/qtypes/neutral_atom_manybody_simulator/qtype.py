"""Descriptor for a site-resolved finite-depth neutral-atom many-body device."""

from __future__ import annotations

from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.neutral_atom_manybody_simulator.device import (
    HiddenNeutralAtomManyBodyConfig,
    PublicNeutralAtomManyBodySpec,
)
from qiqcbench.qsim.qtypes.neutral_atom_manybody_simulator.lab_notebook import (
    NeutralAtomManyBodyLabNotebook,
    build_neutral_atom_manybody_lab_notebook,
)
from qiqcbench.qsim.qtypes.neutral_atom_manybody_simulator.wire import JobRydbergTrajectoryData

_PKG = "qiqcbench.qsim.qtypes.neutral_atom_manybody_simulator"

DESCRIPTOR = QtypeDescriptor(
    qtype="neutral_atom_manybody_simulator",
    public_model=PublicNeutralAtomManyBodySpec,
    hidden_model=HiddenNeutralAtomManyBodyConfig,
    notebook_model=NeutralAtomManyBodyLabNotebook,
    build_lab_notebook=build_neutral_atom_manybody_lab_notebook,
    result_data_models=(JobRydbergTrajectoryData,),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"stroboscopic_rydberg_evolution"}),
    build_simulator_backend=lazy(
        f"{_PKG}.backend", "build_neutral_atom_manybody_simulator_backend"
    ),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
