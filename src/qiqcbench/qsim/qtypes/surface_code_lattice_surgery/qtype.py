"""surface_code_lattice_surgery qtype descriptor (discovered by qtypes/registry.py).

Circuit-level (Stim-class) stabilizer simulation of rotated surface-code patches on a
fixed transmon-style 2D grid with a hidden physically-parameterized noise model. Three
experiment tools on the async job model, gated on the ``lattice_surgery_calibration``
capability. Simulator-only, MCP-only; the engine is run-long stateful (shot + shot-rounds
budgets accumulate across calls). The engine only samples — replay decoders live
verifier-side.
"""

from __future__ import annotations

from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery.device import (
    HiddenLatticeSurgeryConfig,
    PublicLatticeSurgerySpec,
)
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery.lab_notebook import (
    LatticeSurgeryLabNotebook,
    build_surface_code_lattice_surgery_lab_notebook,
)
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery.wire import JobLatticeSurgeryData

_PKG = "qiqcbench.qsim.qtypes.surface_code_lattice_surgery"

DESCRIPTOR = QtypeDescriptor(
    qtype="surface_code_lattice_surgery",
    public_model=PublicLatticeSurgerySpec,
    hidden_model=HiddenLatticeSurgeryConfig,
    notebook_model=LatticeSurgeryLabNotebook,
    build_lab_notebook=build_surface_code_lattice_surgery_lab_notebook,
    result_data_models=(JobLatticeSurgeryData,),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"lattice_surgery_calibration"}),
    build_simulator_backend=lazy(
        f"{_PKG}.backend", "build_surface_code_lattice_surgery_simulator_backend"
    ),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
