"""Self-registering descriptor for the native simultaneous-CZ2 memory Qtype."""

from __future__ import annotations

from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.rydberg_multitarget_surface_code.device import (
    HiddenRydbergMultitargetSurfaceCodeConfig,
    PublicRydbergMultitargetSurfaceCodeSpec,
)
from qiqcbench.qsim.qtypes.rydberg_multitarget_surface_code.lab_notebook import (
    RydbergMultitargetLabNotebook,
    build_rydberg_multitarget_lab_notebook,
)
from qiqcbench.qsim.qtypes.rydberg_multitarget_surface_code.wire import (
    JobCz2CharacterizationData,
    JobMultitargetMemoryData,
)

_PKG = "qiqcbench.qsim.qtypes.rydberg_multitarget_surface_code"

DESCRIPTOR = QtypeDescriptor(
    qtype="rydberg_multitarget_surface_code",
    public_model=PublicRydbergMultitargetSurfaceCodeSpec,
    hidden_model=HiddenRydbergMultitargetSurfaceCodeConfig,
    notebook_model=RydbergMultitargetLabNotebook,
    build_lab_notebook=build_rydberg_multitarget_lab_notebook,
    result_data_models=(JobCz2CharacterizationData, JobMultitargetMemoryData),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"multitarget_stabilizer_readout"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_rydberg_multitarget_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
