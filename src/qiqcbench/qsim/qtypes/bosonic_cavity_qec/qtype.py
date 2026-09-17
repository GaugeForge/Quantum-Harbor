"""bosonic_cavity_qec qtype descriptor (discovered by qtypes/registry.py).

A storage cavity (bosonic mode, Fock cutoff n_max) dispersively coupled to a
transmon ancilla, with SNAP/displacement universal control, a dispersive
photon-number-parity measurement, and mid-circuit measurement + classical
feedback. Simulator-only, MCP-only.
"""

from __future__ import annotations

from qiqcbench.qsim.qtypes.bosonic_cavity_qec.device import (
    HiddenBosonicCavityConfig,
    PublicBosonicCavitySpec,
)
from qiqcbench.qsim.qtypes.bosonic_cavity_qec.lab_notebook import (
    BosonicCavityLabNotebook,
    build_bosonic_cavity_lab_notebook,
)
from qiqcbench.qsim.qtypes.bosonic_cavity_qec.wire import JobBosonicProgramData
from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions

_PKG = "qiqcbench.qsim.qtypes.bosonic_cavity_qec"

DESCRIPTOR = QtypeDescriptor(
    qtype="bosonic_cavity_qec",
    public_model=PublicBosonicCavitySpec,
    hidden_model=HiddenBosonicCavityConfig,
    notebook_model=BosonicCavityLabNotebook,
    build_lab_notebook=build_bosonic_cavity_lab_notebook,
    result_data_models=(JobBosonicProgramData,),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"basic_measurement", "bosonic_qec_control"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_bosonic_cavity_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
