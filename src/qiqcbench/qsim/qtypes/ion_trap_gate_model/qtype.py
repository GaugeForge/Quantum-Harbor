"""ion_trap_gate_model qtype descriptor (discovered by qtypes/registry.py)."""

from __future__ import annotations

from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.ion_trap_gate_model.device import (
    HiddenIonTrapGateModelConfig,
    PublicIonTrapGateModelSpec,
)
from qiqcbench.qsim.qtypes.ion_trap_gate_model.lab_notebook import (
    IonTrapLabNotebook,
    build_ion_trap_lab_notebook,
)
from qiqcbench.qsim.qtypes.ion_trap_gate_model.wire import (
    JobIonBitstringData,
    JobRandomizedMeasurementData,
    JobRandomizedMeasurementDataV2,
)

_PKG = "qiqcbench.qsim.qtypes.ion_trap_gate_model"

DESCRIPTOR = QtypeDescriptor(
    qtype="ion_trap_gate_model",
    public_model=PublicIonTrapGateModelSpec,
    hidden_model=HiddenIonTrapGateModelConfig,
    notebook_model=IonTrapLabNotebook,
    build_lab_notebook=build_ion_trap_lab_notebook,
    result_data_models=(
        JobIonBitstringData,
        JobRandomizedMeasurementData,
        JobRandomizedMeasurementDataV2,
    ),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"basic_measurement", "randomized_measurement"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_ion_trap_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
