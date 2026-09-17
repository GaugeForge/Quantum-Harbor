"""digital_gate_model qtype descriptor (discovered by qtypes/registry.py)."""

from __future__ import annotations

from qiqcbench.qsim.core.wire import JobBitstringData, JobObservableBitstringData
from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.digital_gate_model.capabilities.randomized_measurement.schemas import (
    JobLocalRandomizedMeasurementData,
)
from qiqcbench.qsim.qtypes.digital_gate_model.device import (
    HiddenDigitalConfig,
    PublicDigitalSpec,
)
from qiqcbench.qsim.qtypes.digital_gate_model.lab_notebook import (
    DigitalLabNotebook,
    build_digital_lab_notebook,
)

_PKG = "qiqcbench.qsim.qtypes.digital_gate_model"

DESCRIPTOR = QtypeDescriptor(
    qtype="digital_gate_model",
    public_model=PublicDigitalSpec,
    hidden_model=HiddenDigitalConfig,
    notebook_model=DigitalLabNotebook,
    build_lab_notebook=build_digital_lab_notebook,
    result_data_models=(
        JobBitstringData,
        JobObservableBitstringData,
        JobLocalRandomizedMeasurementData,
    ),
    supported_backend_modes=frozenset({"simulator", "provider_replay", "live_provider"}),
    supported_surfaces=frozenset({"mcp", "qiskit"}),
    supported_capabilities=frozenset(
        {
            "basic_measurement",
            "digital_circuit_execution",
            "digital_vqe",
            "local_randomized_measurement",
        }
    ),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_digital_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
    build_replay_backend=lazy(f"{_PKG}.replay", "build_digital_replay_backend"),
    build_live_provider_backend=lazy(f"{_PKG}.live_ibm", "build_digital_live_provider_backend"),
    run_circuit_sequence=lazy(f"{_PKG}.runner", "run_circuit_sequence"),
    run_circuit_sweep_request=lazy(f"{_PKG}.runner", "run_circuit_sweep_request"),
)
