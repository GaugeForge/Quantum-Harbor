"""randomized_measurement capability for ion_trap_gate_model.

run_ion_circuit (single native-gate circuit -> raw bitstrings) and
run_randomized_measurement_batch (prepare per shot + qsim-private local Haar
bases -> per-setting subsystem bitstrings for second-Renyi estimation).
"""

from qiqcbench.qsim.qtypes.ion_trap_gate_model.capabilities.randomized_measurement.materials import (
    RandomizedMeasurementMaterials,
    load_randomized_measurement_materials,
    validate_ion_circuit_request,
    validate_rm_batch_request,
)
from qiqcbench.qsim.qtypes.ion_trap_gate_model.capabilities.randomized_measurement.runtime import (
    run_ion_circuit_request,
    run_rm_batch_request,
)

__all__ = [
    "RandomizedMeasurementMaterials",
    "load_randomized_measurement_materials",
    "run_ion_circuit_request",
    "run_rm_batch_request",
    "validate_ion_circuit_request",
    "validate_rm_batch_request",
]
