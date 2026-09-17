"""Public physical contract and admission checks for randomized measurement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from ruamel.yaml import YAML

from qiqcbench.qsim.core.wire import SCHEMA_VERSION, GateOp
from qiqcbench.qsim.qtypes.ion_trap_gate_model.device import (
    PublicIonTrapGateModelSpec,
)
from qiqcbench.qsim.qtypes.ion_trap_gate_model.wire import (
    IonCircuitRequest,
    RandomizedMeasurementBatchRequestV2,
)

_yaml = YAML(typ="safe")


class RandomizedMeasurementMaterials(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2] = 2
    randomness_policy: Literal["fresh_private_per_job"] = "fresh_private_per_job"
    local_basis_ensemble: Literal["single_qubit_haar"] = "single_qubit_haar"
    analysis_rotation_model: Literal["ideal"] = "ideal"
    returned_outcomes: Literal["post_readout_raw_bitstrings"] = "post_readout_raw_bitstrings"


def load_randomized_measurement_materials(public_dir: Path) -> RandomizedMeasurementMaterials:
    path = Path(public_dir) / "randomized_measurement_capability.yaml"
    if not path.is_file():
        raise ValueError("randomized-measurement public capability material is missing")
    with path.open(encoding="utf-8") as handle:
        data = _yaml.load(handle)
    return RandomizedMeasurementMaterials.model_validate(data or {})


def _validate_native_circuit(
    circuit: list,
    public: PublicIonTrapGateModelSpec,
) -> None:
    budget = public.budget
    if len(circuit) > budget.max_native_ops_per_circuit:
        raise ValueError(
            f"circuit has {len(circuit)} native ops; public limit is "
            f"max_native_ops_per_circuit={budget.max_native_ops_per_circuit}"
        )
    if any(not isinstance(op, GateOp) for op in circuit):
        raise ValueError("ion-trap circuits may contain only native gate ops")
    supported = set(public.one_qubit_gates) | {public.two_qubit_gate}
    unsupported = sorted({op.name for op in circuit if op.name not in supported})
    if unsupported:
        raise ValueError(f"circuit contains unsupported native gates: {unsupported}")
    if any(isinstance(value, str) for op in circuit for value in op.params):
        raise ValueError("ion-trap circuits require concrete numeric gate parameters")
    ms_count = sum(op.name == public.two_qubit_gate for op in circuit)
    if ms_count > budget.max_ms_gates_per_circuit:
        raise ValueError(
            f"circuit has {ms_count} ms gates; public admission limit is "
            f"max_ms_gates_per_circuit={budget.max_ms_gates_per_circuit}"
        )


def _validate_request_bytes(
    request: IonCircuitRequest | RandomizedMeasurementBatchRequestV2,
    public: PublicIonTrapGateModelSpec,
) -> None:
    serialized = json.dumps(
        request.model_dump(mode="json"),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(serialized) > public.budget.max_ingress_request_bytes:
        raise ValueError(
            f"request uses {len(serialized)} compact JSON bytes; public limit is "
            f"max_ingress_request_bytes={public.budget.max_ingress_request_bytes}"
        )


def validate_ion_circuit_request(
    request: IonCircuitRequest,
    public: PublicIonTrapGateModelSpec,
) -> None:
    _validate_request_bytes(request, public)
    if request.schema_version != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if request.shots > public.max_shots:
        raise ValueError(f"shots={request.shots} exceeds max_shots={public.max_shots}")
    _validate_native_circuit(list(request.circuit), public)
    measured = list(request.measured_qubits)
    if len(set(measured)) != len(measured):
        raise ValueError("measured_qubits must be distinct")
    if any(qubit < 0 or qubit >= public.n_qubits for qubit in measured):
        raise ValueError(
            f"measured_qubits out of range for {public.n_qubits}-qubit device: {measured}"
        )


def validate_rm_batch_request(
    request: RandomizedMeasurementBatchRequestV2,
    public: PublicIonTrapGateModelSpec,
) -> None:
    budget = public.budget
    _validate_request_bytes(request, public)
    if request.schema_version != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    _validate_native_circuit(list(request.prepare_circuit), public)
    if request.n_unitaries > budget.max_randomized_unitaries_per_call:
        raise ValueError(
            f"n_unitaries={request.n_unitaries} exceeds "
            f"max_randomized_unitaries_per_call={budget.max_randomized_unitaries_per_call}"
        )
    if request.shots_per_unitary > budget.max_shots_per_randomized_unitary:
        raise ValueError(
            f"shots_per_unitary={request.shots_per_unitary} exceeds "
            "max_shots_per_randomized_unitary="
            f"{budget.max_shots_per_randomized_unitary}"
        )
    total = request.n_unitaries * request.shots_per_unitary
    if total > budget.max_randomized_shot_records_per_call:
        raise ValueError(
            f"n_unitaries*shots_per_unitary={total} exceeds "
            "max_randomized_shot_records_per_call="
            f"{budget.max_randomized_shot_records_per_call}"
        )
    subsystem = list(request.subsystem_qubits)
    if len(set(subsystem)) != len(subsystem):
        raise ValueError("subsystem_qubits must be distinct")
    if len(subsystem) > budget.max_randomized_subsystem_size:
        raise ValueError(
            f"subsystem has {len(subsystem)} qubits; public limit is "
            f"max_randomized_subsystem_size={budget.max_randomized_subsystem_size}"
        )
    if any(qubit < 0 or qubit >= public.n_qubits for qubit in subsystem):
        raise ValueError(
            f"subsystem_qubits out of range for {public.n_qubits}-qubit device: {subsystem}"
        )


__all__ = [
    "RandomizedMeasurementMaterials",
    "load_randomized_measurement_materials",
    "validate_ion_circuit_request",
    "validate_rm_batch_request",
]
