"""Public material contract for local randomized-measurement experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DepthCycle(_Strict):
    single_qubit_ensemble: Literal["independent_haar_su2"]
    single_qubit_native_compilation: Literal["rz_ry_rz"]
    even_cnot_edges: tuple[tuple[int, int], ...]
    odd_cnot_edges: tuple[tuple[int, int], ...]
    cnot_direction: Literal["lower_index_control_to_higher_index_target"]


class MeasurementProtocol(_Strict):
    ensemble: Literal["independent_local_haar_su2"]
    measured_qubits: tuple[int, ...] = Field(..., min_length=1)
    readout_calibration: Literal["all_zero_and_all_one"]


class SamplingProtocol(_Strict):
    n_random_circuits: int = Field(..., ge=2, le=64)
    n_measurement_bases: int = Field(..., ge=2, le=256)
    shots_per_basis: int = Field(..., ge=2, le=20_000)
    readout_calibration_shots_per_preparation: int = Field(..., ge=2, le=100_000)


class DepthLimits(_Strict):
    minimum: int = Field(..., ge=1)
    maximum: int = Field(..., ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> DepthLimits:
        if self.maximum < self.minimum:
            raise ValueError("depth maximum must be at least the minimum")
        return self


class LocalRandomizedMeasurementProtocol(_Strict):
    schema_version: Literal[1]
    task_id: str = Field(..., min_length=1)
    capability: Literal["local_randomized_measurement"]
    n_qubits: int = Field(..., ge=2, le=10)
    subsystem_qubits: tuple[int, ...] = Field(..., min_length=1)
    initial_state: Literal["all_zero"]
    depth_cycle: DepthCycle
    measurement: MeasurementProtocol
    sampling: SamplingProtocol
    depth_limits: DepthLimits
    max_experiment_jobs: int = Field(..., ge=2, le=32)
    randomness: Literal["fresh_private_per_job"]

    @model_validator(mode="after")
    def _semantic_shape(self) -> LocalRandomizedMeasurementProtocol:
        expected = tuple(range(self.n_qubits))
        if self.measurement.measured_qubits != expected:
            raise ValueError("local randomized measurement must return the full register in order")
        if len(set(self.subsystem_qubits)) != len(self.subsystem_qubits) or any(
            qubit not in expected for qubit in self.subsystem_qubits
        ):
            raise ValueError("subsystem_qubits must be distinct valid qubits")
        for label, edges in (
            ("even", self.depth_cycle.even_cnot_edges),
            ("odd", self.depth_cycle.odd_cnot_edges),
        ):
            touched: set[int] = set()
            for left, right in edges:
                if left < 0 or right >= self.n_qubits or left >= right:
                    raise ValueError(f"{label} CNOT edges must be ordered valid qubit pairs")
                if left in touched or right in touched:
                    raise ValueError(f"{label} CNOT edges must be disjoint")
                touched.update((left, right))
        raw_records = (
            self.sampling.n_random_circuits
            * self.sampling.n_measurement_bases
            * self.sampling.shots_per_basis
            + 2 * self.sampling.readout_calibration_shots_per_preparation
        )
        if raw_records > 1_000_000:
            raise ValueError("one randomized-measurement job exceeds the raw-shot safety bound")
        return self


def load_protocol(public_material_dir: Path) -> LocalRandomizedMeasurementProtocol:
    path = public_material_dir / "measurement_protocol.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("randomized-measurement protocol material is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"randomized-measurement protocol material is unreadable: {exc}") from None
    return LocalRandomizedMeasurementProtocol.model_validate(payload)
