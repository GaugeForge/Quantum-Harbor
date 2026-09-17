"""Strict public/hidden schemas for the Rydberg unrotated surface-code device."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Coordinate(_Strict):
    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)


class DataQubitSpec(_Strict):
    role: str
    coordinate: Coordinate
    simulator_index: int = Field(..., ge=0)


class StabilizerCheckSpec(_Strict):
    check_id: str
    ancilla_role: str
    coordinate: Coordinate
    simulator_index: int = Field(..., ge=0)
    pauli: Literal["X", "Z"]
    data_roles: list[str] = Field(..., min_length=3, max_length=4)
    bulk_design_id: str | None = None

    @model_validator(mode="after")
    def _bulk_is_exactly_weight_four(self) -> StabilizerCheckSpec:
        if (self.bulk_design_id is None) == (len(self.data_roles) == 4):
            raise ValueError("exactly weight-four checks must declare bulk_design_id")
        return self


class LogicalOperatorSpec(_Strict):
    logical_x_data_roles: list[str] = Field(..., min_length=3, max_length=3)
    logical_z_data_roles: list[str] = Field(..., min_length=3, max_length=3)


class NativeGateSpec(_Strict):
    local_layer_us: float = Field(..., gt=0.0)
    cz_us: float = Field(..., gt=0.0)
    cz2_us: float = Field(..., gt=0.0)
    measurement_reset_us: float = Field(..., gt=0.0)


class FixedDecoderSpec(_Strict):
    revision: Literal["unrotated_d3_mwpm_v2"]
    nominal_independent_clifford_error: float = Field(..., gt=0.0, lt=0.1)
    nominal_measurement_flip_probability: float = Field(..., gt=0.0, lt=0.1)
    nominal_reset_flip_probability: float = Field(..., gt=0.0, lt=0.1)
    nominal_correlated_pair_error: float = Field(..., gt=0.0, lt=0.1)


class MultitargetBudgetSpec(_Strict):
    shot_budget: int = Field(..., ge=1)
    experiment_job_budget: int = Field(..., ge=1)
    max_shots_per_job: int = Field(..., ge=1)
    scored_memory_rounds: int = Field(..., ge=3, le=32)
    replay_shots_per_basis: int = Field(..., ge=1)
    replay_confidence_level: float = Field(..., gt=0.5, lt=1.0)
    terminal_logical_failure_probability_max: float = Field(..., gt=0.0, lt=0.5)


class PublicRydbergMultitargetSurfaceCodeSpec(_Strict):
    schema_version: Literal[3] = 3
    device_id: str
    qtype: Literal["rydberg_multitarget_surface_code"] = "rydberg_multitarget_surface_code"
    task_id: str | None = None
    code: Literal["unrotated_surface_code"] = "unrotated_surface_code"
    distance: Literal[3] = 3
    data_qubits: list[DataQubitSpec] = Field(..., min_length=13, max_length=13)
    stabilizer_checks: list[StabilizerCheckSpec] = Field(..., min_length=12, max_length=12)
    logical_operators: LogicalOperatorSpec
    decoded_bases: list[Literal["Z", "X"]] = Field(..., min_length=2, max_length=2)
    native_gates: NativeGateSpec
    fixed_decoder: FixedDecoderSpec
    budgets: MultitargetBudgetSpec
    notes: str = ""

    @model_validator(mode="after")
    def _validate_code_geometry(self) -> PublicRydbergMultitargetSurfaceCodeSpec:
        data_roles = [item.role for item in self.data_qubits]
        ancilla_roles = [item.ancilla_role for item in self.stabilizer_checks]
        check_ids = [item.check_id for item in self.stabilizer_checks]
        bulk_ids = [
            item.bulk_design_id
            for item in self.stabilizer_checks
            if item.bulk_design_id is not None
        ]
        if data_roles != [f"d{index}" for index in range(13)]:
            raise ValueError("data_qubits must be in canonical d0..d12 order")
        if ancilla_roles != [f"m{index}" for index in range(12)]:
            raise ValueError("stabilizer_checks must be in canonical m0..m11 order")
        if len(set(check_ids)) != 12:
            raise ValueError("stabilizer check IDs must be unique")
        if bulk_ids != ["B0", "B1", "B2", "B3"]:
            raise ValueError("weight-four checks must expose canonical B0..B3 design IDs")
        known = set(data_roles)
        if any(set(check.data_roles) - known for check in self.stabilizer_checks):
            raise ValueError("stabilizer support references an unknown data role")
        logical = self.logical_operators
        if set(logical.logical_x_data_roles) - known or set(logical.logical_z_data_roles) - known:
            raise ValueError("logical representative references an unknown data role")
        if self.decoded_bases != ["Z", "X"]:
            raise ValueError("decoded_bases must use canonical Z, X order")
        return self

    @property
    def data_roles(self) -> list[str]:
        return [item.role for item in self.data_qubits]

    @property
    def measurement_ancilla_roles(self) -> list[str]:
        return [item.ancilla_role for item in self.stabilizer_checks]

    @property
    def bulk_checks(self) -> list[StabilizerCheckSpec]:
        return [item for item in self.stabilizer_checks if item.bulk_design_id is not None]


class HiddenRydbergMultitargetNoise(_Strict):
    duration_scale_optimum: float = Field(..., ge=0.90, le=1.10)
    target_phase_compensation_rad_optimum: float = Field(..., ge=-0.30, le=0.30)
    independent_clifford_error: float = Field(..., gt=0.0, lt=0.05)
    measurement_flip_probability: float = Field(..., gt=0.0, lt=0.05)
    reset_flip_probability: float = Field(..., gt=0.0, lt=0.05)
    correlated_pair_error_at_optimum: float = Field(..., gt=0.0, lt=0.05)
    correlated_pair_duration_curvature: float = Field(..., gt=0.0, lt=10.0)
    correlated_pair_phase_curvature: float = Field(..., gt=0.0, lt=10.0)
    idle_depolarization_per_us: float = Field(..., ge=0.0, lt=0.01)
    characterization_readout_flip_probability: float = Field(..., ge=0.0, lt=0.1)


class StaleRydbergMultitargetNotebook(_Strict):
    claimed_cz2_fidelity: str = "0.9965 on the previous calibration sequence"
    duration_advice: str = "use duration_scale=1.0"
    phase_advice: str = "use target_phase_compensation_rad=0.0"
    partition_advice: str = "all legal pairings were treated as equivalent"
    note: str = ""


class HiddenRydbergMultitargetSurfaceCodeConfig(_Strict):
    schema_version: Literal[2] = 2
    device_id: str
    qtype: Literal["rydberg_multitarget_surface_code"] = "rydberg_multitarget_surface_code"
    construction_id: str
    noise: HiddenRydbergMultitargetNoise
    stale_lab_notebook: StaleRydbergMultitargetNotebook = StaleRydbergMultitargetNotebook()


__all__ = [
    "HiddenRydbergMultitargetNoise",
    "HiddenRydbergMultitargetSurfaceCodeConfig",
    "PublicRydbergMultitargetSurfaceCodeSpec",
    "StabilizerCheckSpec",
]
