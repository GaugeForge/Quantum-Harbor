"""Strict public and hidden models for coherent spin-chain control devices."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


ControllerId = Literal["A", "B", "C", "D"]
CONTROLLER_IDS: tuple[ControllerId, ...] = ("A", "B", "C", "D")


def controller_bank_digest(controllers: list[HiddenController] | list[dict]) -> str:
    payload = [
        item.model_dump(mode="json") if isinstance(item, BaseModel) else item
        for item in controllers
    ]
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class ControlBudgets(_Strict):
    max_shots_per_job: int = Field(..., gt=0)
    shot_budget: int = Field(..., gt=0)
    experiment_job_budget: int = Field(..., gt=0)
    max_recorded_bits_per_job: int = Field(..., gt=0)
    max_recorded_bits_per_run: int = Field(..., gt=0)


class PublicSpinChainControlSpec(_Strict):
    schema_version: Literal[2] = 2
    device_id: str
    qtype: Literal["spin_chain_control"] = "spin_chain_control"
    task_id: str
    n_qubits: Literal[3] = 3
    qubit_labels: list[str]
    connectivity: list[tuple[int, int]]
    controller_ids: list[ControllerId]
    controller_bank_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    initial_state: Literal["100"] = "100"
    target_state: Literal["001"] = "001"
    measured_qubits: list[int]
    bit_order: Literal["string_position_i_is_q_i"] = "string_position_i_is_q_i"
    perturbation_name: Literal["x_drive_scale_fraction"] = "x_drive_scale_fraction"
    perturbation_semantics: str
    x_drive_scale_fraction_min: float
    x_drive_scale_fraction_max: float
    x_drive_scale_fraction_resolution: float = Field(..., gt=0.0)
    success_threshold_pre_readout: float = Field(..., gt=0.0, lt=1.0)
    trusted_reference_preparations: list[Literal["000", "100", "010", "001"]]
    measurement_return: Literal["raw_bitstrings"] = "raw_bitstrings"
    budgets: ControlBudgets
    notes: str = ""

    @model_validator(mode="after")
    def _public_contract_is_consistent(self) -> PublicSpinChainControlSpec:
        if self.qubit_labels != ["q0", "q1", "q2"]:
            raise ValueError("spin-chain qubit labels must be q0, q1, q2")
        if self.connectivity != [(0, 1), (1, 2)]:
            raise ValueError("spin-chain connectivity must be the nearest-neighbor line")
        if self.controller_ids != ["A", "B", "C", "D"]:
            raise ValueError("controller bank must contain A, B, C, D in order")
        if self.measured_qubits != [0, 1, 2]:
            raise ValueError("spin-chain measurement order must be q0, q1, q2")
        if self.trusted_reference_preparations != ["000", "100", "010", "001"]:
            raise ValueError("trusted reference preparations are not canonical")
        if not self.x_drive_scale_fraction_min < 0.0 < self.x_drive_scale_fraction_max:
            raise ValueError("structured-control domain must contain zero")
        width = len(self.measured_qubits)
        if self.budgets.max_recorded_bits_per_job < width * self.budgets.max_shots_per_job:
            raise ValueError("per-job raw-bit cap is smaller than a maximum-size job")
        if self.budgets.max_recorded_bits_per_run < width * self.budgets.shot_budget:
            raise ValueError("run-wide raw-bit cap is smaller than the shot budget")
        return self


class HiddenController(_Strict):
    controller_id: ControllerId
    interval_duration_us: float = Field(..., gt=0.0)
    x_drive_rad_per_us: list[float] = Field(..., min_length=32, max_length=32)
    y_drive_rad_per_us: list[float] = Field(..., min_length=32, max_length=32)

    @model_validator(mode="after")
    def _waveform_is_finite(self) -> HiddenController:
        if any(
            not math.isfinite(value)
            for value in [*self.x_drive_rad_per_us, *self.y_drive_rad_per_us]
        ):
            raise ValueError("controller waveforms must be finite")
        return self


class HiddenReadout(_Strict):
    qubit: int = Field(..., ge=0, le=2)
    p_0_to_1: float = Field(..., ge=0.0, lt=1.0)
    p_1_to_0: float = Field(..., ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _invertible(self) -> HiddenReadout:
        if self.p_0_to_1 + self.p_1_to_0 >= 1.0:
            raise ValueError("readout assignment matrix must be invertible")
        return self


class StaleControlNotebook(_Strict):
    last_characterization: str
    recommended_controller_id: ControllerId
    claimed_nominal_success: dict[str, float]
    claimed_local_sensitivity: dict[str, float]
    recommendation_method: str
    drift_note: str
    note: str = ""


class HiddenSpinChainControlConfig(_Strict):
    schema_version: Literal[1] = 1
    device_id: str
    qtype: Literal["spin_chain_control"] = "spin_chain_control"
    seed: int
    exchange_rad_per_us: float = Field(..., gt=0.0)
    controller_preparation_failure: float = Field(..., ge=0.0, lt=1.0)
    run_offset_sigma_fraction: float = Field(..., ge=0.0)
    run_offset_clip_fraction: float = Field(..., ge=0.0)
    drift_peak_fraction: float = Field(..., ge=0.0)
    drift_period_jobs: float = Field(..., gt=2.0)
    drift_phase_rad: float
    controller_bank_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    controllers: list[HiddenController] = Field(..., min_length=4, max_length=4)
    readout: list[HiddenReadout] = Field(..., min_length=3, max_length=3)
    stale_lab_notebook: StaleControlNotebook

    @model_validator(mode="after")
    def _hidden_contract_is_consistent(self) -> HiddenSpinChainControlConfig:
        if [item.controller_id for item in self.controllers] != ["A", "B", "C", "D"]:
            raise ValueError("hidden controllers must be A, B, C, D in order")
        if [item.qubit for item in self.readout] != [0, 1, 2]:
            raise ValueError("hidden readout table must be q0, q1, q2 in order")
        if self.run_offset_clip_fraction < self.run_offset_sigma_fraction:
            raise ValueError("run-offset clip must not be smaller than its sigma")
        if self.controller_bank_sha256 != controller_bank_digest(self.controllers):
            raise ValueError("hidden controller-bank commitment mismatch")
        return self


__all__ = [
    "CONTROLLER_IDS",
    "ControllerId",
    "ControlBudgets",
    "HiddenController",
    "HiddenReadout",
    "HiddenSpinChainControlConfig",
    "PublicSpinChainControlSpec",
    "StaleControlNotebook",
    "controller_bank_digest",
]
