"""Public and hidden schemas for scheduled native-gate transmon devices."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NativeGateDurations(_Strict):
    rz_ns: int = Field(0, ge=0)
    sx_ns: int = Field(..., gt=0)
    x_ns: int = Field(..., gt=0)
    ecr_ns: int = Field(..., gt=0)


class CompilationBudgets(_Strict):
    max_shots_per_job: int = Field(..., gt=0)
    shot_budget: int = Field(..., gt=0)
    experiment_job_budget: int = Field(..., gt=0)
    max_recorded_bits_per_job: int = Field(..., gt=0)
    max_recorded_bits_per_run: int = Field(..., gt=0)


class ClaimedScheduledFidelities(_Strict):
    one_qubit_average: float = Field(..., gt=0.0, le=1.0)
    ecr_average: float = Field(..., gt=0.0, le=1.0)
    readout_average: float = Field(..., gt=0.0, le=1.0)


class HiddenTemporalCoupling(_Strict):
    """How one physical error channel couples to the shared drift clock.

    A zero phase offset and unit sensitivity reproduce the legacy global-v1
    multiplier.  The coupling is hidden physical device state; it contains no
    candidate identity or task score.
    """

    sensitivity: float = Field(1.0, ge=0.0, le=1.5)
    phase_offset_rad: float = 0.0

    @model_validator(mode="after")
    def _values_are_finite(self) -> HiddenTemporalCoupling:
        if not math.isfinite(self.sensitivity) or not math.isfinite(self.phase_offset_rad):
            raise ValueError("temporal coupling values must be finite")
        return self


class PublicScheduledTransmonSpec(_Strict):
    """Agent-visible scheduled-transmon manual; contains no true noise values."""

    schema_version: Literal[1] = 1
    device_id: str
    qtype: Literal["scheduled_transmon_gate_model"] = "scheduled_transmon_gate_model"
    task_id: str
    n_physical_qubits: int = Field(..., ge=2)
    candidate_ids: list[str] = Field(..., min_length=1)
    candidate_bank_file: str
    candidate_bank_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    protocol_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    measured_physical_qubits: list[int] = Field(..., min_length=1)
    undirected_connectivity: list[tuple[int, int]] = Field(..., min_length=1)
    directed_ecr_edges: list[tuple[int, int]] = Field(..., min_length=1)
    native_gates: list[Literal["rz", "sx", "x", "ecr"]]
    native_durations: NativeGateDurations
    mirror_seed_domain: list[int] = Field(..., min_length=3)
    mirror_frame_rule: str
    bit_order: Literal["string_position_i_is_measured_physical_qubits_i"] = (
        "string_position_i_is_measured_physical_qubits_i"
    )
    measurement_return: Literal["raw_unframed_bitstrings"] = "raw_unframed_bitstrings"
    trusted_reference_preparations: list[str] = Field(..., min_length=2)
    claimed_fidelities: ClaimedScheduledFidelities
    budgets: CompilationBudgets
    notes: str = ""

    @model_validator(mode="after")
    def _public_topology_is_consistent(self) -> PublicScheduledTransmonSpec:
        if len(self.candidate_ids) != len(set(self.candidate_ids)):
            raise ValueError("candidate_ids must be unique")
        if len(self.measured_physical_qubits) != len(set(self.measured_physical_qubits)):
            raise ValueError("measured_physical_qubits must be unique")
        if any(q < 0 or q >= self.n_physical_qubits for q in self.measured_physical_qubits):
            raise ValueError("measured physical qubit is outside device range")
        if len(self.mirror_seed_domain) != len(set(self.mirror_seed_domain)):
            raise ValueError("mirror_seed_domain must be unique")
        directed = set(self.directed_ecr_edges)
        for a, b in self.undirected_connectivity:
            if a == b or (a, b) not in directed or (b, a) not in directed:
                raise ValueError("every connectivity link must admit both directed ECRs")
        width = len(self.measured_physical_qubits)
        expected_refs = {"0" * width}
        for index in range(width):
            bits = ["0"] * width
            bits[index] = "1"
            expected_refs.add("".join(bits))
        if set(self.trusted_reference_preparations) != expected_refs:
            raise ValueError("trusted references must be zero plus every single excitation")
        if self.budgets.max_recorded_bits_per_job < width * self.budgets.max_shots_per_job:
            raise ValueError("per-job raw-bit cap is smaller than one maximum-size job")
        if self.budgets.max_recorded_bits_per_run < width * self.budgets.shot_budget:
            raise ValueError("run-wide raw-bit cap is smaller than the shot budget")
        return self


class HiddenDirectedEcrNoise(_Strict):
    edge: tuple[int, int]
    stochastic_rate: float = Field(..., ge=0.0, le=1.0)
    stochastic_pauli_weights: list[float] = Field(..., min_length=15, max_length=15)
    coherent_overrotation_rad: float = Field(..., ge=-0.2, le=0.2)
    stochastic_drift: HiddenTemporalCoupling = Field(default_factory=HiddenTemporalCoupling)
    coherent_drift: HiddenTemporalCoupling = Field(default_factory=HiddenTemporalCoupling)

    @model_validator(mode="after")
    def _pauli_channel_is_valid(self) -> HiddenDirectedEcrNoise:
        if any(not math.isfinite(value) or value < 0.0 for value in self.stochastic_pauli_weights):
            raise ValueError("stochastic Pauli weights must be finite and nonnegative")
        if sum(self.stochastic_pauli_weights) <= 0.0:
            raise ValueError("stochastic Pauli weights must have positive total weight")
        return self


class HiddenConflictClass(_Strict):
    edge_a: tuple[int, int]
    edge_b: tuple[int, int]
    correlated_zz_angle_rad: float = Field(..., ge=0.0, le=0.2)
    drift: HiddenTemporalCoupling = Field(default_factory=HiddenTemporalCoupling)


class HiddenScheduledReadout(_Strict):
    physical_qubit: int = Field(..., ge=0)
    p_0_to_1: float = Field(..., ge=0.0, le=1.0)
    p_1_to_0: float = Field(..., ge=0.0, le=1.0)


class HiddenDrift(_Strict):
    model: Literal["global_v1", "channel_local_v2"] = "global_v1"
    peak_fraction: float = Field(..., ge=0.0, le=0.5)
    period_jobs: float = Field(..., gt=2.0)
    phase_rad: float

    @model_validator(mode="after")
    def _clock_is_finite(self) -> HiddenDrift:
        if not math.isfinite(self.period_jobs) or not math.isfinite(self.phase_rad):
            raise ValueError("drift clock values must be finite")
        return self


class StaleScheduledNotebook(_Strict):
    last_full_calibration: str
    claimed_ecr_fidelity: float = Field(..., gt=0.0, le=1.0)
    claimed_readout_fidelity: float = Field(..., gt=0.0, le=1.0)
    simultaneous_gate_penalty: str
    recommended_fom: str
    recommended_candidate_id: str
    drift_note: str
    note: str = ""
    reliability: str = ""


class HiddenScheduledTransmonConfig(_Strict):
    """True edge, context, coherence, readout, and drift model inside qsim."""

    schema_version: Literal[1, 2] = 1
    device_id: str
    qtype: Literal["scheduled_transmon_gate_model"] = "scheduled_transmon_gate_model"
    seed: int
    one_qubit_depolarizing: float = Field(..., ge=0.0, le=1.0)
    ecr_noise: list[HiddenDirectedEcrNoise] = Field(..., min_length=1)
    conflict_classes: list[HiddenConflictClass] = Field(default_factory=list)
    idle_t2_us: list[float] = Field(..., min_length=1)
    readout: list[HiddenScheduledReadout] = Field(..., min_length=1)
    drift: HiddenDrift
    stale_lab_notebook: StaleScheduledNotebook

    @model_validator(mode="after")
    def _hidden_tables_are_well_formed(self) -> HiddenScheduledTransmonConfig:
        edges = [entry.edge for entry in self.ecr_noise]
        if len(edges) != len(set(edges)) or any(a < 0 or b < 0 or a == b for a, b in edges):
            raise ValueError("hidden directed-ECR edges must be unique, nonnegative links")
        conflicts = [frozenset((entry.edge_a, entry.edge_b)) for entry in self.conflict_classes]
        if len(conflicts) != len(set(conflicts)) or any(
            entry.edge_a == entry.edge_b for entry in self.conflict_classes
        ):
            raise ValueError("hidden conflict classes must be unique pairs of distinct edges")
        readout_qubits = [entry.physical_qubit for entry in self.readout]
        if len(readout_qubits) != len(set(readout_qubits)):
            raise ValueError("hidden readout table must contain unique physical qubits")
        if any(entry.p_0_to_1 + entry.p_1_to_0 >= 1.0 for entry in self.readout):
            raise ValueError("hidden readout confusion matrices must be invertible")
        if any(value <= 0.0 for value in self.idle_t2_us):
            raise ValueError("idle T2 values must be positive")
        couplings = [
            coupling
            for entry in self.ecr_noise
            for coupling in (entry.stochastic_drift, entry.coherent_drift)
        ] + [entry.drift for entry in self.conflict_classes]
        if self.schema_version == 1:
            if self.drift.model != "global_v1" or any(
                coupling.sensitivity != 1.0 or coupling.phase_offset_rad != 0.0
                for coupling in couplings
            ):
                raise ValueError(
                    "scheduled-transmon hidden schema v1 requires the legacy global drift law"
                )
        elif self.drift.model != "channel_local_v2":
            raise ValueError("scheduled-transmon hidden schema v2 requires channel_local_v2 drift")
        if any(self.drift.peak_fraction * coupling.sensitivity >= 1.0 for coupling in couplings):
            raise ValueError("channel-local drift multipliers must remain strictly positive")
        return self


__all__ = [
    "ClaimedScheduledFidelities",
    "CompilationBudgets",
    "HiddenConflictClass",
    "HiddenDirectedEcrNoise",
    "HiddenDrift",
    "HiddenScheduledReadout",
    "HiddenScheduledTransmonConfig",
    "HiddenTemporalCoupling",
    "NativeGateDurations",
    "PublicScheduledTransmonSpec",
    "StaleScheduledNotebook",
]
