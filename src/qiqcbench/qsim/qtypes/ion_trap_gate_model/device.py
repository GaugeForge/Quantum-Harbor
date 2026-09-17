"""Public + hidden device config types for the ion_trap_gate_model qtype.

Public/hidden split mirrors trapped_ion_chain/device.py and
digital_gate_model/device.py: both classes use ``extra='forbid'``, so loading a
hidden YAML into the public type fails by construction.

The agent-visible spec carries claimed fidelities while the
real per-native-gate depolarizing rates and asymmetric readout live only in the
hidden config.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Public spec (visible to agent) ----------


class PublicIonTrapClaimedFidelities(_Strict):
    one_qubit_avg: float = Field(..., gt=0, le=1)
    two_qubit_ms: float = Field(..., gt=0, le=1)
    readout: float = Field(..., gt=0, le=1)


class IonTrapResourceBudget(_Strict):
    """Public compute, evidence, and transport limits; none is a score term."""

    max_experiment_jobs: int = Field(32, gt=0)
    max_randomized_unitaries_per_call: int = Field(200, gt=0)
    max_shots_per_randomized_unitary: int = Field(500, gt=0)
    max_randomized_shot_records_per_call: int = Field(6_000, gt=0)
    max_raw_shot_records_total: int = Field(36_000, gt=0)
    max_native_ops_per_circuit: int = Field(4_500, gt=0)
    max_ms_gates_per_circuit: int = Field(900, gt=0)
    max_randomized_subsystem_size: int = Field(8, gt=0)
    max_ingress_request_bytes: int = Field(4 * 2**20, gt=0)
    max_metadata_calls: int = Field(256, gt=0)
    max_job_result_polls: int = Field(4_096, gt=0)
    max_final_answer_serialized_bytes: int = Field(128 * 2**10, gt=0)
    max_final_answer_submissions: int = Field(4, gt=0)
    max_answer_string_characters: int = Field(16_384, gt=0)

    @model_validator(mode="after")
    def _consistent_limits(self) -> IonTrapResourceBudget:
        if self.max_randomized_shot_records_per_call > self.max_raw_shot_records_total:
            raise ValueError(
                "max_randomized_shot_records_per_call cannot exceed max_raw_shot_records_total"
            )
        if self.max_ms_gates_per_circuit > self.max_native_ops_per_circuit:
            raise ValueError("max_ms_gates_per_circuit cannot exceed max_native_ops_per_circuit")
        return self


class PublicIonTrapGateModelSpec(_Strict):
    """Hardware-manual-style public spec for the ion_trap_gate_model qtype."""

    schema_version: int = 2
    device_id: str
    qtype: Literal["ion_trap_gate_model"] = "ion_trap_gate_model"
    ion_species: str = "171Yb+"
    n_qubits: int = Field(..., ge=1, le=16)
    connectivity: Literal["all_to_all"] = "all_to_all"
    one_qubit_gates: list[str] = Field(
        default_factory=lambda: ["rx", "ry", "rz", "x", "y", "z", "h", "s", "sdg"]
    )
    two_qubit_gate: Literal["ms"] = "ms"
    claimed_fidelities: PublicIonTrapClaimedFidelities
    max_shots: int = Field(20_000, gt=0)
    budget: IonTrapResourceBudget = IonTrapResourceBudget()
    notes: str = (
        "All-to-all trapped-ion gate model. Native single-qubit rotations "
        "rx/ry/rz (universal; equivalent to the GPI/GPI2 native set) plus the "
        "Clifford set, and an arbitrary-angle two-qubit entangler "
        "ms(theta)=exp(-i theta/2 X_i X_j) on any chosen pair. Depolarizing noise "
        "is charged per native gate (the two-qubit MS dominates), so the MS-gate "
        "count drives your achieved fidelity. Claimed fidelities are nominal and "
        "may drift; budget for residual noise. You receive raw per-shot bitstrings."
    )


# ---------- Hidden truth (only inside qsim) ----------


class HiddenIonTrapReadout(_Strict):
    id: int = Field(..., ge=0)
    p_0_to_1: float = Field(..., ge=0, le=1)
    p_1_to_0: float = Field(..., ge=0, le=1)


class StaleIonTrapNotebookEntry(_Strict):
    """Thin, honest calibration claims surfaced via get_lab_notebook.

    Per the task's author intent the notebook carries NO misleading parameters;
    the difficulty is intrinsic to the physics. All fields nullable.
    """

    last_calibrated: str | None = None
    claimed_two_qubit_fidelity: float | None = None
    advisory_notes: str | None = None


class HiddenIonTrapGateModelConfig(_Strict):
    """Hidden truth for the ion_trap_gate_model qtype. Lives only inside qsim.

    ``one_qubit_depolarizing`` / ``ms_depolarizing`` are the per-gate probability
    of a depolarizing Pauli error (≈ 1 − gate fidelity for the dominant term);
    the engine applies a uniformly-random non-identity Pauli with this
    probability after each native gate. Readout is per-qubit asymmetric.
    """

    schema_version: int = 1
    device_id: str
    qtype: Literal["ion_trap_gate_model"] = "ion_trap_gate_model"
    seed: int
    n_qubits: int = Field(..., ge=1, le=16)
    one_qubit_depolarizing: float = Field(..., ge=0, le=1)
    ms_depolarizing: float = Field(..., ge=0, le=1)
    readout: list[HiddenIonTrapReadout] = Field(..., min_length=1)
    stale_lab_notebook: StaleIonTrapNotebookEntry = StaleIonTrapNotebookEntry()

    @field_validator("readout")
    @classmethod
    def _readout_nonempty(cls, v: list[HiddenIonTrapReadout]) -> list[HiddenIonTrapReadout]:
        if not v:
            raise ValueError("readout must be non-empty")
        return v

    @model_validator(mode="after")
    def _check_readout_matches_qubits(self) -> HiddenIonTrapGateModelConfig:
        ids = {r.id for r in self.readout}
        if len(ids) != len(self.readout):
            raise ValueError("readout ids must be unique")
        if ids != set(range(self.n_qubits)):
            raise ValueError(
                f"readout ids must be exactly 0..{self.n_qubits - 1}; got {sorted(ids)}"
            )
        return self


__all__ = [
    "HiddenIonTrapGateModelConfig",
    "HiddenIonTrapReadout",
    "IonTrapResourceBudget",
    "PublicIonTrapClaimedFidelities",
    "PublicIonTrapGateModelSpec",
    "StaleIonTrapNotebookEntry",
]
