"""Public + hidden device config types for the transmon g-f erasure-qutrit qtype.

A ``transmon_erasure_qutrit`` device is a transmon **qutrit** (levels ``g``,
``e``, ``f``) operated as a ``g``-``f`` erasure qubit (``|0_L>=|g>``,
``|1_L>=|f>``, ``|e>`` = erasure/leakage) plus a neighbouring ancilla configured
for mid-circuit erasure detection. The agent prepares a logical state, interleaves
dynamical decoupling with erasure-detection rounds at a chosen cycle time, and
reads out the data qutrit.

Public / hidden are STRUCTURALLY separated: both classes
use ``extra="forbid"``. The PUBLIC spec advertises control ranges and *claimed*
rates only; the true cascaded-decay times, the emergent post-selected bit-flip
lifetime, the detection error rates, and the stale notebook live in the HIDDEN
config inside qsim.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErasureQutritBudget(_Strict):
    """Public transport/evidence bounds; none of these fields is a score term."""

    max_sweep_points_per_call: int = Field(128, ge=1)
    max_final_assignments_per_call: int = Field(2_000_000, ge=1)
    max_syndrome_bits_per_call: int = Field(33_554_432, ge=1)
    max_experiment_jobs: int = Field(4096, ge=1)
    max_final_assignments_total: int = Field(16_777_216, ge=1)
    max_syndrome_bits_total: int = Field(134_217_728, ge=1)
    max_ingress_request_bytes: int = Field(4 * 2**20, ge=1)
    max_metadata_calls: int = Field(16_384, ge=1)
    max_job_result_polls: int = Field(32_768, ge=1)
    max_final_answer_serialized_bytes: int = Field(256 * 2**10, ge=1)
    max_final_answer_submissions: int = Field(8, ge=1)
    max_answer_string_characters: int = Field(16_384, ge=1)

    @model_validator(mode="after")
    def _validate_raw_record_limits(self) -> ErasureQutritBudget:
        if self.max_final_assignments_per_call > self.max_final_assignments_total:
            raise ValueError(
                "max_final_assignments_per_call cannot exceed max_final_assignments_total"
            )
        if self.max_syndrome_bits_per_call > self.max_syndrome_bits_total:
            raise ValueError("max_syndrome_bits_per_call cannot exceed max_syndrome_bits_total")
        return self


class ErasureQutritMeasurementModel(_Strict):
    """Public ordering and assignment semantics; numerical rates stay hidden."""

    mid_circuit_ancilla_records: Literal["post_readout_raw_binary"] = "post_readout_raw_binary"
    final_qutrit_records: Literal["post_readout_raw_three_level_assignment"] = (
        "post_readout_raw_three_level_assignment"
    )
    final_qutrit_assignment_channel: Literal["stationary_asymmetric_classical_confusion"] = (
        "stationary_asymmetric_classical_confusion"
    )
    final_qutrit_assignment_parameters: Literal["hidden"] = "hidden"
    shot_index_alignment: Literal["preserved_within_point"] = "preserved_within_point"


# ---------- Public spec (visible to agent) ----------


class PublicErasureQutritSpec(_Strict):
    """Hardware-manual-style public spec for the g-f erasure-qutrit qtype."""

    schema_version: int = 2
    device_id: str
    qtype: Literal["transmon_erasure_qutrit"] = "transmon_erasure_qutrit"

    data_qubit: str = "Q1"
    ancilla_qubit: str = "Qa"

    # Advertised (range-only) physical decay times; true values are hidden.
    t1_ge_us_range: tuple[float, float] = (30.0, 70.0)
    t1_ef_us_range: tuple[float, float] = (10.0, 30.0)

    # Logical prep + measurement.
    prep_bases: list[str] = Field(default_factory=lambda: ["z", "x"])
    measurement_bases: list[str] = Field(default_factory=lambda: ["z", "x"])
    dd_options: list[str] = Field(default_factory=lambda: ["xy4", "none"])

    # Erasure-detection cadence controls.
    min_cycle_time_us: float = Field(3.5, gt=0)
    max_cycle_time_us: float = Field(12.0, gt=0)
    cycle_time_resolution_us: float = Field(0.01, gt=0)
    max_total_evolution_us: float = Field(300.0, gt=0)
    max_rounds: int = Field(256, ge=1)

    # Claimed (not real) detection rates.
    claimed_false_positive_rate: float = Field(0.025, ge=0, le=1)
    claimed_false_negative_rate: float = Field(0.07, ge=0, le=1)

    measurement_return: Literal["erasure_memory_round_resolved", "erasure_memory"] = (
        "erasure_memory_round_resolved"
    )
    measurement_model: ErasureQutritMeasurementModel = Field(
        default_factory=ErasureQutritMeasurementModel
    )
    max_shots: int = 100_000
    budget: ErasureQutritBudget = Field(default_factory=ErasureQutritBudget)
    notes: str = (
        "Calibrations may be stale. |1_L>=|f> is the second excited state; "
        "relaxation is cascaded and detectable as leakage to |e>. The mid-circuit "
        "erasure-detection ancilla flags leakage but is imperfect. Verify the "
        "logical lifetime experimentally. Results preserve every round's digital "
        "ancilla outcome and the same shot's final qutrit assignment."
    )

    @model_validator(mode="after")
    def _validate_control_and_record_limits(self) -> PublicErasureQutritSpec:
        if self.max_cycle_time_us < self.min_cycle_time_us:
            raise ValueError("max_cycle_time_us cannot be below min_cycle_time_us")
        if self.max_shots > self.budget.max_final_assignments_per_call:
            raise ValueError("max_shots cannot exceed budget.max_final_assignments_per_call")
        from qiqcbench.qsim.qtypes.transmon_erasure_qutrit.model import cadence_grid_us

        cadence_grid_us(
            minimum_us=self.min_cycle_time_us,
            maximum_us=self.max_cycle_time_us,
            resolution_us=self.cycle_time_resolution_us,
        )
        return self


# ---------- Hidden truth (only inside qsim) ----------


class StaleErasureNotebookEntry(_Strict):
    """Optional stale/optimistic calibration written into the lab notebook."""

    last_calibrated: str | None = None
    logical_lifetime_us: float | None = None
    t1_ge_us: float | None = None
    t1_ef_us: float | None = None
    recommended_cycle_time_us: float | None = None
    false_negative_rate: float | None = None
    expected_per_cycle_error: float | None = None
    readout_fidelity_claim: float | None = None
    notes: str = ""


class HiddenErasureQutritConfig(_Strict):
    """Hidden truth for the g-f erasure-qutrit qtype. Lives only inside qsim.

    The post-selected code-space bit-flip lifetime (the task answer) is NOT
    stored here as a constant: it *emerges* from the cascaded-decay + imperfect
    detection rate model the engine simulates.
    """

    schema_version: int = 2
    device_id: str
    qtype: Literal["transmon_erasure_qutrit"] = "transmon_erasure_qutrit"
    seed: int

    # Cascaded relaxation (transmon qutrit).
    t1_ge_us: float = Field(..., gt=0, description="e->g decay time.")
    t1_ef_us: float = Field(..., gt=0, description="f->e (leakage) decay time; ~T1_ge/2.")
    t_phi_us: float = Field(..., gt=0, description="Logical pure-dephasing time (X basis).")
    thermal_pop: float = Field(0.007, ge=0, le=0.2)

    # Imperfect mid-circuit erasure detection.
    false_positive_rate: float = Field(..., ge=0, le=1)
    false_negative_rate: float = Field(..., ge=0, le=1)
    dd_pulse_error: float = Field(
        ..., ge=0, le=1, description="Per-cycle logical bit-flip from imperfect XY4 pulses."
    )
    fn_seepage_coeff: float = Field(
        0.21,
        ge=0,
        le=1,
        description="Fraction of missed leaks that seep to |g> undetected (residual bit-flip).",
    )
    nodd_dephasing_t_phi_us: float = Field(
        40.0, gt=0, description="Fast dephasing time when DD is disabled (X basis)."
    )
    readout_backaction_error_at_reference: float = Field(
        0.0,
        ge=0,
        le=0.5,
        description=(
            "Per-cycle symmetric logical-flip probability from residual readout "
            "photons at readout_backaction_reference_cycle_us."
        ),
    )
    readout_backaction_reference_cycle_us: float = Field(3.5, gt=0)
    readout_backaction_decay_time_us: float = Field(
        0.8,
        gt=0,
        description="Exponential ring-down scale for short-cycle readout backaction.",
    )

    # 3-state readout confusion: confusion[measured][true], columns sum to 1.
    readout_confusion: list[list[float]] = Field(
        ...,
        description="3x3 matrix P(measure i | prepared j) over (g, e, f); columns sum to 1.",
    )

    stale_lab_notebook: StaleErasureNotebookEntry = StaleErasureNotebookEntry()


__all__ = [
    "ErasureQutritBudget",
    "ErasureQutritMeasurementModel",
    "HiddenErasureQutritConfig",
    "PublicErasureQutritSpec",
    "StaleErasureNotebookEntry",
]
