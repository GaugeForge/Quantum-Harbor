"""Public + hidden device config for the bosonic_cavity_qec qtype.

A storage cavity (bosonic mode, Fock cutoff ``n_max``) dispersively coupled to a
transmon ancilla, with SNAP/displacement universal control, a dispersive
photon-number-parity measurement, and mid-circuit measurement + feedback. The
PUBLIC spec advertises control ranges + the *exact ideal unitaries* (so the
encoder/recovery are an offline construction); the true cavity loss rate, ancilla
decoherence, real dispersive coupling, gate infidelities, readout confusion, and
the stale notebook live in the HIDDEN config inside qsim.

Public/hidden are STRUCTURALLY separated: both classes use
``extra="forbid"`` and share no fields.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Public spec (visible to agent) ----------


class BosonicCavityResourceBudget(_Strict):
    """Public task-scoped qsim resource limits; none is a score term."""

    max_ingress_request_bytes: int = Field(1_048_576, gt=0)
    max_metadata_calls: int = Field(256, gt=0)
    max_job_result_polls: int = Field(32_768, gt=0)
    max_final_answer_serialized_bytes: int = Field(8_192, gt=0)
    max_final_answer_submissions: int = Field(1, gt=0)
    max_answer_string_characters: int = Field(2_048, gt=0)
    max_execution_live_branches_per_point: int = Field(256, gt=0)
    max_execution_weighted_branch_ops_per_call: int = Field(1_024, gt=0)
    max_execution_weighted_branch_ops_per_run: int = Field(65_536, gt=0)
    max_execution_distinct_variable_durations_per_call: int = Field(16, gt=0)


class PublicBosonicCavitySpec(_Strict):
    """Hardware-manual-style public spec for the bosonic_cavity_qec qtype."""

    # Schema 2 is the frozen binomial device shape. Schema 3 appends the
    # common-surface resource budget required by GKP recovery without silently
    # changing schema 2.
    schema_version: Literal[2, 3] = 2
    device_id: str
    qtype: Literal["bosonic_cavity_qec"] = "bosonic_cavity_qec"

    cavity_name: str = "storage"
    ancilla_name: str = "ancilla"

    n_max: int = Field(14, ge=6, le=64)

    # Advertised (range-only) physical parameters; true values are hidden.
    tau_s_us_range: tuple[float, float] = (80.0, 220.0)
    ancilla_t1_us_range: tuple[float, float] = (20.0, 45.0)
    ancilla_t2_us_range: tuple[float, float] = (60.0, 160.0)
    nominal_chi_mhz: float = 2.0

    # Control envelope.
    allowed_ops: list[str] = Field(
        default_factory=lambda: [
            "displace",
            "snap",
            "ancilla_rotate",
            "dispersive_wait",
            "ancilla_measure",
            "photon_number_measure",
            "idle",
            "conditional",
        ]
    )
    max_alpha: float = 3.0
    max_snap_len: int = 14
    max_duration_us: float = 200.0
    max_ops_per_program: int = 512
    max_shots: int = 100_000
    max_sweep_points_per_call: int = Field(64, ge=1)
    max_ancilla_measurements_per_call: int = Field(16, ge=1)
    max_recorded_outcomes_per_call: int = Field(2_000_000, ge=1)
    max_experiment_calls: int = Field(10_000, ge=1)
    max_outstanding_jobs: int = Field(8, ge=1)
    max_recorded_outcomes_per_run: int = Field(100_000_000, ge=1)
    max_retained_result_bytes_per_run: int = Field(536_870_912, ge=1)
    budget: BosonicCavityResourceBudget | None = Field(
        None,
        exclude_if=lambda value: value is None,
    )

    # Public nominal op durations (ns). Task resource accounting may bind these
    # exact published values; hidden durations remain the physical replay truth.
    nominal_displace_duration_ns: float = Field(20.0, gt=0, allow_inf_nan=False)
    nominal_snap_duration_ns: float = Field(200.0, gt=0, allow_inf_nan=False)
    nominal_ancilla_rotate_duration_ns: float = Field(20.0, gt=0, allow_inf_nan=False)
    nominal_ancilla_measure_duration_ns: float = Field(400.0, gt=0, allow_inf_nan=False)

    measurement_return: Literal["bosonic_program"] = "bosonic_program"

    notes: str = (
        "Single storage cavity (bosonic mode, Fock cutoff n_max) + a transmon "
        "ancilla (g/e). IDEAL control unitaries, in the control frame:\n"
        "  displace(alpha):       D(alpha)=exp(alpha a_dag - alpha* a) on the cavity.\n"
        "  snap(thetas):          diag(exp(i*theta_n)), theta_n on Fock level n.\n"
        "  ancilla_rotate(t,phi): R=exp(-i*t/2*(cos phi*X + sin phi*Y)) on the ancilla.\n"
        "  dispersive_wait(T):    exp(i*2pi*chi*T*n) on the |e> branch (chi in MHz, "
        "T in us), on top of the storage mode's free evolution over T; the parity "
        "wait T=1/(2*chi) gives (-1)^n.\n"
        "  ancilla_measure:       projective ancilla readout, records a syndrome bit; "
        "optional reset to |g>. With record=false the same projection/reset happens "
        "but no bit is returned, the op cannot drive a conditional, and it does not "
        "multiply the per-point execution cost.\n"
        "  photon_number_measure: photon-number-resolved cavity readout (records n).\n"
        "  idle(T):               free evolution of the storage mode for T us, in the "
        "control frame (ancilla untouched).\n"
        "  conditional(on_index,value,op): apply a non-measurement op iff recorded syndrome "
        "#on_index == value; conditional measurements are not executable with the current "
        "fixed-column result.\n"
        "Worked example -- prepare Fock |2> from |0> (one route): displace, snap, "
        "displace tuned so the residual amplitude lands the population on |2> (a 2-3 "
        "round D-SNAP-D sequence); verify with photon_number_measure. SNAP+displacement "
        "is universal, so any codeword / recovery is reachable by such a sequence.\n"
        "Retained-result capacity is conservatively charged as 96 bytes per "
        "point-shot row plus 40 bytes per recorded scalar outcome.\n"
        "Calibrations may be stale; the notebook lifetime and chi predate the current "
        "cooldown."
    )

    @model_validator(mode="after")
    def _validate_versioned_budget(self) -> PublicBosonicCavitySpec:
        if self.schema_version == 2 and self.budget is not None:
            raise ValueError("bosonic public schema 2 forbids budget")
        if self.schema_version == 3 and self.budget is None:
            raise ValueError("bosonic public schema 3 requires budget")
        return self


# ---------- Hidden truth (only inside qsim) ----------


class StaleBosonicNotebookEntry(_Strict):
    """Optional stale/optimistic calibration written into the lab notebook."""

    last_calibrated: str | None = None
    storage_lifetime_us: float | None = None
    process_fidelity: float | None = None
    lifetime_gain: str | None = None
    recommended_qec: str | None = None
    nominal_chi_mhz: float | None = None
    notes: str = ""


class HiddenBosonicCavityConfig(_Strict):
    """Hidden truth for the bosonic_cavity_qec qtype. Lives only inside qsim.

    The process-fidelity advantage / optimal cadence (the task answer) is NOT
    stored here as a constant: it *emerges* from this open-system model.

    **This is the simulator's noise/error level.** Every field below is a noise
    knob the engine reads directly; the shipped device YAML sets decent non-zero
    defaults. To change the noise strength, edit
    ``configs/devices/binomial_cavity_v0.hidden.example.yaml`` (which carries a
    per-knob tuning guide) -- larger ``*_t1_us``/``*_tphi_us`` and smaller
    ``readout_*`` / ``*_duration_ns`` / ``gate_depolarizing`` mean LESS noise.
    """

    schema_version: int = 1
    device_id: str
    qtype: Literal["bosonic_cavity_qec"] = "bosonic_cavity_qec"
    seed: int

    n_max: int = Field(14, ge=6, le=64)

    # Cavity (the storage mode). cavity_t1_us is the dominant noise knob (the
    # global fidelity ceiling). Keep cavity_tphi_us LARGE: cavity dephasing hits
    # the |0>-|4> code coherence ~16x (it scales as (delta n)^2) and can sink the
    # binomial code on its own. (Thermal heating n_bar is not modeled in v1.)
    cavity_t1_us: float = Field(
        ..., gt=0, description="Single-photon loss time tau_s = 1/kappa (storage loss)."
    )
    cavity_tphi_us: float = Field(
        1500.0, gt=0, description="Cavity pure dephasing time; keep LARGE (sub-dominant)."
    )

    # Idle-time random-displacement diffusion (common-mode drift noise): variance
    # of an isotropic Gaussian phase-space displacement accrued per microsecond of
    # `idle` (alpha units squared, per quadrature). 0 disables the channel (the
    # binomial device default); the GKP recovery device uses a small non-zero rate.
    idle_displacement_variance_per_us: float = Field(0.0, ge=0.0)

    # The storage mode's own dynamics in the control frame (frame edition,
    # 2026-08-26): Fock level n sits at E_n = delta*n + K/2*n(n-1) (kHz) during
    # `idle` and `dispersive_wait`. Nothing public states either value; the agent
    # must discover and calibrate them (number-resolved Ramsey) and undo them with
    # a SNAP. Both 0 = the frame is locked to the cavity (every pre-frame device).
    cavity_detuning_khz: float = Field(
        0.0, description="Detuning of the storage mode from the control frame, kHz (any sign)."
    )
    cavity_self_kerr_khz: float = Field(
        0.0, description="Self-Kerr K, kHz: level n shifts by K/2*n(n-1) (invisible on {0,1})."
    )

    # Ancilla (sets the per-QEC-round cost). LOWER t1/tphi -> noisier QEC rounds.
    ancilla_t1_us: float = Field(..., gt=0)
    ancilla_tphi_us: float = Field(..., gt=0)

    # Real dispersive coupling (chi/2pi in MHz); differs from the nominal value
    # the agent sees, so the parity wait must be re-calibrated.
    chi_mhz: float = Field(..., gt=0)

    # Op durations (ns) -> finite-duration decoherence (HIGHER -> more noise/op).
    displace_duration_ns: float = 20.0
    snap_duration_ns: float = 200.0
    ancilla_rotate_duration_ns: float = 20.0
    ancilla_measure_duration_ns: float = 400.0

    # Extra per-ancilla-gate dephasing (HIGHER -> more noise).
    gate_depolarizing: float = Field(0.002, ge=0, le=0.1)

    # Ancilla parity-readout misassignment (asymmetric): p(measure e | g),
    # p(measure g | e). HIGHER -> more syndrome errors.
    readout_p_e_given_g: float = Field(0.01, ge=0, le=0.5)
    readout_p_g_given_e: float = Field(0.02, ge=0, le=0.5)
    # Photon-number readout: per-level misassignment to a neighbour (tomography).
    number_readout_error: float = Field(0.03, ge=0, le=0.5)

    stale_lab_notebook: StaleBosonicNotebookEntry = StaleBosonicNotebookEntry()


__all__ = [
    "BosonicCavityResourceBudget",
    "HiddenBosonicCavityConfig",
    "PublicBosonicCavitySpec",
    "StaleBosonicNotebookEntry",
]
