"""Public + hidden device config for the ``surface_code_memory`` qtype.

The experiment interface and code geometry are **public and authoritative**; the noise
model is **hidden** (architecture invariant 2: the split is structural — both classes use
``extra="forbid"``). The public spec exposes everything the agent needs to reconstruct the
matching graph (check supports, spatial edges, hook-candidate orientations, the logical
representative, detector indexing) but never the hidden rates or the hidden leakage topology.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------- public ----------------


class SpatialEdgeSpec(_Strict):
    """One data-qubit X-error edge of the matching graph (public topology).

    ``checks`` is ``[s1, s2]`` for a bulk edge or ``[s1]`` for a boundary edge.
    """

    data_qubit: int
    checks: list[int] = Field(..., min_length=1, max_length=2)
    flips_observable: bool


class HookCandidateSpec(_Strict):
    """A schedule-oriented hook-candidate orientation (public). Its RATE is hidden."""

    checks: list[int] = Field(..., min_length=2, max_length=2)  # (s1 @ t, s2 @ t+1)
    dt: Literal[1] = 1
    flips_observable: bool


class ScBudgets(_Strict):
    shot_budget: int = Field(..., ge=1)  # total shots across all run_memory_experiment calls
    max_rounds: int = Field(64, ge=1)
    max_shots_per_call: int = Field(200_000, ge=1)


class SyndromeFeedbackSpec(_Strict):
    """Public v3 contract for causal syndrome-feedback controller programs.

    This is deliberately an *instrument* description, rather than a noise
    description.  It tells an experimenter how the controller is clocked and
    constrained without exposing the drift transition or actuator response.
    """

    contract_version: Literal[3] = 3
    controller_abi_version: Literal["syndrome-controller-jsonl-v1"] = "syndrome-controller-jsonl-v1"
    controller_entrypoint: Literal["controller/run.py"] = "controller/run.py"
    control_epoch_cycles: int = Field(..., ge=1)
    experiment_epochs_per_trajectory: int = Field(..., ge=2)
    deployment_epochs_per_trajectory: int = Field(..., ge=4)
    deployment_warmup_epochs: int = Field(..., ge=0)
    deployment_trajectories: int = Field(..., ge=8)
    deployment_panels: int = Field(..., ge=2)
    detector_families: list[str] = Field(..., min_length=2, max_length=2)
    trim_min: int = Field(..., le=0)
    trim_max: int = Field(..., ge=0)
    trim_slew_per_epoch: int = Field(..., ge=1)
    controller_latency_epochs: Literal[1] = 1
    initial_applied_trim: list[int] = Field(..., min_length=2, max_length=2)
    experiment_trajectory_budget: int = Field(..., ge=1)
    max_trajectories_per_call: int = Field(..., ge=1)
    max_controller_bundle_files: int = Field(..., ge=1)
    max_controller_bundle_bytes: int = Field(..., ge=1)
    max_controller_response_bytes: int = Field(..., ge=128)
    controller_step_timeout_s: float = Field(..., gt=0.0)
    controller_trajectory_timeout_s: float = Field(..., gt=0.0)
    max_requested_trim_abs: float = Field(..., gt=0.0)
    confidence_level: float = Field(..., gt=0.5, lt=1.0)
    max_decoded_logical_error_per_cycle_upper_bound: float = Field(..., gt=0.0, lt=1.0)
    max_actuator_limit_fraction: float = Field(..., ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _feedback_contract_is_coherent(self):
        if self.deployment_warmup_epochs >= self.deployment_epochs_per_trajectory:
            raise ValueError("deployment_warmup_epochs must be shorter than the trajectory")
        if self.deployment_trajectories % self.deployment_panels:
            raise ValueError("deployment_trajectories must divide evenly into panels")
        if self.max_trajectories_per_call > self.experiment_trajectory_budget:
            raise ValueError("max_trajectories_per_call exceeds the experiment budget")
        if self.controller_step_timeout_s >= self.controller_trajectory_timeout_s:
            raise ValueError("controller step timeout must be below the trajectory timeout")
        if any(
            type(value) is not int or value < self.trim_min or value > self.trim_max
            for value in self.initial_applied_trim
        ):
            raise ValueError("initial_applied_trim lies outside the public trim range")
        return self


class LegacyDriftReloqationSpec(_Strict):
    """Frozen v1 contract retained so archived device material still loads."""

    contract_version: Literal[1] = 1
    window_cycles: int = Field(..., ge=16)
    n_tiles: Literal[2] = 2
    stream_window_budget: int = Field(..., ge=1)
    max_windows_per_call: int = Field(..., ge=1)
    recalibration_budget: int = Field(..., ge=1)
    recalibration_downtime_windows: int = Field(..., ge=1)
    relocation_downtime_windows: int = Field(..., ge=1)
    # Warm-up mechanics are public: for warmup_windows after a recalibrated tile
    # returns, its OBSERVED detector statistics carry a decaying transient. The
    # transient's magnitude is hidden and must be measured.
    warmup_windows: int = Field(..., ge=0)
    # Firmware trigger semantics (public, deterministic): during a policy's
    # warm-up hold, a reactive trigger still fires if the predicted LER exceeds
    # emergency_override_factor * breach_threshold_ler.
    emergency_override_factor: float = Field(..., ge=1.0)
    target_logical_error_per_cycle: float = Field(..., gt=0.0, lt=1.0)
    sustained_breach_windows: int = Field(..., ge=1)
    replay_trajectory_windows: int = Field(..., ge=1)
    replay_trajectories: int = Field(..., ge=8)
    replay_pass_min: int = Field(..., ge=1)
    replay_downtime_budget_windows: int = Field(..., ge=1)
    baseline_margin: float = Field(..., gt=0.0, lt=1.0)
    min_stream_windows_for_submission: int = Field(..., ge=1)


class DriftReloqationSpec(_Strict):
    """Public v3 contract for drift characterization and maintenance control.

    Characterization and deployment budgets are separate by construction. All
    action timing, risk-validation rules, and deployment SLAs are public; only
    the stochastic drift process and sensor-transient magnitude remain hidden.

    Version 3 replaces the fixed absolute risk-accuracy cap with a test scaled
    by the 1-sigma the cited evidence supports. A fixed cap is a different
    number of sigma on every instance, so it rejected honest work far more
    often where the drift explored little DFR range; the scaled form makes the
    honest rejection probability the same everywhere and lets more evidence
    always buy a tighter claim.
    """

    contract_version: Literal[3] = 3
    window_cycles: int = Field(..., ge=16)
    n_tiles: Literal[2] = 2
    characterization_stream_window_budget: int = Field(..., ge=1)
    max_windows_per_call: int = Field(..., ge=1)
    characterization_recalibration_budget: int = Field(..., ge=1)
    recalibration_downtime_windows: int = Field(..., ge=1)
    relocation_downtime_windows: int = Field(..., ge=1)
    warmup_observation_windows: int = Field(..., ge=0)
    emergency_override_factor: float = Field(..., ge=1.0)
    target_logical_error_per_cycle: float = Field(..., gt=0.0, lt=1.0)
    sustained_breach_operating_windows: int = Field(..., ge=1)
    deployment_trajectory_windows: int = Field(..., ge=1)
    deployment_trajectories: int = Field(..., ge=8)
    deployment_reliability_min_successes: int = Field(..., ge=1)
    deployment_downtime_budget_windows_per_trajectory: int = Field(..., ge=1)
    deployment_recalibration_budget_per_trajectory: int = Field(..., ge=1)
    risk_validation_dfr_min: float = Field(..., gt=0.0, lt=0.5)
    risk_validation_dfr_max: float = Field(..., gt=0.0, lt=0.5)
    # Accuracy is graded in units of the cited evidence's own 1-sigma on the
    # predicted curve, not in fixed log10 units.
    risk_validation_max_sigma: float = Field(..., gt=0.0)
    # A reported prediction 1-sigma must agree with the verifier's
    # recomputation from the cited evidence to within this multiplicative
    # factor, in both directions.
    risk_evidence_reported_1sigma_factor: float = Field(..., gt=1.0)
    risk_evidence_likelihood_ratio_max: float = Field(..., gt=0.0)
    risk_evidence_intercept_1sigma_max: float = Field(..., gt=0.0)
    risk_evidence_exponent_1sigma_max: float = Field(..., gt=0.0)

    @model_validator(mode="after")
    def _validate_contract(self):
        if self.max_windows_per_call > self.characterization_stream_window_budget:
            raise ValueError(
                "max_windows_per_call cannot exceed the characterization stream budget"
            )
        if self.risk_validation_dfr_min >= self.risk_validation_dfr_max:
            raise ValueError("risk_validation_dfr_min must be smaller than its maximum")
        if self.deployment_reliability_min_successes > self.deployment_trajectories:
            raise ValueError(
                "deployment_reliability_min_successes cannot exceed deployment_trajectories"
            )
        if self.sustained_breach_operating_windows > self.deployment_trajectory_windows:
            raise ValueError(
                "sustained breach duration cannot exceed the deployment trajectory horizon"
            )
        if (
            self.deployment_downtime_budget_windows_per_trajectory
            > self.deployment_trajectory_windows
        ):
            raise ValueError("deployment downtime budget cannot exceed the trajectory horizon")
        return self


class HeraldedLeakageSpec(_Strict):
    """Public contract for the heralded-leakage adaptive-decoding instrument.

    Everything structural is public and authoritative: the LRU schedule, the herald
    stream indexing, the challenge-set design, and the published pass ceiling. What a
    herald physically implies (its retro damage window, spatial extent, and reliability
    in both directions) is hidden physics the agent must characterize from calibration
    data.
    """

    lru_period_rounds: Literal[2] = 2
    lru_phase_formula: Literal["(row + col) % 2"] = "(row + col) % 2"
    lru_rule: str = (
        "each data qubit q at grid (row, col) is measured + reset by the leakage "
        "reduction unit at the END of every round t with t % 2 == (row + col) % 2; "
        "one herald bit is returned per data qubit per LRU event"
    )
    herald_index_formula: Literal["data_qubit * (rounds // 2) + (lru_round // 2)"] = (
        "data_qubit * (rounds // 2) + (lru_round // 2)"
    )
    calibration_rounds_min: int = Field(2, ge=2)
    calibration_rounds_max: int = Field(24, le=64)
    max_shots_per_call: int = Field(..., ge=1)
    challenge_shots: int = Field(..., ge=1)
    challenge_rounds: int = Field(..., ge=2)
    challenge_chunk_shots: int = Field(..., ge=1)
    # Published pass line: total prediction failures on the challenge set (missing
    # predictions count as failures) must be <= this ceiling.
    failure_ceiling: int = Field(..., ge=1)
    notes: str = ""


class PublicSurfaceCodeSpec(_Strict):
    """Agent-visible device specification (no hidden truth)."""

    schema_version: int = 1
    device_id: str
    qtype: Literal["surface_code_memory"] = "surface_code_memory"
    task_id: str | None = None

    code: Literal["rotated_surface_code"] = "rotated_surface_code"
    distance: int = 5
    n_physical_qubits: int = 49  # 2 d^2 - 1
    n_data_qubits: int = 25
    decoded_basis: Literal["Z"] = "Z"
    n_checks: int = 12
    default_rounds: int = 11
    # The submitted DEM is ALWAYS scored at this fixed round count (det_id = check*scoring_rounds
    # + t), independent of the rounds the agent chooses for its own characterization
    # experiments. Fixing the replay layout removes a gaming knob: the per-cycle rate drifts with
    # round count (time-boundary effects), so a free replay-rounds choice would otherwise let a
    # short-experiment DEM clear the floor without modeling the correlations.
    scoring_rounds: int = 11

    # Public geometry: check supports (data-qubit ids), the spatial matching graph, the
    # schedule-oriented hook candidates, and the logical representative.
    check_supports: list[list[int]]
    spatial_edges: list[SpatialEdgeSpec]
    hook_candidates: list[HookCandidateSpec]
    logical_representative_data_qubits: list[int]

    # Detector indexing convention (authoritative): det_id = check * rounds + t.
    detector_index_formula: Literal["check * rounds + t"] = "check * rounds + t"

    # Decoder-submission conventions.
    decoder_types: list[str] = Field(default_factory=lambda: ["mwpm", "belief_matching"])

    # Nominal/stale physical scale; the true rates are hidden.
    nominal_physical_error_rate: float = 0.002

    budgets: ScBudgets
    logical_error_rate_floor: float = 4.5e-3
    replay_shots: int = 1_000_000
    # Ordinary decoder-calibration devices leave this absent.  A feedback
    # task is activated only by capability plus this public instrument contract.
    syndrome_feedback: SyndromeFeedbackSpec | None = None
    # A drift-reloqation task is activated only by capability plus this public
    # instrument contract; ordinary devices leave it absent.
    drift_reloqation: LegacyDriftReloqationSpec | DriftReloqationSpec | None = None
    # A heralded-leakage decoding task is activated only by capability plus this
    # public instrument contract; ordinary devices leave it absent.
    heralded_leakage: HeraldedLeakageSpec | None = None
    notes: str = ""


# ---------------- hidden ----------------


class LeakPair(_Strict):
    s1: int
    s2: int
    dt: Literal[1, 2]
    flips_observable: bool


class HiddenNoiseModel(_Strict):
    """The hidden DEM parameters (frozen, explicit)."""

    p_data: float = Field(..., gt=0.0, lt=1.0)  # single data-qubit X error / round
    p_meas: float = Field(..., gt=0.0, lt=1.0)  # measurement (temporal) error / round
    hook_base: float = Field(..., gt=0.0, lt=1.0)
    hook_multipliers: list[float]  # one per public hook_candidate, in order
    p_leak: float = Field(..., gt=0.0, lt=1.0)
    leak_pairs: list[LeakPair]


class HiddenSyndromeFeedbackDrift(_Strict):
    """Private exogenous drift and actuator parameters for feedback tasks.

    The latent environment evolves around randomly switching persistent regime
    centers and is never modified by the controller. ``actuator_response``
    instead maps the applied trim into an instantaneous correction of that
    state. ``detector_response`` defines residual projections whose squared
    magnitudes become detector-family log-odds penalties on the real DEM
    mechanisms. Logical failures are produced by
    the fixed firmware decoder, so no hand-written decoder-cost coefficients
    belong in this schema.
    """

    initial_state_std: float = Field(..., gt=0.0)
    transition: list[list[float]]
    process_noise_std: list[float]
    shock_probability: float = Field(..., ge=0.0, le=1.0)
    shock_std: list[float]
    regime_centers: list[list[float]] = Field(..., min_length=2)
    regime_switch_probability: float = Field(..., gt=0.0, lt=1.0)
    detector_baseline: list[float]
    detector_response: list[list[float]]
    actuator_response: list[list[float]]
    detector_overdispersion: float = Field(..., ge=0.0)


class HiddenDriftReloqation(_Strict):
    """Private drift-process + fixed-decoder parameters for reloqation tasks."""

    dfr_base: float = Field(..., gt=0.0, lt=0.5)
    dfr_base_spread: float = Field(..., ge=0.0, lt=0.5)
    ler_log10_a: float
    ler_exponent_b: float = Field(..., gt=0.0)
    slow_p_median_cycles: float = Field(..., gt=0.0)
    slow_p_sigma: float = Field(..., ge=0.0)
    jump_rate_per_cycle: float = Field(..., ge=0.0)
    jump_mult_min: float = Field(..., ge=1.0)
    jump_mult_exp_mean: float = Field(..., gt=0.0)
    jump_relax_windows: float = Field(..., gt=0.0)
    burst_rate_per_cycle: float = Field(..., ge=0.0)
    burst_mult: float = Field(..., ge=1.0)
    burst_duration_median_windows: float = Field(..., gt=0.0)
    burst_duration_sigma: float = Field(..., ge=0.0)
    warmup_sensor_mult: float = Field(..., ge=1.0)
    # Legacy v0 diagnostics; the v2 scorer does not consume baseline parameters.
    baseline_b1_period_windows: int | None = Field(None, ge=1)
    baseline_b2_threshold_factor: float | None = Field(None, gt=0.0)
    baseline_b2_cooldown_windows: int | None = Field(None, ge=0)


class HiddenHeraldedLeakage(_Strict):
    """Private leakage-dwell process + herald reliability (arXiv:2411.10343 lineage).

    Leakage is injected by the LRU reset itself (``p_leak_per_lru`` per LRU event); a
    dwell covers exactly the two rounds until the next LRU clears + heralds it.
    """

    p_leak_per_lru: float = Field(..., gt=0.0, lt=1.0)
    b_self: float = Field(..., gt=0.0, lt=1.0)
    b_meas: float = Field(..., gt=0.0, lt=1.0)
    b_partner: float = Field(..., gt=0.0, lt=1.0)
    b_decay_second_round: float = Field(..., gt=0.0, le=1.0)
    herald_false_negative: float = Field(..., ge=0.0, lt=1.0)
    herald_false_positive: float = Field(..., ge=0.0, lt=1.0)
    # Structured unreliability: a few chronically false-flagging LDU channels whose
    # false-positive rate is fp_hot_rate instead of the base rate. A herald-reliability
    # model that treats all channels as identical is diluted by these into near-static
    # performance; the per-channel model is the intended full science.
    fp_hot_qubits: list[int] = Field(default_factory=list)
    fp_hot_rate: float = Field(0.0, ge=0.0, lt=1.0)


class StaleScNotebook(_Strict):
    model_config = ConfigDict(extra="forbid")

    noise_model: str = "i.i.d. depolarizing at the nominal ~0.2% per operation"
    dem_advice: str = "use the nominal single-detector rates; no need to estimate from data"
    decoder_advice: str = "default minimum-weight matching with uniform weights is fine"
    correlation_advice: str = "treat detection events as independent"
    budget_advice: str = "spend all shots evaluating the decoder"
    feedback_advice: str = ""
    drift_advice: str = ""
    dfr_to_ler_claim: str = ""
    recalibration_advice: str = ""
    herald_semantics_advice: str = ""
    herald_reliability_advice: str = ""
    note: str = ""
    # Overridable so a device can avoid announcing its own unreliability. The default
    # preserves the historical string for surface_code_d5_v0 byte-for-byte.
    reliability: str = ""


class HiddenSurfaceCodeConfig(_Strict):
    schema_version: int = 1
    device_id: str
    qtype: Literal["surface_code_memory"] = "surface_code_memory"
    seed: int

    distance: int = 5
    noise: HiddenNoiseModel
    syndrome_feedback_drift: HiddenSyndromeFeedbackDrift | None = None
    drift_reloqation: HiddenDriftReloqation | None = None
    heralded_leakage: HiddenHeraldedLeakage | None = None
    stale_lab_notebook: StaleScNotebook = StaleScNotebook()
    # Per-device MCP initialize string, following the StaleScNotebook.reliability
    # override pattern: empty means the qtype default in ``mcp.py`` (which
    # describes the base memory-calibration contract for surface_code_d5_v0);
    # other devices set it so their MCP initialize string matches their own
    # task contract instead of the decoder-calibration one.
    mcp_instructions: str = ""
