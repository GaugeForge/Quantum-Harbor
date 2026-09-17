"""surface_code_memory control + result wire schemas (qtype-local).

One experiment primitive on the async job model: ``run_memory_experiment``. The device runs
``rounds`` rounds of syndrome extraction over ``shots`` shots under the hidden,
spacetime-correlated noise model and returns **raw per-shot detection events** (a packed bit
array) plus the final logical outcome per shot (for the agent's own self-scoring). It never
returns the noise model, the decoded result, or the logical error rate.

The shot budget accumulates across every call in the run and is echoed back in each result.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from qiqcbench.qsim.core.wire import SCHEMA_VERSION, _Strict


class MemoryExperimentRequest(_Strict):
    schema_version: int = SCHEMA_VERSION
    rounds: int = Field(..., ge=1, le=64)
    shots: int = Field(..., ge=1, le=200_000)
    # logical_input is maintainer-controlled in scoring; the agent may set it for its own
    # self-scoring runs (the device prepares logical |0> and measures Z̄ memory).
    logical_input: Literal[0] = 0


class _BudgetView(_Strict):
    shots_used: int
    shots_cap: int


class JobDetectorData(_Strict):
    kind: Literal["surface_code_detectors"] = "surface_code_detectors"
    rounds: int
    n_detectors: int
    shots: int
    detector_index_formula: Literal["check * rounds + t"] = "check * rounds + t"
    # Raw per-shot detection events, row-major [shots][n_detectors], packed as a hex string
    # of the little-endian bit array per shot joined by ';' — compact and lossless.
    detection_events_b64: str
    # Final logical outcome per shot (0/1), packed the same way (length = shots bits).
    logical_outcomes_b64: str
    budget: _BudgetView


# ---------- syndrome-feedback control ----------


class SyndromeFeedbackController(_Strict):
    """A canonical non-programmable two-mode leaky-integral controller.

    The grammar has no decoder, correction, source-code, future-feature, or
    episode-index field. Firmware alone applies clipping, quantization, slew
    limiting and the public one-epoch actuator latency.
    """

    baseline_detector_rates: list[float] = Field(..., min_length=2, max_length=2)
    gain_matrix: list[list[float]] = Field(..., min_length=2, max_length=2)
    leak: float = Field(..., ge=0.0, le=0.98)
    deadband: float = Field(..., ge=0.0, le=0.03)
    initial_trim: list[int] = Field(..., min_length=2, max_length=2)


class SyndromeControlProbeRequest(_Strict):
    schema_version: int = SCHEMA_VERSION
    constant_trim: list[int] = Field(..., min_length=2, max_length=2)
    trajectories: int = Field(..., ge=1, le=1_000)


class SyndromeFeedbackEpisodeRequest(_Strict):
    schema_version: int = SCHEMA_VERSION
    controller: SyndromeFeedbackController
    episodes: int = Field(..., ge=1, le=1_000)


class _CycleBudgetView(_Strict):
    episodes_used: int
    episodes_cap: int
    qec_cycles_used: int
    qec_cycles_cap: int


class JobSyndromeControlProbeData(_Strict):
    kind: Literal["syndrome_control_probe"] = "syndrome_control_probe"
    constant_trim: list[int]
    episodes: int
    epochs_per_episode: int
    control_epoch_cycles: int
    detector_families: list[str]
    # Packed, raw per-cycle detector events. Row-major episode/epoch/cycle/check.
    detector_events_b64: str
    detector_shape: list[int]
    family_rate_mean: list[float]
    # One fixed-decoder terminal logical outcome per episode, packed losslessly.
    terminal_logical_bits_b64: str
    terminal_logical_shape: list[int]
    terminal_logical_failures: int
    terminal_logical_trials: int
    budget: _CycleBudgetView


class JobSyndromeFeedbackEpisodeData(_Strict):
    kind: Literal["syndrome_feedback_episode"] = "syndrome_feedback_episode"
    controller_digest: str
    episodes: int
    epochs_per_episode: int
    control_epoch_cycles: int
    detector_families: list[str]
    detector_events_b64: str
    detector_shape: list[int]
    # One quantized command per episode/epoch, after firmware clipping/slew.
    command_history: list[list[list[int]]]
    family_rate_mean: list[float]
    saturation_fraction: float
    terminal_logical_bits_b64: str
    terminal_logical_shape: list[int]
    terminal_logical_failures: int
    terminal_logical_trials: int
    budget: _CycleBudgetView


class SyndromeControllerProgramRequest(_Strict):
    """Run the source bundle currently staged at the public submission mount."""

    schema_version: int = SCHEMA_VERSION
    controller_manifest_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    trajectories: int = Field(..., ge=1, le=1_000)


class _ControllerBudgetView(_Strict):
    trajectories_used: int = Field(..., ge=0)
    trajectories_cap: int = Field(..., ge=1)
    qec_cycles_used: int = Field(..., ge=0)
    qec_cycles_cap: int = Field(..., ge=1)


class JobSyndromeControlProbeDataV3(_Strict):
    """Lossless raw constant-trim evidence; no fitted or aggregated rates."""

    kind: Literal["syndrome_control_probe_v3"] = "syndrome_control_probe_v3"
    control_data_schema_version: Literal[3] = 3
    constant_trim: list[int] = Field(..., min_length=2, max_length=2)
    trajectories: int = Field(..., ge=1)
    epochs_per_trajectory: int = Field(..., ge=1)
    control_epoch_cycles: int = Field(..., ge=1)
    detector_events_b64: str
    detector_shape: list[int]
    decoded_logical_bits_b64: str
    decoded_logical_shape: list[int]
    applied_trim_history: list[list[list[int]]]
    budget: _ControllerBudgetView


class JobSyndromeControllerProgramData(_Strict):
    """Lossless raw evidence from one causal controller-program batch."""

    kind: Literal["syndrome_controller_program"] = "syndrome_controller_program"
    control_data_schema_version: Literal[3] = 3
    controller_manifest_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    controller_abi_version: Literal["syndrome-controller-jsonl-v1"] = "syndrome-controller-jsonl-v1"
    trajectories: int = Field(..., ge=1)
    epochs_per_trajectory: int = Field(..., ge=1)
    control_epoch_cycles: int = Field(..., ge=1)
    detector_events_b64: str
    detector_shape: list[int]
    decoded_logical_bits_b64: str
    decoded_logical_shape: list[int]
    requested_trim_history: list[list[list[float]]]
    applied_trim_history: list[list[list[int]]]
    actuator_limited_bits_b64: str
    actuator_limited_shape: list[int]
    budget: _ControllerBudgetView


# ---------- drift-recalibration scheduling ----------


class DriftRiskModel(_Strict):
    """A fixed-decoder DFR-to-LER risk model fitted from raw observations.

    Schema 2 carries the submitter's own 1-sigma on the predicted curve. The
    accuracy check accepts a curve within a multiple of the uncertainty the
    CITED EVIDENCE supports, so the submitter has to know that uncertainty to
    know its own acceptance window; the reported value is bound against the
    verifier's recomputation and never widens the window itself.
    """

    risk_model_schema_version: Literal[2] = 2
    model_type: Literal["dfr_power_law"] = "dfr_power_law"
    log10_ler_per_cycle_intercept_estimate: float = Field(..., ge=-8.0, le=2.0)
    dfr_exponent_estimate: float = Field(..., ge=0.5, le=4.0)
    # Largest 1-sigma of the predicted log10 LER per cycle over the public
    # validation DFR interval, as the submitter's own evidence supports it.
    log10_ler_prediction_1sigma_estimate: float = Field(..., gt=0.0, le=10.0)


class DriftPolicy(_Strict):
    """A canonical non-programmable maintenance policy.

    The grammar has no decoder, correction, source-code, or time-indexed
    lookup-table field: a policy must generalize to fresh drift realizations.
    Firmware alone applies the public cooldown, warm-up-hold, and
    emergency-override semantics.
    """

    policy_schema_version: Literal[2] = 2
    trigger_mode: Literal["scheduled", "reactive_dfr", "hybrid"]
    dfr_buffer_windows: int = Field(..., ge=1, le=64)
    trigger_ler_per_cycle: float = Field(..., gt=0.0, lt=1.0e-2)
    schedule_period_windows: int | None = Field(None, ge=8, le=100_000)
    cooldown_windows: int = Field(..., ge=0, le=4_000)
    post_recalibration_hold_windows: int = Field(..., ge=0, le=4_000)
    trigger_action: Literal["recalibrate_operating_tile", "relocate_then_recalibrate_vacated_tile"]


class DriftStreamRequest(_Strict):
    schema_version: int = SCHEMA_VERSION
    windows: int = Field(..., ge=1, le=1_000)


class DriftControlRequest(_Strict):
    schema_version: int = SCHEMA_VERSION
    action: Literal["recalibrate_operating_tile", "recalibrate_idle_tile", "relocate"]


class _DriftBudgetView(_Strict):
    characterization_stream_windows_used_raw: int = Field(..., ge=0)
    characterization_stream_windows_cap: int = Field(..., ge=1)
    characterization_recalibrations_used_raw: int = Field(..., ge=0)
    characterization_recalibrations_cap: int = Field(..., ge=1)

    @model_validator(mode="after")
    def _usage_does_not_exceed_caps(self):
        if self.characterization_stream_windows_used_raw > self.characterization_stream_windows_cap:
            raise ValueError("characterization stream usage exceeds its cap")
        if self.characterization_recalibrations_used_raw > self.characterization_recalibrations_cap:
            raise ValueError("characterization recalibration usage exceeds its cap")
        return self


class JobDriftStreamData(_Strict):
    kind: Literal["drift_stream"] = "drift_stream"
    drift_data_schema_version: Literal[2] = 2
    window_cycles: int = Field(..., ge=1)
    n_checks: int = Field(..., ge=1)
    windows: int = Field(..., ge=1)
    start_window: int = Field(..., ge=0)  # first returned device-time window
    operating_tile: Literal[0, 1]
    # Raw per-cycle detector events, row-major [window][cycle][check], packed.
    # Windows where the memory was down (recalibration/relocation in progress)
    # carry zero bits and are flagged in down_mask.
    detector_events_b64: str | None = None
    detector_shape: list[int]
    down_mask_b64: str | None = None  # one bit per window: 1 = memory down
    # Fixed firmware decoder: one logical-failure bit per operating window.
    logical_failure_bits_b64: str | None = None
    # Large raw batches are delivered through this poll-gated lossless artifact.
    raw_data_file: str | None = None
    raw_data_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    budget: _DriftBudgetView

    @model_validator(mode="after")
    def _raw_payload_is_inline_or_artifact(self):
        inline = all(
            value is not None
            for value in (
                self.detector_events_b64,
                self.down_mask_b64,
                self.logical_failure_bits_b64,
            )
        )
        artifact = self.raw_data_file is not None and self.raw_data_sha256 is not None
        if inline == artifact:
            raise ValueError("drift raw data must be exactly one of inline or artifact-backed")
        if self.detector_shape != [self.windows, self.window_cycles, self.n_checks]:
            raise ValueError("detector_shape must equal [windows, window_cycles, n_checks]")
        return self


class JobDriftControlData(_Strict):
    kind: Literal["drift_event"] = "drift_event"
    drift_data_schema_version: Literal[2] = 2
    action: Literal["recalibrate_operating_tile", "recalibrate_idle_tile", "relocate"]
    start_window: int = Field(..., ge=0)
    operating_tile_before_action: Literal[0, 1]
    operating_tile_after_action: Literal[0, 1]
    operating_downtime_starts_at_window: int | None = Field(None, ge=0)
    operating_downtime_completes_at_window: int | None = Field(None, ge=0)
    relocation_completes_at_window: int | None = Field(None, ge=0)
    recalibrated_tile: Literal[0, 1]
    recalibration_starts_at_window: int = Field(..., ge=0)
    recalibration_completes_at_window: int = Field(..., ge=0)
    budget: _DriftBudgetView

    @model_validator(mode="after")
    def _timing_shape_matches_action(self):
        if self.recalibration_starts_at_window < self.start_window:
            raise ValueError("recalibration cannot start before the control action")
        if self.recalibration_completes_at_window <= self.recalibration_starts_at_window:
            raise ValueError("recalibration completion must follow its start")
        if self.action == "recalibrate_idle_tile":
            if (
                self.operating_tile_after_action != self.operating_tile_before_action
                or self.recalibrated_tile == self.operating_tile_before_action
                or self.operating_downtime_starts_at_window is not None
                or self.operating_downtime_completes_at_window is not None
                or self.relocation_completes_at_window is not None
            ):
                raise ValueError("idle-tile recalibration timing is inconsistent")
        elif self.action == "recalibrate_operating_tile":
            if (
                self.operating_tile_after_action != self.operating_tile_before_action
                or self.recalibrated_tile != self.operating_tile_before_action
                or self.operating_downtime_starts_at_window != self.start_window
                or self.operating_downtime_completes_at_window
                != self.recalibration_completes_at_window
                or self.relocation_completes_at_window is not None
            ):
                raise ValueError("operating-tile recalibration timing is inconsistent")
        elif (
            self.operating_tile_after_action == self.operating_tile_before_action
            or self.recalibrated_tile != self.operating_tile_before_action
            or self.operating_downtime_starts_at_window != self.start_window
            or self.relocation_completes_at_window != self.operating_downtime_completes_at_window
            or self.recalibration_starts_at_window != self.relocation_completes_at_window
        ):
            raise ValueError("relocation timing is inconsistent")
        return self


# ---------- heralded-leakage decoding ----------


class LeakageMemoryRequest(_Strict):
    schema_version: int = SCHEMA_VERSION
    rounds: int = Field(..., ge=2, le=24)
    shots: int = Field(..., ge=1, le=20_000)

    @field_validator("rounds")
    @classmethod
    def _even_rounds(cls, v: int) -> int:
        if v % 2 != 0:
            raise ValueError("leakage experiments require an even round count (LRU period 2)")
        return v


class ChallengeBatchRequest(_Strict):
    schema_version: int = SCHEMA_VERSION
    chunk_index: int = Field(..., ge=0)


class JobLeakageMemoryData(_Strict):
    kind: Literal["leakage_memory"] = "leakage_memory"
    rounds: int
    n_detectors: int
    shots: int
    detector_index_formula: Literal["check * rounds + t"] = "check * rounds + t"
    # One herald bit per data qubit per LRU event, row-major [shots][n_data * n_slots],
    # herald_id = data_qubit * (rounds // 2) + (lru_round // 2).
    herald_index_formula: Literal["data_qubit * (rounds // 2) + (lru_round // 2)"] = (
        "data_qubit * (rounds // 2) + (lru_round // 2)"
    )
    n_herald_slots: int
    detection_events_b64: str
    herald_events_b64: str
    # Final logical outcome per shot (calibration truth; the device prepares logical |0>).
    logical_outcomes_b64: str
    budget: _BudgetView


class JobLeakageChallengeData(_Strict):
    kind: Literal["leakage_challenge_batch"] = "leakage_challenge_batch"
    challenge_set_id: str
    # sha256 over the FULL challenge detector + herald packed bytes (all chunks).
    challenge_digest: str
    chunk_index: int
    n_chunks: int
    chunk_shots: int
    start_shot: int
    rounds: int
    n_detectors: int
    n_herald_slots: int
    detector_index_formula: Literal["check * rounds + t"] = "check * rounds + t"
    herald_index_formula: Literal["data_qubit * (rounds // 2) + (lru_round // 2)"] = (
        "data_qubit * (rounds // 2) + (lru_round // 2)"
    )
    detection_events_b64: str
    herald_events_b64: str
    # No logical outcomes: the challenge truth is withheld and scored by the verifier.


__all__ = [
    "MemoryExperimentRequest",
    "JobDetectorData",
    "LeakageMemoryRequest",
    "ChallengeBatchRequest",
    "JobLeakageMemoryData",
    "JobLeakageChallengeData",
    "SyndromeFeedbackController",
    "SyndromeControlProbeRequest",
    "SyndromeFeedbackEpisodeRequest",
    "JobSyndromeControlProbeData",
    "JobSyndromeFeedbackEpisodeData",
    "SyndromeControllerProgramRequest",
    "JobSyndromeControlProbeDataV3",
    "JobSyndromeControllerProgramData",
    "DriftRiskModel",
    "DriftPolicy",
    "DriftStreamRequest",
    "DriftControlRequest",
    "JobDriftStreamData",
    "JobDriftControlData",
    "_BudgetView",
]
