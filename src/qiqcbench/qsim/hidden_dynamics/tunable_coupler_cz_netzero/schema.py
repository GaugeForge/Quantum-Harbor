"""Frozen hidden-scorer schema for tunable_coupler_cz_netzero."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CzHiddenScorer(_Strict):
    # schema 2: the scalar ``flux_settle_tau_ns`` /
    # ``tau_settle_rel_tol`` pair became the vector settling truth below, and
    # ``max_programmed_samples`` was added. ``extra="forbid"`` makes any stored
    # schema-1 YAML invalid rather than silently stale — deliberate.
    schema_version: Literal[2] = 2
    device_id: str

    # --- hidden device truth (GHz / ns / Phi_0) ---
    omega_1_max_ghz: float
    omega_2_ghz: float
    omega_c_max_ghz: float
    alpha_1_ghz: float
    alpha_2_ghz: float
    arc_asym: float
    g12_ghz: float
    gg_ghz2: float
    zeta_scale: float
    t1_us: float
    t2_us: float
    # Flux-line short-time settling truth: parallel bank of DISCRETE first-order
    # poles on the 0.4 ns AWG grid (alpha_k = dt/(tau_k+dt), step response
    # s[n] = 1 - sum_k A_k (1-alpha_k)^n), tau ascending.
    flux_settle_amplitudes: tuple[float, ...]
    flux_settle_taus_ns: tuple[float, ...]
    robustness_offset_phi0: float
    # readout_q1/q2 are retained hidden-truth records; ``score_answer`` never
    # reads them, and that is correct design rather than an oversight: gate
    # quality is a property of the gate, and the agent already pays readout
    # error during measurement (the engine applies it).
    readout_q1: tuple[float, float]
    readout_q2: tuple[float, float]
    # derived truth anchors
    j_gate_mhz: float  # |J_eff| at the REFERENCE gate flux (informational; the
    # scorer grades coupling_j against |J_eff| recomputed at the agent's own
    # declared coupler_gate_flux)
    coupler_idle_flux: float  # where J_idle=0 and zeta~0
    coupler_gate_flux: float  # informational record of the reference gate flux
    # (like j_gate_mhz, not consumed by score_answer)
    crossing_flux_q1: float  # informational anchor; not consumed by score_answer

    # --- Stage-A tolerances ---
    alpha_2_rel_tol: float = 0.05
    alpha_2_stale_reject_mhz: float = -300.0
    j_rel_tol: float = 0.25
    # Minimum absolute tolerance on coupling_j. |J_eff| passes through zero on the
    # coupler tuning curve, so a purely relative tolerance is unsatisfiable by ANY
    # reported value at a bias where the truth is near that null -- the gate would
    # be unpassable by construction rather than by measurement. 0.5 MHz is the 1
    # sigma the task's own answer schema asks the agent to quote for this quantity.
    j_abs_tol_mhz: float = 0.5
    j_stale_reject_mhz: float = 7.0
    decoupling_flux_tol: float = 0.03
    # Uniform per-parameter relative band on the reported settling vectors
    # (amplitudes AND taus, compared tau-ascending). Sized from the measured
    # recovery precision of the reference cryoscope + separable-NLS estimator
    # (100 truncation points x 4096 shots, model weight pinned) over 200
    # shot-noise seeds -- per-parameter worst-case errors are all under 6.6%,
    # so +-10% passes every seed with margin while a blind two-pole prior lands
    # all four numbers inside the band on only ~0.1% of draws.
    # A one-pole report fails on shape alone. The residual gate below is a
    # separate standard (graded on the programmed waveform, not the quote).
    settling_rel_tol: float = 0.10
    crossing_flux_tol: float = 0.04

    # --- execution-evidence floor ---
    # Minimum completed run_flux_pulse jobs in the qsim experiment log before any
    # scoring. An anti-gaming backstop against the purely-analytic route (waveform
    # designed offline, device never meaningfully driven), NOT a discriminator --
    # the physics gates do that. Calibration of the value against the observed
    # analytic / weak-real / strong distribution: see
    # construction._MIN_COMPLETED_FLUX_JOBS.
    min_completed_flux_jobs: int = Field(15, ge=1)

    # --- Stage-C thresholds ---
    phi2q_tol_rad: float = 0.05
    leakage_max: float = 1.0e-2
    infidelity_max: float = 1.5e-2
    # Sized from ACHIEVABLE cryoscope precision, not from an oracle-calibrated
    # reference. Re-derived for the two-pole line: the
    # honest reference protocol (100 truncation points x 4096 shots, separable
    # NLS with the model weight pinned) clears this gate on all 200 shot-noise
    # seeds measured, and no predistortion at all still fails it on every legal
    # trajectory measured (the headline trap gives ~1.3e-1 on the reference
    # trajectory).
    #
    # It does NOT independently fail a sloppy settling estimate on the gentlest
    # legal trajectory; a bad quote is rejected by `settling_rel_tol` above, not
    # here. The two settling gates are related but they are not one standard.
    predistortion_residual_max: float = 1.0e-2
    netzero_advantage_min: float = 3.0
    idle_zz_max_khz: float = 10.0
    netzero_integral_max_phi0_ns: float = 0.05
    gate_window_ns: tuple[float, float] = (36.0, 100.0)
    # Hard cap on the replayed waveform length. Redundant
    # defence in depth: with len(programmed) == len(intended) enforced, the
    # public gate window already bounds a legal on-grid waveform to <= 250
    # samples; 300 admits every legal waveform with 20% slack while pinning the
    # worst-case replay cost (linear in sample count) four orders of magnitude
    # below the verifier timeout. Violations are ScorerError (exit 1), never
    # the reserved infrastructure exit.
    max_programmed_samples: int = 300

    # --- plausibility bands ---
    alpha_2_band_mhz: tuple[float, float] = (-400.0, -200.0)
    # j_band_mhz is retained for stored-yaml compatibility but no longer
    # consumed: |J_eff| at a legitimate agent-chosen bias can leave any static
    # band (the tuning curve has a pole), and the relative check against the
    # recomputed truth supersedes it.
    j_band_mhz: tuple[float, float] = (2.0, 30.0)
    # flux_band_phi0 and crossing_flux_tol are likewise retained records, not
    # consumed by score_answer.
    flux_band_phi0: tuple[float, float] = (0.0, 0.45)


__all__ = ["CzHiddenScorer"]
