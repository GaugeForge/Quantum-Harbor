"""Public + hidden device config for the tunable-coupler CZ qtype.

Two transmons + a tunable coupler + a finite-bit-depth AWG and a distorting flux
line. Public/hidden are STRUCTURALLY separated (``extra="forbid"``): the agent
sees nominal/claimed calibration + the AWG nameplate; the true anharmonicity,
couplings, coupler curves, flux-line settling, and readout stay hidden.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Public spec (visible to agent) ----------


class PublicTunableCouplerSpec(_Strict):
    schema_version: int = 1
    device_id: str
    qtype: Literal["transmon_pair_tunable_coupler"] = "transmon_pair_tunable_coupler"

    n_levels: int = Field(4, ge=3, description="Duffing levels per transmon (|2> leakage).")
    omega_1_max_ghz: float = 4.500  # q1 sweet-spot frequency (tunable down via flux)
    omega_2_ghz: float = 4.000  # q2 fixed
    omega_c_max_ghz: float = 6.500  # coupler sweet spot (above both qubits)
    # CLAIMED calibration shown to the agent (previous-cooldown values; the
    # current truth is hidden).
    claimed_alpha_1_mhz: float = -300.0
    claimed_alpha_2_mhz: float = -300.0
    claimed_g1c_mhz: float = 68.0
    claimed_g2c_mhz: float = 68.0
    claimed_g12_mhz: float = 5.0
    claimed_coupler_idle_flux: float = 0.30  # previous cooldown ("J=0, ZZ negligible")

    # AWG nameplate (public) + flux limits
    awg_sample_rate_ghz: float = 2.5  # 0.4 ns grid
    awg_n_bits: int = 16
    awg_full_scale_phi0: float = 0.70
    awg_analog_bw_ghz: float = 0.70
    max_abs_flux_q1: float = 0.45
    gate_window_ns: tuple[float, float] = (36.0, 100.0)

    measurement_return: Literal["level_population"] = "level_population"
    max_shots: int = Field(100_000, gt=0)
    notes: str = (
        "Two transmons (q1 flux-tunable, q2 fixed) + a tunable coupler. Build a CZ "
        "via a net-zero adiabatic flux pulse on the |11>-|02> crossing. You PROGRAM "
        "DAC samples for q1.flux; the flux the qubit sees is your programmed waveform "
        "DISTORTED by the flux line (short-time settling) — characterize it (e.g. a "
        "cryoscope on the quadratic sweet-spot arc) and predistort. Claimed coupling / "
        "anharmonicity values were accepted at the previous cooldown's calibration; "
        "measure the 1<->2 transition, the |11>-|02> "
        "crossing, J, and the coupler J_eff(Phi_c)/zeta(Phi_c) curves in situ. The coupler "
        "flux sets both the coupling and the static ZZ; park it where both vanish. "
        "Report the single-qubit virtual-Z corrections. All run_* tools are async."
    )


# ---------- Hidden truth ----------


class HiddenReadoutLevel(_Strict):
    """Per-qubit asymmetric level readout (rows=true level, simplified 0/1 confusion)."""

    p_0_to_1: float = Field(..., ge=0, le=1)
    p_1_to_0: float = Field(..., ge=0, le=1)
    p_2_misread: float = Field(0.0, ge=0, le=1)


class StaleCouplerNotebookEntry(_Strict):
    last_calibrated: str | None = None
    alpha_2_mhz_claim: float | None = None
    coupling_j_mhz_claim: float | None = None
    g_mhz_claim: str | None = None
    coupler_idle_flux_claim: float | None = None
    pulse_advice: str | None = None
    line_transfer_function_claim: str | None = None
    readout_claim: str | None = None
    notes: str = ""


class HiddenTunableCouplerConfig(_Strict):
    schema_version: int = 1
    device_id: str
    qtype: Literal["transmon_pair_tunable_coupler"] = "transmon_pair_tunable_coupler"
    seed: int

    # qubit/coupler truth (GHz)
    omega_1_max_ghz: float = 4.500
    omega_2_ghz: float = 4.000
    omega_c_max_ghz: float = 6.500
    alpha_1_ghz: float = -0.300
    alpha_2_ghz: float = -0.330  # true (stale claims -0.300)
    arc_asym: float = 0.001  # q1 arc asymmetry (slightly breaks 1/f immunity)
    # constructed coupler curves
    g12_ghz: float = 0.005
    gg_ghz2: float = 0.002537  # ~g1c*g2c/2; pins J_idle=0 @0.266, J_gate=-10MHz @0.36
    zeta_scale: float = 1.0
    # coherence
    t1_us: float = 45.0
    t2_us: float = 32.0
    # flux-line short-time settling (the cryoscope target): parallel bank of
    # DISCRETE first-order poles on the 0.4 ns AWG grid (alpha_k = dt/(tau_k+dt),
    # step response s[n] = 1 - sum_k A_k (1-alpha_k)^n), tau ascending. Edition
    # v3 replaced the v1/v2 single pole: its 2.0 ns constant was
    # simultaneously the code default, the published value, and the
    # natural round-number guess. Values perturbed off the published Foxen/Rol
    # figures.
    flux_settle_amplitudes: tuple[float, ...] = (0.38, 0.11)
    flux_settle_taus_ns: tuple[float, ...] = (1.7, 14.0)
    # quasi-static flux noise + robustness offset
    flux_noise_sigma_phi0: float = 50e-6
    robustness_offset_phi0: float = 300e-6
    # readout
    readout_q1: HiddenReadoutLevel = HiddenReadoutLevel(p_0_to_1=0.020, p_1_to_0=0.030)
    readout_q2: HiddenReadoutLevel = HiddenReadoutLevel(p_0_to_1=0.025, p_1_to_0=0.035)

    stale_lab_notebook: StaleCouplerNotebookEntry = StaleCouplerNotebookEntry()


__all__ = [
    "HiddenReadoutLevel",
    "HiddenTunableCouplerConfig",
    "PublicTunableCouplerSpec",
    "StaleCouplerNotebookEntry",
]
