"""Public + hidden device config types for the dipolar-spin-ensemble qtype.

A ``dipolar_spin_ensemble`` device is a dense, globally-controlled ensemble of
electronic spins (NV centers in diamond) with *no single-spin addressing*. The
agent applies global ``Rx``/``Ry`` rotations (``pi/2`` pulses about ``+-x``,
``+-y``) separated by free evolution, and reads out the *collective*
magnetization ``<Sx>, <Sy>, <Sz>`` of the ensemble.

The system is **disorder-dominated**: each spin sees a static on-site field
``h_i`` drawn from a Gaussian of standard deviation ``W`` (the inhomogeneous
broadening), and the spins interact through the *traceless secular dipolar*
tensor ``c = (-1/2, -1/2, 1)`` with a median scale ``J << W``.

Public / hidden are STRUCTURALLY separated. For this
qtype the traps are *stale calibration* (the notebook under-reports ``W`` and
over-reports ``J``) and *optimistic control* (the notebook claims near-ideal
hard pulses, while the truth has finite pulse duration and a systematic
rotation-angle error). The realized ``W``, ``J``, pulse duration, rotation
error, RNG seed, and cluster-simulation parameters live ONLY in the hidden
config. The native-interaction *form* (the traceless dipolar tensor) is public
physics, as are the nominal Rabi frequency and pulse timing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Public spec (visible to agent) ----------


class PublicDipolarSpec(_Strict):
    """Hardware-manual-style public spec for the dipolar-spin-ensemble qtype.

    Calibration quantities are *claimed/nominal*; the realized disorder and
    interaction live in the hidden config and the stale notebook. The native
    interaction *form* is public: the secular dipolar tensor along the
    quantization axis, ``c = (c_xx, c_yy, c_zz) = (-1/2, -1/2, 1)`` (traceless).
    """

    schema_version: int = 1
    device_id: str
    qtype: Literal["dipolar_spin_ensemble"] = "dipolar_spin_ensemble"
    architecture: str = "NV-center electronic-spin ensemble in diamond"
    control_axes: list[Literal["x", "y"]] = Field(
        default_factory=lambda: ["x", "y"],
        description="Global rotation axes available (pi/2 pulses about +-x, +-y). No local addressing.",
    )
    native_dipolar_coeffs: tuple[float, float, float] = Field(
        (-0.5, -0.5, 1.0),
        description="Secular dipolar tensor (c_xx, c_yy, c_zz) along the quantization axis (traceless).",
    )
    rabi_mhz: float = Field(25.0, gt=0, description="Nominal drive Rabi Omega/2pi in MHz.")
    pi_over_2_duration_ns: float = Field(
        10.0, gt=0, description="Nominal pi/2 pulse duration in ns (= 250 / rabi_mhz)."
    )
    pulse_timing_resolution_ns: float = Field(1.0, gt=0)
    max_sequence_length_ns: float = Field(10_000.0, gt=0)
    spin_density_ppm: float = Field(
        15.0, gt=0, description="Spin density; sets the interaction scale."
    )
    claimed_pulse_quality: str = (
        "near-ideal hard pulses; rotation error < 0.5%; finite duration negligible"
    )
    measurement_return: Literal["collective_magnetization"] = "collective_magnetization"
    max_shots: int = 100_000
    notes: str = (
        "Calibration figures are the values accepted at the last scheduled service "
        "(vendor linewidth and pulse-response fits, not in-situ measurements); the "
        "lab notebook records them with their provenance."
    )


# ---------- Hidden truth (only inside qsim) ----------


class StaleDipolarNotebookEntry(_Strict):
    """Optional stale/optimistic calibration written into the lab notebook."""

    last_calibrated: str | None = None
    w_mhz_claim: float | None = None
    j_khz_claim: float | None = None
    pulse_quality_claim: str | None = None
    recommended_sequence: str | None = None
    notes: str = ""


class HiddenDipolarConfig(_Strict):
    """Hidden truth for the dipolar-spin-ensemble qtype. Lives only inside qsim.

    Holds the realized disorder/interaction scales, the true control
    imperfections (finite pulse duration + systematic rotation error), the
    collective-readout confusion, the cluster-simulation parameters (cluster
    size + number of disorder realizations), the RNG seed, and the stale
    notebook entry.
    """

    schema_version: int = 1
    device_id: str
    qtype: Literal["dipolar_spin_ensemble"] = "dipolar_spin_ensemble"
    seed: int

    # --- realized physics (the answers + traps) ---
    disorder_w_mhz: float = Field(
        ..., gt=0, description="True on-site disorder W/2pi (Gaussian field std) in MHz."
    )
    interaction_j_khz: float = Field(
        ..., gt=0, description="True median dipolar interaction |J0|/2pi in kHz."
    )
    dipolar_coeffs: tuple[float, float, float] = Field(
        (-0.5, -0.5, 1.0),
        description="Realized traceless secular dipolar tensor (must match the public form).",
    )
    rabi_mhz: float = Field(25.0, gt=0, description="True drive Rabi Omega/2pi in MHz.")
    pulse_duration_ns: float = Field(
        10.0, gt=0, description="True pi/2 pulse duration t_p in ns (NOT negligible)."
    )
    rotation_error: float = Field(
        0.03,
        description="Systematic fractional over-rotation applied to every pulse on error injection.",
    )
    t1_us: float = Field(5000.0, gt=0)

    # --- collective readout confusion (raw vs corrected) ---
    readout_p_plus_to_minus: float = Field(0.02, ge=0, le=1)
    readout_p_minus_to_plus: float = Field(0.05, ge=0, le=1)

    # --- cluster simulation parameters ---
    n_spins: int = Field(6, ge=2, le=10, description="Simulated cluster size (exact statevector).")
    n_realizations: int = Field(
        160, ge=1, description="Number of on-site-disorder realizations averaged."
    )

    stale_lab_notebook: StaleDipolarNotebookEntry = StaleDipolarNotebookEntry()


__all__ = [
    "HiddenDipolarConfig",
    "PublicDipolarSpec",
    "StaleDipolarNotebookEntry",
]
