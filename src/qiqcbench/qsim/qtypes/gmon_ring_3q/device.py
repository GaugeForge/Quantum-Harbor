"""Public + hidden device config types for the gmon-ring (pulse-level) qtype.

A three-qubit superconducting "gmon" ring (Q1, Q2, Q3) with one tunable coupler
per edge (CP12, CP23, CP31). The agent controls single-qubit XY rotations and the
parametric modulation of each coupler g_jk(t) = A_jk * env(t) * cos(2*pi*f_jk*t + phi_jk).

Public and hidden are STRUCTURALLY separated: both use
``extra="forbid"`` so a hidden-truth field can never be absorbed by the public spec
and vice versa. Mirrors the split in ``transmon_pulse/device.py`` and
``digital_gate_model/device.py``.

Task note: the qubit frequencies ``omega_j`` are PUBLIC control settings (the agent
needs them to compute the resonant modulation frequencies Delta_jk = |omega_j - omega_k|),
so the public spec carries exact frequencies. The hidden config repeats them only for
engine self-containment; they are not a hidden lever. The hidden lever is the realized
coupler hopping ``g0_hz`` (the Stage-A calibration-recovery target), the true on-site
anharmonicity / U, T1/T2, and the asymmetric readout model.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Public spec (visible to agent) ----------


class PublicGmonQubit(_Strict):
    id: str
    frequency_hz: float = Field(
        ..., description="Exact qubit frequency (public control setting, not hidden truth)."
    )
    claimed_anharmonicity_hz: float = Field(
        ...,
        description="Nominal/claimed anharmonicity; sets the on-site interaction scale U.",
    )


class PublicGmonCoupler(_Strict):
    id: str
    qubits: tuple[str, str] = Field(..., description="The two qubit ids this coupler links.")


class GmonControlLimits(_Strict):
    """Advertised pulse/modulation envelope the agent designs against (Stage B/C)."""

    timing_resolution_ns: float = 1.0
    max_sequence_ns: float = 4000.0
    max_modulation_freq_hz: float = 200.0e6
    coupling_min_hz: float = -55.0e6
    coupling_max_hz: float = 5.0e6
    max_amp: float = Field(1.0, description="Modulation amplitude is normalized to [0, max_amp].")
    supported_envelopes: list[str] = ["constant", "raised_cosine"]
    supported_rotation_axes: list[str] = ["x", "y", "z"]


class PublicGmonRingSpec(_Strict):
    """Hardware-manual-style public spec for the gmon-ring qtype."""

    schema_version: int = 1
    device_id: str
    qtype: Literal["gmon_ring_3q"] = "gmon_ring_3q"
    qubits: list[PublicGmonQubit] = Field(..., min_length=1)
    couplers: list[PublicGmonCoupler] = Field(..., min_length=1)
    control_limits: GmonControlLimits = GmonControlLimits()
    measurement_return: Literal["iq"] = "iq"
    max_shots: int = Field(100_000, gt=0)
    notes: str = (
        "Coupler calibration may have drifted since the last calibration; verify the "
        "hopping g0 experimentally before relying on the lab-notebook value."
    )


# ---------- Hidden truth (only inside qsim) ----------


class HiddenGmonQubit(_Strict):
    id: str
    frequency_hz: float
    anharmonicity_hz: float
    t1_s: float = Field(..., gt=0)
    t2_s: float = Field(..., gt=0)


class HiddenGmonCoupler(_Strict):
    id: str
    qubits: tuple[str, str]
    g0_hz: float = Field(
        ...,
        description=(
            "Realized effective hopping at modulation amplitude amp=1.0. The Stage-A "
            "calibration-recovery target; the stale lab notebook claims a drifted value."
        ),
    )
    phase_offset_rad: float = Field(
        0.0,
        description=(
            "Hidden Peierls-phase calibration offset added to the agent's commanded "
            "coupler phase (the phase analog of the g0 drift). The realized synthetic "
            "flux differs from the commanded one by the ring-sum of these offsets. "
            "Default 0.0 leaves a perfectly-calibrated device (e.g. gmon_ring_3q_v0)."
        ),
    )


class HiddenGmonReadout(_Strict):
    id: str
    p_0_to_1: float = Field(..., ge=0, le=1)
    p_1_to_0: float = Field(..., ge=0, le=1)
    iq_0_mean: tuple[float, float]
    iq_1_mean: tuple[float, float]
    iq_sigma: float = Field(..., gt=0)


class StaleGmonNotebookEntry(_Strict):
    """Stale calibration claims surfaced via get_lab_notebook."""

    g0_mhz_claim: float | None = None
    last_calibrated: str | None = None
    readout_fidelity_claim: float | None = None
    phase_offset_rad_claim: float | None = Field(
        None,
        description=(
            "Optimistic/stale claim for the coupler Peierls-phase calibration offset "
            "(the lab believes phases are calibrated to ~0); the true per-coupler "
            "offset is hidden and may have drifted."
        ),
    )
    notes: str | None = None


class HiddenGmonRingConfig(_Strict):
    """Hidden truth for the gmon-ring qtype. Lives only inside the qsim container."""

    schema_version: int = 1
    device_id: str
    qtype: Literal["gmon_ring_3q"] = "gmon_ring_3q"
    seed: int
    qubits: list[HiddenGmonQubit] = Field(..., min_length=1)
    couplers: list[HiddenGmonCoupler] = Field(..., min_length=1)
    readout: list[HiddenGmonReadout] = Field(..., min_length=1)
    stale_lab_notebook: StaleGmonNotebookEntry = StaleGmonNotebookEntry()


__all__ = [
    "GmonControlLimits",
    "HiddenGmonCoupler",
    "HiddenGmonQubit",
    "HiddenGmonReadout",
    "HiddenGmonRingConfig",
    "PublicGmonCoupler",
    "PublicGmonQubit",
    "PublicGmonRingSpec",
    "StaleGmonNotebookEntry",
]
