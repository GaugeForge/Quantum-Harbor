"""Effective pulse model for a fixed-parity four-Majorana qubit.

The computational basis consists of two same-total-parity states of two coupled
Kitaev segments (for example ``|ee>`` and ``|oo>`` in the global-even manifold).
The simulator retains one aggregate non-computational level ``|2>`` for bulk
excitations and poisoning.  It is an explicitly effective three-level control
model, not a microscopic claim that one qutrit is a complete Majorana device.
It captures three error mechanisms a pulse gate must negotiate:

* **mis-calibration** -- the realized drive rate is ``amp * omega_scale`` with a
  HIDDEN/stale ``omega_scale``; an uncalibrated pulse over/under-rotates;
* **coherent bulk-excitation leakage** -- transverse control also couples
  ``|1>-|2>`` (strength ``lambda``), so fast/strong pulses populate the
  non-computational sector;
* **charge noise + poisoning** -- per-shot detuning/amplitude fluctuations and a
  parity-changing process.  Poisoning leaves the fixed-parity computational
  manifold; it is not a logical bit flip.

The agent submits pulse SEGMENTS; the engine builds the piecewise-constant
propagator and samples level-resolved outcomes. The task-specific RB action draws
and compiles random Cliffords inside qsim; the verifier independently replays the
submitted pulse artifact. This module is the shared dynamics source, pure numpy.

Units: energies in micro-eV, times in ns, ``hbar = 0.6582119569 ueV*ns`` so a
segment propagator is ``expm(-i H tau / hbar)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HBAR_UEV_NS = 0.6582119569


@dataclass(frozen=True)
class PulseParams:
    """Hidden pulse-physics parameters of the device."""

    omega_scale_ueV: float  # realized Rabi rate at amp=1 (the hidden calibration)
    e_gap_ueV: float  # leakage level (bulk quasiparticle) energy ~ 2t
    leakage_lambda: float  # |1>-|2> drive coupling relative to |0>-|1>
    charge_noise_sigma_ueV: float  # per-shot detuning fluctuation (dephasing)
    amp_noise_frac: float  # per-shot fractional drive-amplitude fluctuation
    poisoning_per_ns: float  # parity-flip (leakage) rate per ns -> time-dependent floor


@dataclass(frozen=True)
class PulseSegment:
    """One piecewise-constant drive segment.

    ``amp`` is the DAC drive amplitude (dimensionless, 0..1); the realized Rabi
    rate is ``amp * omega_scale_ueV``. ``phase_rad`` sets the rotation axis in the
    x-y plane. Positive ``detuning_ueV`` generates a positive logical-Z
    rotation. ``drag`` is retained as the public wire name for a constant
    quadrature ratio; it is not assumed to be a derivative-DRAG waveform.
    """

    amp: float
    phase_rad: float
    detuning_ueV: float
    duration_ns: float
    drag: float = 0.0


def _segment_props_batched(
    seg: PulseSegment, p: PulseParams, amp_scales: np.ndarray, det_offsets: np.ndarray
) -> np.ndarray:
    """Per-shot (S,3,3) propagators for one segment, vectorized over shots.

    ``amp_scales`` / ``det_offsets`` are per-shot (length S); the realized Rabi
    rate is ``amp * amp_scale * omega_scale`` and the qubit splitting carries the
    charge-noise ``det_offset``. ``drag`` adds a constant quadrature component
    on the orthogonal axis. Returns the unitary propagator for each shot.
    """
    s = amp_scales.shape[0]
    omega = seg.amp * amp_scales * p.omega_scale_ueV  # (S,)
    drag_q = seg.drag * omega
    c = np.exp(-1j * seg.phase_rad)
    drive01 = 0.5 * (omega + 1j * drag_q) * c  # (S,)
    drive12 = 0.5 * p.leakage_lambda * (omega + 1j * drag_q) * c
    h = np.zeros((s, 3, 3), dtype=complex)
    h[:, 0, 1] = np.conj(drive01)
    h[:, 1, 0] = drive01
    h[:, 1, 2] = np.conj(drive12)
    h[:, 2, 1] = drive12
    logical_detuning = seg.detuning_ueV + det_offsets
    # H_logical = +(detuning/2) Z, with Z = diag(1,-1).  The aggregate
    # bulk-excitation level keeps a fixed gap relative to |1>.
    h[:, 0, 0] = 0.5 * logical_detuning
    h[:, 1, 1] = -0.5 * logical_detuning
    h[:, 2, 2] = p.e_gap_ueV - 0.5 * logical_detuning
    h = 0.5 * (h + np.conj(np.transpose(h, (0, 2, 1))))
    evals, evecs = np.linalg.eigh(h)  # (S,3), (S,3,3)
    phase = np.exp(-1j * evals * seg.duration_ns / HBAR_UEV_NS)  # (S,3)
    return evecs @ (phase[:, :, None] * np.conj(np.transpose(evecs, (0, 2, 1))))


def _evolve_batched(
    segments: list[PulseSegment], p: PulseParams, amp_scales: np.ndarray, det_offsets: np.ndarray
) -> np.ndarray:
    """Final (S,3) states from |0> for S shots with per-shot amp/detuning offsets."""
    s = amp_scales.shape[0]
    psi = np.zeros((s, 3), dtype=complex)
    psi[:, 0] = 1.0
    for seg in segments:
        u = _segment_props_batched(seg, p, amp_scales, det_offsets)
        psi = np.einsum("sij,sj->si", u, psi)
    return psi


def evolve(
    segments: list[PulseSegment],
    p: PulseParams,
    *,
    amp_scale: float = 1.0,
    det_offset: float = 0.0,
) -> np.ndarray:
    """Final 3-level state from |0> under the segment sequence (no shot noise)."""
    psi = _evolve_batched(segments, p, np.array([amp_scale]), np.array([det_offset]))
    return psi[0]


def propagator(segments: list[PulseSegment], p: PulseParams) -> np.ndarray:
    """Noise-free 3x3 propagator of the segment sequence (no per-shot draws)."""
    u = np.eye(3, dtype=complex)
    ones = np.array([1.0])
    zeros = np.array([0.0])
    for seg in segments:
        u = _segment_props_batched(seg, p, ones, zeros)[0] @ u
    return u


def sample_outcomes(
    segments: list[PulseSegment],
    p: PulseParams,
    rng: np.random.Generator,
    shots: int,
) -> np.ndarray:
    """Per-shot level outcomes (0/1/2), vectorized over shots.

    Each shot draws a fresh detuning offset (charge noise) and amplitude scale,
    evolves, samples a charge state, then a time-dependent poisoning event may
    leave the fixed-total-parity computational manifold.
    """
    if not segments:
        return np.zeros(shots, dtype=np.uint8)
    det = (
        rng.normal(0.0, p.charge_noise_sigma_ueV, shots)
        if p.charge_noise_sigma_ueV > 0
        else np.zeros(shots)
    )
    amp = 1.0 + rng.normal(0.0, p.amp_noise_frac, shots) if p.amp_noise_frac > 0 else np.ones(shots)
    psi = _evolve_batched(segments, p, amp, det)
    probs = np.abs(psi) ** 2
    probs /= probs.sum(axis=1, keepdims=True)
    u = rng.random(shots)
    lvl = (u[:, None] >= np.cumsum(probs, axis=1)).sum(axis=1).astype(np.uint8)  # 0/1/2
    total_ns = sum(s.duration_ns for s in segments)
    p_pois = 1.0 - np.exp(-p.poisoning_per_ns * total_ns)
    if p_pois > 0:
        poisoned = (rng.random(shots) < p_pois) & (lvl != 2)
        lvl[poisoned] = 2
    return lvl


# --------------------------------------------------------------------------- #
# Helpers for the reference solver / agent: build a calibrated rotation pulse.
# --------------------------------------------------------------------------- #


def rotation_pulse(
    angle_rad: float,
    *,
    axis_phase_rad: float,
    amp: float,
    omega_scale_ueV: float,
    drag: float = 0.0,
    n_seg: int = 1,
) -> list[PulseSegment]:
    """A constant-amplitude pulse of the given rotation angle about an x-y axis.

    Duration is solved from ``angle = omega * tau / hbar`` with
    ``omega = amp * omega_scale_ueV`` (the caller supplies its *calibrated*
    omega_scale estimate). Splitting into ``n_seg`` equal segments is a
    convenience for callers constructing segmented envelopes; identical
    segments alone do not create a derivative waveform.
    """
    omega = amp * omega_scale_ueV
    tau = angle_rad * HBAR_UEV_NS / omega
    seg = PulseSegment(
        amp=amp,
        phase_rad=axis_phase_rad,
        detuning_ueV=0.0,
        duration_ns=tau / n_seg,
        drag=drag,
    )
    return [seg] * n_seg


__all__ = [
    "HBAR_UEV_NS",
    "PulseParams",
    "PulseSegment",
    "evolve",
    "propagator",
    "sample_outcomes",
    "rotation_pulse",
]
