"""Pure-numpy control stack for the tunable-coupler CZ qtype (shared engine+verifier).

The agent does not apply flux directly: it programs DAC samples on a finite-rate,
finite-bit-depth AWG, and the flux at the qubit is the programmed waveform
distorted by the flux line. So "the waveform you program is not the flux the qubit
sees" (the headline trap) is built into scoring:

    realized = flux_line( dac_quantize( programmed ) )

The flux line models the dominant short-time **settling**: a parallel bank of
discrete first-order poles plus a direct feed-through, with unit-step response

    s[n] = 1 - sum_k A_k * (1 - alpha_k)^n,   alpha_k = dt / (tau_k + dt)

on the AWG grid — the first nanoseconds of any edge are off by tens of percent
unless predistorted. As a rational filter this is ``H(z) = b(z)/a(z)``, so the
exact inverse (predistortion) is the same recursion with numerator and
denominator swapped, bounded by the finite **DAC bit depth** (a large overshoot
consumes dynamic range). A cryoscope recovers the settling in situ via the
Ramsey phase on the quadratic arc.

The earlier single pole is the exact special case
``amplitudes=(1.0,), taus_ns=(2.0,)``. The scalar constant was simultaneously
the dataclass default, the published value, and the most natural
round-number guess, so a no-cryoscope submission could predistort by guessing
it; the multi-exponential line, perturbed off the published
Foxen/Rol values, is mandatory at release.

The longer-timescale layer — the bias-tee high-pass tail and the
depth->=20 repeated-gate history dependence — remains a documented deferred
extension ("the control-stack engine ... is separable ...
stage-2 can land it incrementally").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AwgSpec:
    sample_rate_ghz: float = 2.5  # GSa/s -> 0.4 ns grid
    n_bits: int = 16
    full_scale_phi0: float = 0.5  # +-0.5 Phi0 DAC range
    analog_bw_ghz: float = 0.7  # -3 dB (nameplate; informational)


@dataclass(frozen=True)
class FluxLineModel:
    """Hidden flux-line short-settling response (the cryoscope target).

    ``taus_ns`` are DISCRETE poles on the AWG grid: ``alpha_k = dt/(tau_k+dt)``,
    step response ``s[n] = 1 - sum_k A_k (1-alpha_k)^n``. Quoting the
    continuous-time equivalent of a discrete tau differs by ~10% at tau ~ 2 ns,
    so every comparison against literature values must convert first. No
    defaults on purpose: the v2 default was the hidden truth.
    """

    amplitudes: tuple[float, ...]
    taus_ns: tuple[float, ...]


def dac_quantize(samples: np.ndarray, spec: AwgSpec) -> np.ndarray:
    """Round to the AWG's finite bit depth over its full-scale range (clips overshoot)."""
    levels = (1 << spec.n_bits) - 1
    fs = spec.full_scale_phi0
    x = np.clip(samples, -fs, fs)
    return np.round((x + fs) / (2 * fs) * levels) / levels * (2 * fs) - fs


def line_coeffs(dt_ns: float, line: FluxLineModel) -> tuple[np.ndarray, np.ndarray]:
    """Rational transfer function ``(b, a)`` (coefficients in z^-1) of the line.

    ``H(z) = c0 + sum_k A_k alpha_k / (1 - beta_k z^-1)`` with the feed-through
    ``c0 = 1 - sum_k A_k`` and ``beta_k = 1 - alpha_k``, brought over the common
    denominator ``a(z) = prod_k (1 - beta_k z^-1)``. DC gain is exactly unity by
    construction (``s[inf] = 1``).
    """
    if not line.amplitudes or len(line.amplitudes) != len(line.taus_ns):
        raise ValueError("FluxLineModel needs equal-length, non-empty amplitudes/taus_ns")
    alphas = [dt_ns / (tau + dt_ns) for tau in line.taus_ns]
    betas = [1.0 - alpha for alpha in alphas]
    feed = 1.0 - float(sum(line.amplitudes))
    a = np.array([1.0])
    for beta in betas:
        a = np.convolve(a, [1.0, -beta])
    b = feed * a
    for k, (amp_k, alpha_k) in enumerate(zip(line.amplitudes, alphas, strict=True)):
        term = np.array([amp_k * alpha_k])
        for j, beta in enumerate(betas):
            if j != k:
                term = np.convolve(term, [1.0, -beta])
        b[: len(term)] += term
    return b, a


def _lfilt(b: np.ndarray, a: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """Causal IIR ``a[0] y[n] = sum_i b[i] x[n-i] - sum_{j>=1} a[j] y[n-j]``.

    Zero initial state: the filter starts at the pre-pulse DC level (q1 parked
    at its sweet spot, zero flux), so a square step settles exactly as a
    physical flux line does (an earlier filter seeded its accumulator — the v1 filter seeded its
    accumulator at the first programmed sample, which erased the transient of
    any waveform starting at its own first value).
    """
    x = np.asarray(samples, dtype=float)
    y = np.empty_like(x)
    nb, na = len(b), len(a)
    for n in range(len(x)):
        acc = 0.0
        for i in range(min(nb, n + 1)):
            acc += b[i] * x[n - i]
        for j in range(1, min(na, n + 1)):
            acc -= a[j] * y[n - j]
        y[n] = acc / a[0]
    return y


def realized_flux(
    programmed: np.ndarray, dt_ns: float, awg: AwgSpec, line: FluxLineModel
) -> np.ndarray:
    """Programmed DAC waveform -> actual flux at the qubit (the scored transform)."""
    x = dac_quantize(np.asarray(programmed, dtype=float), awg)
    b, a = line_coeffs(dt_ns, line)
    return _lfilt(b, a, x)


def predistort(intended: np.ndarray, dt_ns: float, line: FluxLineModel) -> np.ndarray:
    """Inverse settling filter so realized(predistort(intended)) ~ intended.

    The exact inverse of the rational line filter: the same recursion with
    ``b`` and ``a`` swapped, starting from the pre-pulse DC level. The
    high-boost is what consumes DAC dynamic range (the agent must keep the
    overshoot representable at ``n_bits``). Stable only when the line estimate
    is minimum-phase — callers fitting a line from data must check
    ``inverse_is_stable`` before predistorting with it.
    """
    b, a = line_coeffs(dt_ns, line)
    return _lfilt(a, b, np.asarray(intended, dtype=float))


def inverse_zero_magnitudes(dt_ns: float, line: FluxLineModel) -> np.ndarray:
    """|z| of the line's zeros (= poles of the inverse filter)."""
    b, _ = line_coeffs(dt_ns, line)
    trimmed = np.trim_zeros(np.asarray(b, dtype=float), "b")
    if len(trimmed) <= 1:
        return np.array([])
    return np.abs(np.roots(trimmed))


def inverse_is_stable(dt_ns: float, line: FluxLineModel) -> bool:
    """Whether predistortion with this line estimate is a stable recursion.

    A fitted ``b`` polynomial can land zeros outside the unit circle, and the
    inverse then diverges (routine at short fit windows). Estimates
    failing this check must be rejected before they are used to predistort.
    """
    zeros = inverse_zero_magnitudes(dt_ns, line)
    return bool(len(zeros) == 0 or np.max(zeros) < 1.0 - 1e-9)


def cryoscope_phase(realized: np.ndarray, dt_ns: float, arc_coeff_ghz: float) -> np.ndarray:
    """Accumulated Ramsey phase phi(tau) = 2*pi * integral of Delta f_Q(realized(t)).

    On the quadratic sweet-spot arc ``Delta f_Q(phi) = arc_coeff * phi^2``; the agent
    differentiates phi(tau) and inverts the quadratic arc to recover the settling
.
    """
    return 2 * math.pi * np.cumsum(arc_coeff_ghz * np.asarray(realized) ** 2) * dt_ns


__all__ = [
    "AwgSpec",
    "FluxLineModel",
    "cryoscope_phase",
    "dac_quantize",
    "inverse_is_stable",
    "inverse_zero_magnitudes",
    "line_coeffs",
    "predistort",
    "realized_flux",
]
