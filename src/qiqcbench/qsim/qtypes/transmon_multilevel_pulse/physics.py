"""Pure-numpy 4-level Duffing transmon propagator.

This module is the single source of truth for the multilevel-transmon physics.
The qsim engine uses it for the agent's live runs; the reference DRAG solver and
the (self-contained) Harbor verifier use the SAME math so there is no
engine-vs-verifier physics drift. Keep it numpy-only and dependency-light so the
verifier can vendor a copy verbatim.

Conventions
-----------
- Rotating frame at ``omega_01``; detuning sign ``delta = omega_01 - omega_drive``.
- Hamiltonian (rad/ns)::

      H(t) = alpha * diag(0,0,1,3)
           + delta(t) * N
           + (1/2)[Omega_x(t)(a+a_dag) + Omega_y(t) i(a_dag-a)]

- Drive quadratures arrive in DAC units; the Rabi rate is
  ``Omega/2pi [MHz] = k * A_dac / (1 + comp * |A_dac|^2)`` where
  ``A_dac = hypot(x_dac, y_dac)`` (compression on the combined magnitude).
- Detuning arrives in Hz (cyclic): ``delta_ang = 2*pi*delta_hz*1e-9`` rad/ns.
- Lindblad collapse: relaxation ``sqrt(1/T1) a`` (a carries the sqrt(n) ladder so
  level n decays n/T1), dephasing ``sqrt(2/T_phi) N`` with
  ``1/T_phi = 1/T2 - 1/(2 T1)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

_TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class DuffingParams:
    """Physical parameters of the 4-level Duffing transmon."""

    anharmonicity_hz: float
    rabi_mhz_per_dac: float
    amp_compression: float = 0.0
    t1_s: float = math.inf
    t2_s: float = math.inf
    n_levels: int = 4


def operators(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (a, a_dag, N, anh) for an n-level oscillator (anh = diag n(n-1)/2)."""
    a = np.zeros((n, n), dtype=complex)
    for i in range(1, n):
        a[i - 1, i] = math.sqrt(i)
    ad = a.conj().T
    nop = np.diag(np.arange(n, dtype=float)).astype(complex)
    anh = np.diag([0.5 * k * (k - 1) for k in range(n)]).astype(complex)
    return a, ad, nop, anh


def dac_to_rabi_radns(amp_dac: np.ndarray, params: DuffingParams) -> np.ndarray:
    """Convert a DAC magnitude array to a Rabi rate (rad/ns), with compression."""
    k = params.rabi_mhz_per_dac
    comp = params.amp_compression
    mhz = k * amp_dac / (1.0 + comp * amp_dac**2)
    return _TWO_PI * mhz * 1e-3  # MHz -> rad/ns


def _collapse_ops(params: DuffingParams) -> list[np.ndarray]:
    a, _ad, nop, _anh = operators(params.n_levels)
    ops: list[np.ndarray] = []
    t1_ns = params.t1_s * 1e9
    t2_ns = params.t2_s * 1e9
    if math.isfinite(t1_ns) and t1_ns > 0:
        ops.append(math.sqrt(1.0 / t1_ns) * a)
    inv_tphi = 0.0
    if math.isfinite(t2_ns) and t2_ns > 0:
        inv_tphi = 1.0 / t2_ns - (1.0 / (2.0 * t1_ns) if math.isfinite(t1_ns) else 0.0)
    if inv_tphi > 0:
        ops.append(math.sqrt(2.0 * inv_tphi) * nop)
    return ops


def static_hamiltonian(params: DuffingParams) -> np.ndarray:
    """Anharmonicity drift term (rad/ns); detuning/drive are added per-sample."""
    _a, _ad, _n, anh = operators(params.n_levels)
    alpha_radns = _TWO_PI * params.anharmonicity_hz * 1e-9
    return alpha_radns * anh


def _hamiltonian_sample(
    ox_radns: float, oy_radns: float, det_hz: float, params: DuffingParams, h_anh: np.ndarray
) -> np.ndarray:
    a, ad, nop, _anh = operators(params.n_levels)
    delta_radns = _TWO_PI * det_hz * 1e-9
    drive = 0.5 * (ox_radns * (a + ad) + oy_radns * (1j * (ad - a)))
    return h_anh + delta_radns * nop + drive


def _lindbladian(h: np.ndarray, c_ops: list[np.ndarray]) -> np.ndarray:
    d = h.shape[0]
    eye = np.eye(d, dtype=complex)
    lind = -1j * (np.kron(eye, h) - np.kron(h.T, eye))
    for c in c_ops:
        cdc = c.conj().T @ c
        lind += np.kron(c.conj(), c) - 0.5 * np.kron(eye, cdc) - 0.5 * np.kron(cdc.T, eye)
    return lind


def _vec(rho: np.ndarray) -> np.ndarray:
    return rho.reshape(-1, order="F")


def _unvec(v: np.ndarray, d: int) -> np.ndarray:
    return v.reshape((d, d), order="F")


def open_superoperator(
    ox_dac: np.ndarray,
    oy_dac: np.ndarray,
    det_hz: np.ndarray,
    dt_ns: float,
    params: DuffingParams,
) -> np.ndarray:
    """Total Lindblad propagator (d^2 x d^2) for piecewise-constant envelopes."""
    d = params.n_levels
    h_anh = static_hamiltonian(params)
    c_ops = _collapse_ops(params)
    amp = np.hypot(ox_dac, oy_dac)
    rabi = dac_to_rabi_radns(amp, params)
    # split combined Rabi back onto the two quadratures by DAC ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        ox_radns = np.where(amp > 0, rabi * ox_dac / amp, 0.0)
        oy_radns = np.where(amp > 0, rabi * oy_dac / amp, 0.0)
    u_total = np.eye(d * d, dtype=complex)
    for ox, oy, det in zip(ox_radns, oy_radns, det_hz, strict=True):
        h = _hamiltonian_sample(float(ox), float(oy), float(det), params, h_anh)
        u_total = expm(_lindbladian(h, c_ops) * dt_ns) @ u_total
    return u_total


def coherent_unitary(
    ox_dac: np.ndarray,
    oy_dac: np.ndarray,
    det_hz: np.ndarray,
    dt_ns: float,
    params: DuffingParams,
) -> np.ndarray:
    """Closed-system unitary (no collapse) for phase-error analysis."""
    d = params.n_levels
    h_anh = static_hamiltonian(params)
    amp = np.hypot(ox_dac, oy_dac)
    rabi = dac_to_rabi_radns(amp, params)
    with np.errstate(divide="ignore", invalid="ignore"):
        ox_radns = np.where(amp > 0, rabi * ox_dac / amp, 0.0)
        oy_radns = np.where(amp > 0, rabi * oy_dac / amp, 0.0)
    u = np.eye(d, dtype=complex)
    for ox, oy, det in zip(ox_radns, oy_radns, det_hz, strict=True):
        h = _hamiltonian_sample(float(ox), float(oy), float(det), params, h_anh)
        u = expm(-1j * h * dt_ns) @ u
    return u


def propagate_density(rho0: np.ndarray, u_super: np.ndarray) -> np.ndarray:
    """Apply a precomputed open superoperator to a density matrix."""
    d = rho0.shape[0]
    return _unvec(u_super @ _vec(rho0.astype(complex)), d)


def final_populations(rho: np.ndarray) -> np.ndarray:
    """Real, clipped, normalized diagonal populations."""
    pops = np.clip(np.real(np.diag(rho)), 0.0, None)
    s = pops.sum()
    return pops / s if s > 0 else pops


# ---------- analysis helpers (verifier + reference) ----------

_AXIAL_STATES: dict[str, np.ndarray] = {}


def _qubit_ket(n: int, c0: complex, c1: complex) -> np.ndarray:
    ket = np.zeros(n, dtype=complex)
    ket[0], ket[1] = c0, c1
    return ket


def axial_density_matrices(n: int) -> dict[str, np.ndarray]:
    """Six axial input states (+-x,+-y,+-z) as n-level density matrices."""
    r = 1.0 / math.sqrt(2.0)
    kets = {
        "+z": _qubit_ket(n, 1, 0),
        "-z": _qubit_ket(n, 0, 1),
        "+x": _qubit_ket(n, r, r),
        "-x": _qubit_ket(n, r, -r),
        "+y": _qubit_ket(n, r, 1j * r),
        "-y": _qubit_ket(n, r, -1j * r),
    }
    return {k: np.outer(v, v.conj()) for k, v in kets.items()}


def leakage_six_axial(u_super: np.ndarray, params: DuffingParams) -> float:
    """Input-averaged population outside {|0>,|1>} after the gate (open system)."""
    states = axial_density_matrices(params.n_levels)
    leak = 0.0
    for rho0 in states.values():
        rho = propagate_density(rho0, u_super)
        pops = final_populations(rho)
        leak += float(pops[2:].sum())
    return leak / len(states)


def coherent_phase_error(u_coh: np.ndarray, theta: float) -> float:
    """AC-Stark Z-phase error of the gate vs an ideal x-rotation by ``theta``.

    Projects the coherent propagator onto {0,1}, unitarizes, removes the ideal
    x-rotation, and reads the residual Z-rotation angle. ~0 for a phase-clean
    gate; nonzero for a derivative-only (no-detuning) DRAG pulse.
    """
    w = u_coh[:2, :2]
    uu, _s, vh = np.linalg.svd(w)
    wu = uu @ vh
    det = np.linalg.det(wu)
    if abs(det) < 1e-12:
        return float("nan")
    wu_su = wu / np.sqrt(det)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    eye = np.eye(2, dtype=complex)
    r_ideal = math.cos(theta / 2) * eye - 1j * math.sin(theta / 2) * x
    e = wu_su @ r_ideal.conj().T
    phi = np.angle(e[0, 0]) - np.angle(e[1, 1])
    # wrap to (-pi, pi]
    return float((phi + math.pi) % (2 * math.pi) - math.pi)


__all__ = [
    "DuffingParams",
    "axial_density_matrices",
    "coherent_phase_error",
    "coherent_unitary",
    "dac_to_rabi_radns",
    "final_populations",
    "leakage_six_axial",
    "open_superoperator",
    "operators",
    "propagate_density",
    "static_hamiltonian",
]
