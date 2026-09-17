"""Pure-numpy physics for the tunable-coupler CZ qtype.

Single source of truth for the two-transmon + tunable-coupler CZ physics, shared
by the qsim engine, the reference solver, and the (self-contained) Harbor
verifier so there is no engine-vs-verifier drift. Numpy + scipy only, so the
verifier can vendor it.

Model
-----
The CZ is driven by the ``|11> <-> |02>`` avoided crossing of a flux-tunable
qubit ``q1`` and a fixed qubit ``q2`` (4-level Duffing each, 16-dim joint space).
The tunable coupler is folded into a constructed effective exchange ``J_eff(Phi_c)``
and a static parasitic ``zeta(Phi_c)`` (the agent-facing ``J_eff`` formula is an
approximation; the engine uses the constructed self-consistent curves). The
exchange term ``J(a1^dag a2 + a1 a2^dag)`` naturally carries the bosonic
``<11|H|02> = sqrt(2) J`` matrix element. Flux-pulsing ``q1`` toward the crossing
(``omega_1 - omega_2 = alpha_2``) and back imprints a conditional phase
``phi_2Q = phi_00 - phi_01 - phi_10 + phi_11`` (target ``pi``) on a net-zero
trajectory; the single-qubit phases ``phi_01``/``phi_10`` are removed by virtual-Z.

All frequencies in GHz, fluxes in ``Phi_0``, times in ns; phase accrues as
``2*pi*f*t``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

_TWO_PI = 2.0 * math.pi
_LEVELS = 4


def _ladder(n: int) -> np.ndarray:
    a = np.zeros((n, n), dtype=complex)
    for i in range(1, n):
        a[i - 1, i] = math.sqrt(i)
    return a


_A = _ladder(_LEVELS)
_I = np.eye(_LEVELS, dtype=complex)
_A1 = np.kron(_A, _I)
_A2 = np.kron(_I, _A)
_AD1, _AD2 = _A1.conj().T, _A2.conj().T
_N1, _N2 = _AD1 @ _A1, _AD2 @ _A2
_EYE = np.eye(_LEVELS * _LEVELS, dtype=complex)
# index of basis |i,j> (q1=i, q2=j)
IDX = {(i, j): i * _LEVELS + j for i in range(_LEVELS) for j in range(_LEVELS)}
COMP = [(0, 0), (0, 1), (1, 0), (1, 1)]


@dataclass(frozen=True)
class CouplerParams:
    """Constructed effective coupler curves J_eff(Phi_c) and zeta(Phi_c).

    Both use the dispersive form ``c0 + c1 * (1/Delta_1c + 1/Delta_2c)`` with the
    coupler frequency on its own flux arc, tuned so J_eff and zeta have the
    intended zeros/values (J=0 + zeta~0 near the idle flux,
    J~10 MHz at the gate flux, zeta~60 kHz at the stale idle).
    """

    omega_c_max: float = 6.5
    g12: float = 0.005  # GHz (direct exchange)
    gg: float = 0.00255  # GHz^2 ~ g1c*g2c/2
    omega_1_gate: float = 3.670  # q1 freq at the crossing (gate config)
    omega_1_idle: float = 4.500  # q1 freq at idle (sweet spot)
    omega_2: float = 4.000
    # static-ZZ construction
    zeta_scale: float = 1.0


def coupler_freq(phi_c: float, omega_c_max: float = 6.5) -> float:
    return omega_c_max * math.sqrt(abs(math.cos(math.pi * phi_c)))


def j_eff(phi_c: float, p: CouplerParams, omega_1: float | None = None) -> float:
    """Effective q1-q2 exchange (GHz) at a given q1 frequency.

    ``omega_1=None`` uses the gate config (q1 near the crossing); pass
    ``p.omega_1_idle`` for the decoupling (idle) value. Decoupling and gate
    coupling are at different coupler fluxes because q1's frequency differs.
    """
    w1 = p.omega_1_gate if omega_1 is None else omega_1
    wc = coupler_freq(phi_c, p.omega_c_max)
    d1 = w1 - wc
    d2 = p.omega_2 - wc
    return p.g12 + p.gg * (1.0 / d1 + 1.0 / d2)  # GHz


def zeta_idle(phi_c: float, p: CouplerParams) -> float:
    """Static parasitic ZZ (GHz) at idle (q1 at sweet spot) vs coupler flux.

    Tied to the idle effective coupling (``zeta ~ 2 J_idle^2 [1/(d12-a2) -
    1/(d12+a1)]``) so it nulls at the decoupling flux (where ``J_idle=0``) and
    grows at the stale idle — the agent must park the coupler where BOTH vanish.
    """
    j = j_eff(phi_c, p, omega_1=p.omega_1_idle)  # idle J_eff
    delta12 = p.omega_1_idle - p.omega_2
    a1, a2 = -0.30, -0.33
    zz_direct = 2.0 * j * j * (1.0 / (delta12 - a2) - 1.0 / (delta12 + a1))
    return p.zeta_scale * zz_direct  # GHz


def flux_arc(phi: float, omega_max: float = 4.5, asym: float = 0.0) -> float:
    """Transmon frequency on a (slightly asymmetric) SQUID arc.

    ``omega(phi) = omega_max * sqrt(|cos(pi*phi*(1+asym*sign(phi)))|)``. Quadratic
    near the sweet spot (required for the cryoscope). ``asym`` (~1%) slightly
    breaks the ``omega(phi)=omega(-phi)`` symmetry the 1/f immunity relies on.
    """
    eff = phi * (1.0 + asym * (1.0 if phi >= 0 else -1.0))
    c = abs(math.cos(math.pi * eff))
    return omega_max * math.sqrt(max(0.0, c))


def flux_to_freq(flux: np.ndarray, omega_max: float = 4.5, asym: float = 0.0) -> np.ndarray:
    return np.array([flux_arc(float(f), omega_max, asym) for f in flux])


def freq_to_flux(omega: float, omega_max: float = 4.5) -> float:
    """Invert the symmetric arc (positive branch) to a flux magnitude in Phi_0."""
    c = min(1.0, max(0.0, (omega / omega_max) ** 2))
    return math.acos(c) / math.pi


@dataclass(frozen=True)
class TransmonPairParams:
    omega_1_max: float = 4.5
    omega_2: float = 4.0
    alpha_1: float = -0.30
    alpha_2: float = -0.33
    arc_asym: float = 0.01  # q1 arc asymmetry (~1%)
    t1_us: float = 45.0
    t2_us: float = 32.0


def _hamiltonian(omega_1: float, j: float, p: TransmonPairParams) -> np.ndarray:
    h = omega_1 * _N1 + p.omega_2 * _N2
    h = h + 0.5 * p.alpha_1 * (_N1 @ (_N1 - _EYE)) + 0.5 * p.alpha_2 * (_N2 @ (_N2 - _EYE))
    h = h + j * (_AD1 @ _A2 + _A1 @ _AD2)
    return h


def cz_unitary(
    realized_flux_q1: np.ndarray, j: float, dt_ns: float, p: TransmonPairParams
) -> np.ndarray:
    """Coherent 16-dim propagator for a q1 flux trajectory at fixed exchange ``j``.

    ``realized_flux_q1`` is the ACTUAL flux at the qubit (after the control stack);
    ``j = J_eff(Phi_c_gate)`` is held constant during the gate.
    """
    omega1 = flux_to_freq(realized_flux_q1, p.omega_1_max, p.arc_asym)
    u = _EYE.copy()
    for w1 in omega1:
        u = expm(-1j * _TWO_PI * _hamiltonian(float(w1), j, p) * dt_ns) @ u
    return u


def conditional_phase_and_leakage(u: np.ndarray) -> tuple[float, float, float, float]:
    """Return (phi_2Q in [0,2pi), L1 input-averaged leakage, phi_01, phi_10).

    ``phi_2Q`` is computed from the diagonal phases; ``phi_01``/``phi_10`` are the
    single-qubit phases removed by virtual-Z; ``L1`` is the computational-subspace
    population lost to leakage, averaged over the four computational inputs.
    """

    def ang(s: tuple[int, int]) -> float:
        return float(np.angle(u[IDX[s], IDX[s]]))

    p00, p01, p10, p11 = ang((0, 0)), ang((0, 1)), ang((1, 0)), ang((1, 1))
    phi_2q = (p00 - p01 - p10 + p11) % _TWO_PI
    leak = 0.0
    for s in COMP:
        v = np.zeros(_LEVELS * _LEVELS, dtype=complex)
        v[IDX[s]] = 1.0
        out = u @ v
        leak += 1.0 - sum(abs(out[IDX[c]]) ** 2 for c in COMP)
    return phi_2q, leak / 4.0, (p01 - p00) % _TWO_PI, (p10 - p00) % _TWO_PI


def cz_gate_infidelity(u: np.ndarray, phi_01: float, phi_10: float) -> float:
    """Average gate infidelity of the computational 4x4 block vs CZ=diag(1,1,1,-1),
    after removing the reported single-qubit virtual-Z phases.

    The corrected unitary is ``VZ^dag U_comp`` with the virtual-Z frame
    ``VZ = diag(1, e^{i phi_01}, e^{i phi_10}, e^{i(phi_01+phi_10)})`` on the basis
    ``(00,01,10,11)``. ``1 - F_avg`` with
    ``F_avg = (|Tr(M)|^2 + Tr(M M^dag)) / (d(d+1))``, d=4, ``M = CZ^dag VZ^dag U_comp``.
    """
    d = 4
    u_comp = np.array([[u[IDX[a], IDX[b]] for b in COMP] for a in COMP], dtype=complex)
    vz = np.diag([1.0, np.exp(1j * phi_01), np.exp(1j * phi_10), np.exp(1j * (phi_01 + phi_10))])
    cz = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)
    m = cz.conj().T @ (vz.conj().T @ u_comp)
    f = (abs(np.trace(m)) ** 2 + np.real(np.trace(m @ m.conj().T))) / (d * (d + 1))
    if not math.isfinite(f):
        # A non-finite reported virtual-Z poisons the trace, and
        # ``max(0.0, 1.0 - nan)`` returns 0.0, a perfect gate. No fidelity demonstrated means infidelity 1, for any caller.
        return 1.0
    return float(max(0.0, 1.0 - f))


def decoherence_infidelity(duration_ns: float, p: TransmonPairParams) -> float:
    """First-order incoherent infidelity floor from T1/T2 over the gate window."""
    t1 = p.t1_us * 1e3
    t2 = p.t2_us * 1e3
    # ~ average of relaxation + dephasing over the two qubits, first order.
    return float(0.5 * duration_ns * (1.0 / t1 + 1.0 / t2))


def reference_netzero_flux(
    t_ramp_ns: float,
    t_hold_ns: float,
    delta_hold_ghz: float,
    dt_ns: float,
    p: TransmonPairParams,
    coupler: CouplerParams,
) -> np.ndarray:
    """Intended net-zero q1 flux trajectory for the reference CZ.

    Two mirror halves: a smooth ramp to a hold at ``omega_cross + delta_hold`` and
    back, with the flux sign flipped in the second half (net-zero; the even arc
    keeps omega identical, giving the 1/f spin-echo immunity).
    """
    w_cross = coupler.omega_1_gate
    w_hold = w_cross + delta_hold_ghz
    n_half = int(round((2 * t_ramp_ns + t_hold_ns) / dt_ns))
    t_total = 2 * t_ramp_ns + t_hold_ns
    half = np.zeros(n_half)
    for i in range(n_half):
        t = (i + 0.5) * dt_ns
        if t < t_ramp_ns:
            e = 0.5 * (1 - math.cos(math.pi * t / t_ramp_ns))
        elif t > t_total - t_ramp_ns:
            e = 0.5 * (1 - math.cos(math.pi * (t_total - t) / t_ramp_ns))
        else:
            e = 1.0
        # shape the FREQUENCY trajectory (smooth ramp w1max -> w_hold), then invert
        # the arc to flux: a fast-adiabatic agent controls the realized frequency.
        omega = p.omega_1_max + (w_hold - p.omega_1_max) * e
        half[i] = freq_to_flux(omega, p.omega_1_max)
    return np.concatenate([half, -half])  # net-zero


__all__ = [
    "COMP",
    "CouplerParams",
    "IDX",
    "TransmonPairParams",
    "conditional_phase_and_leakage",
    "coupler_freq",
    "cz_gate_infidelity",
    "cz_unitary",
    "decoherence_infidelity",
    "flux_arc",
    "flux_to_freq",
    "j_eff",
    "reference_netzero_flux",
    "zeta_idle",
]
