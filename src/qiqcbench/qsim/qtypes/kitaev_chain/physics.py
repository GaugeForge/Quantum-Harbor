"""Free-fermion (Bogoliubov-de Gennes) core for the Kitaev-chain qtype.

A Kitaev chain of ``M`` sites is a quadratic (free-fermion) Hamiltonian

    H = sum_i mu_i c_i^dag c_i
        + sum_b ( t_b c_b^dag c_{b+1} + Delta_b c_b^dag c_{b+1}^dag + h.c. )

with on-site potentials ``mu_i`` (i = 0..M-1) and per-bond complex hopping
``t_b`` and pairing ``Delta_b = |Delta_b| e^{i phi_b}`` (b = 0..M-2). Because the
model is quadratic, every quantity we need is exact and cheap: the
single-particle spectrum, the Majorana-edge-mode (parity) gap, the gap of any
contiguous sub-chain, and the minimal-chain charge-readout signals all come
from a ``2M x 2M`` Bogoliubov-de Gennes (BdG) matrix.

This module is the single source of truth shared by the engine and the
verifier. It is pure numpy and never touches the agent surface.

BdG convention follows Bordin et al. (arXiv:2402.19382, Supp.): Nambu basis
``Psi = (c_0..c_{M-1}, c_0^dag..c_{M-1}^dag)`` and

    h_BdG = [[ T,      P   ],
             [ P^dag, -T^* ]]

with ``T`` Hermitian (``T_ii = mu_i``, ``T_{b,b+1} = t_b``) and ``P``
antisymmetric (``P_{b,b+1} = -Delta_b``). Eigenvalues come in +/- E pairs; the
lowest positive eigenvalue is the lowest Bogoliubov quasiparticle, and the
even/odd ground-state energy splitting (the Majorana parity gap) is twice it.
"""

from __future__ import annotations

import numpy as np

_SQRT_EPS = 1e-12


def build_bdg(
    mu: np.ndarray,
    t: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    """Build the ``2M x 2M`` Hermitian BdG matrix.

    Parameters
    ----------
    mu : (M,) real on-site potentials.
    t  : (M-1,) complex bond hoppings ``t_b``.
    delta : (M-1,) complex bond pairings ``Delta_b = |Delta_b| e^{i phi_b}``.
    """
    mu = np.asarray(mu, dtype=float)
    t = np.asarray(t, dtype=complex)
    delta = np.asarray(delta, dtype=complex)
    m = mu.shape[0]
    if t.shape[0] != m - 1 or delta.shape[0] != m - 1:
        raise ValueError(f"expected {m - 1} bonds, got t={t.shape}, delta={delta.shape}")

    tt = np.zeros((m, m), dtype=complex)  # single-particle block T (Hermitian)
    pp = np.zeros((m, m), dtype=complex)  # pairing block P (antisymmetric)
    for i in range(m):
        tt[i, i] = mu[i]
    for b in range(m - 1):
        tt[b, b + 1] = t[b]
        tt[b + 1, b] = np.conj(t[b])
        pp[b, b + 1] = -delta[b]
        pp[b + 1, b] = delta[b]

    h = np.zeros((2 * m, 2 * m), dtype=complex)
    h[:m, :m] = tt
    h[:m, m:] = pp
    h[m:, :m] = pp.conj().T
    h[m:, m:] = -tt.conj()
    # Hermitize against floating round-off.
    return (h + h.conj().T) / 2.0


def _positive_spectrum(h: np.ndarray) -> np.ndarray:
    """The M physical (non-negative) BdG quasiparticle energies, ascending.

    ``h`` is ``2M x 2M`` with particle-hole symmetry, so its eigenvalues come in
    ``+/- E`` pairs. The physical single-particle spectrum is the upper half:
    ``pos[0]`` is the lowest quasiparticle (the near-zero Majorana edge mode at a
    topological sweet spot) and ``pos[1]`` is the lowest *bulk* excitation.
    """
    vals = np.sort(np.linalg.eigvalsh(h))  # ascending, length 2M
    m = vals.shape[0] // 2
    return vals[m:]


def edge_splitting(mu: np.ndarray, t: np.ndarray, delta: np.ndarray) -> float:
    """Even/odd ground-state energy splitting (Majorana parity gap), in input units.

    Equals ``2 * (lowest positive BdG eigenvalue)``. At a clean sweet spot this
    is exponentially small in the chain length; any protection-breaking defect
    lifts it.
    """
    return float(2.0 * abs(_positive_spectrum(build_bdg(mu, t, delta))[0]))


def bulk_gap(mu: np.ndarray, t: np.ndarray, delta: np.ndarray) -> float:
    """Gap to the lowest *bulk* excitation (the second positive eigenvalue).

    At a clean sweet spot this is ``~2|t|``; a phase defect suppresses it (the
    gap-closing physics of Bordin et al.), giving a non-local handle a single
    amplitude-balanced bond does not betray locally.
    """
    pos = _positive_spectrum(build_bdg(mu, t, delta))
    return float(pos[1]) if pos.shape[0] > 1 else float(pos[0])


def subchain_bulk_gap(
    mu: np.ndarray,
    t: np.ndarray,
    delta: np.ndarray,
    a: int,
    b: int,
) -> float:
    """Bulk gap (lowest bulk excitation) of the contiguous sites ``a..b``.

    The sub-chain keeps the actual couplings on its internal bonds. A phase
    defect suppresses the bulk gap of any sub-chain (of length >= 3, since a
    2-site phase is gaugeable) that *spans* the defect bond, while sub-chains
    excluding it stay at the full ``~2|t|`` -- this localizes a phase defect
    non-locally. Returns ``inf`` for a single-site window (no excitation).
    """
    if not (0 <= a <= b < mu.shape[0]):
        raise ValueError("invalid sub-chain endpoints")
    sub_mu = np.asarray(mu, dtype=float)[a : b + 1]
    sub_t = np.asarray(t, dtype=complex)[a:b]
    sub_delta = np.asarray(delta, dtype=complex)[a:b]
    if sub_mu.shape[0] < 2:
        return float("inf")
    pos = _positive_spectrum(build_bdg(sub_mu, sub_t, sub_delta))
    return float(pos[1]) if pos.shape[0] > 1 else float(pos[0])


def bond_imbalance(t_b: complex, delta_b: complex) -> float:
    """Amplitude imbalance ``| |t_b| - |Delta_b| |`` on a single bond."""
    return float(abs(abs(t_b) - abs(delta_b)))


def bond_tilt(t_b: complex, delta_b: complex) -> float:
    """Indistinguishability-line tilt ``(|Delta_b| - |t_b|) / (|Delta_b| + |t_b|)``.

    Zero only at the local sweet spot ``|t_b| = |Delta_b|`` (van Loo et al.,
    arXiv:2507.01606, Fig. 4); its magnitude/sign read off the amplitude
    imbalance and which of CAR/ECT dominates.
    """
    a = abs(t_b)
    d = abs(delta_b)
    denom = a + d
    return float((d - a) / denom) if denom > _SQRT_EPS else 0.0


def fit_power_law(mu_values: np.ndarray, splittings: np.ndarray) -> tuple[float, float]:
    """Fit ``log dE = p * log mu + c`` over the given common-mode detunings.

    Returns ``(p, c)``. Points with non-positive splitting are dropped (they are
    below numerical resolution); at least two valid points are required.
    """
    mu_values = np.asarray(mu_values, dtype=float)
    splittings = np.asarray(splittings, dtype=float)
    ok = (mu_values > 0) & (splittings > 0) & np.isfinite(splittings)
    if ok.sum() < 2:
        raise ValueError("need >=2 positive (mu, splitting) points to fit an exponent")
    x = np.log(mu_values[ok])
    y = np.log(splittings[ok])
    p, c = np.polyfit(x, y, 1)
    return float(p), float(c)


# ---------------------------------------------------------------------------
# Minimal (2-dot) charge-readout window (van Loo et al., arXiv:2507.01606)
#
# A long chain is base-band pulsed so only one bond's two dots are resonant.
# That 2-dot window has effective couplings (t, Delta) = (|t_b|, |Delta_b|).
# Scanning the two dot detunings (mu_LD, mu_RD) -> (eps, delta) with
#   eps   = (mu_LD - mu_RD)/2   (interdot detuning axis)
#   delta = (mu_LD + mu_RD)/2   (common-mode axis, the RF drive axis)
# the even branch (|00>,|11> via CAR) carries quantum capacitance while the odd
# branch (|01>,|10> via ECT) is flat; the local charge sensor only resolves
# parity off the indistinguishability line mu_LD = (Delta-t)/(Delta+t) mu_RD.
# ---------------------------------------------------------------------------


def quantum_capacitance_even(delta_cm: float, big_delta: float) -> float:
    """Even-branch quantum capacitance magnitude ``Delta^2 / (delta^2+Delta^2)^{3/2}``.

    Up to the (state-independent) lever-arm prefactor, which cancels in any
    parity *contrast*. Odd branch is identically zero.
    """
    d2 = float(big_delta) ** 2
    return d2 / (delta_cm**2 + d2) ** 1.5 if d2 > 0 else 0.0


def charge_sensor_qr_even(delta_cm: float, big_delta: float) -> float:
    """Average charge on the right dot in the even ground state."""
    big_delta = abs(big_delta)
    root = float(np.sqrt(delta_cm**2 + big_delta**2))
    if root + delta_cm < _SQRT_EPS:
        return 0.0
    return big_delta**2 / (2.0 * root) / (root + delta_cm)


def charge_sensor_qr_odd(eps: float, t: float) -> float:
    """Average charge on the right dot in the odd ground state."""
    t = abs(t)
    root = float(np.sqrt(eps**2 + t**2))
    if root + eps < _SQRT_EPS:
        return 0.0
    return t**2 / (2.0 * root) / (root + eps)


def local_charge_contrast(mu_ld: float, mu_rd: float, t: float, big_delta: float) -> float:
    """Local-sensor parity contrast ``|Q_R,odd - Q_R,even|`` at one grid point.

    Vanishes on the indistinguishability line, which is flat (mu_LD = 0) only at
    the sweet spot ``t = Delta``.
    """
    delta_cm = 0.5 * (mu_ld + mu_rd)
    eps = 0.5 * (mu_rd - mu_ld)  # see engine.run_charge_stability sign note
    return abs(charge_sensor_qr_odd(eps, t) - charge_sensor_qr_even(delta_cm, big_delta))


def assignment_error_from_snr(snr: float) -> float:
    """Single-shot parity assignment error ``[1 - erf(SNR/sqrt2)] / 2``."""
    from math import erf, sqrt

    snr = max(float(snr), 0.0)
    return 0.5 * (1.0 - erf(snr / sqrt(2.0)))


__all__ = [
    "build_bdg",
    "edge_splitting",
    "bulk_gap",
    "subchain_bulk_gap",
    "bond_imbalance",
    "bond_tilt",
    "fit_power_law",
    "quantum_capacitance_even",
    "charge_sensor_qr_even",
    "charge_sensor_qr_odd",
    "local_charge_contrast",
    "assignment_error_from_snr",
]
