"""Pure-numpy bosonic-cavity + dispersive-ancilla physics.

Single source of truth for the ``bosonic_cavity_qec`` qtype. The qsim engine uses
it for the agent's live runs; the reference solver and the (self-contained) Harbor
verifier use the SAME math so there is no engine-vs-verifier physics drift. Keep it
numpy-only and dependency-light (scipy only for the displacement ``expm``) so the
verifier can vendor a copy verbatim.

Conventions
-----------
- A storage cavity (Fock space truncated at ``N = n_max``) is dispersively coupled
  to a 2-level transmon **ancilla** (``g=0``, ``e=1``). The **joint** Hilbert space
  is ordered ``kron(cavity, ancilla)`` so the basis index of ``|n>_cav |a>_anc`` is
  ``n*2 + a``.
- Universal cavity control is ``{displace, snap}`` (Krastanov/Heeres); the
  dispersive interaction ``H = -chi * a_dag a (x) |e><e|`` is the parity-mapping
  primitive. ``chi`` is an **angular** rate ``2*pi*chi_mhz`` (rad/us); a free
  evolution of duration ``t`` (us) imprints phase ``exp(i*chi*t*n)`` on the ``|e>``
  branch, so the parity wait ``t = pi/chi = 1/(2*chi_mhz)`` gives ``(-1)^n``.
- Single-photon loss is the bosonic amplitude-damping channel with
  ``gamma = 1 - exp(-dt/tau_s)``; its no-jump Kraus operator ``E_0`` carries the
  deterministic back-action automatically (no separate handling needed).

Binomial "kitten" code (arXiv:1805.09072 Eq. 1 & 2)::

    |0_L> = (|0> + |4>)/sqrt(2)        |1_L> = |2>

A single photon loss maps ``|0_L> -> |3>`` and ``|1_L> -> |1>`` (odd parity);
``recovery_unitary`` restores ``|3> -> |0_L>``, ``|1> -> |1_L>`` exactly.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy.linalg import expm

_SQRT2 = math.sqrt(2.0)

# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


def cavity_operators(n_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(a, a_dag, n_op)`` for an ``n_max``-level cavity mode."""
    a = np.zeros((n_max, n_max), dtype=complex)
    for n in range(1, n_max):
        a[n - 1, n] = math.sqrt(n)
    a_dag = a.conj().T
    n_op = np.diag(np.arange(n_max, dtype=float)).astype(complex)
    return a, a_dag, n_op


def lift_cavity(op_cav: np.ndarray, n_max: int) -> np.ndarray:
    """Embed a cavity operator into the joint (cavity (x) ancilla) space."""
    return np.kron(op_cav, np.eye(2, dtype=complex))


def lift_ancilla(op_anc: np.ndarray, n_max: int) -> np.ndarray:
    """Embed a 2x2 ancilla operator into the joint space."""
    return np.kron(np.eye(n_max, dtype=complex), op_anc)


# ---------------------------------------------------------------------------
# Ideal control unitaries (cavity-only unless noted; lift with lift_cavity)
# ---------------------------------------------------------------------------


def displace_unitary(alpha: complex, n_max: int) -> np.ndarray:
    """Cavity displacement ``D(alpha) = exp(alpha a_dag - alpha* a)`` (N x N)."""
    a, a_dag, _ = cavity_operators(n_max)
    return expm(alpha * a_dag - np.conjugate(alpha) * a)


def snap_unitary(thetas: list[float] | np.ndarray, n_max: int) -> np.ndarray:
    """SNAP gate ``diag(exp(i*theta_n))`` (N x N). ``thetas`` shorter than
    ``n_max`` is zero-padded (no phase on higher Fock levels)."""
    phases = np.zeros(n_max, dtype=float)
    th = np.asarray(thetas, dtype=float)
    k = min(len(th), n_max)
    phases[:k] = th[:k]
    return np.diag(np.exp(1j * phases))


def ancilla_rotation(theta: float, phi: float) -> np.ndarray:
    """Single-qubit ancilla rotation ``R(theta, phi) = exp(-i*theta/2 *
    (cos phi * X + sin phi * Y))`` (2 x 2)."""
    nx, ny = math.cos(phi), math.sin(phi)
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return np.array([[c, -1j * s * (nx - 1j * ny)], [-1j * s * (nx + 1j * ny), c]], dtype=complex)


def dispersive_phase_unitary(n_max: int, chi_t: float) -> np.ndarray:
    """Joint dispersive evolution ``exp(i*chi_t*n (x) |e><e|)`` (2N x 2N, diagonal).

    ``chi_t`` is the accumulated phase per photon on the ``|e>`` branch
    (``chi_t = 2*pi*chi_mhz*t_us``). The ``|g>`` branch is untouched; ``chi_t = pi``
    realizes the parity kick ``(-1)^n``.
    """
    diag = np.ones(2 * n_max, dtype=complex)
    for n in range(n_max):
        diag[n * 2 + 1] = np.exp(1j * chi_t * n)  # ancilla = e
    return np.diag(diag)


def ancilla_projectors(n_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Joint projectors ``(P_g, P_e)`` onto the ancilla ground / excited subspace."""
    pg = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    pe = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=complex)
    return lift_ancilla(pg, n_max), lift_ancilla(pe, n_max)


def ancilla_reset_kraus(n_max: int) -> list[np.ndarray]:
    """Kraus ops that reset the ancilla to ``|g>`` (cavity untouched)."""
    g_from_g = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)  # |g><g|
    g_from_e = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)  # |g><e|
    return [lift_ancilla(g_from_g, n_max), lift_ancilla(g_from_e, n_max)]


# ---------------------------------------------------------------------------
# Open-system Kraus channels (joint space)
# ---------------------------------------------------------------------------


def amplitude_damping_kraus(n_max: int, gamma: float) -> list[np.ndarray]:
    """Bosonic single-photon-loss Kraus set on the cavity (lifted to joint space).

    ``E_k = sum_{n>=k} sqrt(C(n,k) gamma^k (1-gamma)^(n-k)) |n-k><n|`` for
    ``k = 0 .. n_max-1``. ``E_0`` is the no-jump (deterministic back-action) op.
    """
    gamma = float(np.clip(gamma, 0.0, 1.0))
    one_minus = 1.0 - gamma
    kraus: list[np.ndarray] = []
    for k in range(n_max):
        e = np.zeros((n_max, n_max), dtype=complex)
        for n in range(k, n_max):
            coeff = math.comb(n, k) * (gamma**k) * (one_minus ** (n - k))
            if coeff > 0:
                e[n - k, n] = math.sqrt(coeff)
        if np.any(e):
            kraus.append(lift_cavity(e, n_max))
    return kraus


def cavity_dephasing_mask(n_max: int, t_us: float, tphi_us: float) -> np.ndarray:
    """Joint phase-damping mask for cavity pure dephasing over ``t_us``.

    Exact and trace-preserving for any time: number-basis coherences decay as
    ``rho_{nm} *= exp(-(n-m)^2 * t / tphi)`` (diagonal untouched). Returned as a
    ``(2 n_max) x (2 n_max)`` elementwise multiplier on the joint density matrix
    (the same cavity factor on both ancilla blocks).
    """
    if not math.isfinite(tphi_us) or tphi_us <= 0 or t_us <= 0:
        return np.ones((2 * n_max, 2 * n_max), dtype=float)
    n = np.arange(n_max)
    diff2 = (n[:, None] - n[None, :]) ** 2
    cav = np.exp(-diff2 * (t_us / tphi_us))
    # expand to joint (cavity (x) ancilla): coherence n*2+a , m*2+b uses cav[n,m]
    mask = np.ones((2 * n_max, 2 * n_max), dtype=float)
    for a in range(2):
        for b in range(2):
            mask[a::2, b::2] = cav
    return mask


def cavity_free_evolution_propagators(
    n_max: int,
    t_us: float,
    kappa_per_us: float,
    tphi_us: float,
    delta_mhz: float,
    kerr_mhz: float,
) -> list[np.ndarray]:
    """Exact cavity free-evolution propagators, one per coherence chain ``d = n - m``.

    A wait of ``t_us`` evolves the storage mode under its OWN Hamiltonian in the
    control frame, ``H = 2*pi * sum_n E_n |n><n|`` with
    ``E_n = delta * n + kerr/2 * n (n-1)`` (``delta`` = detuning of the mode from
    the control frame, ``kerr`` = self-Kerr; both in cycles/us like ``chi``),
    together with single-photon loss ``kappa`` and pure dephasing ``1/tphi``::

        d/dt rho_{n,m} = -i 2pi (E_n - E_m) rho_{n,m} - kappa/2 (n + m) rho_{n,m}
                         - (n-m)^2 / tphi rho_{n,m} + kappa sqrt((n+1)(m+1)) rho_{n+1,m+1}

    Loss couples the chain ``{rho_{n,n-d}}`` only to itself, so chain ``d`` is one
    ``(n_max-d) x (n_max-d)`` matrix exponential. At ``delta = kerr = 0`` this
    equals ``amplitude_damping_kraus`` composed with ``cavity_dephasing_mask`` to
    machine precision (the linear detuning commutes with loss too); the Kerr term
    does not commute with loss -- a photon lost at an unknown time randomises the
    phase -- which is why the closed-form Kraus route does not extend to it.
    ``props[d]`` acts on the vector ``(rho_{d,0}, rho_{d+1,1}, ...)``; the ``m > n``
    chains use the complex conjugate (Hermiticity).
    """
    n = np.arange(n_max, dtype=float)
    omega = 2.0 * math.pi * (delta_mhz * n + 0.5 * kerr_mhz * n * (n - 1.0))  # rad/us
    deph = 0.0 if (not math.isfinite(tphi_us) or tphi_us <= 0) else 1.0 / tphi_us
    props: list[np.ndarray] = []
    for d in range(n_max):
        size = n_max - d
        gen = np.zeros((size, size), dtype=complex)
        for i in range(size):
            nn = i + d
            gen[i, i] = (
                -1j * (omega[nn] - omega[nn - d]) - 0.5 * kappa_per_us * (2 * nn - d) - deph * d * d
            )
            if i + 1 < size:
                gen[i, i + 1] = kappa_per_us * math.sqrt((nn + 1) * (nn + 1 - d))
        props.append(expm(gen * t_us))
    return props


def apply_cavity_free_evolution(rho: np.ndarray, props: list[np.ndarray], n_max: int) -> np.ndarray:
    """Apply ``cavity_free_evolution_propagators`` output to a joint
    ``(2 n_max) x (2 n_max)`` density matrix: every ancilla block ``(a, b)`` carries
    the same cavity evolution (the ancilla channels commute with it)."""
    r4 = rho.reshape(n_max, 2, n_max, 2)
    out = np.zeros_like(r4)
    for a in range(2):
        for b in range(2):
            block = r4[:, a, :, b]
            new = np.zeros_like(block)
            for d in range(n_max):
                lo = np.arange(d, n_max)
                hi = np.arange(0, n_max - d)
                new[lo, hi] = props[d] @ block[lo, hi]
                if d > 0:
                    new[hi, lo] = props[d].conj() @ block[hi, lo]
            out[:, a, :, b] = new
    return out.reshape(2 * n_max, 2 * n_max)


def ancilla_amplitude_damping_kraus(n_max: int, p: float) -> list[np.ndarray]:
    """Ancilla ``T1`` relaxation (``|e> -> |g>``) with jump probability ``p``."""
    p = float(np.clip(p, 0.0, 1.0))
    a0 = np.array([[1.0, 0.0], [0.0, math.sqrt(1.0 - p)]], dtype=complex)
    a1 = np.array([[0.0, math.sqrt(p)], [0.0, 0.0]], dtype=complex)
    return [lift_ancilla(a0, n_max), lift_ancilla(a1, n_max)]


def ancilla_dephasing_kraus(n_max: int, lam: float) -> list[np.ndarray]:
    """Ancilla pure dephasing with strength ``lam`` (phase-flip channel)."""
    lam = float(np.clip(lam, 0.0, 1.0))
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    b0 = math.sqrt(1.0 - lam / 2.0) * np.eye(2, dtype=complex)
    b1 = math.sqrt(lam / 2.0) * z
    return [lift_ancilla(b0, n_max), lift_ancilla(b1, n_max)]


def gaussian_displacement_kraus(n_max: int, sigma: float, nodes: int = 5) -> list[np.ndarray]:
    """Random-displacement (Gaussian diffusion) channel as a weighted mixture.

    Isotropic Gaussian displacement noise of per-quadrature standard deviation
    ``sigma`` (alpha units): ``rho -> E_{x,y~N(0,sigma^2)} D(x+iy) rho D^dag``.
    Discretized with a Gauss-Hermite product grid (exact for polynomial moments
    up to order ``2*nodes-1``); returned as ``sqrt(weight)``-scaled displacement
    operators lifted to the joint space so ``apply_kraus`` applies the channel.
    A mixture of unitaries is exactly CPTP and does not grow the branch tree.
    """
    if sigma <= 0.0:
        return []
    h, w = np.polynomial.hermite.hermgauss(nodes)
    xs = _SQRT2 * sigma * h
    ws = w / math.sqrt(math.pi)
    kraus: list[np.ndarray] = []
    for xi, wi in zip(xs, ws, strict=True):
        for yj, wj in zip(xs, ws, strict=True):
            weight = float(wi * wj)
            if weight < 1e-12:
                continue
            d = displace_unitary(complex(xi, yj), n_max)
            kraus.append(math.sqrt(weight) * lift_cavity(d, n_max))
    return kraus


def apply_kraus(rho: np.ndarray, kraus: list[np.ndarray]) -> np.ndarray:
    """Apply a Kraus channel ``rho -> sum_k K_k rho K_k^dag`` (joint space)."""
    out = np.zeros_like(rho)
    for k in kraus:
        out += k @ rho @ k.conj().T
    return out


def apply_unitary(rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Apply a unitary ``rho -> U rho U^dag``."""
    return u @ rho @ u.conj().T


# ---------------------------------------------------------------------------
# Codewords + recovery
# ---------------------------------------------------------------------------


def _ket(n_max: int, idx: int) -> np.ndarray:
    v = np.zeros(n_max, dtype=complex)
    v[idx] = 1.0
    return v


def binomial_codewords(n_max: int) -> tuple[np.ndarray, np.ndarray]:
    """``(|0_L>, |1_L>)`` for the binomial kitten code (needs ``n_max >= 5``)."""
    if n_max < 5:
        raise ValueError("binomial kitten code needs n_max >= 5")
    zero_l = (_ket(n_max, 0) + _ket(n_max, 4)) / _SQRT2
    one_l = _ket(n_max, 2)
    return zero_l, one_l


def fock_codewords(n_max: int) -> tuple[np.ndarray, np.ndarray]:
    """``(|0_L>, |1_L>) = (|0>, |1>)`` for the trivial Fock encoding."""
    return _ket(n_max, 0), _ket(n_max, 1)


def recovery_unitary(n_max: int) -> np.ndarray:
    """Exact binomial single-loss recovery ``U_R`` (cavity-only, N x N).

    Acts as the orthogonal swaps ``|1> <-> |2>`` and ``|3> <-> |0_L>`` (identity
    elsewhere), so ``U_R(alpha|3> + beta|1>) = alpha|0_L> + beta|1_L>`` restores the
    logical state after a detected single photon loss.
    """
    u = np.eye(n_max, dtype=complex)
    # swap |1> <-> |2>
    u[1, 1] = u[2, 2] = 0.0
    u[2, 1] = 1.0
    u[1, 2] = 1.0
    # swap |3> <-> |0_L> = (|0>+|4>)/sqrt2 inside span{|0>,|3>,|4>}
    zero_l = (_ket(n_max, 0) + _ket(n_max, 4)) / _SQRT2
    zero_l_perp = (_ket(n_max, 0) - _ket(n_max, 4)) / _SQRT2
    three = _ket(n_max, 3)
    # remove the identity action on |0>,|3>,|4> and install the swap
    for i in (0, 3, 4):
        u[:, i] = 0.0
        u[i, :] = 0.0
    block = np.outer(zero_l, three) + np.outer(three, zero_l) + np.outer(zero_l_perp, zero_l_perp)
    u = u + block
    return u


def parity_projectors(n_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Cavity even/odd photon-number-parity projectors (lifted to joint space)."""
    even = np.zeros((n_max, n_max), dtype=complex)
    odd = np.zeros((n_max, n_max), dtype=complex)
    for n in range(n_max):
        if n % 2 == 0:
            even[n, n] = 1.0
        else:
            odd[n, n] = 1.0
    return lift_cavity(even, n_max), lift_cavity(odd, n_max)


# ---------------------------------------------------------------------------
# Logical readout + process fidelity
# ---------------------------------------------------------------------------


def axial_logical_states(
    codewords: tuple[np.ndarray, np.ndarray],
) -> dict[str, np.ndarray]:
    """Six axial logical input cavity density matrices (+-x,+-y,+-z)."""
    z0, z1 = codewords
    r = 1.0 / _SQRT2
    kets = {
        "+z": z0,
        "-z": z1,
        "+x": r * (z0 + z1),
        "-x": r * (z0 - z1),
        "+y": r * (z0 + 1j * z1),
        "-y": r * (z0 - 1j * z1),
    }
    return {k: np.outer(v, v.conj()) for k, v in kets.items()}


def logical_block(rho_cav: np.ndarray, codewords: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Project a cavity density matrix onto the 2x2 logical code subspace.

    The (sub-unit) trace of the block is the in-code population; leakage out of the
    code space reduces it and counts as logical error.
    """
    z0, z1 = codewords
    basis = np.stack([z0, z1], axis=1)  # N x 2
    return basis.conj().T @ rho_cav @ basis


def logical_pauli_expectations(
    rho_cav: np.ndarray, codewords: tuple[np.ndarray, np.ndarray]
) -> tuple[float, float, float, float]:
    """Return ``(<X>, <Y>, <Z>, in_code_trace)`` for a projected cavity state."""
    block = logical_block(rho_cav, codewords)
    tr = float(np.real(np.trace(block)))
    ex = 2.0 * float(np.real(block[0, 1]))
    ey = 2.0 * float(np.imag(block[1, 0]))
    ez = float(np.real(block[0, 0] - block[1, 1]))
    return ex, ey, ez, tr


def entanglement_fidelity(
    channel: Callable[[np.ndarray], np.ndarray],
    codewords: tuple[np.ndarray, np.ndarray],
) -> float:
    """Exact logical-memory process (entanglement) fidelity to the identity.

    ``channel`` maps a *cavity* density matrix to a cavity density matrix (the full
    encode-free logical-memory channel: loss + QEC). ``F = (1/4) sum_{i,j in {0,1}}
    <i_L| channel(|i_L><j_L|) |j_L>`` for the codeword basis.
    """
    z = codewords
    f = 0.0 + 0.0j
    for i in range(2):
        for j in range(2):
            op = np.outer(z[i], z[j].conj())
            out = channel(op)
            f += z[i].conj() @ out @ z[j]
    return float(np.real(f) / 4.0)


def process_fidelity_from_axial(
    expectations: dict[str, tuple[float, float, float, float]],
) -> float:
    """Process (entanglement) fidelity from six-axial logical tomography.

    ``expectations[axis] = (<X>, <Y>, <Z>, in_code_trace)`` measured for logical
    input ``axis``. This is the exact Pauli-transfer-matrix trace identity for a
    (possibly leaky) logical channel::

        R_PP = (<P>_{+P} - <P>_{-P}) / 2      (P in {X, Y, Z})
        R_II = mean in-code survival over the six axial inputs
        F    = (R_II + R_xx + R_yy + R_zz) / 4

    The ``R_II`` term penalizes leakage out of the code space (population in the
    odd-parity error states), so the binomial code is *not* over-credited; without
    it the metric would mis-order the high-photon binomial code above Fock.
    """
    rxx = (expectations["+x"][0] - expectations["-x"][0]) / 2.0
    ryy = (expectations["+y"][1] - expectations["-y"][1]) / 2.0
    rzz = (expectations["+z"][2] - expectations["-z"][2]) / 2.0
    r_ii = sum(expectations[ax][3] for ax in expectations) / len(expectations)
    return (r_ii + rxx + ryy + rzz) / 4.0


__all__ = [
    "amplitude_damping_kraus",
    "ancilla_amplitude_damping_kraus",
    "ancilla_dephasing_kraus",
    "ancilla_projectors",
    "ancilla_reset_kraus",
    "apply_kraus",
    "apply_unitary",
    "axial_logical_states",
    "binomial_codewords",
    "cavity_dephasing_mask",
    "cavity_operators",
    "dispersive_phase_unitary",
    "displace_unitary",
    "entanglement_fidelity",
    "fock_codewords",
    "gaussian_displacement_kraus",
    "lift_ancilla",
    "lift_cavity",
    "logical_block",
    "logical_pauli_expectations",
    "parity_projectors",
    "process_fidelity_from_axial",
    "recovery_unitary",
    "snap_unitary",
]
