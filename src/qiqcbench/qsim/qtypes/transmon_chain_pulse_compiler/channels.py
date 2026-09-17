"""Numpy-only channel algebra for the chain pulse-compiler qtype.

This module is the single source of truth for the chain-compiler physics. The
qsim engine uses it for the agent's live runs; the reference solver and the
(self-contained) Harbor verifier use the SAME math so there is no
engine-vs-verifier physics drift. Keep it numpy-only and dependency-light so the
verifier can vendor a copy verbatim (mirrors transmon_multilevel_pulse/physics.py).

Model
-----
The scored object is the SPAM-inclusive average-gate fidelity of the implemented
program against an ideal target unitary (``QFT_5``), at ``d = 2^n`` over the
computational subspace. The channel is built as a ``d^2 x d^2`` superoperator:

    E = E_readout  o  E_decoherence  o  E_gates  o  E_prep

and the process (entanglement) fidelity against a unitary ``U`` is the
superoperator overlap

    F_pro = (1/d^2) * Tr[ S_U^dag @ S_E ],   S_U = kron(conj(U), U).

Conventions
-----------
- Qubit 0 is the least-significant bit of the basis index (matches the digital
  engine and ``ideal_qft_unitary``).
- Vectorization is column-stacking (``order='F'``); for ``rho -> A rho B`` the
  superoperator is ``kron(B.T, A)``, so a unitary channel ``rho -> U rho U^dag``
  has superoperator ``kron(conj(U), U)``.
- Leakage to ``|2>`` is modeled as a trace-DECREASING CPTP map on the
  computational subspace (a single Kraus ``diag(...,sqrt(1-p))`` on the ``|11>``
  of the edge, no compensating Kraus): the leaked population leaves the
  computational subspace and counts DIRECTLY as infidelity (the F_pro prefactor
  is fixed at ``1/d^2`` — never renormalize by the surviving trace).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
_PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULIS_1Q = (_PAULI_X, _PAULI_Y, _PAULI_Z)


# ---------- index / embedding helpers (qubit 0 = LSB) ----------


def embed_operator(m_local: np.ndarray, qubits: list[int], n: int) -> np.ndarray:
    """Embed a k-qubit operator acting on ``qubits`` into the full n-qubit space.

    ``qubits[a]`` corresponds to bit ``a`` of the local basis index (LSB-first).
    Works for any matrix (unitary or Kraus). Spectator qubits act as identity.
    """
    k = len(qubits)
    if m_local.shape != (1 << k, 1 << k):
        raise ValueError(f"local shape {m_local.shape} != ({1 << k},{1 << k}) for {k} qubits")
    if any(q < 0 or q >= n for q in qubits):
        raise ValueError(f"qubit index out of range for n={n}: {qubits}")
    if len(set(qubits)) != k:
        raise ValueError(f"repeated qubit index in {qubits}")
    if k == n and qubits == list(range(n)):
        return m_local.astype(complex, copy=True)

    dim = 1 << n
    other = [q for q in range(n) if q not in qubits]
    out = np.zeros((dim, dim), dtype=complex)
    for combo in range(1 << len(other)):
        mask = 0
        for a, q in enumerate(other):
            mask |= ((combo >> a) & 1) << q
        for i_loc in range(1 << k):
            i_full = mask
            for a in range(k):
                i_full |= ((i_loc >> a) & 1) << qubits[a]
            for j_loc in range(1 << k):
                j_full = mask
                for a in range(k):
                    j_full |= ((j_loc >> a) & 1) << qubits[a]
                out[i_full, j_full] = m_local[i_loc, j_loc]
    return out


def _vec(rho: np.ndarray) -> np.ndarray:
    return rho.reshape(-1, order="F")


def _unvec(v: np.ndarray, d: int) -> np.ndarray:
    return v.reshape((d, d), order="F")


def unitary_superop(u: np.ndarray) -> np.ndarray:
    """Superoperator of ``rho -> U rho U^dag`` (column-stacking convention)."""
    return np.kron(u.conj(), u)


def kraus_superop(kraus: list[np.ndarray]) -> np.ndarray:
    """Superoperator of ``rho -> sum_k K_k rho K_k^dag``."""
    return sum(np.kron(k.conj(), k) for k in kraus)


# ---------- gate unitaries (computational subspace) ----------


def rz_unitary(theta: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]], dtype=complex)


def rx_unitary(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry_unitary(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def cphase_local(theta: float) -> np.ndarray:
    """Local 2-qubit controlled-phase ``diag(1,1,1,e^{i theta})`` (qa=bit0, qb=bit1)."""
    return np.diag([1.0, 1.0, 1.0, np.exp(1j * theta)]).astype(complex)


# ---------- noise channel superoperators ----------


def depol_1q_superop(q: int, p: float, n: int) -> np.ndarray:
    """Single-qubit depolarizing channel on qubit ``q`` at rate ``p``."""
    dim = 1 << n
    if p <= 0:
        return np.eye(dim * dim, dtype=complex)
    s = (1.0 - p) * np.eye(dim * dim, dtype=complex)
    for pauli in _PAULIS_1Q:
        e = embed_operator(pauli, [q], n)
        s = s + (p / 3.0) * unitary_superop(e)
    return s


def depol_2q_superop(a: int, b: int, p: float, n: int) -> np.ndarray:
    """Two-qubit depolarizing channel on edge ``(a, b)`` at rate ``p``."""
    dim = 1 << n
    if p <= 0:
        return np.eye(dim * dim, dtype=complex)
    s = (1.0 - p) * np.eye(dim * dim, dtype=complex)
    paulis = (np.eye(2, dtype=complex), *_PAULIS_1Q)
    for pa in paulis:
        for pb in paulis:
            if pa is paulis[0] and pb is paulis[0]:
                continue  # identity term excluded from the 15 nontrivial Paulis
            # local 2-qubit basis: a is bit 0 (LSB), b is bit 1 -> kron(pb, pa).
            e = embed_operator(np.kron(pb, pa), [a, b], n)
            s = s + (p / 15.0) * unitary_superop(e)
    return s


def leakage_superop(a: int, b: int, p_leak: float, n: int) -> np.ndarray:
    """Trace-decreasing leakage on the ``|11>`` of edge ``(a, b)``.

    A single Kraus ``diag(1,1,1,sqrt(1-p_leak))`` on the local 2-qubit subspace:
    ``p_leak`` of the ``|11>`` population leaks to ``|2>`` (out of the computational
    subspace) and is lost. No compensating Kraus, so the channel is sub-unital and
    the leaked population counts as infidelity.
    """
    dim = 1 << n
    if p_leak <= 0:
        return np.eye(dim * dim, dtype=complex)
    k0_local = np.diag([1.0, 1.0, 1.0, math.sqrt(max(0.0, 1.0 - p_leak))]).astype(complex)
    k0 = embed_operator(k0_local, [a, b], n)
    return unitary_superop(k0)


def amp_phase_damping_kraus(t_ns: float, t1_ns: float, t2_ns: float) -> list[np.ndarray]:
    """Single-qubit amplitude + pure-dephasing Kraus for active time ``t_ns``.

    ``gamma = 1 - exp(-t/T1)`` (relaxation); pure dephasing
    ``lam = 1 - exp(-t/T_phi)`` with ``1/T_phi = 1/T2 - 1/(2 T1)`` (clamped >= 0).
    """
    gamma = 0.0 if (t1_ns <= 0 or not math.isfinite(t1_ns)) else 1.0 - math.exp(-t_ns / t1_ns)
    inv_tphi = 0.0
    if t2_ns > 0 and math.isfinite(t2_ns):
        inv_tphi = 1.0 / t2_ns - (1.0 / (2.0 * t1_ns) if t1_ns > 0 else 0.0)
    inv_tphi = max(0.0, inv_tphi)
    lam = 1.0 - math.exp(-t_ns * inv_tphi)
    a0 = np.array([[1.0, 0.0], [0.0, math.sqrt(max(0.0, 1.0 - gamma))]], dtype=complex)
    a1 = np.array([[0.0, math.sqrt(max(0.0, gamma))], [0.0, 0.0]], dtype=complex)
    p0 = np.array([[1.0, 0.0], [0.0, math.sqrt(max(0.0, 1.0 - lam))]], dtype=complex)
    p1 = np.array([[0.0, 0.0], [0.0, math.sqrt(max(0.0, lam))]], dtype=complex)
    # Compose phase-damping after amplitude-damping: {p_i a_j}.
    return [pp @ aa for pp in (p0, p1) for aa in (a0, a1)]


def damping_superop(q: int, t_ns: float, t1_ns: float, t2_ns: float, n: int) -> np.ndarray:
    """Embed single-qubit amplitude+phase damping for qubit ``q`` into the full space."""
    dim = 1 << n
    if t_ns <= 0:
        return np.eye(dim * dim, dtype=complex)
    kraus = [embed_operator(k, [q], n) for k in amp_phase_damping_kraus(t_ns, t1_ns, t2_ns)]
    return kraus_superop(kraus)


def prep_superop(prep_error: float, n: int) -> np.ndarray:
    """Per-qubit state-prep bit-flip (residual ``|1>`` population) at the input."""
    dim = 1 << n
    s = np.eye(dim * dim, dtype=complex)
    if prep_error <= 0:
        return s
    for q in range(n):
        xq = embed_operator(_PAULI_X, [q], n)
        s_q = (1.0 - prep_error) * np.eye(dim * dim, dtype=complex)
        s_q = s_q + prep_error * unitary_superop(xq)
        s = s_q @ s
    return s


def readout_qubit_kraus(p01: float, p10: float) -> list[np.ndarray]:
    """Asymmetric computational-basis readout as a CPTP channel, identity at 0 rates.

    Action on ``rho = [[a, b], [c, d]]``::

        |0><0| -> (1-p01)|0><0| + p01|1><1|
        |1><1| -> p10 |0><0| + (1-p10)|1><1|
        off-diagonals scaled by  sqrt((1-p01)(1-p10))  (max CP coherence factor)

    Three Kraus realize this exactly (sum_k K^dag K = I, identity at zero rates):
    ``diag(sqrt(1-p01), sqrt(1-p10))`` (keeps populations + scales coherence),
    ``sqrt(p01)|1><0|`` (0->1), ``sqrt(p10)|0><1|`` (1->0). Diagonal in the
    computational basis (Z); not a coherent X-error or depolarizing map.
    """
    e01 = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)  # |0><1|
    e10 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)  # |1><0|
    k_keep = np.diag([math.sqrt(max(0.0, 1.0 - p01)), math.sqrt(max(0.0, 1.0 - p10))]).astype(
        complex
    )
    return [k_keep, math.sqrt(max(0.0, p01)) * e10, math.sqrt(max(0.0, p10)) * e01]


def readout_superop(readout_rates: list[tuple[float, float]], n: int) -> np.ndarray:
    """Asymmetric per-qubit readout CPTP, identity at zero rates.

    Composes the per-qubit channels (each a 3-Kraus asymmetric confusion embedded
    into the full space) so it joins the other Hilbert-space superoperators.
    """
    dim = 1 << n
    s = np.eye(dim * dim, dtype=complex)
    for q, (p01, p10) in enumerate(readout_rates):
        if p01 <= 0 and p10 <= 0:
            continue
        embedded = [embed_operator(k, [q], n) for k in readout_qubit_kraus(p01, p10)]
        s = kraus_superop(embedded) @ s
    return s


# ---------- physics container + circuit -> channel ----------


@dataclass(frozen=True)
class ChainPhysics:
    """Realized device physics (after applying the agent's calibration)."""

    n_qubits: int
    t1_ns: tuple[float, ...]
    t2_ns: tuple[float, ...]
    rx_ry_duration_ns: float
    rx_ry_depol: float
    cphase_duration_ns: float
    cphase_depol: float
    cphase_leakage: float  # realized leakage per CPhase (well-tuned floor * factor)
    over_rotation: float  # realized factor: shipped * agent_correction
    prep_error: float
    readout_rates: tuple[tuple[float, float], ...]


def per_qubit_active_ns(circuit: list[dict], phys: ChainPhysics) -> list[float]:
    """Per-qubit BUSY time: the summed durations of the gates acting on each qubit.

    Each gate's duration is canonical (virtual rz: 0; rx/ry/physical-rz:
    rx_ry_duration; cphase: cphase_duration). A qubit is charged only for the
    gates that touch it; spectator idling and wall-clock layer time are NOT
    charged, so reschedulings of the same gate list yield identical active
    times (documented v1 simplification; a spectator-idling/wall-clock model is
    deferred to a next-edition decision). ``layer`` feeds only the validity
    check: raises if a layer puts two gates on a shared qubit (physically
    impossible parallelism).
    """
    # layer -> qubit -> duration
    layers: dict[int, dict[int, float]] = {}
    for g in circuit:
        qubits = g["qubits"]
        layer = int(g.get("layer", 0))
        dur = _gate_duration_ns(g, phys)
        slot = layers.setdefault(layer, {})
        for q in qubits:
            if q in slot:
                raise ValueError(
                    f"layer {layer} assigns two gates to qubit {q} (impossible parallelism)"
                )
            slot[q] = dur
    active = [0.0] * phys.n_qubits
    for slot in layers.values():
        for q, dur in slot.items():
            active[q] += dur
    return active


def _gate_duration_ns(g: dict, phys: ChainPhysics) -> float:
    gtype = g["type"]
    if gtype == "rz":
        return 0.0 if g.get("virtual", False) else phys.rx_ry_duration_ns
    if gtype in ("rx", "ry"):
        return phys.rx_ry_duration_ns
    if gtype == "cphase":
        return phys.cphase_duration_ns
    raise ValueError(f"unknown gate type {gtype!r}")


def coherent_unitary(circuit: list[dict], phys: ChainPhysics) -> np.ndarray:
    """Compose the circuit's coherent unitary (d x d) using realized CPhase angles.

    Cheap (d=2^n matmuls). The noiseless validity gate forms its superoperator from
    this once, avoiding the full noisy-channel composition.
    """
    n = phys.n_qubits
    u_total = np.eye(1 << n, dtype=complex)
    for g in circuit:
        gtype = g["type"]
        angle = float(g.get("angle", 0.0))
        if gtype == "rz":
            local = rz_unitary(angle)
        elif gtype == "rx":
            local = rx_unitary(angle)
        elif gtype == "ry":
            local = ry_unitary(angle)
        elif gtype == "cphase":
            local = cphase_local(phys.over_rotation * angle)
        else:
            raise ValueError(f"unknown gate type {gtype!r}")
        u_total = embed_operator(local, list(g["qubits"]), n) @ u_total
    return u_total


def _noise_cache(circuit: list[dict], phys: ChainPhysics) -> dict:
    """Precompute the distinct noise superoperators the circuit needs (built once)."""
    n = phys.n_qubits
    cache: dict = {"d1": {}, "d2": {}, "leak": {}}
    for g in circuit:
        gtype = g["type"]
        qubits = g["qubits"]
        if gtype in ("rx", "ry") or (gtype == "rz" and not g.get("virtual", False)):
            q = qubits[0]
            if q not in cache["d1"]:
                cache["d1"][q] = depol_1q_superop(q, phys.rx_ry_depol, n)
        elif gtype == "cphase":
            key = (qubits[0], qubits[1])
            if key not in cache["d2"]:
                cache["d2"][key] = depol_2q_superop(qubits[0], qubits[1], phys.cphase_depol, n)
                cache["leak"][key] = leakage_superop(qubits[0], qubits[1], phys.cphase_leakage, n)
    return cache


def gate_channel_superop(circuit: list[dict], phys: ChainPhysics) -> np.ndarray:
    """Coherent gates + per-gate depolarizing + per-CPhase leakage (no decoherence)."""
    n = phys.n_qubits
    dim = 1 << n
    cache = _noise_cache(circuit, phys)
    s = np.eye(dim * dim, dtype=complex)
    for g in circuit:
        gtype = g["type"]
        qubits = g["qubits"]
        angle = float(g.get("angle", 0.0))
        if gtype == "rz":
            u = embed_operator(rz_unitary(angle), [qubits[0]], n)
            s = unitary_superop(u) @ s  # frame change: free
            if not g.get("virtual", False):
                s = cache["d1"][qubits[0]] @ s
        elif gtype in ("rx", "ry"):
            local = rx_unitary(angle) if gtype == "rx" else ry_unitary(angle)
            u = embed_operator(local, [qubits[0]], n)
            s = unitary_superop(u) @ s
            s = cache["d1"][qubits[0]] @ s
        elif gtype == "cphase":
            a, b = qubits[0], qubits[1]
            u = embed_operator(cphase_local(phys.over_rotation * angle), [a, b], n)
            s = unitary_superop(u) @ s
            s = cache["leak"][(a, b)] @ s
            s = cache["d2"][(a, b)] @ s
        else:
            raise ValueError(f"unknown gate type {gtype!r}")
    return s


def decoherence_superop(circuit: list[dict], phys: ChainPhysics) -> np.ndarray:
    """Per-qubit amplitude+phase damping keyed to each qubit's scheduled active time."""
    n = phys.n_qubits
    dim = 1 << n
    active = per_qubit_active_ns(circuit, phys)
    s = np.eye(dim * dim, dtype=complex)
    for q in range(n):
        s = damping_superop(q, active[q], phys.t1_ns[q], phys.t2_ns[q], n) @ s
    return s


def full_channel_superop(
    circuit: list[dict],
    phys: ChainPhysics,
    *,
    include_prep: bool = True,
    include_decoherence: bool = True,
    include_readout: bool = True,
) -> np.ndarray:
    """Compose the SPAM-inclusive channel ``E_ro o E_dec o E_gates o E_prep``."""
    n = phys.n_qubits
    dim = 1 << n
    s = np.eye(dim * dim, dtype=complex)
    if include_prep:
        s = prep_superop(phys.prep_error, n) @ s
    s = gate_channel_superop(circuit, phys) @ s
    if include_decoherence:
        s = decoherence_superop(circuit, phys) @ s
    if include_readout:
        s = readout_superop(list(phys.readout_rates), n) @ s
    return s


def process_fidelity_to_unitary(superop: np.ndarray, u: np.ndarray) -> float:
    """Entanglement (process) fidelity of a channel superoperator vs a unitary.

    ``F_pro = (1/d^2) Re Tr[ S_U^dag S ]`` with ``S_U = kron(conj(U), U)``. The
    prefactor is fixed at ``1/d^2`` so leakage (trace deficit) counts as infidelity.
    """
    d = u.shape[0]
    s_u = unitary_superop(u)
    return float(np.real(np.trace(s_u.conj().T @ superop)) / (d * d))


def channel_average_surviving_trace(superop: np.ndarray, d: int) -> float:
    """Average surviving trace ``Tbar = Tr[E(I/d)]`` of a channel superoperator.

    Equals the Haar average over pure inputs of the output trace (by linearity).
    Exactly 1 for a trace-preserving channel; leakage (the trace-decreasing
    Kraus map) pushes it below 1. Column-stacking convention:
    ``Tr[E(rho)] = vec(I)^dag S vec(rho)``.
    """
    vec_id = np.eye(d, dtype=complex).reshape(-1, order="F")
    return float(np.real(vec_id.conj() @ superop @ vec_id) / d)


def avg_gate_fidelity(f_pro: float, d: int, tbar: float = 1.0) -> float:
    """Average-gate fidelity from the process fidelity (trace-decreasing form).

    ``F_avg = (d*F_pro + Tbar)/(d+1)`` with ``Tbar`` the average surviving trace
    of the channel (:func:`channel_average_surviving_trace`). For a
    trace-preserving channel ``Tbar = 1`` and this reduces to Nielsen's formula
    ``(d*F_pro + 1)/(d+1)``; for a leaky channel ``Tbar < 1``, so the leaked
    population earns no survival credit and F_avg sits strictly below the
    trace-preserving value by ``(1 - Tbar)/(d+1)``.
    """
    return (d * f_pro + tbar) / (d + 1.0)


def ideal_qft_unitary(n: int) -> np.ndarray:
    """Standard ``N=2**n`` DFT unitary in the engine's integer-index basis.

    ``U[k, j] = exp(2*pi*i * j * k / N) / sqrt(N)`` with qubit 0 the LSB.
    """
    dim = 1 << n
    j = np.arange(dim)
    k = j.reshape(-1, 1)
    omega = np.exp(2j * np.pi / dim)
    return omega ** (j * k) / np.sqrt(dim)


__all__ = [
    "ChainPhysics",
    "avg_gate_fidelity",
    "channel_average_surviving_trace",
    "coherent_unitary",
    "cphase_local",
    "decoherence_superop",
    "depol_1q_superop",
    "depol_2q_superop",
    "embed_operator",
    "full_channel_superop",
    "gate_channel_superop",
    "ideal_qft_unitary",
    "leakage_superop",
    "per_qubit_active_ns",
    "process_fidelity_to_unitary",
    "rx_unitary",
    "ry_unitary",
    "rz_unitary",
    "unitary_superop",
]
