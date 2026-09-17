"""Pure-numpy physics core for the transmon_repeater_array qtype.

Shared by the engine (sampling raw bitstrings) and the hidden-dynamics reference
scorer (recomputing the best achievable final fidelity), so there is a single source
of truth and no engine<->verifier parity drift. NO qiqcbench imports here.

Model:
- Each distributed Bell pair is a 2-qubit density matrix (4x4), qubit 0 = Alice
  (MSB), qubit 1 = Bob (LSB); computational basis index = 2*a + b.
- A near-ideal |Phi+> is generated locally (small depolarizing ``p_gen``), then each
  half is SWAP-transported through ``n_swap_per_arm`` noisy SWAPs (per-SWAP
  depolarizing ``p_swap`` + a ``t_swap_ns`` T1/T2 dwell) to the two ends -> the
  distributed pair arrives at a hidden fidelity F0 < 1.
- Purification (two pairs -> one pair): each end (Alice, Bob) applies ANY local unitary
  to its two held qubits (built from a small gate set), then the second pair is measured
  and the first pair is kept when the two outcomes agree (post-selection) -- converting
  two noisy pairs into one with some success probability. Nothing acts across the two
  ends. Iterating this raises fidelity; noisy local gates + a depth-growing memory dwell
  cap it (rise -> peak -> fall). The best local-unitary choice (a quarter-turn rotation
  before the within-end CNOT) maximizes the final fidelity.

Qubit layout in a purification round (16-dim): (A1, B1, A2, B2) = global (0, 1, 2, 3),
MSB = A1. Alice owns party-local qubits {0->A1=0, 1->A2=2}; Bob owns {0->B1=1, 1->B2=3}.
A "circuit op" is a tuple: ("rx"|"ry"|"rz", q_local, angle_rad), a 0-arg Clifford
("h"|"s"|"sdg"|"x"|"y"|"z", q_local, 0.0), or ("cnot", control_local, target_local).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --- constants -------------------------------------------------------------

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)

_S2 = 1.0 / math.sqrt(2.0)
PHI_PLUS = np.array([_S2, 0, 0, _S2], dtype=complex)  # (|00>+|11>)/sqrt2  -- target
PHI_MINUS = np.array([_S2, 0, 0, -_S2], dtype=complex)
PSI_PLUS = np.array([0, _S2, _S2, 0], dtype=complex)
PSI_MINUS = np.array([0, _S2, -_S2, 0], dtype=complex)

_PHI_PLUS_DM = np.outer(PHI_PLUS, PHI_PLUS.conj())

BASES = ("ZZ", "XX", "YY")
GATE_NAMES = ("rx", "ry", "rz", "h", "s", "sdg", "x", "y", "z", "cnot")

# party-local qubit index -> global 4-qubit index
ALICE_MAP = {0: 0, 1: 2}  # A1, A2
BOB_MAP = {0: 1, 1: 3}  # B1, B2


@dataclass(frozen=True)
class RepeaterParams:
    """Hidden physical parameters of the repeater array (the answer is emergent)."""

    p_gen: float
    n_swap_per_arm: int
    p_swap: float
    t_swap_ns: float
    t1_us: float
    t2_us: float
    p_cnot: float
    p_1q: float
    readout_p01: float
    readout_p10: float
    t_mem_per_round_ns: float
    dwell_growth: str = "exp"
    # Non-isotropic transport noise: per SWAP hop each qubit is additionally
    # flipped about ONE unit axis ``bias_axis`` (device frame, same on both arms) with
    # probability ``p_bias``. The axis is a per-attempt realized hidden instance.
    p_bias: float = 0.0
    bias_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)


# --- single-qubit gates + channels ----------------------------------------


def _rot(axis: str, ang: float) -> np.ndarray:
    c, s = math.cos(ang / 2.0), math.sin(ang / 2.0)
    if axis == "x":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if axis == "y":
        return np.array([[c, -s], [s, c]], dtype=complex)
    return np.array([[np.exp(-1j * ang / 2.0), 0], [0, np.exp(1j * ang / 2.0)]], dtype=complex)


_CLIFFORD_1Q = {
    "h": _S2 * np.array([[1, 1], [1, -1]], dtype=complex),
    "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    "sdg": np.array([[1, 0], [0, -1j]], dtype=complex),
    "x": _X,
    "y": _Y,
    "z": _Z,
}


def _gate_matrix(name: str, angle: float) -> np.ndarray:
    if name in ("rx", "ry", "rz"):
        return _rot(name[1], angle)
    return _CLIFFORD_1Q[name]


def _depol_1q_kraus(p: float) -> list[np.ndarray]:
    if p <= 0:
        return [_I2.copy()]
    return [
        math.sqrt(1.0 - p) * _I2,
        math.sqrt(p / 3.0) * _X,
        math.sqrt(p / 3.0) * _Y,
        math.sqrt(p / 3.0) * _Z,
    ]


def _bias_kraus(p: float, axis) -> list[np.ndarray]:
    """rho -> (1 - p) rho + p (n.sigma) rho (n.sigma): a flip about the unit axis ``n``."""
    if p <= 0:
        return [_I2.copy()]
    nx, ny, nz = (float(v) for v in axis)
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("bias_axis must be a non-zero finite vector")
    sig = (nx * _X + ny * _Y + nz * _Z) / norm
    return [math.sqrt(1.0 - p) * _I2, math.sqrt(p) * sig]


def _t1t2_kraus(t_us: float, t1_us: float, t2_us: float) -> list[np.ndarray]:
    if t_us <= 0:
        return [_I2.copy()]
    gamma = 1.0 - math.exp(-t_us / t1_us) if t1_us > 0 else 0.0
    k0 = np.array([[1.0, 0.0], [0.0, math.sqrt(max(0.0, 1.0 - gamma))]], dtype=complex)
    k1 = np.array([[0.0, math.sqrt(max(0.0, gamma))], [0.0, 0.0]], dtype=complex)
    amp = [k0, k1]
    inv_tphi = (1.0 / t2_us if t2_us > 0 else 0.0) - (1.0 / (2.0 * t1_us) if t1_us > 0 else 0.0)
    if inv_tphi <= 0:
        return amp
    lam = math.exp(-t_us * inv_tphi)
    deph = [math.sqrt((1.0 + lam) / 2.0) * _I2, math.sqrt((1.0 - lam) / 2.0) * _Z]
    return [d @ a for d in deph for a in amp]


def _embed(op2: np.ndarray, q: int, n: int) -> np.ndarray:
    mats = [op2 if i == q else _I2 for i in range(n)]
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def _apply_1q_channel(rho: np.ndarray, kraus: list[np.ndarray], q: int, n: int) -> np.ndarray:
    if len(kraus) == 1 and np.allclose(kraus[0], _I2):
        return rho
    out = np.zeros_like(rho)
    for k in kraus:
        e = _embed(k, q, n)
        out += e @ rho @ e.conj().T
    return out


def _depol_2q(rho4: np.ndarray, p: float) -> np.ndarray:
    if p <= 0:
        return rho4
    return (1.0 - p) * rho4 + p * np.eye(4, dtype=complex) / 4.0


def fidelity_phi_plus(rho4: np.ndarray) -> float:
    return float(np.real(PHI_PLUS.conj() @ rho4 @ PHI_PLUS))


# --- distributed pair + memory --------------------------------------------


def distributed_pair(params: RepeaterParams) -> np.ndarray:
    rho = _PHI_PLUS_DM.copy()
    rho = _depol_2q(rho, params.p_gen)
    swap_depol = _depol_1q_kraus(params.p_swap)
    swap_bias = _bias_kraus(params.p_bias, params.bias_axis)
    swap_dwell = _t1t2_kraus(params.t_swap_ns / 1000.0, params.t1_us, params.t2_us)
    for _ in range(params.n_swap_per_arm):
        for q in (0, 1):
            rho = _apply_1q_channel(rho, swap_depol, q, 2)
            rho = _apply_1q_channel(rho, swap_bias, q, 2)
            rho = _apply_1q_channel(rho, swap_dwell, q, 2)
    return rho


def _dwell_ns(level: int, params: RepeaterParams) -> float:
    if params.dwell_growth == "linear":
        g = float(level - 1)
    else:
        g = float(2 ** (level - 1))
    return params.t_mem_per_round_ns * g


def memory_dwell(rho4: np.ndarray, t_ns: float, params: RepeaterParams) -> np.ndarray:
    if t_ns <= 0:
        return rho4
    kraus = _t1t2_kraus(t_ns / 1000.0, params.t1_us, params.t2_us)
    rho = _apply_1q_channel(rho4, kraus, 0, 2)
    rho = _apply_1q_channel(rho, kraus, 1, 2)
    return rho


# --- agent-designed local circuits + the 2->1 purification round -----------


def _cnot4(control: int, target: int) -> np.ndarray:
    u = np.zeros((16, 16), dtype=complex)
    for old in range(16):
        bits = [(old >> 3) & 1, (old >> 2) & 1, (old >> 1) & 1, old & 1]
        if bits[control]:
            bits[target] ^= 1
        new = (bits[0] << 3) | (bits[1] << 2) | (bits[2] << 1) | bits[3]
        u[new, old] = 1.0
    return u


def _apply_party_circuit(
    rho16: np.ndarray, ops: list[tuple], party_map: dict[int, int], params: RepeaterParams
) -> np.ndarray:
    """Apply a party-local circuit (party-local qubit indices 0/1) to the 16-dim state."""
    for op in ops:
        name = op[0]
        if name == "cnot":
            c, t = party_map[op[1]], party_map[op[2]]
            u = _cnot4(c, t)
            rho16 = u @ rho16 @ u.conj().T
            gate = _depol_1q_kraus(params.p_cnot)
            rho16 = _apply_1q_channel(rho16, gate, c, 4)
            rho16 = _apply_1q_channel(rho16, gate, t, 4)
        else:
            q = party_map[op[1]]
            ang = float(op[2]) if len(op) > 2 else 0.0
            u = _embed(_gate_matrix(name, ang), q, 4)
            rho16 = u @ rho16 @ u.conj().T
            rho16 = _apply_1q_channel(rho16, _depol_1q_kraus(params.p_1q), q, 4)
    return rho16


def _reduce_pair1_given_target(rho16: np.ndarray, a2: int, b2: int) -> np.ndarray:
    t = rho16.reshape(2, 2, 2, 2, 2, 2, 2, 2)
    block = t[:, :, a2, b2, :, :, a2, b2]
    return block.reshape(4, 4)


def purify_round(
    rho_in: np.ndarray, alice_ops: list[tuple], bob_ops: list[tuple], params: RepeaterParams
) -> tuple[np.ndarray, float]:
    """One 2->1 round on two copies of ``rho_in``: apply the agent's local unitaries at
    each end, measure the second pair (A2,B2) in Z with readout confusion, keep the first
    pair (A1,B1) when the two outcomes agree. Returns (rho_kept, success_prob)."""
    rho16 = np.kron(rho_in, rho_in)
    rho16 = _apply_party_circuit(rho16, alice_ops, ALICE_MAP, params)
    rho16 = _apply_party_circuit(rho16, bob_ops, BOB_MAP, params)
    r01, r10 = params.readout_p01, params.readout_p10

    def p_meas(meas: int, true: int) -> float:
        if true == 0:
            return (1.0 - r01) if meas == 0 else r01
        return r10 if meas == 0 else (1.0 - r10)

    kept = np.zeros((4, 4), dtype=complex)
    for a2 in (0, 1):
        for b2 in (0, 1):
            cond = _reduce_pair1_given_target(rho16, a2, b2)
            p_keep = p_meas(0, a2) * p_meas(0, b2) + p_meas(1, a2) * p_meas(1, b2)
            kept += cond * p_keep
    yield_prob = float(np.real(np.trace(kept)))
    if yield_prob <= 1e-12:
        return _PHI_PLUS_DM.copy(), 0.0
    return kept / yield_prob, yield_prob


def recurrence_states(
    params: RepeaterParams, depth: int, alice_ops: list[tuple], bob_ops: list[tuple]
) -> tuple[list[np.ndarray], list[float]]:
    """states[l] = the kept output pair after l rounds of the agent's circuit (the pair you
    would measure if you stopped at depth l); keeps[l] = that round's success probability."""
    rho0 = distributed_pair(params)
    states = [rho0]
    keeps = [1.0]
    rho_kept = rho0
    for level in range(1, depth + 1):
        rho_in = memory_dwell(rho_kept, _dwell_ns(level, params), params)
        rho_kept, keep = purify_round(rho_in, alice_ops, bob_ops, params)
        states.append(rho_kept)
        keeps.append(keep)
    return states, keeps


# --- measurement sampling (engine raw bitstrings) -------------------------

# Per-end pre-measurement rotations: a setting is a two-letter string, first letter
# Alice's basis, second Bob's (the nine Pauli settings = full two-qubit tomography).
_ROT_1Q = {
    "Z": _I2,
    "X": _S2 * np.array([[1, 1], [1, -1]], dtype=complex),  # H
    "Y": _S2 * np.array([[1, -1j], [1, 1j]], dtype=complex),  # H S^dag
}
SETTINGS = tuple(a + b for a in "XYZ" for b in "XYZ")
_BASIS_ROT = {"ZZ": _ROT_1Q["Z"], "XX": _ROT_1Q["X"], "YY": _ROT_1Q["Y"]}  # same-basis alias


def _setting_unitary(basis: str) -> np.ndarray:
    if len(basis) != 2 or basis[0] not in _ROT_1Q or basis[1] not in _ROT_1Q:
        raise ValueError(f"unknown measurement setting {basis!r}; expected one of {SETTINGS}")
    return np.kron(_ROT_1Q[basis[0]], _ROT_1Q[basis[1]])


def measure_counts(
    rho4: np.ndarray, basis: str, shots: int, params: RepeaterParams, rng: np.random.Generator
) -> dict[str, int]:
    u = _setting_unitary(basis)
    rho = u @ rho4 @ u.conj().T
    probs = np.clip(np.real(np.diag(rho)), 0.0, None)
    total = probs.sum()
    probs = probs / total if total > 0 else np.full(4, 0.25)
    counts = {f"{i >> 1}{i & 1}": 0 for i in range(4)}
    draws = rng.choice(4, size=shots, p=probs)
    r01, r10 = params.readout_p01, params.readout_p10
    for idx in draws:
        a, b = (idx >> 1) & 1, idx & 1
        a = (1 if rng.random() < r01 else 0) if a == 0 else (0 if rng.random() < r10 else 1)
        b = (1 if rng.random() < r01 else 0) if b == 0 else (0 if rng.random() < r10 else 1)
        counts[f"{a}{b}"] += 1
    return counts


def measured_correlator(rho4: np.ndarray, basis: str, params: RepeaterParams) -> float:
    u = _setting_unitary(basis)
    rho = u @ rho4 @ u.conj().T
    p = np.clip(np.real(np.diag(rho)), 0.0, None)
    total = p.sum()
    p = p / total if total > 0 else np.full(4, 0.25)
    r01, r10 = params.readout_p01, params.readout_p10

    def pm(meas: int, true: int) -> float:
        if true == 0:
            return (1.0 - r01) if meas == 0 else r01
        return r10 if meas == 0 else (1.0 - r10)

    corr = 0.0
    for idx in range(4):
        a, b = (idx >> 1) & 1, idx & 1
        for m in (0, 1):
            for n in (0, 1):
                corr += p[idx] * pm(m, a) * pm(n, b) * ((-1) ** (m + n))
    return corr


def expected_raw_fidelity(rho4: np.ndarray, params: RepeaterParams) -> float:
    """Exact raw Pauli-correlator fidelity F = (1 + <ZZ> + <XX> - <YY>)/4 (readout-depressed)."""
    return (
        1.0
        + measured_correlator(rho4, "ZZ", params)
        + measured_correlator(rho4, "XX", params)
        - measured_correlator(rho4, "YY", params)
    ) / 4.0


def raw_fidelity_curve(
    params: RepeaterParams, alice_ops: list[tuple], bob_ops: list[tuple], max_depth: int
) -> list[float]:
    """Raw (readout-depressed) measured fidelity per depth for the agent's circuit."""
    states, _ = recurrence_states(params, max_depth, alice_ops, bob_ops)
    return [expected_raw_fidelity(s, params) for s in states]


def best_raw_fidelity(
    params: RepeaterParams, alice_ops: list[tuple], bob_ops: list[tuple], max_depth: int = 8
) -> tuple[float, int]:
    """(F_best_raw, k*) = depth-maximising raw measured fidelity for the agent's circuit."""
    curve = raw_fidelity_curve(params, alice_ops, bob_ops, max_depth)
    k_star = int(np.argmax(curve))
    return curve[k_star], k_star


def fidelity_from_correlator_counts(counts_by_basis: dict[str, dict[str, int]]) -> float:
    def corr(basis: str) -> float:
        c = counts_by_basis[basis]
        n = sum(c.values())
        if n == 0:
            return 0.0
        even = c.get("00", 0) + c.get("11", 0)
        odd = c.get("01", 0) + c.get("10", 0)
        return (even - odd) / n

    return (1.0 + corr("ZZ") + corr("XX") - corr("YY")) / 4.0
