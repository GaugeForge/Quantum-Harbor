"""Pauli-transfer-matrix forward core for the 5-qubit cycle-error-recon device.

Single source of forward truth shared by the engine (evidence generation) and the
hidden scorer (predictive validation). n=5, the 4^5 = 1024 Hermitian Pauli basis.

Conventions:
  - qubit 0 is the LEFTMOST factor of a Pauli label and the OUTERMOST Kronecker
    factor: P = kron(P0, P1, P2, P3, P4); the returned 5-bit counts have leftmost
    bit = q0.
  - per-qubit Hermitian Paulis I, X, Y, Z; Y = [[0,-i],[i,0]].
  - PTM: state rho <-> Pauli vector v with v_P = Tr(P rho); channel C with PTM has
    (C v)_P = Tr(P C(rho)); composition C = A.B -> PTM(A) @ PTM(B).
  - one physical noisy cycle: ideal cycle G, then coherent U_Q(theta) =
    exp(-i theta Q/2), then the stochastic Pauli channel -> M = S @ Ucoh @ G.

The noisy-cycle PTM is <=2 nonzeros per column (Clifford permutes; coherent rotation
mixes a Pauli with one partner; stochastic is diagonal), so folded blocks and depth-32
workloads are exact and cheap via sparse matvecs.
"""

from __future__ import annotations

import itertools
import re

import numpy as np
import scipy.sparse as sp

N = 5
DIM = 2**N
NP = 4**N  # 1024

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_SINGLE = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}
_LETTERS = "IXYZ"
_XZ = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}


def _kron_string(letters: str) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]])
    for ch in letters:
        out = np.kron(out, _SINGLE[ch])
    return out


PAULI_STRINGS: list[str] = ["".join(t) for t in itertools.product(_LETTERS, repeat=N)]
INDEX_OF: dict[str, int] = {s: i for i, s in enumerate(PAULI_STRINGS)}
_BMAT = np.stack([_kron_string(s).reshape(-1) for s in PAULI_STRINGS])

_XMASK = np.zeros(NP, dtype=np.int64)
_ZMASK = np.zeros(NP, dtype=np.int64)
for _i, _s in enumerate(PAULI_STRINGS):
    _xm = _zm = 0
    for _q, _ch in enumerate(_s):
        _x, _z = _XZ[_ch]
        _xm |= _x << _q
        _zm |= _z << _q
    _XMASK[_i] = _xm
    _ZMASK[_i] = _zm
_MASK_TO_INDEX = {(int(_XMASK[i]), int(_ZMASK[i])): i for i in range(NP)}


def label_to_string(label: str) -> str:
    """'Z1Z2' -> 'IZZII' (length 5, q0 leftmost). '' / 'none' / 'I' -> 'IIIII'."""
    if label in ("", "none", None, "I", "IIIII"):
        if label == "IIIII":
            return label
        return "IIIII"
    if len(label) == 5 and set(label) <= set("IXYZ"):
        return label
    chars = ["I"] * N
    for ch, idx in re.findall(r"([XYZ])(\d+)", label):
        q = int(idx)
        if not 0 <= q < N:
            raise ValueError(f"qubit index {q} out of range in {label!r}")
        chars[q] = ch
    return "".join(chars)


def pauli_matrix(label: str) -> np.ndarray:
    return _kron_string(label_to_string(label))


def _popcount(a: np.ndarray) -> np.ndarray:
    out = np.zeros_like(a)
    x = a.copy()
    while x.any():
        out += x & 1
        x >>= 1
    return out


def commute_signs(label: str) -> np.ndarray:
    """+1/-1 per Pauli index: +1 if it commutes with ``label``, -1 if it anticommutes."""
    qi = INDEX_OF[label_to_string(label)]
    anti = (_popcount(_XMASK & int(_ZMASK[qi])) + _popcount(_ZMASK & int(_XMASK[qi]))) & 1
    return np.where(anti == 1, -1.0, 1.0)


def _ptm_from_unitary(u: np.ndarray) -> np.ndarray:
    conj = u.conj().T
    mvec = np.empty_like(_BMAT)
    for q in range(NP):
        bq = _BMAT[q].reshape(DIM, DIM)
        mvec[q] = (u @ bq @ conj).reshape(-1)
    return ((_BMAT.conj() @ mvec.T) / DIM).real


# ---- ideal cycle G = CZ01 CZ12 CZ23 CZ34 ----

_G_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4)]


def _cz(i: int, j: int) -> np.ndarray:
    diag = np.ones(DIM, dtype=complex)
    for k in range(DIM):
        bi = (k >> (N - 1 - i)) & 1
        bj = (k >> (N - 1 - j)) & 1
        if bi and bj:
            diag[k] = -1.0
    return np.diag(diag)


def _ideal_cycle_matrix() -> np.ndarray:
    g = np.eye(DIM, dtype=complex)
    for i, j in _G_EDGES:
        g = _cz(i, j) @ g
    return g


G_MAT = _ideal_cycle_matrix()
G_PTM = _ptm_from_unitary(G_MAT)
_G_SPARSE = sp.csc_matrix(G_PTM)
# G conjugation as a Pauli-index permutation: G P_i G = +- P_{G_PERM[i]}.
G_PERM = np.argmax(np.abs(G_PTM), axis=0)

# Public symplectic masks (for dressing-frame / recovery tracking in the engine).
XMASK = _XMASK
ZMASK = _ZMASK


def commute_signs_from_mask(dx: int, dz: int) -> np.ndarray:
    """+1/-1 per Pauli index for commutation with the Pauli having masks (dx, dz)."""
    anti = (_popcount(_XMASK & int(dz)) + _popcount(_ZMASK & int(dx))) & 1
    return np.where(anti == 1, -1.0, 1.0)


# ---- coherent PTM (analytic, cached structure) ----

_COH_COO: dict[str, tuple] = {}


def _coherent_coo(generator: str):
    if generator in _COH_COO:
        return _COH_COO[generator]
    qi = INDEX_OF[label_to_string(generator)]
    xq, zq = int(_XMASK[qi]), int(_ZMASK[qi])
    qmat = _BMAT[qi].reshape(DIM, DIM)
    anti = ((_popcount(_XMASK & zq) + _popcount(_ZMASK & xq)) & 1).astype(bool)
    anti_idx, off_j, off_sign = [], [], []
    for i in np.where(anti)[0]:
        j = _MASK_TO_INDEX[(int(_XMASK[i]) ^ xq, int(_ZMASK[i]) ^ zq)]
        r = (-1j * qmat @ _BMAT[i].reshape(DIM, DIM)).reshape(-1)  # U P U^dag = cos P + sin*(-iQP)
        anti_idx.append(int(i))
        off_j.append(int(j))
        off_sign.append(float(np.sign(np.real(_BMAT[j].conj() @ r) / DIM)))
    _COH_COO[generator] = (np.array(anti_idx), np.array(off_j), np.array(off_sign))
    return _COH_COO[generator]


def _coherent_sparse(generator: str, theta: float) -> sp.csc_matrix:
    if generator in ("none", "", None) or theta == 0.0:
        return sp.identity(NP, format="csc")
    anti_idx, off_j, off_sign = _coherent_coo(generator)
    c, s = np.cos(theta), np.sin(theta)
    diag = np.ones(NP)
    diag[anti_idx] = c
    rows = np.concatenate([np.arange(NP), off_j])
    cols = np.concatenate([np.arange(NP), anti_idx])
    data = np.concatenate([diag, off_sign * s])
    return sp.csc_matrix((data, (rows, cols)), shape=(NP, NP))


def _stochastic_diag(prob: dict[str, float]) -> np.ndarray:
    diag = np.ones(NP) * (1.0 - sum(prob.values()))
    for label, p in prob.items():
        diag += p * commute_signs(label)
    return diag


def noisy_cycle(prob: dict[str, float], generator: str, theta: float) -> sp.csc_matrix:
    """One physical noisy cycle PTM M = S @ Ucoh @ G (sparse, <=2 nnz/col)."""
    s_diag = sp.diags(_stochastic_diag(prob), format="csc")
    return (s_diag @ _coherent_sparse(generator, theta) @ _G_SPARSE).tocsc()


def folded_cer_orbit_signal(
    prob: dict[str, float],
    generator: str,
    theta: float,
    probe: str,
    *,
    fold_factor: int,
    block_repetitions: int,
) -> float:
    """Exact pre-readout mean parity for one randomized folded-CER row.

    Pauli randomization diagonalizes the error of each folded block, but the
    nontrivial ideal Clifford still carries the probe around its Pauli orbit.
    Consequently the repeated signal is the product of the folded-block
    fidelities encountered along that orbit, not one fidelity raised to the
    repetition count.
    """
    matrix = noisy_cycle(prob, generator, theta)
    probe_index = INDEX_OF[label_to_string(probe)]
    orbit_fidelities: dict[int, float] = {}
    signal = 1.0
    current = probe_index
    for _ in range(block_repetitions):
        if current not in orbit_fidelities:
            basis = np.zeros(NP)
            basis[current] = 1.0
            evolved = basis
            for _ in range(fold_factor):
                evolved = matrix @ evolved
            orbit_fidelities[current] = float((_G_SPARSE.T @ basis) @ evolved)
        signal *= orbit_fidelities[current]
        current = int(G_PERM[current])
    return float(signal)


# ---- observables / states ----

_BLOCH = {
    "+": (1, 0, 0),
    "-": (-1, 0, 0),
    "0": (0, 0, 1),
    "1": (0, 0, -1),
    "i": (0, 1, 0),
    "j": (0, -1, 0),
}
_EIGENSTATE = {"X": "+", "Y": "i", "Z": "0", "I": "0"}  # +1 eigenstate char per single-qubit Pauli
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_SDG = np.array([[1, 0], [0, -1j]], dtype=complex)
_BASIS_GATE = {
    "Z": np.eye(2, dtype=complex),
    "X": _H,
    "Y": _H @ _SDG,
    "I": np.eye(2, dtype=complex),
}


def product_state_vector(states: str) -> np.ndarray:
    """Pauli vector of a 5-char product state over {+,-,0,1,i,j}."""
    comp = []
    for ch in states:
        sx, sy, sz = _BLOCH[ch]
        comp.append({"I": 1.0, "X": sx, "Y": sy, "Z": sz})
    v = np.empty(NP)
    for i, s in enumerate(PAULI_STRINGS):
        val = 1.0
        for q, letter in enumerate(s):
            val *= comp[q][letter]
            if val == 0.0:
                break
        v[i] = val
    return v


def probe_eigenstate(probe: str) -> str:
    """5-char +1 product eigenstate of a probe Pauli string (q outside support -> |0>)."""
    s = label_to_string(probe)
    return "".join(_EIGENSTATE[ch] for ch in s)


def readout_confusion(r01, r10) -> np.ndarray:
    mats = [np.array([[1 - r01[q], r10[q]], [r01[q], 1 - r10[q]]], float) for q in range(N)]
    m = mats[0]
    for q in range(1, N):
        m = np.kron(m, mats[q])  # q0 outermost (leftmost)
    return m


def basis_measure_distribution(v: np.ndarray, basis: str, conf: np.ndarray) -> np.ndarray:
    """Computational-basis 32-outcome distribution: reconstruct rho from Pauli vector v,
    rotate each qubit's measurement basis to Z, read the diagonal, apply readout confusion."""
    rho = (_BMAT.T @ v).reshape(DIM, DIM) / DIM
    r = np.array([[1.0 + 0j]])
    for q in range(N):
        r = np.kron(r, _BASIS_GATE[basis[q]])
    rho_m = r @ rho @ r.conj().T
    probs = np.clip(np.real(np.diag(rho_m)), 0, None)
    probs = probs / probs.sum()
    return conf @ probs
