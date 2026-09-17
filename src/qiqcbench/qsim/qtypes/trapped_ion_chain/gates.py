"""Trapped-ion-chain gate library.

Convention: qubit 0 is the LEAST significant bit of the basis index, matching
the digital_gate_model engine. A length-n bitstring with leftmost char acting
on qubit 0 has basis index sum_{i} (bit_i << i).

MS gate: U_MS(theta) = exp(-i (theta/2) sum_{i<j in active} X_i X_j).
The pairwise XX form matches the H_MS = J sum_{i<j} X_i X_j; the
GMS collective (sum X_i)^2 differs by N*I diagonal terms (for N>=3) and is
not the convention here. Verified against librarian-gate-conventions.md.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

_TOL = 1e-12
_MS_MAX_N = 14

PAULI_I = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _rx(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]], dtype=complex)


ONE_QUBIT_GATE_MATRICES: dict[str, np.ndarray | Callable[[float], np.ndarray]] = {
    "i": PAULI_I,
    "x": PAULI_X,
    "y": PAULI_Y,
    "z": PAULI_Z,
    "h": (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex),
    "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    "sdg": np.array([[1, 0], [0, -1j]], dtype=complex),
    "rx": _rx,
    "ry": _ry,
    "rz": _rz,
}


def embed_unitary(U_local: np.ndarray, qubits: list[int], n_total: int) -> np.ndarray:
    """Embed a local k-qubit unitary on `qubits` into the n_total-qubit Hilbert space.

    Qubit-0-as-LSB convention. For 1q: returns I^(n-q-1) ⊗ U ⊗ I^q so that
    bit q of the basis index is acted on by U.
    For multi-qubit: uses index-permutation to apply U_local on the listed qubits.

    Mirrors digital_gate_model/gates.py::embed_unitary.
    """
    dim = 1 << n_total
    k = len(qubits)
    if U_local.shape != (1 << k, 1 << k):
        raise ValueError(
            f"U_local has shape {U_local.shape}, expected ({1 << k}, {1 << k}) for {k} qubits"
        )
    if any(q < 0 or q >= n_total for q in qubits):
        raise ValueError(f"Qubit index out of range for n={n_total}: {qubits}")
    if len(set(qubits)) != k:
        raise ValueError(f"Repeated qubit index in {qubits}")
    if k == 1:
        # Fast path: kron I⊗U⊗I according to qubit-0-as-LSB convention.
        q = qubits[0]
        left = np.eye(1 << (n_total - q - 1), dtype=complex)
        right = np.eye(1 << q, dtype=complex)
        return np.kron(left, np.kron(U_local, right))
    # General path: index-permutation embedding.
    # Reshape state vector as tensor with each axis = one qubit (axis 0 = qubit n-1, etc.).
    # We construct U_full element by element by tracking which basis indices map.
    other = [q for q in range(n_total) if q not in qubits]
    perm = qubits + other  # active qubits first, then the rest

    def _basis_to_perm(idx: int) -> int:
        out = 0
        for new_bit, src_q in enumerate(perm):
            out |= ((idx >> src_q) & 1) << new_bit
        return out

    def _perm_to_basis(idx_p: int) -> int:
        out = 0
        for new_bit, src_q in enumerate(perm):
            out |= ((idx_p >> new_bit) & 1) << src_q
        return out

    U_full = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        col_p = _basis_to_perm(col)
        col_active = col_p & ((1 << k) - 1)
        col_rest = col_p >> k
        # Apply U_local on the active part.
        for row_active in range(1 << k):
            amp = U_local[row_active, col_active]
            if abs(amp) < _TOL:
                continue
            row_p = row_active | (col_rest << k)
            row = _perm_to_basis(row_p)
            U_full[row, col] += amp
    return U_full


def ms_unitary(n_qubits: int, theta: float, active: list[int] | None = None) -> np.ndarray:
    """Global Mølmer-Sørensen entangler over the active ion subset.

    U_MS(theta) = exp(-i (theta/2) sum_{i<j in active} X_i X_j).

    Implementation: H is diagonal in the X-basis. Apply Hadamards on all active
    qubits (H_act); in that basis each X_i becomes Z_i, and Σ_{i<j} Z_i Z_j has
    eigenvalues Σ_{i<j} z_i z_j where z_i ∈ {+1, -1}: bit i unset → +1,
    bit i set → -1. So the unitary is H_act† · diag(exp(-i θ/2 · eig)) · H_act.

    Complexity: O(4^N) to materialize the dense unitary; the eigenvalue vector
    itself costs O(N · 2^N). Memory: O(4^N) for the returned dense unitary.
    Engine-facing dense use is capped at 14 ions.

    All X-string pair operators X_iX_j and X_kX_l commute pairwise
    (X² = I → all products of X/I matrices commute), so the diagonalization
    is exact and order-independent.
    """
    if n_qubits > _MS_MAX_N:
        raise ValueError(
            f"ms_unitary supports up to {_MS_MAX_N} qubits as a dense unitary; "
            f"got {n_qubits}. For larger systems use direct state-vector application "
            f"of the X-basis diagonal phase (Task 3 engine path)."
        )
    if active is None:
        active = list(range(n_qubits))
    if any(q < 0 or q >= n_qubits for q in active):
        raise ValueError(
            f"ms_unitary active={active} contains out-of-range qubit for n_qubits={n_qubits}"
        )
    if len(set(active)) != len(active):
        raise ValueError(f"ms_unitary active={active} contains duplicate qubits")
    dim = 1 << n_qubits

    eigvals = np.zeros(dim, dtype=float)
    for idx in range(dim):
        z_sum = 0
        for q in active:
            z_sum += 1 if ((idx >> q) & 1) == 0 else -1
        eigvals[idx] = (z_sum * z_sum - len(active)) / 2

    hadamard = ONE_QUBIT_GATE_MATRICES["h"]
    H_active = np.eye(dim, dtype=complex)
    for q in active:
        H_active = embed_unitary(hadamard, [q], n_qubits) @ H_active

    phase = np.exp(-1j * (theta / 2) * eigvals)
    return H_active.conj().T @ (H_active * phase[:, None])
