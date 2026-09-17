"""Gate library for the digital qtype.

`GATE_REGISTRY` maps wire-protocol gate names to their numpy unitary (constant
matrix or callable taking parameters), the parameter count, and the gate's
qubit arity.

Conventions:
- For 2-qubit gates with `qubits=[qa, qb]`, qa corresponds to bit 0 (LSB) of
  the local 4x4 basis and qb to bit 1. CNOT(qa, qb) treats qa as control,
  qb as target.
- 1-qubit unitaries act on the single qubit in `qubits`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

PAULI_I = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
HADAMARD = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
S_GATE = np.array([[1, 0], [0, 1j]], dtype=complex)
SDG_GATE = np.array([[1, 0], [0, -1j]], dtype=complex)
T_GATE = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
TDG_GATE = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)


def rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
        dtype=complex,
    )


# 2-qubit gates: basis |q0 q1⟩ with q0 as bit 0 (LSB).
# Index map: |00⟩=0, |01⟩=1 (q0=1,q1=0), |10⟩=2 (q0=0,q1=1), |11⟩=3.

# CNOT: q0 = control, q1 = target. Flips q1 iff q0 = 1.
# |00⟩→|00⟩ (0→0); |01⟩(q0=1,q1=0)→|11⟩(q0=1,q1=1) (1→3);
# |10⟩(q0=0,q1=1)→|10⟩ (2→2); |11⟩(q0=1,q1=1)→|01⟩(q0=1,q1=0) (3→1).
CNOT_GATE = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
    ],
    dtype=complex,
)

CZ_GATE = np.diag([1, 1, 1, -1]).astype(complex)

# SWAP: |q0 q1⟩ ↔ |q1 q0⟩, swaps idx 1 ↔ idx 2.
SWAP_GATE = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=complex,
)


GateMatrix = np.ndarray | Callable[..., np.ndarray]


GATE_REGISTRY: dict[str, dict] = {
    "i": {"matrix": PAULI_I, "param_count": 0, "n_qubits": 1},
    "x": {"matrix": PAULI_X, "param_count": 0, "n_qubits": 1},
    "y": {"matrix": PAULI_Y, "param_count": 0, "n_qubits": 1},
    "z": {"matrix": PAULI_Z, "param_count": 0, "n_qubits": 1},
    "h": {"matrix": HADAMARD, "param_count": 0, "n_qubits": 1},
    "s": {"matrix": S_GATE, "param_count": 0, "n_qubits": 1},
    "sdg": {"matrix": SDG_GATE, "param_count": 0, "n_qubits": 1},
    "t": {"matrix": T_GATE, "param_count": 0, "n_qubits": 1},
    "tdg": {"matrix": TDG_GATE, "param_count": 0, "n_qubits": 1},
    "rx": {"matrix": rx, "param_count": 1, "n_qubits": 1},
    "ry": {"matrix": ry, "param_count": 1, "n_qubits": 1},
    "rz": {"matrix": rz, "param_count": 1, "n_qubits": 1},
    "cx": {"matrix": CNOT_GATE, "param_count": 0, "n_qubits": 2},
    "cz": {"matrix": CZ_GATE, "param_count": 0, "n_qubits": 2},
    "swap": {"matrix": SWAP_GATE, "param_count": 0, "n_qubits": 2},
}


def embed_unitary(U_local: np.ndarray, qubits: list[int], n: int) -> np.ndarray:
    """Embed a k-qubit unitary acting on `qubits` into the full n-qubit space.

    `qubits[a]` corresponds to bit a of the local basis index (LSB-first).
    Spectator qubits act as identity. Returns a 2^n x 2^n complex matrix.
    """
    k = len(qubits)
    if U_local.shape != (1 << k, 1 << k):
        raise ValueError(
            f"U_local shape {U_local.shape} != ({1 << k}, {1 << k}) for {k} qubits"
        )
    if any(q < 0 or q >= n for q in qubits):
        raise ValueError(f"Qubit index out of range for n={n}: {qubits}")
    if len(set(qubits)) != k:
        raise ValueError(f"Repeated qubit index in {qubits}")

    if k == n and qubits == list(range(n)):
        return U_local.astype(complex, copy=True)

    dim = 1 << n
    other_qubits = [q for q in range(n) if q not in qubits]
    n_other = len(other_qubits)

    U_full = np.zeros((dim, dim), dtype=complex)
    for other_combo in range(1 << n_other):
        # Spectator-qubit bit values for this block.
        other_mask = 0
        for a, q in enumerate(other_qubits):
            other_mask |= ((other_combo >> a) & 1) << q
        for i_local in range(1 << k):
            i_full = other_mask
            for a in range(k):
                i_full |= ((i_local >> a) & 1) << qubits[a]
            for j_local in range(1 << k):
                j_full = other_mask
                for a in range(k):
                    j_full |= ((j_local >> a) & 1) << qubits[a]
                U_full[i_full, j_full] = U_local[i_local, j_local]
    return U_full
