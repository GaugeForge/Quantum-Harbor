"""Circuit representation + dense reference simulator for the bounded-gate qtype.

The hidden device is a fixed n-qubit circuit built from Clifford gates
(H, S, nearest-neighbour CX/CZ) plus a bounded number of single-qubit
non-Clifford rotations ``exp(-i * x[angle] * P / 2)`` driven by a small input
vector ``x``. The circuit itself is hidden truth; agents only see shot data.

Conventions (match the digital track):
- qubit 0 is the least-significant bit of a dense statevector index;
- two-qubit gates act on adjacent qubits only (keeps the runtime MPS exact);
- measuring in basis P applies U with U^dag Z U = P, then measures Z
  (X -> H, Y -> H @ S^dag).

The dense path here is a verification reference for small n (unit tests and
hidden-dynamics cross-checks). The runtime engine is the MPS in ``sim.py``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_DENSE_QUBITS = 14


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CircuitOp(_Strict):
    """One gate of the hidden bounded-gate circuit.

    ``gate`` in {"h", "s"}: single-qubit Clifford on ``q``.
    ``gate`` in {"cx", "cz"}: two-qubit Clifford on adjacent (q, q2);
    for ``cx`` the control is ``q`` and the target is ``q2``.
    ``gate == "rot"``: exp(-i * x[angle] * P_axis / 2) on ``q``.
    """

    gate: Literal["h", "s", "cx", "cz", "rot"]
    q: int = Field(..., ge=0)
    q2: int | None = Field(None, ge=0)
    axis: Literal["x", "y", "z"] | None = None
    angle: int | None = Field(None, ge=0)

    @model_validator(mode="after")
    def _check_shape(self) -> CircuitOp:
        if self.gate in ("cx", "cz"):
            if self.q2 is None or abs(self.q2 - self.q) != 1:
                raise ValueError(f"{self.gate} requires q2 adjacent to q")
            if self.axis is not None or self.angle is not None:
                raise ValueError(f"{self.gate} takes no axis/angle")
        elif self.gate == "rot":
            if self.axis is None or self.angle is None:
                raise ValueError("rot requires axis and angle index")
            if self.q2 is not None:
                raise ValueError("rot takes no q2")
        else:
            if self.q2 is not None or self.axis is not None or self.angle is not None:
                raise ValueError(f"{self.gate} takes only q")
        return self


def max_qubit(ops: list[CircuitOp]) -> int:
    hi = 0
    for op in ops:
        hi = max(hi, op.q, op.q2 if op.q2 is not None else 0)
    return hi


def n_angles(ops: list[CircuitOp]) -> int:
    return 1 + max((op.angle for op in ops if op.angle is not None), default=-1)


# ---------- gate matrices ----------

_SQ2 = 1.0 / np.sqrt(2.0)
I2 = np.eye(2, dtype=complex)
H = np.array([[_SQ2, _SQ2], [_SQ2, -_SQ2]], dtype=complex)
S = np.array([[1.0, 0.0], [0.0, 1j]], dtype=complex)
SDG = S.conj().T
PAULI = {
    "x": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "y": np.array([[0.0, -1j], [1j, 0.0]], dtype=complex),
    "z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}
# Measuring basis P == measuring Z after applying BASIS_ROTATIONS[P]
# (U^dag Z U = P; for y: (H Sdg)^dag Z (H Sdg) = S X Sdg = Y).
BASIS_ROTATIONS = {"x": H, "y": H @ SDG, "z": I2}
BASIS_ORDER = ("x", "y", "z")
BASIS_INDEX = {name: k for k, name in enumerate(BASIS_ORDER)}


def rotation_matrix(axis: str, theta: float) -> np.ndarray:
    return np.cos(theta / 2.0) * I2 - 1j * np.sin(theta / 2.0) * PAULI[axis]


def one_qubit_unitary(op: CircuitOp, x: np.ndarray) -> np.ndarray:
    if op.gate == "h":
        return H
    if op.gate == "s":
        return S
    if op.gate == "rot":
        assert op.axis is not None and op.angle is not None
        return rotation_matrix(op.axis, float(x[op.angle]))
    raise ValueError(f"not a one-qubit gate: {op.gate}")


def two_qubit_unitary(op: CircuitOp) -> np.ndarray:
    """4x4 unitary indexed by (hi, lo) bit pairs, hi = max(q, q2)."""
    assert op.q2 is not None
    u = np.zeros((4, 4), dtype=complex)
    hi, lo = max(op.q, op.q2), min(op.q, op.q2)
    for b_hi in (0, 1):
        for b_lo in (0, 1):
            bits = {hi: b_hi, lo: b_lo}
            if op.gate == "cz":
                phase = -1.0 if b_hi == 1 and b_lo == 1 else 1.0
                out_bits = dict(bits)
            elif op.gate == "cx":
                out_bits = dict(bits)
                if bits[op.q] == 1:
                    out_bits[op.q2] ^= 1
                phase = 1.0
            else:
                raise ValueError(f"not a two-qubit gate: {op.gate}")
            row = 2 * out_bits[hi] + out_bits[lo]
            col = 2 * b_hi + b_lo
            u[row, col] += phase
    return u


# ---------- dense reference simulator (small n only) ----------


def _apply_1q_dense(state: np.ndarray, u: np.ndarray, q: int, n: int) -> np.ndarray:
    psi = state.reshape(2 ** (n - 1 - q), 2, 2**q)
    return np.einsum("ab,hbl->hal", u, psi).reshape(-1)


def _apply_2q_dense(state: np.ndarray, u4: np.ndarray, q_lo: int, n: int) -> np.ndarray:
    psi = state.reshape(2 ** (n - 2 - q_lo), 2, 2, 2**q_lo)
    u = u4.reshape(2, 2, 2, 2)  # (hi', lo', hi, lo)
    return np.einsum("abcd,xcdl->xabl", u, psi).reshape(-1)


def dense_state(ops: list[CircuitOp], x: np.ndarray, n: int) -> np.ndarray:
    """Exact statevector of the circuit output (reference; n <= MAX_DENSE_QUBITS)."""
    if n > MAX_DENSE_QUBITS:
        raise ValueError(f"dense reference limited to n <= {MAX_DENSE_QUBITS}")
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0
    for op in ops:
        if op.gate in ("cx", "cz"):
            assert op.q2 is not None
            state = _apply_2q_dense(state, two_qubit_unitary(op), min(op.q, op.q2), n)
        else:
            state = _apply_1q_dense(state, one_qubit_unitary(op, x), op.q, n)
    return state


def dense_pauli_expectation(state: np.ndarray, pauli: dict[int, str], n: int) -> float:
    """<state| prod_q P_q |state> for a sparse Pauli given as {qubit: axis}."""
    phi = state
    for q, axis in pauli.items():
        phi = _apply_1q_dense(phi, PAULI[axis], q, n)
    return float(np.real(np.vdot(state, phi)))
