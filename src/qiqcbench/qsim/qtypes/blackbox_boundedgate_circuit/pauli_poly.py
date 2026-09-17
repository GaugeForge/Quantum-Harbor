"""Closed-form trigonometric polynomials for bounded-gate circuit observables.

For a circuit of Clifford gates plus R rotation gates exp(-i * x[k] * P / 2),
the Heisenberg back-propagation of any Pauli observable branches only at
anticommuting rotations, so

    <0..0| U(x)^dag  O  U(x) |0..0>

is exactly a signed sum of monomials  prod_k cos(x_k)^c_k sin(x_k)^s_k  with
at most 2^R terms. This module extracts that polynomial symbolically — it is
the verifier-side truth engine (never agent-visible) and is cross-checked
against the dense reference simulator in unit tests.

Pauli bookkeeping uses the symplectic W-representation W(a, b) = X^a Z^b
(per-qubit products, X before Z), with an explicit complex phase carried per
term; Y = i * X * Z. Clifford conjugation tables are generated numerically
from the gate matrices at import time, so no hand-derived signs enter.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.circuit import (
    PAULI,
    CircuitOp,
    H,
    S,
    two_qubit_unitary,
)

DEFAULT_MAX_TERMS = 200_000

_AXIS_BITS = {"x": (1, 0), "y": (1, 1), "z": (0, 1)}
# W-representation prefactor: Y = i * X * Z = i * W(1, 1).
_AXIS_PHASE = {"x": 1.0 + 0j, "y": 1j, "z": 1.0 + 0j}


def _w_matrix_1q(xbit: int, zbit: int) -> np.ndarray:
    m = np.eye(2, dtype=complex)
    if xbit:
        m = m @ PAULI["x"]
    if zbit:
        m = m @ PAULI["z"]
    return m


def _match_w(m: np.ndarray, n_qubits_local: int) -> tuple[tuple[int, ...], complex]:
    """Express m as phase * W(bits) over the local qubits; raise if not a Pauli."""
    dim = 2**n_qubits_local
    for code in range(4**n_qubits_local):
        bits = []
        w = np.eye(1, dtype=complex)
        c = code
        for _ in range(n_qubits_local):
            xbit, zbit = (c >> 0) & 1, (c >> 1) & 1
            c >>= 2
            bits.extend([xbit, zbit])
            w = np.kron(_w_matrix_1q(xbit, zbit), w)  # earlier qubit = low kron factor
        phase = np.trace(w.conj().T @ m) / dim
        if abs(abs(phase) - 1.0) < 1e-9 and np.allclose(m, phase * w, atol=1e-9):
            return tuple(bits), complex(phase)
    raise ValueError("matrix is not proportional to a Pauli — non-Clifford conjugation?")


def _build_1q_table(u: np.ndarray) -> dict[tuple[int, int], tuple[int, int, complex]]:
    table: dict[tuple[int, int], tuple[int, int, complex]] = {}
    for xbit in (0, 1):
        for zbit in (0, 1):
            m = u.conj().T @ _w_matrix_1q(xbit, zbit) @ u
            bits, phase = _match_w(m, 1)
            table[(xbit, zbit)] = (bits[0], bits[1], phase)
    return table


def _build_2q_table(u4: np.ndarray) -> dict[tuple[int, ...], tuple[tuple[int, ...], complex]]:
    """Conjugation table for W ops on an adjacent (lo, hi) pair.

    Local key/value bit order: (x_lo, z_lo, x_hi, z_hi). ``u4`` is indexed by
    (hi, lo) pairs, matching circuit.two_qubit_unitary — i.e. the hi qubit is
    the high kron factor.
    """
    table: dict[tuple[int, ...], tuple[tuple[int, ...], complex]] = {}
    for x_lo in (0, 1):
        for z_lo in (0, 1):
            for x_hi in (0, 1):
                for z_hi in (0, 1):
                    w = np.kron(_w_matrix_1q(x_hi, z_hi), _w_matrix_1q(x_lo, z_lo))
                    m = u4.conj().T @ w @ u4
                    bits, phase = _match_w(m, 2)
                    table[(x_lo, z_lo, x_hi, z_hi)] = (bits, phase)
    return table


_CONJ_1Q = {"h": _build_1q_table(H), "s": _build_1q_table(S)}
_CZ_OP = CircuitOp(gate="cz", q=0, q2=1)
_CX_LOW_CONTROL = CircuitOp(gate="cx", q=0, q2=1)
_CX_HIGH_CONTROL = CircuitOp(gate="cx", q=1, q2=0)
_CONJ_2Q = {
    "cz": _build_2q_table(two_qubit_unitary(_CZ_OP)),
    "cx_low_control": _build_2q_table(two_qubit_unitary(_CX_LOW_CONTROL)),
    "cx_high_control": _build_2q_table(two_qubit_unitary(_CX_HIGH_CONTROL)),
}


@dataclass(frozen=True)
class TrigTerm:
    coeff: float
    cos_pow: tuple[int, ...]
    sin_pow: tuple[int, ...]


class TrigPoly:
    """f(x) = sum_t coeff_t * prod_k cos(x_k)^c_tk * sin(x_k)^s_tk."""

    def __init__(self, d: int, terms: list[TrigTerm]):
        self.d = d
        self.terms = terms

    def evaluate_batch(self, xs: np.ndarray) -> np.ndarray:
        xs = np.atleast_2d(np.asarray(xs, dtype=float))
        if xs.shape[1] != self.d:
            raise ValueError(f"expected inputs of dimension {self.d}, got {xs.shape[1]}")
        cos_x, sin_x = np.cos(xs), np.sin(xs)
        out = np.zeros(xs.shape[0])
        for term in self.terms:
            factor = np.full(xs.shape[0], term.coeff)
            for k in range(self.d):
                if term.cos_pow[k]:
                    factor = factor * cos_x[:, k] ** term.cos_pow[k]
                if term.sin_pow[k]:
                    factor = factor * sin_x[:, k] ** term.sin_pow[k]
            out += factor
        return out

    def evaluate(self, x: np.ndarray) -> float:
        return float(self.evaluate_batch(np.asarray(x, dtype=float)[None, :])[0])

    def angles_touched(self) -> set[int]:
        touched: set[int] = set()
        for term in self.terms:
            for k in range(self.d):
                if term.cos_pow[k] or term.sin_pow[k]:
                    touched.add(k)
        return touched

    def is_constant(self, tol: float = 1e-12) -> bool:
        return all(
            (not any(t.cos_pow) and not any(t.sin_pow)) or abs(t.coeff) <= tol for t in self.terms
        )

    def to_jsonable(self) -> dict:
        return {
            "d": self.d,
            "terms": [
                {"coeff": t.coeff, "cos_pow": list(t.cos_pow), "sin_pow": list(t.sin_pow)}
                for t in self.terms
            ],
        }

    @classmethod
    def from_jsonable(cls, data: dict) -> TrigPoly:
        return cls(
            int(data["d"]),
            [
                TrigTerm(
                    coeff=float(t["coeff"]),
                    cos_pow=tuple(int(v) for v in t["cos_pow"]),
                    sin_pow=tuple(int(v) for v in t["sin_pow"]),
                )
                for t in data["terms"]
            ],
        )


def _popcount_parity(mask: int) -> int:
    return bin(mask).count("1") & 1


def _accumulate(
    store: dict[tuple[int, int, tuple[tuple[int, ...], tuple[int, ...]]], complex],
    key: tuple[int, int, tuple[tuple[int, ...], tuple[int, ...]]],
    c: complex,
) -> None:
    acc = store.get(key, 0j) + c
    if acc == 0:
        store.pop(key, None)
    else:
        store[key] = acc


def backpropagate_pauli(
    ops: list[CircuitOp],
    d: int,
    pauli: dict[int, str],
    *,
    max_terms: int = DEFAULT_MAX_TERMS,
) -> TrigPoly:
    """Exact trig polynomial for <0|U(x)^dag (prod_q P_q) U(x)|0>."""
    a_mask = 0
    b_mask = 0
    phase = 1.0 + 0j
    for q, axis in pauli.items():
        xbit, zbit = _AXIS_BITS[axis]
        a_mask |= xbit << q
        b_mask |= zbit << q
        phase *= _AXIS_PHASE[axis]

    zero_mono = ((0,) * d, (0,) * d)
    # terms: (a_mask, b_mask, (cos_pow, sin_pow)) -> complex coeff
    terms: dict[tuple[int, int, tuple[tuple[int, ...], tuple[int, ...]]], complex] = {
        (a_mask, b_mask, zero_mono): phase
    }

    for op in reversed(ops):
        # Exact causal-cone skip: a gate whose qubits carry identity in every
        # term commutes with all of them (Cliffords fix identity; a rotation's
        # generator commutes with identity), so it cannot change anything.
        union = 0
        for a, b, _mono in terms:
            union |= a | b
        gate_mask = 1 << op.q
        if op.q2 is not None:
            gate_mask |= 1 << op.q2
        if gate_mask & union == 0:
            continue

        new_terms: dict[tuple[int, int, tuple[tuple[int, ...], tuple[int, ...]]], complex] = {}
        _add = functools.partial(_accumulate, new_terms)

        if op.gate in ("h", "s"):
            table = _CONJ_1Q[op.gate]
            q = op.q
            for (a, b, mono), coeff in terms.items():
                local = ((a >> q) & 1, (b >> q) & 1)
                nx, nz, ph = table[local]
                na = (a & ~(1 << q)) | (nx << q)
                nb = (b & ~(1 << q)) | (nz << q)
                _add((na, nb, mono), coeff * ph)
        elif op.gate in ("cx", "cz"):
            assert op.q2 is not None
            lo, hi = min(op.q, op.q2), max(op.q, op.q2)
            if op.gate == "cz":
                table = _CONJ_2Q["cz"]
            elif op.q == lo:
                table = _CONJ_2Q["cx_low_control"]
            else:
                table = _CONJ_2Q["cx_high_control"]
            for (a, b, mono), coeff in terms.items():
                local = ((a >> lo) & 1, (b >> lo) & 1, (a >> hi) & 1, (b >> hi) & 1)
                (nx_lo, nz_lo, nx_hi, nz_hi), ph = table[local]
                na = (a & ~((1 << lo) | (1 << hi))) | (nx_lo << lo) | (nx_hi << hi)
                nb = (b & ~((1 << lo) | (1 << hi))) | (nz_lo << lo) | (nz_hi << hi)
                _add((na, nb, mono), coeff * ph)
        else:  # rot: exp(-i x[k] P / 2); backward step O -> cos O + sin * (i P O)
            assert op.axis is not None and op.angle is not None
            k = op.angle
            if k >= d:
                raise ValueError(f"rot angle index {k} out of range for d={d}")
            p_xbit, p_zbit = _AXIS_BITS[op.axis]
            px = p_xbit << op.q
            pz = p_zbit << op.q
            p_phase = _AXIS_PHASE[op.axis]
            for (a, b, mono), coeff in terms.items():
                commutes = (_popcount_parity(a & pz) ^ _popcount_parity(b & px)) == 0
                if commutes:
                    _add((a, b, mono), coeff)
                    continue
                cos_pow, sin_pow = mono
                cos_mono = (_bump(cos_pow, k), sin_pow)
                sin_mono = (cos_pow, _bump(sin_pow, k))
                _add((a, b, cos_mono), coeff)
                # i * P * O = i * p_phase * (-1)^{|pz & a|} * W(px^a, pz^b)
                sign = -1.0 if _popcount_parity(pz & a) else 1.0
                _add(
                    (a ^ px, b ^ pz, sin_mono),
                    coeff * 1j * p_phase * sign,
                )
        terms = new_terms
        if len(terms) > max_terms:
            raise RuntimeError(f"trig-poly term count {len(terms)} exceeded max_terms={max_terms}")

    # Project onto |0..0>: <0|W(a, b)|0> = 1 if a == 0 else 0.
    poly_terms: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
    for (a, _b, mono), coeff in terms.items():
        if a != 0:
            continue
        if abs(coeff.imag) > 1e-9:
            raise RuntimeError(f"non-real surviving coefficient {coeff}")
        poly_terms[mono] = poly_terms.get(mono, 0.0) + coeff.real

    return TrigPoly(
        d,
        [
            TrigTerm(coeff=c, cos_pow=mono[0], sin_pow=mono[1])
            for mono, c in sorted(poly_terms.items())
            if abs(c) > 1e-14
        ],
    )


def _bump(powers: tuple[int, ...], k: int) -> tuple[int, ...]:
    return powers[:k] + (powers[k] + 1,) + powers[k + 1 :]
