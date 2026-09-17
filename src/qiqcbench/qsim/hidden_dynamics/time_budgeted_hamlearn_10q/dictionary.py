"""Public 111-term Pauli dictionary for ``time_budgeted_hamlearn_10q``.

Pure *public* knowledge: the term ordering, term<->Pauli-string
mapping, term weights, and a cached builder for the full Hilbert-space Pauli
operators used to assemble the hidden Hamiltonian.

Note: ``full_pauli_matrix`` returns **dense** 2^10 x 2^10 operators (and the
oracle does a dense ``eigh``). That is fine for this fixed 10-qubit feasibility
gate; promotion to a sparse operator/evolution substrate is deferred until a
reusable black-box-dynamics qtype needs it.

This module is shared helper code for construction / oracle / scorer /
material generation. It is deliberately **outside** the reference solver's
reachable surface: the solver reconstructs equivalent term knowledge from the
``get_device_spec()`` payload and never imports this module.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "EDGE_PAIR_ORDER",
    "N_QUBITS",
    "N_TERMS",
    "TERM_ORDER",
    "TERM_ORDER_CONVENTION",
    "full_pauli_matrix",
    "index_to_term",
    "pauli_string",
    "term_to_index",
    "term_weight",
]

N_QUBITS = 10
N_TERMS = 111

# §8c literal — encodes the required coefficient ordering.
TERM_ORDER_CONVENTION = "one_local_xyz_then_edge_local_xx_xy_xz_yx_yy_yz_zx_zy_zz"

# Order of the nine two-local Pauli pairs on each edge.
EDGE_PAIR_ORDER: tuple[tuple[str, str], ...] = (
    ("X", "X"),
    ("X", "Y"),
    ("X", "Z"),
    ("Y", "X"),
    ("Y", "Y"),
    ("Y", "Z"),
    ("Z", "X"),
    ("Z", "Y"),
    ("Z", "Z"),
)


def _build_term_order() -> list[str]:
    terms: list[str] = []
    # 30 one-local terms: Xi, Yi, Zi for i = 0..9
    for i in range(N_QUBITS):
        for axis in ("X", "Y", "Z"):
            terms.append(f"{axis}{i}")
    # 81 nearest-neighbour two-local terms on edges (i, i+1)
    for i in range(N_QUBITS - 1):
        for a, b in EDGE_PAIR_ORDER:
            terms.append(f"{a}{i}{b}{i + 1}")
    return terms


TERM_ORDER: tuple[str, ...] = tuple(_build_term_order())
assert len(TERM_ORDER) == N_TERMS, f"expected {N_TERMS} terms, got {len(TERM_ORDER)}"

_TERM_TO_INDEX = {term: idx for idx, term in enumerate(TERM_ORDER)}


def term_to_index(term: str) -> int:
    """Index of ``term`` in the canonical ordering."""
    return _TERM_TO_INDEX[term]


def index_to_term(index: int) -> str:
    """Term name at position ``index`` in the canonical ordering."""
    return TERM_ORDER[index]


def _parse_factors(term: str) -> list[tuple[str, int]]:
    """Parse e.g. ``"Z6Z7"`` -> ``[("Z", 6), ("Z", 7)]``."""
    factors: list[tuple[str, int]] = []
    i = 0
    while i < len(term):
        axis = term[i]
        if axis not in "XYZ":
            raise ValueError(f"bad term {term!r}: axis {axis!r}")
        qubit = int(term[i + 1])
        factors.append((axis, qubit))
        i += 2
    return factors


def pauli_string(term: str) -> str:
    """Length-10 ``I/X/Y/Z`` Pauli string for a term (``"Z6Z7"`` -> ``"IIIIIIZZII"``)."""
    chars = ["I"] * N_QUBITS
    for axis, qubit in _parse_factors(term):
        chars[qubit] = axis
    return "".join(chars)


def term_weight(term: str) -> int:
    """Number of non-identity factors (1 for one-local, 2 for two-local)."""
    return len(_parse_factors(term))


_PAULI_2x2: dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

_MATRIX_CACHE: dict[str, np.ndarray] = {}


def full_pauli_matrix(term: str) -> np.ndarray:
    """Dense ``2^10 x 2^10`` operator for ``term`` (qubit 0 is the leftmost factor).

    Cached: construction / oracle reuse these to assemble the hidden
    Hamiltonian ``M = sum_P omega_P P``.
    """
    key = pauli_string(term)
    cached = _MATRIX_CACHE.get(key)
    if cached is not None:
        return cached
    op = np.array([[1.0 + 0j]])
    for char in key:
        op = np.kron(op, _PAULI_2x2[char])
    _MATRIX_CACHE[key] = op
    return op
