"""Steane [[7,1,3]] code geometry, the |H_L> target, the classical in-basis Hamming
decode (the "QEC at readout"), and a reference FT-prep op-list.

This module is the **shared engine<->verifier physics** (no parity drift): both the
simulator engine and the verifier import the same stabilizers, decode table, and
F(|H_L>) formula. Every function here is validated noiselessly:
  - |H_L> is a +1 eigenstate of all 6 stabilizers; logical Bloch = (1/sqrt2, 0, 1/sqrt2).
  - the in-basis Hamming decode corrects a single error in the Z / X / Y bases.
  - REFERENCE_PREP produces |H_L> with overlap 1.0.

Convention: paper qubits 1..7 -> 0-indexed 0..6. Natural-order Hamming generators
(equivalent to Goto's Methods group on the same labels; S_z2 = g_z2 . g_z3):
    Z/X supports (0-indexed):  S*1 {3,4,5,6}   S*2 {1,2,5,6}   S*3 {0,2,4,6}
Reading the {3,4,5,6}/{1,2,5,6}/{0,2,4,6} parities as the bits worth 4/2/1, the 3-bit
syndrome equals the flipped qubit's index 1..7 (textbook Hamming).
"""

from __future__ import annotations

import math

import numpy as np

N_DATA = 7

# Natural-order Hamming supports, MSB->LSB (bit worth 4, 2, 1).
STAB_SUPPORTS: tuple[tuple[int, ...], ...] = ((3, 4, 5, 6), (1, 2, 5, 6), (0, 2, 4, 6))

# Ideal single-qubit |H> Bloch vector (the magic-state target), in (X, Y, Z).
H_BLOCH = (1.0 / math.sqrt(2.0), 0.0, 1.0 / math.sqrt(2.0))


# ---------------------------------------------------------------------------
# exact located-erasure correctability
# ---------------------------------------------------------------------------


def _support_mask(support: tuple[int, ...]) -> int:
    return sum(1 << role for role in support)


_STEANE_SUPPORT_MASKS = tuple(_support_mask(support) for support in STAB_SUPPORTS)
_STEANE_SYMPLECTIC_GENERATORS = tuple(
    [(support, 0) for support in _STEANE_SUPPORT_MASKS]
    + [(0, support) for support in _STEANE_SUPPORT_MASKS]
)


def _steane_stabilizer_group() -> frozenset[tuple[int, int]]:
    group: set[tuple[int, int]] = {(0, 0)}
    for generator_x, generator_z in _STEANE_SYMPLECTIC_GENERATORS:
        group.update((old_x ^ generator_x, old_z ^ generator_z) for old_x, old_z in tuple(group))
    return frozenset(group)


_STEANE_STABILIZERS = _steane_stabilizer_group()


def _symplectic_commutes(left: tuple[int, int], right: tuple[int, int]) -> bool:
    left_x, left_z = left
    right_x, right_z = right
    return ((left_x & right_z).bit_count() + (left_z & right_x).bit_count()) % 2 == 0


def _steane_nontrivial_logical_supports() -> frozenset[int]:
    supports: set[int] = set()
    for x_mask in range(1 << N_DATA):
        for z_mask in range(1 << N_DATA):
            pauli = (x_mask, z_mask)
            if pauli in _STEANE_STABILIZERS:
                continue
            if all(
                _symplectic_commutes(pauli, generator)
                for generator in _STEANE_SYMPLECTIC_GENERATORS
            ):
                supports.add(x_mask | z_mask)
    return frozenset(supports)


STEANE_NONTRIVIAL_LOGICAL_SUPPORTS = _steane_nontrivial_logical_supports()


def _erasure_mask_is_correctable(erasure_mask: int) -> bool:
    return not any(
        logical_support & ~erasure_mask == 0
        for logical_support in STEANE_NONTRIVIAL_LOGICAL_SUPPORTS
    )


STEANE_ERASURE_CORRECTABLE: tuple[bool, ...] = tuple(
    _erasure_mask_is_correctable(erasure_mask) for erasure_mask in range(1 << N_DATA)
)


def steane_erasure_correctable(erasure_mask: int) -> bool:
    """Return exact Steane correctability for a seven-role located-erasure support.

    For a stabilizer code, a known erasure support is correctable exactly when it
    contains no support of a nontrivial logical Pauli. This is support-sensitive:
    some three-role Steane erasures are correctable and the seven weight-three
    logical-line supports are not.
    """

    if isinstance(erasure_mask, bool) or not isinstance(erasure_mask, int):
        raise TypeError("Steane erasure mask must be an integer")
    if not 0 <= erasure_mask < 1 << N_DATA:
        raise ValueError("Steane erasure mask must fit exactly seven physical roles")
    return STEANE_ERASURE_CORRECTABLE[erasure_mask]


# ---------------------------------------------------------------------------
# classical in-basis Hamming decode (the QEC at readout) — pure integer logic
# ---------------------------------------------------------------------------


def syndrome(bits: list[int]) -> int:
    """3-bit syndrome (0 = no detected error; 1..7 = paper index of the single error).

    `bits` are the per-qubit transversal-measurement outcomes (0/1), 0-indexed q0..q6.
    Works identically for Z / X / Y bases — each basis's transversal measurement reveals
    its own in-basis stabilizer parities from the same 7 bits.
    """
    s = 0
    for weight, sup in zip((4, 2, 1), STAB_SUPPORTS, strict=True):
        if sum(bits[k] for k in sup) & 1:
            s += weight
    return s


def decode_corrected_logical(bits: list[int]) -> int:
    """Return the corrected logical operator value (+1/-1) = product of (-1)^bit after the
    single-error Hamming correction. This is the QEC-decoded logical estimator."""
    s = syndrome(bits)
    corrected = list(bits)
    if s != 0:
        corrected[s - 1] ^= 1  # paper qubit s -> 0-indexed s-1
    return 1 if (sum(corrected) & 1) == 0 else -1


def naive_logical(bits: list[int]) -> int:
    """The UNcorrected logical estimator = bare product of (-1)^bit (no decode).
    Reporting this instead of `decode_corrected_logical` is failure mode #3."""
    return 1 if (sum(bits) & 1) == 0 else -1


def fidelity_from_bloch(xl: float, yl: float, zl: float) -> float:
    """atom_goto_magic = <H|rho_L|H> = (1 + r . s)/2 with r = (1/sqrt2, 0, 1/sqrt2)."""
    rx, ry, rz = H_BLOCH
    return 0.5 * (1.0 + rx * xl + ry * yl + rz * zl)


# ---------------------------------------------------------------------------
# exact statevector construction of |H_L> (verifier-side ground truth / tests)
# ---------------------------------------------------------------------------

_X = np.array([[0, 1], [1, 0]], dtype=complex)


def _apply_1q(state: np.ndarray, op: np.ndarray, k: int, n: int = N_DATA) -> np.ndarray:
    return np.tensordot(op, state, axes=([1], [k])).transpose(
        list(range(1, k + 1)) + [0] + list(range(k + 1, n))
    )


def ideal_h_logical() -> np.ndarray:
    """The exact |H_L> as a length-2^7 complex statevector (big-endian: q0 is the MSB)."""
    zero = np.zeros((2,) * N_DATA, dtype=complex)
    zero[(0,) * N_DATA] = 1.0
    p0 = zero.copy()
    for sup in STAB_SUPPORTS:  # |0_L> = prod (I + X_sup)/2 |0>^7
        s = p0
        for k in sup:
            s = _apply_1q(s, _X, k)
        p0 = 0.5 * (p0 + s)
    p0 /= np.linalg.norm(p0)
    p1 = p0
    for k in range(N_DATA):  # |1_L> = X^7 |0_L>
        p1 = _apply_1q(p1, _X, k)
    th = math.pi / 8
    hl = math.cos(th) * p0 + math.sin(th) * p1
    return (hl / np.linalg.norm(hl)).reshape(-1)


# ---------------------------------------------------------------------------
# reference FT-prep op-list (the F_opt anchor / reference solver)
# ---------------------------------------------------------------------------
# Verified: executing this on |0>^7 yields |H_L> with overlap 1.0. CNOT(c,t) is
# expressed natively as H_t . CZ(c,t) . H_t (the engine's entangler is the Rydberg CZ).
# Op tuples: ("ry", q, angle) | ("h", q) | ("cnot", c, t). The reference solver / tests
# expand "cnot" into the native [h t; cz c t; h t] for the gate-level program.

_PI_4 = math.pi / 4
_ANCHORS = {3: STAB_SUPPORTS[0], 1: STAB_SUPPORTS[1], 0: STAB_SUPPORTS[2]}

REFERENCE_PREP_LOGICAL: list[tuple] = (
    [("ry", 2, _PI_4), ("cnot", 2, 4), ("cnot", 2, 5)]  # magic on q2, spread to {4,5}
    + [("h", a) for a in _ANCHORS]  # anchors -> |+>
    + [("cnot", a, t) for a, sup in _ANCHORS.items() for t in sup if t != a]  # encode
)

__all__ = [
    "N_DATA",
    "STAB_SUPPORTS",
    "H_BLOCH",
    "syndrome",
    "decode_corrected_logical",
    "naive_logical",
    "fidelity_from_bloch",
    "ideal_h_logical",
    "REFERENCE_PREP_LOGICAL",
    "STEANE_ERASURE_CORRECTABLE",
    "STEANE_NONTRIVIAL_LOGICAL_SUPPORTS",
    "steane_erasure_correctable",
]
