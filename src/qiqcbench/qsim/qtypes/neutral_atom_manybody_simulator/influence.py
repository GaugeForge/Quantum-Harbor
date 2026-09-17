"""Closed-form loss-influence calculus for the fixed depth-6 Rydberg brickwork.

A lost atom leaves its trap: from the layer it disappears it takes no rotation and its
two bonds carry no ZZ gate, so the remaining atoms evolve under a different circuit.
Everything in this module is a consequence of the *published* protocol -- no hidden
parameter enters here. The hidden loss rates and readout channel are applied by the
caller.

Two structural facts make the whole task tractable in closed form:

* The last ZZ layer is diagonal in the measured basis and therefore cannot move a
  Z-basis outcome. The remaining ``DEPTH - 1`` ZZ layers each widen the backward
  support of the central pair by one site, so a loss can only matter while its site
  still lies inside that support. This bounds both *where* and *how late* a loss can
  matter -- and it makes the reach shrink as the loss happens later.
* The resulting support is ten sites wide, and only eight of them are free (the two
  target atoms must survive to be read out), so the influence configurations form a
  finite set that can be enumerated exactly once and then reused for every shot and
  every candidate selection rule.
"""

from __future__ import annotations

import itertools

import numpy as np

DEPTH = 6
TARGET_OFFSETS = (0, 1)
#: Absence-layer sentinel meaning "this site's loss cannot move the observable".
NO_EFFECT = DEPTH


def max_relevant_absence_layer(offset: int) -> int:
    """Latest absence layer at which a loss at ``offset`` can still move the observable.

    Returns a negative value when a loss at that offset can never matter.
    """
    return min(DEPTH - 2 + offset, DEPTH - 1 - offset)


#: Target-relative offsets whose occupancy can change the scored correlation.
INFLUENCE_OFFSETS: tuple[int, ...] = tuple(
    offset for offset in range(-DEPTH, DEPTH + 2) if max_relevant_absence_layer(offset) >= 0
)
#: The subset a selection rule has to reason about; the two target atoms are always
#: required to survive, so their loss never appears in a retained shot.
FREE_OFFSETS: tuple[int, ...] = tuple(
    offset for offset in INFLUENCE_OFFSETS if offset not in TARGET_OFFSETS
)
#: Number of distinguishable absence classes per free offset: ``0..max_relevant``
#: plus one class for "lost too late, or not lost at all".
CLASS_COUNTS: tuple[int, ...] = tuple(
    max_relevant_absence_layer(offset) + 2 for offset in FREE_OFFSETS
)
CONFIGURATION_COUNT = int(np.prod(CLASS_COUNTS))

_WINDOW = len(INFLUENCE_OFFSETS)
_TARGET_INDEX = INFLUENCE_OFFSETS.index(TARGET_OFFSETS[0])
_FREE_INDICES = tuple(INFLUENCE_OFFSETS.index(offset) for offset in FREE_OFFSETS)


def _ry(angle: float) -> np.ndarray:
    half = angle / 2.0
    return np.array([[np.cos(half), -np.sin(half)], [np.sin(half), np.cos(half)]], dtype=complex)


def _site_signs(n: int) -> np.ndarray:
    """``(+1, -1)`` pattern per site over the flat computational basis."""
    index = np.arange(1 << n)
    return np.stack([1.0 - 2.0 * ((index >> (n - 1 - site)) & 1) for site in range(n)])


def _simulate_pair(
    absent_from: np.ndarray, rotation: tuple, zz: tuple, signs: np.ndarray
) -> np.ndarray:
    """Exact joint distribution of the two target atoms under one loss configuration."""
    n = _WINDOW
    psi = np.zeros(1 << n, dtype=complex)
    psi[0] = 1.0
    for layer, (theta, beta) in enumerate(zip(rotation, zz, strict=True)):
        present = absent_from > layer
        gate = _ry(theta)
        for site in range(n):
            if present[site]:
                block = psi.reshape(1 << site, 2, 1 << (n - site - 1))
                psi = np.einsum("ab,ibj->iaj", gate, block, optimize=True).reshape(-1)
        accumulator = None
        for site in range(layer % 2, n - 1, 2):
            if present[site] and present[site + 1]:
                pair = signs[site] * signs[site + 1]
                accumulator = pair if accumulator is None else accumulator + pair
        if accumulator is not None:
            psi = psi * np.exp(-1j * beta * accumulator)
    probability = np.abs(psi) ** 2
    left, right = signs[_TARGET_INDEX], signs[_TARGET_INDEX + 1]
    joint = np.empty(4)
    for outcome, (a, b) in enumerate(itertools.product((1.0, -1.0), repeat=2)):
        joint[outcome] = probability[(left == a) & (right == b)].sum()
    return joint


def configuration_index(classes: np.ndarray) -> np.ndarray:
    """Mixed-radix index of one or many free-offset class assignments."""
    classes = np.asarray(classes)
    index = np.zeros(classes.shape[:-1], dtype=np.int64)
    for position, count in enumerate(CLASS_COUNTS):
        index = index * count + classes[..., position]
    return index


def _configuration_classes() -> np.ndarray:
    """Every configuration, in ``configuration_index`` order."""
    grids = np.meshgrid(*[np.arange(count) for count in CLASS_COUNTS], indexing="ij")
    return np.stack([grid.reshape(-1) for grid in grids], axis=-1)


def build_pair_distributions(
    rotation_angles: tuple[float, ...], zz_angles: tuple[float, ...]
) -> np.ndarray:
    """``(CONFIGURATION_COUNT, 4)`` exact target-pair distributions for one protocol.

    Outcome order is ``(Z=+1,+1), (+1,-1), (-1,+1), (-1,-1)``.
    """
    rotation, zz = tuple(rotation_angles), tuple(zz_angles)
    signs = _site_signs(_WINDOW)
    classes = _configuration_classes()
    table = np.empty((CONFIGURATION_COUNT, 4))
    for row, assignment in enumerate(classes):
        absent_from = np.full(_WINDOW, NO_EFFECT + 1, dtype=int)
        for position, site in enumerate(_FREE_INDICES):
            layer = int(assignment[position])
            if layer <= max_relevant_absence_layer(FREE_OFFSETS[position]):
                absent_from[site] = layer
        table[row] = _simulate_pair(absent_from, rotation, zz, signs)
    return table


def all_present_classes() -> np.ndarray:
    """The configuration in which no loss can move the observable."""
    return np.array([count - 1 for count in CLASS_COUNTS])


def absence_to_class(absent_from: np.ndarray) -> np.ndarray:
    """Map raw absence layers on the free offsets to their influence classes.

    ``absent_from`` carries one entry per free offset (last axis), using ``NO_EFFECT``
    or above for atoms that survive far enough not to matter.
    """
    limits = np.array([max_relevant_absence_layer(offset) for offset in FREE_OFFSETS])
    classes = np.minimum(absent_from, limits + 1)
    return classes


__all__ = [
    "CLASS_COUNTS",
    "all_present_classes",
    "CONFIGURATION_COUNT",
    "DEPTH",
    "FREE_OFFSETS",
    "INFLUENCE_OFFSETS",
    "NO_EFFECT",
    "TARGET_OFFSETS",
    "absence_to_class",
    "build_pair_distributions",
    "configuration_index",
    "max_relevant_absence_layer",
]
