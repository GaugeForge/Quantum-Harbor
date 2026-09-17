"""Chip layout + rotated-patch geometry for ``surface_code_lattice_surgery``.

One fixed transmon-style 2D grid hosts every experiment configuration on published subsets
of the *same* physical qubits (so the hidden per-qubit noise is shared across layouts):

- ``d3_control``   — rotated d=3 patch C   (data x in {1,3,5},   y in {1,3,5})
- ``d3_intermediate`` — rotated d=3 patch INT (data x in {1,3,5}, y in {9,11,13})
- ``d3_target``    — rotated d=3 patch T   (data x in {9,11,13}, y in {9,11,13})
- ``d5``           — rotated d=5 patch     (data x in {17..25},  y in {1..9}), dedicated region
- routing row R1 (x in {1,3,5}, y=7): activated only by the C-INT (ZZ) merge
- routing col R2 (x=7, y in {9,11,13}): activated only by the INT-T (XX) merge

Conventions (global, all patches):

- Data qubits sit at odd (x, y); ancillas at even (x, y). An ancilla at (x, y) couples to
  the data qubits at (x±1, y±1) present in its patch.
- Plaquette basis type is a global checkerboard: **Z** if ``((x + y) / 2) % 2 == 0`` else
  **X**. Left/right patch boundaries host only Z-type weight-2 plaquettes, top/bottom only
  X-type — so the logical Z̄ is a horizontal data row (terminating on the left/right
  boundaries) and X̄ is a vertical data column.
- Merging two patches vertically (C over INT) makes horizontal Z̄ rows of both patches
  representatives of the same merged coset → the merge measures ``Z̄_C Z̄_INT`` (rough
  merge; routing row prepared in |+>). Merging horizontally (INT beside T) measures
  ``X̄_INT X̄_T`` (smooth merge; routing column prepared in |0>).

By construction no memory layout contains a routing-region qubit;
``routing_disjointness_ok()`` asserts it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Coord = tuple[int, int]


def plaquette_type(pos: Coord) -> str:
    """Global checkerboard: 'Z' or 'X' for an even-coordinate plaquette position."""
    x, y = pos
    return "Z" if ((x + y) // 2) % 2 == 0 else "X"


@dataclass(frozen=True)
class Plaquette:
    pos: Coord  # even (x, y)
    basis: str  # 'Z' | 'X'
    data: tuple[Coord, ...]  # supported data-qubit coords (2 or 4)


@dataclass(frozen=True)
class PatchGeometry:
    """A rotated (h rows x w cols) surface-code region of data qubits."""

    name: str
    x0: int  # min data x (odd)
    y0: int  # min data y (odd)
    w: int  # data columns
    h: int  # data rows
    plaquettes: tuple[Plaquette, ...] = field(default=(), compare=False)

    @property
    def data(self) -> list[Coord]:
        return [(self.x0 + 2 * i, self.y0 + 2 * j) for j in range(self.h) for i in range(self.w)]

    def z_logical_row(self, row: int = 0) -> list[Coord]:
        """Horizontal Z̄ representative on data row ``row`` (weight w)."""
        y = self.y0 + 2 * row
        return [(self.x0 + 2 * i, y) for i in range(self.w)]

    def x_logical_col(self, col: int = 0) -> list[Coord]:
        """Vertical X̄ representative on data column ``col`` (weight h)."""
        x = self.x0 + 2 * col
        return [(x, self.y0 + 2 * j) for j in range(self.h)]


def build_patch(name: str, x0: int, y0: int, w: int, h: int) -> PatchGeometry:
    data = {(x0 + 2 * i, y0 + 2 * j) for i in range(w) for j in range(h)}
    xmin, xmax = x0 - 1, x0 + 2 * w - 1
    ymin, ymax = y0 - 1, y0 + 2 * h - 1
    plaquettes: list[Plaquette] = []
    for x in range(xmin, xmax + 1, 2):
        for y in range(ymin, ymax + 1, 2):
            support = tuple(
                sorted(
                    q
                    for q in ((x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1), (x + 1, y + 1))
                    if q in data
                )
            )
            basis = plaquette_type((x, y))
            if len(support) == 4:
                plaquettes.append(Plaquette((x, y), basis, support))
            elif len(support) == 2:
                on_lr = x == xmin or x == xmax
                on_tb = y == ymin or y == ymax
                # weight-2 plaquettes: Z on left/right boundaries, X on top/bottom.
                if on_lr and not on_tb and basis == "Z":
                    plaquettes.append(Plaquette((x, y), basis, support))
                elif on_tb and not on_lr and basis == "X":
                    plaquettes.append(Plaquette((x, y), basis, support))
    geo = PatchGeometry(name=name, x0=x0, y0=y0, w=w, h=h, plaquettes=tuple(plaquettes))
    n_expected = w * h - 1
    if len(plaquettes) != n_expected:
        raise AssertionError(
            f"{name}: expected {n_expected} plaquettes for {h}x{w} rotated patch, "
            f"got {len(plaquettes)}"
        )
    return geo


# --------------------------------------------------------------------------- #
# The chip: fixed patch placements.
# --------------------------------------------------------------------------- #

PATCH_C = build_patch("c", 1, 1, 3, 3)
PATCH_INT = build_patch("int", 1, 9, 3, 3)
PATCH_T = build_patch("t", 9, 9, 3, 3)
PATCH_D5 = build_patch("d5", 17, 1, 5, 5)

# Merged patches (the code during a merge window).
MERGED_ZZ = build_patch("merged_zz", 1, 1, 3, 7)  # C + routing row + INT (vertical)
MERGED_XX = build_patch("merged_xx", 1, 9, 7, 3)  # INT + routing col + T (horizontal)

ROUTING_ZZ: tuple[Coord, ...] = ((1, 7), (3, 7), (5, 7))  # routing row R1
ROUTING_XX: tuple[Coord, ...] = ((7, 9), (7, 11), (7, 13))  # routing col R2

MEMORY_PATCHES: dict[str, PatchGeometry] = {
    "d3_control": PATCH_C,
    "d3_intermediate": PATCH_INT,
    "d3_target": PATCH_T,
    "d5": PATCH_D5,
}


def _chip_coords() -> list[Coord]:
    coords: set[Coord] = set()
    for patch in (PATCH_C, PATCH_INT, PATCH_T, PATCH_D5, MERGED_ZZ, MERGED_XX):
        coords.update(patch.data)
        coords.update(p.pos for p in patch.plaquettes)
    return sorted(coords, key=lambda c: (c[1], c[0]))


CHIP_COORDS: tuple[Coord, ...] = tuple(_chip_coords())
COORD_TO_ID: dict[Coord, int] = {c: i for i, c in enumerate(CHIP_COORDS)}
N_CHIP_QUBITS: int = len(CHIP_COORDS)


def qubit_id(coord: Coord) -> int:
    return COORD_TO_ID[coord]


def is_data(coord: Coord) -> bool:
    return coord[0] % 2 == 1


def routing_qubit_ids() -> set[int]:
    """All qubits (data + ancilla) that appear ONLY in merged layouts."""
    memory_coords: set[Coord] = set()
    for patch in MEMORY_PATCHES.values():
        memory_coords.update(patch.data)
        memory_coords.update(p.pos for p in patch.plaquettes)
    return {COORD_TO_ID[c] for c in CHIP_COORDS if c not in memory_coords}


def routing_disjointness_ok() -> bool:
    """Brief §13.2 gate: no memory layout touches a routing-region qubit."""
    routing_data = set(ROUTING_ZZ) | set(ROUTING_XX)
    for patch in MEMORY_PATCHES.values():
        if routing_data & set(patch.data):
            return False
        for p in patch.plaquettes:
            if routing_data & set(p.data):
                return False
    return True


# --------------------------------------------------------------------------- #
# Extraction schedule (public): 4 interaction sub-steps per round.
# X-plaquettes traverse their data in an N-shaped order, Z-plaquettes in its transpose
# (Z-shaped) — the standard hook-suppressing pairing. The sub-step order fixes the
# EMERGENT hook orientations; the hidden rates make them inhomogeneous.
# --------------------------------------------------------------------------- #

_ORDER_X: tuple[Coord, ...] = ((1, -1), (-1, -1), (1, 1), (-1, 1))
_ORDER_Z: tuple[Coord, ...] = ((1, -1), (1, 1), (-1, -1), (-1, 1))


def interaction_substeps(plaquettes: list[Plaquette]) -> list[list[tuple[Coord, Coord, str]]]:
    """Per sub-step [0..3], the (ancilla_pos, data_pos, basis) interactions in that step."""
    steps: list[list[tuple[Coord, Coord, str]]] = [[], [], [], []]
    for p in plaquettes:
        order = _ORDER_X if p.basis == "X" else _ORDER_Z
        for k, (dx, dy) in enumerate(order):
            d = (p.pos[0] + dx, p.pos[1] + dy)
            if d in p.data:
                steps[k].append((p.pos, d, p.basis))
    return steps


__all__ = [
    "CHIP_COORDS",
    "COORD_TO_ID",
    "Coord",
    "MEMORY_PATCHES",
    "MERGED_XX",
    "MERGED_ZZ",
    "N_CHIP_QUBITS",
    "PATCH_C",
    "PATCH_D5",
    "PATCH_INT",
    "PATCH_T",
    "Plaquette",
    "PatchGeometry",
    "ROUTING_XX",
    "ROUTING_ZZ",
    "build_patch",
    "interaction_substeps",
    "is_data",
    "plaquette_type",
    "qubit_id",
    "routing_disjointness_ok",
    "routing_qubit_ids",
]
