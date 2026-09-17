"""Rotated surface-code geometry + detector-error-model sampler (pure numpy).

The whole qtype operates at the **detector / DEM level** — there is no pulse or
stabilizer-circuit simulation. We model one decoded basis (the Z-type checks, which detect
X errors) of a rotated distance-``d`` surface code, run ``T`` rounds of syndrome
extraction, and return raw per-shot **detection events** (the parity of consecutive
same-check measurements) plus the final logical outcome.

Geometry (``d`` odd):

- Data qubits on a ``d × d`` grid; ``data_id(r, c) = r*d + c`` (``d²`` of them).
- Z-checks: interior weight-4 faces ``(r, c)`` with ``(r+c)`` even, plus weight-2 boundary
  faces on the top/bottom edges. For ``d=5`` this gives the ``(d²−1)/2 = 12`` checks of the
  decoded basis (``n = 2d²−1 = 49`` physical qubits in total).
- An X error on a data qubit flips the Z-checks adjacent to it: bulk qubits flip 2 checks
  (a "spatial edge" of the matching graph), boundary qubits flip 1 (a "boundary edge").
- Logical observable: the logical Z̄ representative is data **column 0**; an error mechanism
  flips the observable iff its data qubit lies in column 0. The minimum-weight undetectable
  logical X chain is a horizontal row (weight ``d``) → code distance ``d``.

Detector indexing (public, authoritative): ``det_id(check, t) = check * T + t`` with
``t ∈ [0, T)``, so ``n_detectors = n_checks * T``.

Detector-error-model (DEM): a list of independent mechanisms, each ``{detectors, p,
flips_observable}`` flipping 1 or 2 detectors. Every mechanism here is **graphlike**
(≤2 detectors), so a weighted matching decoder consumes the DEM directly. The hidden noise
adds spacetime-correlated mechanisms an i.i.d. (single-detector + spatial + temporal) DEM
cannot represent:

- **hooks** — a single mid-extraction fault flips two data qubits → a diagonal
  ``(s1, t)``–``(s2, t+1)`` edge whose orientation follows the public schedule but whose
  rate is spatially inhomogeneous (hidden);
- **leakage** — long-range correlations on a hidden subset of check pairs across 2–3 rounds;
- **correlated measurement** — handled by the ordinary temporal edges.

This module is pure numpy (no decoder dependency): the base memory-calibration
engine only samples. Decoder-calibration scoring lives in the verifier; a
feedback capability may pass these sampled records to its private fixed
firmware decoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

BOUNDARY = -1  # sentinel: a mechanism touching one detector + the boundary


@dataclass(frozen=True)
class Geometry:
    d: int
    checks: tuple[frozenset[int], ...]  # data-qubit support per check id
    # spatial edges: (s1, s2_or_BOUNDARY, data_id, flips_observable)
    spatial: tuple[tuple[int, int, int, int], ...]
    logical_column: int = 0

    @property
    def n_checks(self) -> int:
        return len(self.checks)

    @property
    def n_data(self) -> int:
        return self.d * self.d

    @property
    def bulk_pairs(self) -> list[tuple[int, int, int, int]]:
        """Spatial edges between two real checks (hook-candidate orientations)."""
        return [e for e in self.spatial if e[1] != BOUNDARY]

    @property
    def boundary_edges(self) -> list[tuple[int, int, int, int]]:
        return [e for e in self.spatial if e[1] == BOUNDARY]


def build_geometry(d: int = 5) -> Geometry:
    if d % 2 == 0:
        raise ValueError("distance d must be odd")

    def data_id(r: int, c: int) -> int:
        return r * d + c

    checks: list[frozenset[int]] = []
    # interior weight-4 Z faces: (r+c) even
    for r in range(d - 1):
        for c in range(d - 1):
            if (r + c) % 2 == 0:
                checks.append(
                    frozenset(
                        {data_id(r, c), data_id(r, c + 1), data_id(r + 1, c), data_id(r + 1, c + 1)}
                    )
                )
    # top boundary weight-2 (virtual row -1): c odd
    for c in range(d - 1):
        if c % 2 == 1:
            checks.append(frozenset({data_id(0, c), data_id(0, c + 1)}))
    # bottom boundary weight-2 (virtual row d-1): ((d-1)+c) even -> c even for odd d
    for c in range(d - 1):
        if ((d - 1) + c) % 2 == 0:
            checks.append(frozenset({data_id(d - 1, c), data_id(d - 1, c + 1)}))

    data_to_checks: dict[int, list[int]] = {q: [] for q in range(d * d)}
    for si, sup in enumerate(checks):
        for q in sup:
            data_to_checks[q].append(si)

    spatial: list[tuple[int, int, int, int]] = []
    for q in range(d * d):
        cs = data_to_checks[q]
        _, c = divmod(q, d)
        flips = 1 if c == 0 else 0
        if len(cs) == 2:
            spatial.append((cs[0], cs[1], q, flips))
        elif len(cs) == 1:
            spatial.append((cs[0], BOUNDARY, q, flips))
        # len 0: corner qubit touching no Z-check — undetectable, contributes no edge
    return Geometry(d=d, checks=tuple(checks), spatial=tuple(spatial))


def geometry_from_public_spec(public: Any) -> Geometry:
    """Build and validate the executable geometry from the public contract.

    The checked-in geometry is agent-visible and authoritative.  Consuming it
    here prevents the distance/check/topology fields from becoming decorative
    configuration while still failing closed if they no longer describe the
    supported rotated-code sector.
    """
    d = int(public.distance)
    expected = build_geometry(d)
    if int(public.n_data_qubits) != d * d:
        raise ValueError("public n_data_qubits is inconsistent with distance")
    if int(public.n_physical_qubits) != 2 * d * d - 1:
        raise ValueError("public n_physical_qubits is inconsistent with distance")

    checks = tuple(frozenset(int(q) for q in support) for support in public.check_supports)
    if len(checks) != int(public.n_checks):
        raise ValueError("public check_supports length does not match n_checks")
    if checks != expected.checks:
        raise ValueError("public check_supports do not match the rotated-code geometry")

    spatial: list[tuple[int, int, int, int]] = []
    for edge in public.spatial_edges:
        edge_checks = [int(s) for s in edge.checks]
        if any(s < 0 or s >= len(checks) for s in edge_checks):
            raise ValueError("public spatial edge references an invalid check")
        if len(edge_checks) == 1:
            s1, s2 = edge_checks[0], BOUNDARY
        else:
            s1, s2 = edge_checks
        spatial.append((s1, s2, int(edge.data_qubit), int(bool(edge.flips_observable))))
    if tuple(spatial) != expected.spatial:
        raise ValueError("public spatial_edges do not match check supports")

    hooks = tuple(
        (
            int(candidate.checks[0]),
            int(candidate.checks[1]),
            int(bool(candidate.flips_observable)),
        )
        for candidate in public.hook_candidates
    )
    expected_hooks = tuple((s1, s2, flips) for s1, s2, _q, flips in expected.bulk_pairs)
    if hooks != expected_hooks:
        raise ValueError("public hook_candidates do not match the spatial geometry")

    logical = sorted(int(q) for q in public.logical_representative_data_qubits)
    expected_logical = sorted(q for _s1, _s2, q, flips in expected.spatial if flips)
    if logical != expected_logical:
        raise ValueError("public logical representative is inconsistent with spatial edges")
    if str(public.detector_index_formula) != "check * rounds + t":
        raise ValueError("unsupported detector indexing formula")
    return Geometry(d=d, checks=checks, spatial=tuple(spatial), logical_column=0)


def det_id(check: int, t: int, T: int) -> int:
    return check * T + t


# --------------------------------------------------------------------------- #
# Hidden DEM construction (the noise model the sampler draws from).
# --------------------------------------------------------------------------- #


@dataclass
class HiddenDemParams:
    """Frozen, explicit hidden DEM parameters (seed-derived but materialized as data so the
    engine and the verifier reproduce the identical model without RNG-order fragility)."""

    p_data: float
    p_meas: float
    hook_base: float
    hook_multipliers: list[float]  # one per geometry.bulk_pairs, in order
    p_leak: float
    leak_pairs: list[tuple[int, int, int, int]]  # (s1, s2, dt, flips_observable)
    extra_metadata: dict = field(default_factory=dict)


def build_hidden_dem(
    geo: Geometry, h: HiddenDemParams, T: int
) -> list[tuple[int, int, float, int]]:
    """Full hidden DEM as a list of ``(det_a, det_b_or_BOUNDARY, p, flips_observable)``."""
    dem: list[tuple[int, int, float, int]] = []
    # single-detector + spatial (data-qubit X errors), every round
    for t in range(T):
        for s1, s2, _q, flips in geo.spatial:
            a = det_id(s1, t, T)
            b = BOUNDARY if s2 == BOUNDARY else det_id(s2, t, T)
            dem.append((a, b, h.p_data, flips))
    # temporal (measurement) errors
    for t in range(T - 1):
        for s in range(geo.n_checks):
            dem.append((det_id(s, t, T), det_id(s, t + 1, T), h.p_meas, 0))
    # hook (spacetime-correlated, schedule-oriented, inhomogeneous rate)
    pairs = geo.bulk_pairs
    if len(h.hook_multipliers) != len(pairs):
        raise ValueError(
            f"hook_multipliers has {len(h.hook_multipliers)} entries; expected {len(pairs)}"
        )
    for (s1, s2, _q, flips), mult in zip(pairs, h.hook_multipliers, strict=True):
        rate = h.hook_base * mult
        for t in range(T - 1):
            dem.append((det_id(s1, t, T), det_id(s2, t + 1, T), rate, flips))
    # leakage (hidden long-range correlations)
    for s1, s2, dt, flips in h.leak_pairs:
        for t in range(T - dt):
            dem.append((det_id(s1, t, T), det_id(s2, t + dt, T), h.p_leak, flips))
    return dem


# --------------------------------------------------------------------------- #
# Sampler.
# --------------------------------------------------------------------------- #


def sample_detectors(
    dem: list[tuple[int, int, float, int]],
    n_shots: int,
    n_detectors: int,
    rng: np.random.Generator,
    *,
    chunk: int = 200_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample raw detection events. Returns ``(syndrome[n_shots, n_detectors] uint8,
    logical[n_shots] uint8)``. Chunked over shots to bound memory."""
    ps = np.array([e[2] for e in dem], dtype=float)
    a_idx = np.array([e[0] for e in dem], dtype=np.int64)
    b_idx = np.array([e[1] for e in dem], dtype=np.int64)
    flips = np.array([e[3] for e in dem], dtype=np.uint8)
    has_b = b_idx != BOUNDARY

    syn = np.zeros((n_shots, n_detectors), dtype=np.uint8)
    obs = np.zeros(n_shots, dtype=np.uint8)
    for start in range(0, n_shots, chunk):
        end = min(start + chunk, n_shots)
        m = end - start
        fired = rng.random((m, len(dem))) < ps[None, :]
        sblk = syn[start:end]
        for ei in range(len(dem)):
            f = fired[:, ei]
            if not f.any():
                continue
            sblk[f, a_idx[ei]] ^= 1
            if has_b[ei]:
                sblk[f, b_idx[ei]] ^= 1
            if flips[ei]:
                obs[start:end][f] ^= 1
    return syn, obs


__all__ = [
    "BOUNDARY",
    "Geometry",
    "HiddenDemParams",
    "build_geometry",
    "geometry_from_public_spec",
    "build_hidden_dem",
    "det_id",
    "sample_detectors",
]
