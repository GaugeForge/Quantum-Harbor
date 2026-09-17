"""Heralded-leakage dwell overlay for the ``surface_code_memory`` qtype (pure numpy).

Extends the DEM-level sampler with a classical per-data-qubit leakage dwell process
(arXiv:2411.10343 lineage): leakage is injected by the LRU reset itself — at each LRU
event of a data qubit, with probability ``p_leak_per_lru`` the reset leaves the qubit
leaked, and the dwell covers exactly the TWO following rounds until the next LRU clears
it. LRU schedule (public): each data qubit ``q`` at grid position ``(row, col)`` is
measured + reset by the leakage reduction unit at the END of every round ``t`` with
``t % 2 == (row + col) % 2``. While a qubit is leaked, extra graphlike error mechanisms
fire each round:

- **self**: the qubit's own spatial matching-graph edge (an X error on the leaked qubit);
- **meas**: each adjacent check's temporal edge (the check measurement corrupted through
  the interaction with the leaked qubit);
- **partner**: the spatial edge of every OTHER data qubit sharing an adjacent check (the
  correlated-damage image of two-qubit gates with a leaked partner), once per shared check.

At the dwell-ending LRU event a herald bit is emitted with probability
``1 - herald_false_negative``; every other LRU event emits a false herald with probability
``herald_false_positive``. A herald at LRU round ``r`` therefore implies damage over
rounds ``{r-1, r}`` — the retro window the notebook misdescribes. Herald stream indexing
(public, authoritative):
``herald_id = data_qubit * (rounds // 2) + (t // 2)`` where ``t`` is the LRU round —
uniform because consecutive LRU rounds of one qubit differ by 2 and ``t // 2`` enumerates
its slots for either phase. Leakage experiments therefore require an EVEN round count.

Everything here is pure (params + rng in, arrays out): the engine and the scorer draw the
same realization from the same seed list through :func:`sample_leakage_run` — one full
deterministic pass, never chunk-lazy — so a served-challenge digest can be cross-checked
by regeneration.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from qiqcbench.qsim.qtypes.surface_code_memory import surface_code as SC

LRU_PERIOD_ROUNDS = 2


@dataclass(frozen=True)
class HeraldedLeakageParams:
    p_leak_per_lru: float
    b_self: float
    b_meas: float
    b_partner: float
    # Damage is strongest right after the leaking reset and partially relaxes: in the
    # second dwell round every b_* is scaled by this factor.
    b_decay_second_round: float
    herald_false_negative: float
    # Scalar, or a per-data-qubit array (structured unreliability: chronic false
    # flaggers).
    herald_false_positive: float | np.ndarray


@dataclass(frozen=True)
class LeakageAdjacency:
    """Precomputed vectorized adjacency for the dwell-damage overlay."""

    d: int
    n_data: int
    phase: np.ndarray  # [n_data] LRU phase (row+col) % 2
    # own spatial edge per data qubit (BOUNDARY/-1 sentinel; -2 = no edge, corner qubit)
    self_s1: np.ndarray
    self_s2: np.ndarray
    self_flips: np.ndarray
    # (data qubit, adjacent check) pairs
    adj_q: np.ndarray
    adj_s: np.ndarray
    # partner triples: (leaked qubit, partner qubit) once per shared check; damage lands on
    # the PARTNER's own spatial edge
    tri_q: np.ndarray
    tri_s1: np.ndarray
    tri_s2: np.ndarray
    tri_flips: np.ndarray


NO_EDGE = -2


def build_adjacency(geo: SC.Geometry) -> LeakageAdjacency:
    d = geo.d
    n_data = geo.n_data
    phase = np.array(
        [(r + c) % 2 for r, c in (divmod(q, d) for q in range(n_data))], dtype=np.int64
    )

    self_s1 = np.full(n_data, NO_EDGE, dtype=np.int64)
    self_s2 = np.full(n_data, NO_EDGE, dtype=np.int64)
    self_flips = np.zeros(n_data, dtype=np.uint8)
    for s1, s2, q, flips in geo.spatial:
        self_s1[q] = s1
        self_s2[q] = s2  # may be BOUNDARY (-1)
        self_flips[q] = flips

    adj_pairs: list[tuple[int, int]] = []
    tri: list[tuple[int, int]] = []
    for si, sup in enumerate(geo.checks):
        for q in sorted(sup):
            adj_pairs.append((q, si))
            for q2 in sorted(sup):
                if q2 != q:
                    tri.append((q, q2))
    adj_q = np.array([p[0] for p in adj_pairs], dtype=np.int64)
    adj_s = np.array([p[1] for p in adj_pairs], dtype=np.int64)
    tri_q = np.array([p[0] for p in tri], dtype=np.int64)
    tri_partner = np.array([p[1] for p in tri], dtype=np.int64)
    return LeakageAdjacency(
        d=d,
        n_data=n_data,
        phase=phase,
        self_s1=self_s1,
        self_s2=self_s2,
        self_flips=self_flips,
        adj_q=adj_q,
        adj_s=adj_s,
        tri_q=tri_q,
        tri_s1=self_s1[tri_partner],
        tri_s2=self_s2[tri_partner],
        tri_flips=self_flips[tri_partner],
    )


def n_herald_slots(rounds: int) -> int:
    if rounds % 2 != 0:
        raise ValueError("leakage experiments require an even round count")
    return rounds // 2


def _xor_edge(
    syn: np.ndarray,
    obs: np.ndarray,
    fire: np.ndarray,
    s1: int,
    s2: int,
    flips: int,
    t: int,
    rounds: int,
) -> None:
    if s1 == NO_EDGE or not fire.any():
        return
    syn[fire, SC.det_id(s1, t, rounds)] ^= 1
    if s2 != SC.BOUNDARY and s2 != NO_EDGE:
        syn[fire, SC.det_id(s2, t, rounds)] ^= 1
    if flips:
        obs[fire] ^= 1


def sample_leakage_run(
    geo: SC.Geometry,
    adj: LeakageAdjacency,
    base_dem: list[tuple[int, int, float, int]],
    leak: HeraldedLeakageParams,
    rounds: int,
    shots: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One full deterministic pass. Returns ``(syn[shots, n_det] uint8,
    heralds[shots, n_data, rounds//2] uint8, obs[shots] uint8)``."""
    n_slots = n_herald_slots(rounds)
    n_det = geo.n_checks * rounds
    syn, obs = SC.sample_detectors(base_dem, shots, n_det, rng)
    heralds = np.zeros((shots, adj.n_data, n_slots), dtype=np.uint8)
    fp_by_qubit = np.broadcast_to(
        np.asarray(leak.herald_false_positive, dtype=float), (adj.n_data,)
    )

    # dwell age: -1 = sealed, 0 = first round after the leaking reset, 1 = second round
    age = np.full((shots, adj.n_data), -1, dtype=np.int8)
    for t in range(rounds):
        leaked = age >= 0
        # per-site damage scale: 1.0 at age 0, b_decay_second_round at age 1
        scale = np.where(age == 1, leak.b_decay_second_round, 1.0)
        # self damage
        r_self = rng.random((shots, adj.n_data))
        fire_self = leaked & (r_self < leak.b_self * scale)
        for q in range(adj.n_data):
            _xor_edge(
                syn,
                obs,
                fire_self[:, q],
                int(adj.self_s1[q]),
                int(adj.self_s2[q]),
                int(adj.self_flips[q]),
                t,
                rounds,
            )
        # adjacent-check measurement damage (temporal edge; none at the last round)
        if t < rounds - 1:
            r_meas = rng.random((shots, len(adj.adj_q)))
            fire_meas = leaked[:, adj.adj_q] & (r_meas < leak.b_meas * scale[:, adj.adj_q])
            for j in range(len(adj.adj_q)):
                f = fire_meas[:, j]
                if not f.any():
                    continue
                s = int(adj.adj_s[j])
                syn[f, SC.det_id(s, t, rounds)] ^= 1
                syn[f, SC.det_id(s, t + 1, rounds)] ^= 1
        # partner damage
        r_par = rng.random((shots, len(adj.tri_q)))
        fire_par = leaked[:, adj.tri_q] & (r_par < leak.b_partner * scale[:, adj.tri_q])
        for j in range(len(adj.tri_q)):
            _xor_edge(
                syn,
                obs,
                fire_par[:, j],
                int(adj.tri_s1[j]),
                int(adj.tri_s2[j]),
                int(adj.tri_flips[j]),
                t,
                rounds,
            )
        # LRU events at the end of round t: herald + clear the ending dwell, then the
        # reset itself may inject a fresh dwell that covers the next two rounds.
        lru_qs = np.nonzero(adj.phase == (t % 2))[0]
        r_h = rng.random((shots, len(lru_qs)))
        r_inj = rng.random((shots, len(lru_qs)))
        slot = t // 2
        for k, q in enumerate(lru_qs):
            was_leaked = age[:, q] >= 0
            herald = np.where(
                was_leaked,
                r_h[:, k] < (1.0 - leak.herald_false_negative),
                r_h[:, k] < fp_by_qubit[q],
            )
            heralds[:, q, slot] = herald.astype(np.uint8)
            age[:, q] = np.where(r_inj[:, k] < leak.p_leak_per_lru, 0, -1)
        # non-LRU qubits advance one dwell round
        non_lru = np.nonzero(adj.phase != (t % 2))[0]
        cols = age[:, non_lru]
        age[:, non_lru] = np.where(cols >= 0, cols + 1, -1)
    return syn, heralds, obs


# --------------------------------------------------------------------------- #
# Canonical digests (serve-time binding + prediction binding).
# --------------------------------------------------------------------------- #


def pack_bits_b64(arr: np.ndarray) -> str:
    import base64

    packed = np.packbits(np.ascontiguousarray(arr.reshape(-1), dtype=np.uint8))
    return base64.b64encode(packed.tobytes()).decode("ascii")


def unpack_bits_b64(b64: str, n_bits: int) -> np.ndarray:
    import base64

    raw = np.frombuffer(base64.b64decode(b64, validate=True), dtype=np.uint8)
    bits = np.unpackbits(raw)
    if len(bits) < n_bits or len(bits) - n_bits >= 8:
        raise ValueError(f"packed payload holds {len(bits)} bits; expected {n_bits}")
    if bits[n_bits:].any():
        raise ValueError("padding bits past the declared length must be zero")
    return bits[:n_bits]


def challenge_digest(syn: np.ndarray, heralds: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.packbits(np.ascontiguousarray(syn.reshape(-1), dtype=np.uint8)).tobytes())
    h.update(np.packbits(np.ascontiguousarray(heralds.reshape(-1), dtype=np.uint8)).tobytes())
    return h.hexdigest()


def predictions_digest(bits: np.ndarray) -> str:
    return hashlib.sha256(
        np.packbits(np.ascontiguousarray(bits.reshape(-1), dtype=np.uint8)).tobytes()
    ).hexdigest()


__all__ = [
    "LRU_PERIOD_ROUNDS",
    "NO_EDGE",
    "HeraldedLeakageParams",
    "LeakageAdjacency",
    "build_adjacency",
    "challenge_digest",
    "n_herald_slots",
    "pack_bits_b64",
    "predictions_digest",
    "sample_leakage_run",
    "unpack_bits_b64",
]
