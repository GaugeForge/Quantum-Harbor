"""Stim circuit builders for ``surface_code_lattice_surgery``.

Builds, for every experiment configuration, a noiseless *skeleton* ``stim.Circuit`` plus
full record/detector/observable bookkeeping:

- a generic **detector rule engine**: a detector is emitted for every measurement-record
  parity that is deterministic under the noiseless circuit — consecutive same-ancilla
  measurements, merge/split transition combinations (support diffs justified by fresh
  resets or measured-out records in the matching basis), first-round measurements over
  freshly-initialized data, and final data-readout closures. Every emitted detector is
  validated by ``stim`` when the detector error model is built (a nondeterministic
  detector raises), so the rules cannot silently drift from the circuit.
- the lattice-surgery **byproduct frame**: the CNOT output observables are folded
  combinations of transversal readout parities and the named parity groups
  (``m_zz``/``m_xx``/``m_z_int``/split records). The fold coefficients are *solved* over
  GF(2) from noiseless samples and validated against the ideal-CNOT truth table, so the
  public frame-convention table is correct by construction.
- per-round **segments** (instruction ranges) so the sampler can step a
  ``stim.FlipSimulator`` round by round and layer the classical leakage machine between
  rounds, plus per-round two-qubit-gate partner maps for leakage spraying.

The skeleton carries no noise. Noise is inserted by :func:`with_noise` from an explicit
:class:`NoiseParams` (nominal → the public matching-graph adjacency; hidden → the exact
circuit-level DEM used by maintainer anchors). Asymmetric readout classification noise and
leakage live *outside* the stim circuit, in the sampler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations

import numpy as np
import stim

from qiqcbench.qsim.qtypes.surface_code_lattice_surgery import layout as L

Coord = L.Coord

CNOT_CONFIGS = ("z", "x", "bell_zz", "bell_xx")
MERGE_ROUNDS = 3  # d rounds per merge window in the fixed CNOT schedule
CNOT_ROUNDS = 9  # 1 prep + 3 merged-ZZ + 1 + 3 merged-XX + 1
TRANSITION_PRE_ROUNDS = 2
TRANSITION_POST_ROUNDS = 2

# Ideal CNOT truth table. Z-basis entry (a, b) -> (c_out, t_out) = (a, a^b).
# X-basis entry (s_c, s_t) in |+/-> labels (0=+) -> (s_c^s_t, s_t) (target acts as control).
IDEAL_Z = {(a, b): (a, a ^ b) for a in (0, 1) for b in (0, 1)}
IDEAL_X = {(sc, st): (sc ^ st, st) for sc in (0, 1) for st in (0, 1)}


@dataclass(frozen=True)
class MeasRecord:
    qubit: Coord
    round: int  # global round marker at measurement time
    kind: str  # "anc" | "split" | "data"
    basis: str  # 'Z' | 'X'


@dataclass(frozen=True)
class DetectorDef:
    recs: tuple[int, ...]  # record indices (ancilla/split first, then data recs)
    key: tuple  # stable content key (for the CNOT global superset)
    coord: tuple  # (x, y, t) metadata


@dataclass(frozen=True)
class ObservableDef:
    name: str
    recs: tuple[int, ...]


@dataclass
class Segment:
    """One steppable chunk of the circuit (instruction index range) = one round or one
    prep/measure boundary. ``cx_pairs`` lists the two-qubit interactions inside it."""

    start: int
    end: int
    round_index: int
    cx_pairs: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class CircuitBundle:
    name: str
    skeleton: stim.Circuit
    records: list[MeasRecord]
    detectors: list[DetectorDef]
    observables: list[ObservableDef]
    segments: list[Segment]
    active_qubits: list[int]  # chip qubit ids used by this configuration
    rounds: int
    named_parities: dict[str, tuple[int, ...]]  # e.g. m_zz -> record indices
    frame_table: dict[str, list[str]]  # output obs name -> named parity groups folded in
    meas_qubit_of_record: list[int]  # chip qubit id per record (for readout noise)

    @property
    def n_detectors(self) -> int:
        return len(self.detectors)

    def detector_matrix(self) -> np.ndarray:
        """[n_detectors, n_records] uint8 membership matrix."""
        m = np.zeros((len(self.detectors), len(self.records)), dtype=np.uint8)
        for i, det in enumerate(self.detectors):
            for r in det.recs:
                m[i, r] ^= 1
        return m

    def observable_matrix(self) -> np.ndarray:
        m = np.zeros((len(self.observables), len(self.records)), dtype=np.uint8)
        for i, obs in enumerate(self.observables):
            for r in obs.recs:
                m[i, r] ^= 1
        return m


# --------------------------------------------------------------------------- #
# Builder.
# --------------------------------------------------------------------------- #


class _Builder:
    def __init__(self, name: str):
        self.name = name
        self.circuit = stim.Circuit()
        self.records: list[MeasRecord] = []
        self.detectors: list[DetectorDef] = []
        self.observables: list[ObservableDef] = []
        self.segments: list[Segment] = []
        self.round_marker = 0
        # plaquette pos -> list of (rec_idx, support frozenset, round)
        self._history: dict[Coord, list[tuple[int, frozenset, int]]] = {}
        # data coord -> (basis, reset_round) of the latest fresh reset (never invalidated
        # explicitly; a reset is "fresh for round t" iff no plaquette round touched the
        # qubit between the reset and t).
        self._reset_log: dict[Coord, tuple[str, int]] = {}
        self._touched_round: dict[Coord, int] = {}  # last plaquette round touching the qubit
        # data coord -> list of (round, basis, rec_idx) measure-out events
        self._measured_out: dict[Coord, list[tuple[int, str, int]]] = {}
        self._final_recs: dict[Coord, int] = {}
        self.active: set[Coord] = set()
        self._seg_start = 0
        self._seg_pairs: list[tuple[int, int]] = []

    # ---- low-level emission ----

    def _ids(self, coords: list[Coord]) -> list[int]:
        return [L.qubit_id(c) for c in coords]

    def _append(self, op: str, coords: list[Coord]) -> None:
        if coords:
            self.circuit.append(op, self._ids(coords))

    def close_segment(self) -> None:
        end = len(self.circuit)
        if end > self._seg_start:
            self.segments.append(
                Segment(
                    start=self._seg_start,
                    end=end,
                    round_index=self.round_marker,
                    cx_pairs=list(self._seg_pairs),
                )
            )
        self._seg_start = end
        self._seg_pairs = []

    # ---- prep / measure-out ----

    def reset(self, coords: list[Coord], basis: str) -> None:
        coords = sorted(coords)
        self._append("R" if basis == "Z" else "RX", coords)
        for c in coords:
            self._reset_log[c] = (basis, self.round_marker)
            self._touched_round.pop(c, None)
        self.active.update(coords)

    def pauli_frame(self, coords: list[Coord], op: str) -> None:
        """Noiseless logical-state prep flips (X̄ / Z̄ representatives)."""
        self._append(op, sorted(coords))

    def measure_out(self, coords: list[Coord], basis: str, kind: str) -> dict[Coord, int]:
        coords = sorted(coords)
        self._append("M" if basis == "Z" else "MX", coords)
        out: dict[Coord, int] = {}
        for c in coords:
            idx = len(self.records)
            self.records.append(MeasRecord(c, self.round_marker, kind, basis))
            self._measured_out.setdefault(c, []).append((self.round_marker, basis, idx))
            out[c] = idx
            if kind == "data":
                self._final_recs[c] = idx
        self.active.difference_update(coords)
        return out

    # ---- one syndrome-extraction round ----

    def round(self, plaquettes: list[L.Plaquette]) -> None:
        plaquettes = sorted(plaquettes, key=lambda p: (p.pos[1], p.pos[0]))
        ancs = [p.pos for p in plaquettes]
        x_ancs = [p.pos for p in plaquettes if p.basis == "X"]
        self._append("R", ancs)
        self._append("H", x_ancs)
        for step in L.interaction_substeps(plaquettes):
            targets: list[int] = []
            for anc, data, basis in sorted(step, key=lambda s: (s[0][1], s[0][0])):
                a, d = L.qubit_id(anc), L.qubit_id(data)
                # X-plaquette: CX ancilla->data; Z-plaquette: CX data->ancilla.
                targets.extend([a, d] if basis == "X" else [d, a])
                self._seg_pairs.append((a, d))
            if targets:
                self.circuit.append("CX", targets)
            self.circuit.append("TICK", [])
        self._append("H", x_ancs)
        self._append("M", ancs)

        base = len(self.records)
        for p in plaquettes:
            self.records.append(MeasRecord(p.pos, self.round_marker, "anc", "Z"))
        # detector rules
        for j, p in enumerate(plaquettes):
            rec_idx = base + j
            self._emit_round_detectors(p, rec_idx)
            self._history.setdefault(p.pos, []).append(
                (rec_idx, frozenset(p.data), self.round_marker)
            )
        for p in plaquettes:
            for c in p.data:
                self._touched_round[c] = self.round_marker
        self.active.update(ancs)
        self.round_marker += 1
        self.close_segment()

    def _fresh_in(self, c: Coord, basis: str, since_round: int | None) -> bool:
        """Was ``c`` freshly reset in ``basis`` and untouched by any plaquette round since
        (strictly after ``since_round`` when given)?"""
        info = self._reset_log.get(c)
        if info is None or info[0] != basis:
            return False
        reset_round = info[1]
        if since_round is not None and reset_round < since_round:
            return False
        touched = self._touched_round.get(c)
        return touched is None or touched < reset_round

    def _measured_out_recs(self, c: Coord, basis: str, since_round: int) -> list[int] | None:
        for rnd, b, idx in self._measured_out.get(c, []):
            if rnd >= since_round and b == basis:
                return [idx]
        return None

    def _emit_round_detectors(self, p: L.Plaquette, rec_idx: int) -> None:
        hist = self._history.get(p.pos, [])
        support = frozenset(p.data)
        t = self.round_marker
        if not hist:
            # first measurement: deterministic iff every supported data qubit is fresh in
            # the plaquette basis.
            if all(self._fresh_in(c, p.basis, None) for c in p.data):
                self._add_detector((rec_idx,), ("first", p.pos), p.pos, t)
            return
        prev_rec, prev_support, prev_round = hist[-1]
        recs: list[int] = [prev_rec, rec_idx]
        for c in support ^ prev_support:
            if c in support:
                # qubit added: needs a fresh matching-basis reset after the previous round
                if self._fresh_in(c, p.basis, prev_round + 1):
                    continue
                return
            # qubit removed: needs a matching-basis measure-out after the previous round
            out = self._measured_out_recs(c, p.basis, prev_round + 1)
            if out is None:
                return
            recs.extend(out)
        self._add_detector(tuple(recs), ("pair", p.pos, prev_round, t), p.pos, t)

    def _add_detector(self, recs: tuple[int, ...], key: tuple, pos: Coord, t: int) -> None:
        self.detectors.append(DetectorDef(recs=recs, key=key, coord=(pos[0], pos[1], t)))
        targets = [stim.target_rec(r - len(self.records)) for r in recs]
        self.circuit.append("DETECTOR", targets, [pos[0], pos[1], t])

    # ---- closures ----

    def final_readout_detectors(self, plaquettes: list[L.Plaquette], basis: str) -> None:
        """After a final transversal data readout in ``basis``: close every matching-type
        plaquette chain against the data readout records."""
        for p in sorted(plaquettes, key=lambda q: (q.pos[1], q.pos[0])):
            if p.basis != basis:
                continue
            hist = self._history.get(p.pos)
            if not hist:
                continue
            prev_rec, prev_support, _ = hist[-1]
            if not all(c in self._final_recs for c in prev_support):
                continue
            recs = (prev_rec, *(self._final_recs[c] for c in sorted(prev_support)))
            self._add_detector(recs, ("final", p.pos, basis), p.pos, self.round_marker)

    def add_observable(self, name: str, recs: tuple[int, ...]) -> None:
        idx = len(self.observables)
        self.observables.append(ObservableDef(name=name, recs=recs))
        targets = [stim.target_rec(r - len(self.records)) for r in recs]
        self.circuit.append("OBSERVABLE_INCLUDE", targets, idx)

    def finish(
        self,
        rounds: int,
        named_parities: dict[str, tuple[int, ...]] | None = None,
        frame_table: dict[str, list[str]] | None = None,
    ) -> CircuitBundle:
        self.close_segment()
        used = sorted(
            {L.qubit_id(c) for c in self._reset_log} | {L.qubit_id(r.qubit) for r in self.records}
        )
        return CircuitBundle(
            name=self.name,
            skeleton=self.circuit,
            records=self.records,
            detectors=self.detectors,
            observables=self.observables,
            segments=self.segments,
            active_qubits=used,
            rounds=rounds,
            named_parities=named_parities or {},
            frame_table=frame_table or {},
            meas_qubit_of_record=[L.qubit_id(r.qubit) for r in self.records],
        )


# --------------------------------------------------------------------------- #
# Experiment builders.
# --------------------------------------------------------------------------- #


def build_memory(layout_name: str, rounds: int) -> CircuitBundle:
    """Single-patch Z-basis memory: prep |0..0>, ``rounds`` extraction rounds, transversal
    M_Z, observable = Z̄ (data row 0)."""
    patch = L.MEMORY_PATCHES[layout_name]
    b = _Builder(f"memory_{layout_name}_r{rounds}")
    b.reset(patch.data, "Z")
    b.close_segment()
    plaqs = list(patch.plaquettes)
    for _ in range(rounds):
        b.round(plaqs)
    b.measure_out(patch.data, "Z", "data")
    b.final_readout_detectors(plaqs, "Z")
    b.add_observable("obs_logical", tuple(b._final_recs[c] for c in patch.z_logical_row(0)))
    b.close_segment()
    return b.finish(rounds)


def _merged_new_plaquettes(window: str) -> tuple[list[L.Plaquette], list[L.Plaquette]]:
    """(new same-type seam plaquettes carrying the joint parity, all merged plaquettes)."""
    if window == "zz":
        merged, parts, basis = L.MERGED_ZZ, (L.PATCH_C, L.PATCH_INT), "Z"
    else:
        merged, parts, basis = L.MERGED_XX, (L.PATCH_INT, L.PATCH_T), "X"
    old = {(p.pos, p.basis) for part in parts for p in part.plaquettes}
    new = [p for p in merged.plaquettes if p.basis == basis and (p.pos, p.basis) not in old]
    return new, list(merged.plaquettes)


def build_merged_memory(window: str, rounds: int, include_transitions: bool) -> CircuitBundle:
    if window not in ("zz", "xx"):
        raise ValueError(f"window must be 'zz' or 'xx', got {window!r}")
    basis = "Z" if window == "zz" else "X"
    merged = L.MERGED_ZZ if window == "zz" else L.MERGED_XX
    routing = list(L.ROUTING_ZZ if window == "zz" else L.ROUTING_XX)
    routing_basis = "X" if window == "zz" else "Z"  # rough merge |+>, smooth merge |0>
    parts = (L.PATCH_C, L.PATCH_INT) if window == "zz" else (L.PATCH_INT, L.PATCH_T)
    name = f"merged_{window}_r{rounds}" + ("_trans" if include_transitions else "")
    b = _Builder(name)

    if not include_transitions:
        b.reset(merged.data, basis)
        b.close_segment()
        for _ in range(rounds):
            b.round(list(merged.plaquettes))
        b.measure_out(merged.data, basis, "data")
        b.final_readout_detectors(list(merged.plaquettes), basis)
        rep = merged.z_logical_row(0) if basis == "Z" else merged.x_logical_col(0)
        b.add_observable("obs_logical", tuple(b._final_recs[c] for c in rep))
        b.close_segment()
        return b.finish(rounds)

    part_data = [c for part in parts for c in part.data]
    part_plaqs = [p for part in parts for p in part.plaquettes]
    b.reset(part_data, basis)
    b.close_segment()
    for _ in range(TRANSITION_PRE_ROUNDS):
        b.round(part_plaqs)
    b.reset(routing, routing_basis)
    b.close_segment()
    for _ in range(rounds):
        b.round(list(merged.plaquettes))
    b.measure_out(routing, routing_basis, "split")
    b.close_segment()
    for _ in range(TRANSITION_POST_ROUNDS):
        b.round(part_plaqs)
    b.measure_out(part_data, basis, "data")
    b.final_readout_detectors(part_plaqs, basis)
    rep_patch = parts[0]
    rep = rep_patch.z_logical_row(0) if basis == "Z" else rep_patch.x_logical_col(0)
    b.add_observable("obs_logical", tuple(b._final_recs[c] for c in rep))
    b.close_segment()
    return b.finish(TRANSITION_PRE_ROUNDS + rounds + TRANSITION_POST_ROUNDS)


def _cnot_skeleton(config: str, entry: tuple[int, int]) -> tuple[_Builder, dict]:
    """Build the fixed CNOT schedule for one reporting configuration + logical input.

    ``entry``: Z-basis (a, b) prep of |a>|b>; X-basis (s_c, s_t) prep of |±>|±>;
    bell configs ignore ``entry`` (prep C |+>, T |0>).
    """
    if config not in CNOT_CONFIGS:
        raise ValueError(f"unknown cnot config {config!r}")
    b = _Builder(f"cnot_{config}_{entry[0]}{entry[1]}")
    c_patch, i_patch, t_patch = L.PATCH_C, L.PATCH_INT, L.PATCH_T
    sep_plaqs = list(c_patch.plaquettes) + list(i_patch.plaquettes) + list(t_patch.plaquettes)
    zz_plaqs = list(L.MERGED_ZZ.plaquettes) + list(t_patch.plaquettes)
    xx_plaqs = list(c_patch.plaquettes) + list(L.MERGED_XX.plaquettes)

    if config == "z":
        c_basis = t_basis = "Z"
    elif config == "x":
        c_basis = t_basis = "X"
    else:  # bell: C |+>, T |0>; readout basis differs per config
        c_basis, t_basis = "X", "Z"
    b.reset(c_patch.data, c_basis)
    b.reset(i_patch.data, "X")  # INT always |+>
    b.reset(t_patch.data, t_basis)
    # logical input frame flips
    if config == "z":
        if entry[0]:
            b.pauli_frame(c_patch.x_logical_col(0), "X")
        if entry[1]:
            b.pauli_frame(t_patch.x_logical_col(0), "X")
    elif config == "x":
        if entry[0]:
            b.pauli_frame(c_patch.z_logical_row(0), "Z")
        if entry[1]:
            b.pauli_frame(t_patch.z_logical_row(0), "Z")
    b.close_segment()

    b.round(sep_plaqs)  # r0: establish all three patches

    # --- ZZ window: rough merge C-INT (routing row |+>) ---
    new_zz, _ = _merged_new_plaquettes("zz")
    b.reset(list(L.ROUTING_ZZ), "X")
    b.close_segment()
    first_zz_round = b.round_marker
    for _ in range(MERGE_ROUNDS):
        b.round(zz_plaqs)
    split_zz = b.measure_out(list(L.ROUTING_ZZ), "X", "split")
    b.close_segment()
    b.round(sep_plaqs)  # r4

    # --- XX window: smooth merge INT-T (routing col |0>) ---
    new_xx, _ = _merged_new_plaquettes("xx")
    b.reset(list(L.ROUTING_XX), "Z")
    b.close_segment()
    first_xx_round = b.round_marker
    for _ in range(MERGE_ROUNDS):
        b.round(xx_plaqs)
    split_xx = b.measure_out(list(L.ROUTING_XX), "Z", "split")
    b.close_segment()
    b.round(sep_plaqs)  # r8

    # --- final readout ---
    int_recs = b.measure_out(i_patch.data, "Z", "data")
    if config in ("z", "bell_zz"):
        cr_basis, tr_basis = "Z", "Z"
    elif config == "x":
        cr_basis, tr_basis = "X", "X"
    else:  # bell_xx
        cr_basis, tr_basis = "X", "X"
    c_recs = b.measure_out(c_patch.data, cr_basis, "data")
    t_recs = b.measure_out(t_patch.data, tr_basis, "data")
    b.final_readout_detectors(list(i_patch.plaquettes), "Z")
    b.final_readout_detectors(list(c_patch.plaquettes), cr_basis)
    b.final_readout_detectors(list(t_patch.plaquettes), tr_basis)
    b.close_segment()

    # named parity groups (record sets; PUBLIC definitions)
    def first_round_recs(plaqs: list[L.Plaquette], rnd: int) -> tuple[int, ...]:
        out = []
        for p in plaqs:
            for rec, _sup, r in b._history[p.pos]:
                if r == rnd:
                    out.append(rec)
                    break
        return tuple(sorted(out))

    named = {
        "m_zz": first_round_recs(new_zz, first_zz_round),
        "m_xx": first_round_recs(new_xx, first_xx_round),
        "m_z_int": tuple(int_recs[c] for c in i_patch.z_logical_row(0)),
    }
    # Split byproducts enter through INDIVIDUAL routing-qubit records (the one crossing
    # the relevant logical representative), so each split record is its own named group.
    for k, c in enumerate(sorted(split_zz)):
        named[f"split_zz_{k}"] = (split_zz[c],)
    for k, c in enumerate(sorted(split_xx)):
        named[f"split_xx_{k}"] = (split_xx[c],)
    ctx = {
        "c_recs": c_recs,
        "t_recs": t_recs,
        "cr_basis": cr_basis,
        "tr_basis": tr_basis,
        "named": named,
    }
    return b, ctx


def _readout_parity_recs(
    patch: L.PatchGeometry, recs: dict[Coord, int], basis: str, index: int = 0
) -> tuple[int, ...]:
    rep = patch.z_logical_row(index) if basis == "Z" else patch.x_logical_col(index)
    return tuple(recs[c] for c in rep)


def _solve_frame(
    b: _Builder, raw_targets: dict[str, tuple[int, ...]], named: dict[str, tuple[int, ...]]
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Solve, per output observable, which named parity groups fold into the raw readout
    parity to make it noiseless-deterministic. Returns (frame_table, constants)."""
    sampler = b.circuit.compile_sampler(seed=12345)
    shots = sampler.sample(shots=256).astype(np.uint8)  # [shots, n_records]
    group_names = list(named)
    group_cols = {g: np.bitwise_xor.reduce(shots[:, list(named[g])], axis=1) for g in group_names}
    table: dict[str, list[str]] = {}
    consts: dict[str, int] = {}
    for name, recs in raw_targets.items():
        target = np.bitwise_xor.reduce(shots[:, list(recs)], axis=1)
        solutions: list[tuple[int, tuple[str, ...]]] = []
        for k in range(len(group_names) + 1):
            for combo in combinations(group_names, k):
                v = target.copy()
                for g in combo:
                    v ^= group_cols[g]
                if np.all(v == v[0]):
                    solutions.append((k, combo))
            if solutions:
                break  # minimal-cardinality solution(s)
        if not solutions:
            raise AssertionError(
                f"{b.name}: no frame fold makes output observable {name} deterministic"
            )
        _, combo = sorted(solutions)[0]
        table[name] = list(combo)
        v = target.copy()
        for g in combo:
            v ^= group_cols[g]
        consts[name] = int(v[0])
    return table, consts


def _raw_targets(config: str, ctx: dict) -> dict[str, tuple[int, ...]]:
    # Output observables are BASIS-TAGGED public names (obs_z_*/obs_x_*): a physical
    # error's logical action differs per readout basis, so the superset DEM's
    # flips_observables must distinguish them.
    if config in ("z", "x"):
        return {
            f"obs_{config}_c_out": _readout_parity_recs(L.PATCH_C, ctx["c_recs"], ctx["cr_basis"]),
            f"obs_{config}_t_out": _readout_parity_recs(L.PATCH_T, ctx["t_recs"], ctx["tr_basis"]),
        }
    # Bell configs: C is prepped |+> (its Z-plaquette values are random), so the Z̄_C
    # readout representative must be the row m_zz uses (C's bottom row) for the folded
    # observable to be deterministic. Symmetrically T (prepped |0>) keeps its m_xx-aligned
    # X̄ column (col 0, the T column m_xx's X̄_T factor lives on).
    c_index = 2 if config == "bell_zz" else 0
    joint = _readout_parity_recs(
        L.PATCH_C, ctx["c_recs"], ctx["cr_basis"], index=c_index
    ) + _readout_parity_recs(L.PATCH_T, ctx["t_recs"], ctx["tr_basis"])
    return {f"obs_{config}": joint}


@lru_cache(maxsize=8)
def _solved_fold(config: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Solve the frame fold once per configuration (on the canonical (0,0) entry). The
    fold — hence the observable definitions and the DEM observable columns — is shared by
    every battery entry of the configuration (entries differ only by noiseless Pauli
    frame flips, which cannot change the fold)."""
    b, ctx = _cnot_skeleton(config, (0, 0))
    table, _ = _solve_frame(b, _raw_targets(config, ctx), ctx["named"])
    return tuple((name, tuple(groups)) for name, groups in table.items())


def build_cnot(config: str, entry: tuple[int, int] = (0, 0)) -> CircuitBundle:
    """The fixed lattice-surgery CNOT schedule for one reporting configuration.

    Output observables (deterministic; frame convention folded in, solved + validated):
    ``z``/``x``: ``obs_out_c``, ``obs_out_t``. ``bell_zz``/``bell_xx``: ``obs_bell``.
    """
    b, ctx = _cnot_skeleton(config, entry)
    named = ctx["named"]
    raw = _raw_targets(config, ctx)
    frame_table = {name: list(groups) for name, groups in _solved_fold(config)}
    sampler = b.circuit.compile_sampler(seed=12345)
    shots = sampler.sample(shots=64).astype(np.uint8)
    consts: dict[str, int] = {}
    for name, recs in raw.items():
        folded = list(recs)
        for g in frame_table[name]:
            folded.extend(named[g])
        v = np.bitwise_xor.reduce(shots[:, folded], axis=1)
        if not np.all(v == v[0]):
            raise AssertionError(f"{b.name}: folded observable {name} is not deterministic")
        consts[name] = int(v[0])
        b.add_observable(name, tuple(folded))
    b.close_segment()
    bundle = b.finish(CNOT_ROUNDS, named_parities=named, frame_table=frame_table)
    bundle.frame_table["_constants"] = consts  # noiseless observable values
    return bundle


@lru_cache(maxsize=64)
def cnot_bundle(config: str, a: int = 0, bb: int = 0) -> CircuitBundle:
    return build_cnot(config, (a, bb))


@lru_cache(maxsize=64)
def memory_bundle(layout_name: str, rounds: int) -> CircuitBundle:
    return build_memory(layout_name, rounds)


@lru_cache(maxsize=64)
def merged_bundle(window: str, rounds: int, include_transitions: bool) -> CircuitBundle:
    return build_merged_memory(window, rounds, include_transitions)


# --------------------------------------------------------------------------- #
# Noise insertion.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NoiseParams:
    """Explicit Pauli noise inserted into a skeleton. Keys are chip qubit ids; two-qubit
    rates are keyed by the unordered pair. Asymmetric readout + leakage are NOT here —
    they live in the sampler (outside stim)."""

    idle_px: dict[int, float]
    idle_py: dict[int, float]
    idle_pz: dict[int, float]
    p_cz: dict[tuple[int, int], float]
    p_reset: dict[int, float]
    p_meas: dict[int, float]  # symmetric measurement flip (nominal / DEM paths)


def noisy_segment_circuits(bundle: CircuitBundle, noise: NoiseParams) -> list[stim.Circuit]:
    """Per-segment noisy circuits: X/Z_ERROR after resets, DEPOLARIZE2 after each
    two-qubit gate, X/Z_ERROR before measurements, and one PAULI_CHANNEL_1 idle step per
    active data qubit at the end of every syndrome round. Segment boundaries match
    ``bundle.segments`` so the sampler can step a FlipSimulator round by round."""
    out: list[stim.Circuit] = []
    for seg in bundle.segments:
        c = stim.Circuit()
        for inst in bundle.skeleton[seg.start : seg.end].flattened():
            name = inst.name
            if name in ("R", "RX"):
                qs = [t.value for t in inst.targets_copy()]
                c.append(name, qs)
                err = "X_ERROR" if name == "R" else "Z_ERROR"
                for q in qs:
                    p = noise.p_reset.get(q, 0.0)
                    if p > 0:
                        c.append(err, [q], p)
            elif name == "CX":
                ts = [t.value for t in inst.targets_copy()]
                c.append("CX", ts)
                for i in range(0, len(ts), 2):
                    pair = (min(ts[i], ts[i + 1]), max(ts[i], ts[i + 1]))
                    p = noise.p_cz.get(pair, 0.0)
                    if p > 0:
                        c.append("DEPOLARIZE2", [ts[i], ts[i + 1]], p)
            elif name in ("M", "MX"):
                qs = [t.value for t in inst.targets_copy()]
                err = "X_ERROR" if name == "M" else "Z_ERROR"
                for q in qs:
                    p = noise.p_meas.get(q, 0.0)
                    if p > 0:
                        c.append(err, [q], p)
                c.append(name, qs)
            else:
                c.append(inst)
        if seg.cx_pairs:
            # one idle step per syndrome round, on the data qubits in play this round
            data_ids = sorted({d for (_a, d) in seg.cx_pairs})
            for q in data_ids:
                px = noise.idle_px.get(q, 0.0)
                py = noise.idle_py.get(q, 0.0)
                pz = noise.idle_pz.get(q, 0.0)
                if px + py + pz > 0:
                    c.append("PAULI_CHANNEL_1", [q], [px, py, pz])
        out.append(c)
    return out


def with_noise(bundle: CircuitBundle, noise: NoiseParams) -> stim.Circuit:
    """The full noisy circuit (concatenated noisy segments) — for DEM derivation."""
    total = stim.Circuit()
    for c in noisy_segment_circuits(bundle, noise):
        total += c
    return total


def uniform_noise(
    bundle: CircuitBundle,
    *,
    p_idle: float,
    p_cz: float,
    p_reset: float,
    p_meas: float,
) -> NoiseParams:
    """Uniform (nominal) noise over the bundle's active qubits — the public/nominal model."""
    qs = bundle.active_qubits
    pairs: set[tuple[int, int]] = set()
    for seg in bundle.segments:
        for a, d in seg.cx_pairs:
            pairs.add((min(a, d), max(a, d)))
    third = p_idle / 3.0
    return NoiseParams(
        idle_px={q: third for q in qs},
        idle_py={q: third for q in qs},
        idle_pz={q: third for q in qs},
        p_cz={p: p_cz for p in pairs},
        p_reset={q: p_reset for q in qs},
        p_meas={q: p_meas for q in qs},
    )


__all__ = [
    "CNOT_CONFIGS",
    "CNOT_ROUNDS",
    "CircuitBundle",
    "DetectorDef",
    "IDEAL_X",
    "IDEAL_Z",
    "MERGE_ROUNDS",
    "MeasRecord",
    "NoiseParams",
    "ObservableDef",
    "Segment",
    "TRANSITION_POST_ROUNDS",
    "TRANSITION_PRE_ROUNDS",
    "build_cnot",
    "build_memory",
    "build_merged_memory",
    "cnot_bundle",
    "memory_bundle",
    "merged_bundle",
    "uniform_noise",
    "with_noise",
]
