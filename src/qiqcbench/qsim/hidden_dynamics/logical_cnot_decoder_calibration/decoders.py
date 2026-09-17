"""Replay decoders for ``logical_cnot_decoder_calibration``.

A submitted DEM is a list of mechanisms ``{detectors: [ids], p, flips_observables:
[names]}`` where every mechanism flips 1 or 2 detectors (graphlike). Memory DEMs use the
d=5 scoring-layout local detector ids; the CNOT DEM uses the public superset ids and is
projected onto each reporting configuration (mechanisms touching detectors inactive in a
configuration are dropped for that configuration — the public rule).

``flips_observables`` may name output observables directly (``obs_out_c``/``obs_out_t``/
``obs_bell``) and/or the named parity groups (``m_zz``/``m_xx``/``m_z_int``/
``split_*``); the public byproduct-frame convention folds parity-group flips into the
output observables linearly over GF(2).

Decoders (public ``decoder_type``): ``mwpm`` (PyMatching / Sparse Blossom) and
``belief_matching``. Because submissions are restricted to graphlike mechanisms, the
correlated edges enter the weighted matching graph directly and BP-reweighted matching
coincides with weighted MWPM — both route through the same Sparse-Blossom matcher (as in
the parent task; ``union_find`` stays deferred). PyMatching is a qsim-/verifier-side
dependency only.
"""

from __future__ import annotations

import math

import numpy as np

from qiqcbench.qsim.hidden_dynamics.logical_cnot_decoder_calibration import construction as C
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery import circuits as CC
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery import metadata as MD
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery.noise_model import (
    exact_circuit_noise,
    sampler_noise_from_hidden,
)
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery.sampler import (
    SamplerNoise,
    sample_bundle,
)

SUPPORTED_DECODERS = ("mwpm", "belief_matching")

PARITY_GROUPS = (
    "m_zz",
    "m_xx",
    "m_z_int",
    "split_zz_0",
    "split_zz_1",
    "split_zz_2",
    "split_xx_0",
    "split_xx_1",
    "split_xx_2",
)
OUTPUT_NAMES = (
    "obs_z_c_out",
    "obs_z_t_out",
    "obs_x_c_out",
    "obs_x_t_out",
    "obs_bell_zz",
    "obs_bell_xx",
    "obs_logical",
)
KNOWN_OBSERVABLES = OUTPUT_NAMES + PARITY_GROUPS


def _edge_weight(p: float) -> float:
    p = min(max(float(p), 1e-9), 0.4999)
    return float(math.log((1.0 - p) / p))


def normalize_mechanism(mech: dict) -> tuple[tuple[int, ...], float, frozenset[str]]:
    """Validate + normalize a submitted mechanism. Raises ValueError if malformed."""
    if not isinstance(mech, dict):
        raise ValueError("mechanism must be an object")
    dets = mech.get("detectors")
    if not isinstance(dets, (list, tuple)) or not (1 <= len(dets) <= 2):
        raise ValueError(f"mechanism must flip 1 or 2 detectors, got {dets!r}")
    dets = tuple(int(d) for d in dets)
    p = float(mech["p"])
    if not (0.0 < p < 0.5):
        raise ValueError(f"mechanism p must be in (0, 0.5), got {p}")
    flips = mech.get("flips_observables")
    if flips is None and "flips_observable" in mech:  # memory-DEM boolean alias
        flips = ["obs_logical"] if mech["flips_observable"] else []
    if flips is None:
        flips = []
    if not isinstance(flips, (list, tuple)):
        raise ValueError(f"flips_observables must be a list of names, got {flips!r}")
    names = frozenset(str(f) for f in flips)
    unknown = names - set(KNOWN_OBSERVABLES)
    if unknown:
        raise ValueError(f"unknown observable names in flips_observables: {sorted(unknown)}")
    return dets, p, names


# --------------------------------------------------------------------------- #
# Matching construction.
# --------------------------------------------------------------------------- #


def consolidate(
    mechanisms: list[tuple[tuple[int, ...], float, set[int]]],
) -> list[tuple[tuple[int, ...], float, frozenset[int]]]:
    """One edge per detector set: XOR-combine parallel mechanisms' probabilities and keep
    the DOMINANT component's fault set. (PyMatching's own parallel-edge merge keeps the
    first edge's fault ids, silently discarding later observable flips — consolidating
    here makes the semantics explicit and deterministic.)"""
    acc: dict[tuple[int, ...], tuple[float, float, frozenset[int]]] = {}
    for dets, p, fids in mechanisms:
        key = tuple(sorted(dets))
        p_total, p_max, best = acc.get(key, (0.0, -1.0, frozenset()))
        p_total = p_total * (1.0 - p) + p * (1.0 - p_total)
        if p > p_max:
            p_max, best = p, frozenset(fids)
        acc[key] = (p_total, p_max, best)
    return [(dets, pt, best) for dets, (pt, _pm, best) in sorted(acc.items())]


def build_matching(mechanisms: list[tuple[tuple[int, ...], float, set[int]]], n_detectors: int):
    """mechanisms: (local detector ids, p, fault ids). Returns a PyMatching matcher."""
    import pymatching

    m = pymatching.Matching()
    for dets, p, fids in consolidate(mechanisms):
        w = _edge_weight(p)
        if len(dets) == 1:
            m.add_boundary_edge(dets[0], fault_ids=set(fids), weight=w)
        else:
            a, b = dets
            if a == b:
                continue
            m.add_edge(a, b, fault_ids=set(fids), weight=w)
    m.set_boundary_nodes(set())
    if m.num_detectors < n_detectors:
        # ensure the matcher accepts full-width syndromes (isolated detectors got no edge
        # only if validation was skipped; scoring validation requires full coverage)
        m.add_boundary_edge(n_detectors - 1, weight=_edge_weight(1e-6))
        m.set_boundary_nodes(set())
    return m


def decode_batch(matching, syndrome: np.ndarray, n_faults: int) -> np.ndarray:
    width = matching.num_detectors
    if syndrome.shape[1] > width:
        syndrome = syndrome[:, :width]
    elif syndrome.shape[1] < width:
        pad = np.zeros((syndrome.shape[0], width - syndrome.shape[1]), dtype=np.uint8)
        syndrome = np.concatenate([syndrome, pad], axis=1)
    preds = np.asarray(matching.decode_batch(syndrome)).reshape(syndrome.shape[0], -1)
    out = np.zeros((syndrome.shape[0], n_faults), dtype=np.uint8)
    k = min(n_faults, preds.shape[1])
    out[:, :k] = preds[:, :k].astype(np.uint8)
    return out


def memory_mechanisms(dem: list[dict]) -> list[tuple[tuple[int, ...], float, set[int]]]:
    out = []
    for mech in dem:
        dets, p, names = normalize_mechanism(mech)
        out.append((dets, p, {0} if "obs_logical" in names else set()))
    return out


def project_cnot_mechanisms(
    dem: list[dict], config: str
) -> list[tuple[tuple[int, ...], float, set[int]]]:
    """Project superset-id mechanisms onto one reporting configuration: map global ->
    local detector columns (drop mechanisms touching inactive detectors) and fold the
    declared observable flips through the public frame convention."""
    bundle = CC.cnot_bundle(config, 0, 0)
    meta = _cnot_meta()
    columns = meta["configs"][config]["detector_columns"]
    local_of = {g: i for i, g in enumerate(columns)}
    obs_names = [o.name for o in bundle.observables]
    frame = {k: v for k, v in bundle.frame_table.items() if k != "_constants"}
    out = []
    for mech in dem:
        dets, p, names = normalize_mechanism(mech)
        if any(d not in local_of for d in dets):
            continue
        fids: set[int] = set()
        for k, obs in enumerate(obs_names):
            flip = obs in names
            for g in frame.get(obs, []):
                if g in names:
                    flip = not flip
            if flip:
                fids.add(k)
        out.append((tuple(local_of[d] for d in dets), p, fids))
    return out


_CNOT_META_CACHE: dict | None = None


def _cnot_meta() -> dict:
    global _CNOT_META_CACHE
    if _CNOT_META_CACHE is None:
        _CNOT_META_CACHE = MD.cnot_superset_metadata()
    return _CNOT_META_CACHE


# --------------------------------------------------------------------------- #
# Fresh-shot replay.
# --------------------------------------------------------------------------- #


def per_cycle(eps_total: float, rounds: int) -> float:
    eps_total = min(max(eps_total, 0.0), 0.5 - 1e-12)
    return 0.5 * (1.0 - (1.0 - 2.0 * eps_total) ** (1.0 / rounds))


def true_noise() -> SamplerNoise:
    return sampler_noise_from_hidden(C.build_hidden_config())


def replay_memory(
    dem: list[dict],
    decoder_type: str,
    *,
    layout: str = "d5",
    rounds: int | None = None,
    shots: int = C.REPLAY_SHOTS_MEMORY,
    rng: np.random.Generator,
    noise: SamplerNoise | None = None,
) -> tuple[float, float]:
    """Sample fresh hidden shots of the fixed scoring layout, decode with the submitted
    DEM, return ``(eps_total, eps_per_cycle)``."""
    if decoder_type not in SUPPORTED_DECODERS:
        raise ValueError(f"unsupported decoder_type {decoder_type!r}")
    rounds = C.MEMORY_SCORING_ROUNDS if rounds is None else rounds
    bundle = CC.memory_bundle(layout, rounds)
    noise = noise or true_noise()
    matching = build_matching(memory_mechanisms(dem), bundle.n_detectors)
    errs = 0
    res = sample_bundle(bundle, noise, shots, rng)
    pred = decode_batch(matching, res.detection_events, 1)
    errs = int(np.sum(pred[:, 0] != res.observable_flips[:, 0]))
    eps_total = errs / shots
    return eps_total, per_cycle(eps_total, rounds)


def replay_cnot_entry(
    dem: list[dict],
    decoder_type: str,
    config: str,
    entry: str,
    *,
    shots: int,
    rng: np.random.Generator,
    noise: SamplerNoise | None = None,
) -> float:
    """Per-entry replayed error: fraction of fresh shots whose frame-corrected logical
    outcome differs from the ideal CNOT output in either bit."""
    if decoder_type not in SUPPORTED_DECODERS:
        raise ValueError(f"unsupported decoder_type {decoder_type!r}")
    a, b = int(entry[0]), int(entry[1])
    bundle = CC.cnot_bundle(config, a, b)
    noise = noise or true_noise()
    mechs = project_cnot_mechanisms(dem, config)
    matching = build_matching(mechs, bundle.n_detectors)
    res = sample_bundle(bundle, noise, shots, rng)
    n_obs = len(bundle.observables)
    pred = decode_batch(matching, res.detection_events, n_obs)
    fails = np.any(pred != res.observable_flips, axis=1)
    return float(np.mean(fails))


def replay_cnot_battery(
    dem: list[dict],
    decoder_type: str,
    *,
    shots_per_entry: int = C.REPLAY_SHOTS_PER_ENTRY,
    rng: np.random.Generator,
    noise: SamplerNoise | None = None,
) -> dict:
    noise = noise or true_noise()
    entries: dict[str, float] = {}
    for basis, entry in C.CNOT_BATTERY:
        e = replay_cnot_entry(
            dem, decoder_type, basis, entry, shots=shots_per_entry, rng=rng, noise=noise
        )
        entries[f"{basis}:{entry}"] = e
    avg = float(np.mean(list(entries.values())))
    return {"entries": entries, "average": avg, "max_entry": max(entries.values())}


def replay_bell_witness(
    dem: list[dict],
    decoder_type: str,
    *,
    shots: int = C.REPLAY_SHOTS_BELL,
    rng: np.random.Generator,
    noise: SamplerNoise | None = None,
) -> dict:
    """Ungated derived figure: decode fresh bell_zz/bell_xx shots with the
    submitted CNOT DEM + decoder and report F_bell >= (<XX> + <ZZ>)/2."""
    noise = noise or true_noise()
    vals = {}
    for config in ("bell_zz", "bell_xx"):
        e = replay_cnot_entry(dem, decoder_type, config, "00", shots=shots, rng=rng, noise=noise)
        vals[config] = 1.0 - 2.0 * e
    f_bell = 0.5 * (vals["bell_zz"] + vals["bell_xx"])
    return {"zz": vals["bell_zz"], "xx": vals["bell_xx"], "f_bell_lower_bound": f_bell}


def reference_d3_dem(layout: str = "d3_control") -> list[dict]:
    """Verifier-side reference d=3 DEM for the ungated Lambda figure: the exact circuit
    DEM of the memory layout under the hidden Pauli noise (leakage excluded)."""
    bundle = CC.memory_bundle(layout, C.MEMORY_SCORING_ROUNDS)
    noisy = CC.with_noise(bundle, exact_circuit_noise(C.build_hidden_config()))
    edges = MD.dem_edges(noisy, [o.name for o in bundle.observables])
    return [
        {
            "detectors": e["detectors"],
            "p": min(e["p_nominal"], 0.4999),
            "flips_observables": e["flips_observables"],
        }
        for e in edges
        if e["detectors"]
    ]


__all__ = [
    "KNOWN_OBSERVABLES",
    "PARITY_GROUPS",
    "SUPPORTED_DECODERS",
    "build_matching",
    "decode_batch",
    "memory_mechanisms",
    "normalize_mechanism",
    "per_cycle",
    "project_cnot_mechanisms",
    "reference_d3_dem",
    "replay_bell_witness",
    "replay_cnot_battery",
    "replay_cnot_entry",
    "replay_memory",
    "true_noise",
]
