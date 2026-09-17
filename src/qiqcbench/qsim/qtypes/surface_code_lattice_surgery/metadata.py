"""Public machine-readable metadata for ``surface_code_lattice_surgery``.

Everything here is derivable from PUBLIC knowledge only — the chip map, the schedules,
and the *nominal* uniform noise scale. The matching-graph adjacency is the detector
topology of the public schedule under nominal noise (rates are the agent's to estimate);
the hidden inhomogeneity, the routing degradation, and the leakage topology are absent by
construction.

The CNOT spacetime detector set is a shared indexed **superset** over the four reporting
configurations: detectors are keyed by content (ancilla, rounds, kind), the union is
enumerated once, and each configuration lists its active detector ids in the exact column
order of the returned bit arrays.
"""

from __future__ import annotations

import stim

from qiqcbench.qsim.qtypes.surface_code_lattice_surgery import circuits as CC
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery import layout as L
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery.circuits import CircuitBundle, NoiseParams

NOMINAL_IDLE = 0.0035
NOMINAL_CZ = 0.0035
NOMINAL_RESET = 0.002
NOMINAL_MEAS = 0.008


def nominal_noise(bundle: CircuitBundle) -> NoiseParams:
    return CC.uniform_noise(
        bundle, p_idle=NOMINAL_IDLE, p_cz=NOMINAL_CZ, p_reset=NOMINAL_RESET, p_meas=NOMINAL_MEAS
    )


def _xor_combine(p1: float, p2: float) -> float:
    return p1 * (1.0 - p2) + p2 * (1.0 - p1)


def dem_edges(noisy: stim.Circuit, obs_names: list[str]) -> list[dict]:
    """Graphlike edge list from a noisy circuit's decomposed DEM: each edge flips 1-2
    detectors (+ observables by name). Duplicate edges are XOR-combined."""
    dem = noisy.detector_error_model(
        decompose_errors=True, ignore_decomposition_failures=True, flatten_loops=True
    )
    acc: dict[tuple, float] = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        groups: list[list] = [[]]
        for t in inst.targets_copy():
            if t.is_separator():
                groups.append([])
            else:
                groups[-1].append(t)
        for g in groups:
            dets = tuple(sorted(t.val for t in g if t.is_relative_detector_id()))
            obs = tuple(sorted(t.val for t in g if t.is_logical_observable_id()))
            if not dets or len(dets) > 2:
                continue  # undetectable or non-graphlike residue — not matchable edges
            key = (dets, obs)
            acc[key] = _xor_combine(acc.get(key, 0.0), p)
    # One edge per detector set (a weighted matching graph has one edge per pair):
    # XOR-combine parallel components and keep the DOMINANT component's observable action.
    by_dets: dict[tuple, tuple[float, float, tuple]] = {}  # dets -> (p_total, p_max, obs)
    for (dets, obs), p in acc.items():
        p_total, p_max, best_obs = by_dets.get(dets, (0.0, -1.0, ()))
        p_total = _xor_combine(p_total, p)
        if p > p_max:
            p_max, best_obs = p, obs
        by_dets[dets] = (p_total, p_max, best_obs)
    edges = []
    for dets in sorted(by_dets):
        p_total, _p_max, obs = by_dets[dets]
        edges.append(
            {
                "detectors": list(dets),
                "p_nominal": round(p_total, 8),
                "flips_observables": [obs_names[i] for i in obs],
            }
        )
    return edges


def _detector_table(bundle: CircuitBundle) -> list[dict]:
    out = []
    for i, d in enumerate(bundle.detectors):
        out.append(
            {
                "id": i,
                "ancilla": [d.coord[0], d.coord[1]],
                "round": d.coord[2],
                "kind": d.key[0],
                "basis": L.plaquette_type((d.coord[0], d.coord[1])),
            }
        )
    return out


def memory_metadata(layout_name: str, rounds: int) -> dict:
    bundle = CC.memory_bundle(layout_name, rounds)
    noisy = CC.with_noise(bundle, nominal_noise(bundle))
    return {
        "layout": layout_name,
        "rounds": rounds,
        "n_detectors": bundle.n_detectors,
        "detector_order_rule": (
            "round-0 block: Z-ancillas in (y,x) order; then per round t=1..rounds-1: all "
            "ancillas in (y,x) order; then final block: Z-ancillas in (y,x) order. The "
            "same rule applies at any `rounds` you run."
        ),
        "detectors": _detector_table(bundle),
        "observables": [o.name for o in bundle.observables],
        "matching_graph_nominal": dem_edges(noisy, [o.name for o in bundle.observables]),
    }


def merged_metadata(window: str, rounds: int, include_transitions: bool) -> dict:
    bundle = CC.merged_bundle(window, rounds, include_transitions)
    noisy = CC.with_noise(bundle, nominal_noise(bundle))
    return {
        "window": window,
        "rounds": rounds,
        "include_transitions": include_transitions,
        "n_detectors": bundle.n_detectors,
        "detector_order_rule": (
            "detectors are emitted per round in (y,x) ancilla order, phases in schedule "
            "order (separate rounds, merged rounds, separate rounds), then the final "
            "readout block; only deterministic parities appear (first-round detectors "
            "exist only for the basis matching the preparation; merge/split transition "
            "detectors appear where the support change is justified by the routing "
            "preparation/measure-out)."
        ),
        "detectors": _detector_table(bundle),
        "observables": [o.name for o in bundle.observables],
        "matching_graph_nominal": dem_edges(noisy, [o.name for o in bundle.observables]),
    }


def cnot_superset_metadata() -> dict:
    """The shared indexed CNOT detector superset + per-configuration active columns,
    adjacency, named parity definitions, and the public byproduct-frame convention."""
    bundles = {cfg: CC.cnot_bundle(cfg, 0, 0) for cfg in CC.CNOT_CONFIGS}
    key_info: dict[tuple, tuple] = {}
    for bundle in bundles.values():
        for d in bundle.detectors:
            key_info.setdefault(d.key, d.coord)
    ordered = sorted(
        key_info, key=lambda k: (key_info[k][2], key_info[k][1], key_info[k][0], str(k))
    )
    gid_of = {k: i for i, k in enumerate(ordered)}

    superset = []
    for k in ordered:
        x, y, t = key_info[k]
        superset.append(
            {
                "id": gid_of[k],
                "ancilla": [x, y],
                "round": t,
                "kind": k[0],
                "basis": L.plaquette_type((x, y)),
            }
        )

    configs: dict[str, dict] = {}
    for cfg, bundle in bundles.items():
        columns = [gid_of[d.key] for d in bundle.detectors]
        noisy = CC.with_noise(bundle, nominal_noise(bundle))
        obs_names = [o.name for o in bundle.observables]
        edges = dem_edges(noisy, obs_names)
        for e in edges:
            e["detectors"] = [columns[i] for i in e["detectors"]]
        named = {
            name: [
                {
                    "qubit": [bundle.records[r].qubit[0], bundle.records[r].qubit[1]],
                    "round": bundle.records[r].round,
                    "kind": bundle.records[r].kind,
                }
                for r in recs
            ]
            for name, recs in bundle.named_parities.items()
        }
        configs[cfg] = {
            "detector_columns": columns,
            "n_detectors": bundle.n_detectors,
            "observables": obs_names,
            "frame_convention": {k: v for k, v in bundle.frame_table.items() if k != "_constants"},
            "named_parities": named,
            "matching_graph_nominal": edges,
        }

    return {
        "n_superset_detectors": len(superset),
        "rounds": CC.CNOT_ROUNDS,
        "merge_rounds": CC.MERGE_ROUNDS,
        "schedule": (
            "prep (C,T per config; INT |+>) -> 1 separate round -> rough merge C-INT "
            "(routing row |+>, 3 merged rounds, split via M_X) -> 1 separate round -> "
            "smooth merge INT-T (routing col |0>, 3 merged rounds, split via M_Z) -> "
            "1 separate round -> M_Z(INT) + transversal readout of C,T in the "
            "configuration basis"
        ),
        "superset_detectors": superset,
        "configs": configs,
    }


def frame_convention_summary() -> dict[str, dict[str, list[str]]]:
    return {
        cfg: {k: v for k, v in CC.cnot_bundle(cfg, 0, 0).frame_table.items() if k != "_constants"}
        for cfg in CC.CNOT_CONFIGS
    }


__all__ = [
    "NOMINAL_CZ",
    "NOMINAL_IDLE",
    "NOMINAL_MEAS",
    "NOMINAL_RESET",
    "cnot_superset_metadata",
    "dem_edges",
    "frame_convention_summary",
    "memory_metadata",
    "merged_metadata",
    "nominal_noise",
]
