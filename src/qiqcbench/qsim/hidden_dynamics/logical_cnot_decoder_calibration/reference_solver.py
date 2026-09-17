"""Data-driven reference solver for ``logical_cnot_decoder_calibration``.

Smoke test for the harness, NOT a tuned baseline. It does what the agent is asked to do:

1. run single-patch memory experiments (bulk characterization) and lattice-surgery CNOT
   runs in both battery bases (merged-layout characterization — CNOT runs double as
   characterization data);
2. estimate every public matching-graph edge rate from the two-point detector
   correlations ``p_ij = (<x_i x_j> - <x_i><x_j>) / ((1-2<x_i>)(1-2<x_j>))`` (Spitz et
   al.), DISCOVER non-adjacency correlated pairs (the hidden leakage structure) under a
   Bonferroni threshold over all candidate pairs, and take single-detector rates as the
   residual of the parity-product identity (never the raw mean);
3. assemble the d=5 memory DEM (scoring layout) + the CNOT superset DEM and self-measure
   both logical rates on held-out data.

It samples through the same engine sampler the MCP tools use (equivalent to calling the
experiment tools) and stays within the public shot + shot-rounds budgets.
"""

from __future__ import annotations

import numpy as np

from qiqcbench.qsim.hidden_dynamics.logical_cnot_decoder_calibration import construction as C
from qiqcbench.qsim.hidden_dynamics.logical_cnot_decoder_calibration import decoders as D
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery import circuits as CC
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery import metadata as MD
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery.sampler import sample_bundle

CNOT_BATTERY = C.CNOT_BATTERY


def _split_two_shots(total_shots: int, *, label: str) -> tuple[int, int]:
    if not isinstance(total_shots, int) or isinstance(total_shots, bool) or total_shots < 2:
        raise ValueError(f"{label} must be an integer of at least 2 shots")
    quotient, remainder = divmod(total_shots, 2)
    return quotient + remainder, quotient


def _erfinv(y: float) -> float:
    from scipy.special import erfinv

    return float(erfinv(y))


def _moments(dets: np.ndarray, chunk: int = 100_000) -> tuple[np.ndarray, np.ndarray]:
    n, d = dets.shape
    x = dets.mean(axis=0)
    coxy = np.zeros((d, d), dtype=np.float64)
    for s in range(0, n, chunk):
        blk = dets[s : s + chunk].astype(np.float64)
        coxy += blk.T @ blk
    coxy /= n
    return x, coxy - np.outer(x, x)


def estimate_mechanisms(
    dets: np.ndarray,
    public_edges: list[dict],
    *,
    det_rounds: np.ndarray | None = None,
    obs_flips: np.ndarray | None = None,
    obs_names: list[str] | None = None,
    alpha: float = 0.01,
    p_floor: float = 1e-6,
    min_dt_discover: int = 2,
    p_min_discover: float = 3e-4,
    p_min_obs_assoc: float = 5e-3,
) -> list[dict]:
    """Estimate a graphlike DEM (local detector ids) from raw detection events.

    ``public_edges``: the layout's nominal matching-graph adjacency (detector pairs /
    boundary edges with their ``flips_observables``). Rates are re-estimated from data;
    non-adjacency pairs clearing the Bonferroni threshold are added as discovered
    correlated edges (observable-flip bits unknown -> empty, the standard honest
    approximation); every detector receives a residual single-detector mechanism so the
    matching graph always covers the full detector set.

    Discovery is restricted to pairs with round separation >= ``min_dt_discover`` (when
    ``det_rounds`` is given): a single circuit fault only correlates detectors within one
    round of each other (those pairs live in the public adjacency and are captured by the
    rate estimation — including the pairwise shadows of genuine hyperedge decompositions),
    while the leakage state machine's lifetime produces the long-range dt>=2 structure
    that must be DISCOVERED. This is the load-bearing physics of the discovery step.

    When ``obs_flips``/``obs_names`` are given, per-detector OBSERVABLE-flip associations
    are estimated too: the residual of the detector<->observable two-point correlation
    after subtracting what the adjacency's flipping mechanisms already explain becomes a
    single-detector observable-flipping mechanism. This is what corrects the (nearly
    detector-silent) mechanisms that flip a joint-parity record — e.g. merge-window
    leakage flipping ``m_zz`` — and it is exactly the kind of two-point structure a
    single-detector rate rescale can never provide.
    """
    n_shots, n_det = dets.shape
    x, cov = _moments(dets)
    denom = 1.0 - 2.0 * x
    denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)

    known_pairs: dict[tuple[int, int], list[str]] = {}
    boundary_flips: dict[int, list[str]] = {}
    boundary_p_nominal: dict[int, float] = {}
    for e in public_edges:
        ds = e["detectors"]
        flips = list(e.get("flips_observables", []))
        if len(ds) == 2:
            key = (min(ds), max(ds))
            known_pairs.setdefault(key, flips)
        elif len(ds) == 1:
            boundary_flips.setdefault(ds[0], flips)
            boundary_p_nominal.setdefault(ds[0], float(e.get("p_nominal", p_floor)))

    mechanisms: list[dict] = []
    pair_p: dict[tuple[int, int], float] = {}

    for (i, j), flips in known_pairs.items():
        p = float(cov[i, j] / (denom[i] * denom[j]))
        p = min(max(p, p_floor), 0.45)
        pair_p[(i, j)] = p
        mechanisms.append({"detectors": [i, j], "p": p, "flips_observables": flips})

    n_pairs = n_det * (n_det - 1) // 2
    z = float(np.sqrt(2.0) * _erfinv(1.0 - alpha / max(n_pairs, 1)))
    iu = np.triu_indices(n_det, k=1)
    sigma = np.sqrt(np.maximum(x[iu[0]] * x[iu[1]], 1e-12) / n_shots) + 1e-12
    signif = cov[iu] > z * sigma
    if det_rounds is not None:
        dt = np.abs(det_rounds[iu[0]] - det_rounds[iu[1]])
        signif &= dt >= min_dt_discover
    for i, j, s in zip(iu[0], iu[1], signif, strict=True):
        if not s or (int(i), int(j)) in known_pairs:
            continue
        p = float(cov[i, j] / (denom[i] * denom[j]))
        if p < p_min_discover:
            continue
        p = min(p, 0.45)
        pair_p[(int(i), int(j))] = p
        mechanisms.append({"detectors": [int(i), int(j)], "p": p, "flips_observables": []})

    # Residual single-detector rates via the parity-product identity — placed ONLY on the
    # public boundary topology (a detector's marginal is inflated by multi-detector
    # events whose pairwise decomposition never sums exactly, so giving EVERY detector a
    # residual boundary edge fabricates escape routes and decodes far worse). Detectors
    # without a public boundary edge get a p_floor placeholder for full graph coverage.
    log_prod = np.zeros(n_det)
    for (i, j), p in pair_p.items():
        term = np.log(max(1.0 - 2.0 * p, 1e-9))
        log_prod[i] += term
        log_prod[j] += term
    z_boundary = float(np.sqrt(2.0) * _erfinv(1.0 - alpha / max(n_det, 1)))
    for i in range(n_det):
        if i in boundary_flips:
            ratio = (1.0 - 2.0 * x[i]) / max(np.exp(log_prod[i]), 1e-9)
            p_res = min(max(float(0.5 * (1.0 - ratio)), p_floor), 0.45)
            # Trust the residual only when it exceeds the nominal boundary rate
            # SIGNIFICANTLY (the same false-discovery discipline applied to extra
            # pairs): the parity-product residual absorbs estimate noise and unmodeled
            # correlated mass as independent boundary mass, fabricating cheap escape
            # routes at hundreds of detectors. Insignificant residuals snap to the
            # nominal rate; genuine excess (e.g. the seam leak, >> nominal) is kept.
            sigma = float(
                np.sqrt(max(x[i] * (1.0 - x[i]), 1e-12) / n_shots) / max(abs(denom[i]), 1e-6)
            )
            p_nom = min(max(boundary_p_nominal.get(i, p_floor), p_floor), 0.45)
            p = p_res if (p_res - p_nom) > z_boundary * sigma else p_nom
            flips = boundary_flips[i]
        else:
            p, flips = p_floor, []
        mechanisms.append({"detectors": [i], "p": p, "flips_observables": flips})

    # detector <-> observable flip associations (residual over the modeled mechanisms)
    if obs_flips is not None and obs_names is not None and obs_flips.size:
        y = obs_flips.astype(np.float64)
        ybar = y.mean(axis=0)
        n_obs = y.shape[1]
        z_obs = float(np.sqrt(2.0) * _erfinv(1.0 - alpha / max(n_det * n_obs, 1)))
        xf64 = dets.astype(np.float64)
        cov_xy = (xf64.T @ y) / n_shots - np.outer(x, ybar)
        for k in range(n_obs):
            dy = 1.0 - 2.0 * ybar[k]
            if abs(dy) < 1e-6:
                dy = 1e-6
            explained = np.zeros(n_det)
            for mech in mechanisms:
                if obs_names[k] in mech["flips_observables"]:
                    for d in mech["detectors"]:
                        explained[d] += mech["p"]
            for i in range(n_det):
                sig = np.sqrt(max(x[i] * ybar[k] * (1 - ybar[k]), 1e-12) / n_shots) + 1e-12
                if cov_xy[i, k] < z_obs * sig:
                    continue
                assoc = float(cov_xy[i, k] / (denom[i] * dy))
                resid = assoc - float(explained[i])
                if resid > p_min_obs_assoc:
                    mechanisms.append(
                        {
                            "detectors": [i],
                            "p": min(resid, 0.45),
                            "flips_observables": [obs_names[k]],
                            "_assoc": True,
                        }
                    )
    return mechanisms


# --------------------------------------------------------------------------- #
# Data collection (equivalent to the agent's experiment calls; budget-accounted).
# --------------------------------------------------------------------------- #


def collect_memory(layout: str, rounds: int, shots: int, rng) -> tuple[np.ndarray, np.ndarray]:
    bundle = CC.memory_bundle(layout, rounds)
    res = sample_bundle(bundle, D.true_noise(), shots, rng)
    return res.detection_events, res.observable_flips[:, 0]


def collect_cnot(
    config: str, shots: int, rng, logical_input: str = "00"
) -> tuple[np.ndarray, np.ndarray]:
    a, b = (int(bit) for bit in logical_input)
    bundle = CC.cnot_bundle(config, a, b)
    res = sample_bundle(bundle, D.true_noise(), shots, rng)
    return res.detection_events, res.observable_flips


def evaluate_cnot_battery(
    dem_cnot: list[dict], total_shots: int, rng
) -> tuple[float, dict[str, float]]:
    """Self-evaluate all eight public truth-table entries within one fixed budget."""
    n_entries = len(CNOT_BATTERY)
    if total_shots < n_entries:
        raise ValueError(f"total_shots must be at least {n_entries} for the CNOT battery")
    shots_per_entry, remainder = divmod(total_shots, n_entries)
    entries: dict[str, float] = {}
    for index, (config, logical_input) in enumerate(CNOT_BATTERY):
        shots = shots_per_entry + (1 if index < remainder else 0)
        a, b = (int(bit) for bit in logical_input)
        bundle = CC.cnot_bundle(config, a, b)
        mechs = D.project_cnot_mechanisms(dem_cnot, config)
        matching = D.build_matching(mechs, bundle.n_detectors)
        dets, observables = collect_cnot(
            config,
            shots,
            rng,
            logical_input=logical_input,
        )
        prediction = D.decode_batch(matching, dets, len(bundle.observables))
        entries[f"{config}:{logical_input}"] = float(
            np.mean(np.any(prediction != observables, axis=1))
        )
    return float(np.mean(list(entries.values()))), entries


def build_cnot_superset_dem(
    per_config_mechs: dict[str, list[dict]],
) -> list[dict]:
    """Merge per-config local-id mechanism lists into one superset-id DEM (shared keys
    averaged)."""
    meta = D._cnot_meta()
    ps_acc: dict[tuple, list] = {}
    flips_acc: dict[tuple, set] = {}
    for config, mechs in per_config_mechs.items():
        columns = meta["configs"][config]["detector_columns"]
        for m in mechs:
            gids = tuple(sorted(columns[d] for d in m["detectors"]))
            ps_acc.setdefault(gids, []).append(m["p"])
            # Output observables are basis-tagged, so the same physical mechanism carries
            # the UNION of its per-configuration observable actions.
            flips_acc.setdefault(gids, set()).update(m["flips_observables"])
    out = []
    for gids in sorted(ps_acc):
        out.append(
            {
                "detectors": list(gids),
                "p": float(np.mean(ps_acc[gids])),
                "flips_observables": sorted(flips_acc[gids]),
            }
        )
    return out


def solve(
    *,
    seed: int = 20260707,
    mem_shots: int = 500_000,
    d3_shots: int = 60_000,
    cnot_shots: int = 320_000,
    eval_shots: int = 120_000,
    alpha: float = 0.01,
) -> dict:
    """Full data-driven pipeline -> a ready-to-score answer dict."""
    rng = np.random.default_rng(seed)

    # --- Stage B: bulk characterization (single-patch memory) ---
    dets5, obs5 = collect_memory("d5", C.MEMORY_SCORING_ROUNDS, mem_shots, rng)
    m5_meta = MD.memory_metadata("d5", C.MEMORY_SCORING_ROUNDS)
    dem_d5 = estimate_mechanisms(
        dets5,
        m5_meta["matching_graph_nominal"],
        det_rounds=np.array([d["round"] for d in m5_meta["detectors"]]),
        alpha=alpha,
    )

    dets3, obs3 = collect_memory("d3_control", C.MEMORY_SCORING_ROUNDS, d3_shots, rng)
    m3_meta = MD.memory_metadata("d3_control", C.MEMORY_SCORING_ROUNDS)
    dem_d3 = estimate_mechanisms(
        dets3,
        m3_meta["matching_graph_nominal"],
        det_rounds=np.array([d["round"] for d in m3_meta["detectors"]]),
        alpha=alpha,
    )

    # --- Stage C: merged-layout characterization (CNOT runs in both bases) ---
    per_config: dict[str, list[dict]] = {}
    meta = D._cnot_meta()
    round_of_gid = {d["id"]: d["round"] for d in meta["superset_detectors"]}
    from qiqcbench.qsim.qtypes.surface_code_lattice_surgery import layout as LL

    merged_only_anc = {
        tuple(LL.CHIP_COORDS[q]) for q in LL.routing_qubit_ids() if LL.CHIP_COORDS[q][0] % 2 == 0
    }
    anc_of_gid = {d["id"]: tuple(d["ancilla"]) for d in meta["superset_detectors"]}
    cnot_config_shots = _split_two_shots(cnot_shots, label="cnot_shots")
    for config, config_shots in zip(("z", "x"), cnot_config_shots, strict=True):
        dcfg, ocfg = collect_cnot(config, config_shots, rng)
        columns = meta["configs"][config]["detector_columns"]
        mechs = estimate_mechanisms(
            dcfg,
            _local_edges(meta, config),
            det_rounds=np.array([round_of_gid[g] for g in columns]),
            obs_flips=ocfg,
            obs_names=meta["configs"][config]["observables"],
            alpha=alpha,
        )
        # Detector->observable associations decide the ATTRIBUTION of the merged-only
        # seam boundary singles: a residual association is accepted only with the
        # clean merge-window joint-parity signature — the detector's ancilla is a
        # merged-only seam ancilla whose basis matches a parity group folded into that
        # output (Z-basis seam ancillas carry m_zz, X-basis carry m_xx) — and is then
        # WRITTEN ONTO that detector's boundary single as the parity-group flip (the
        # public frame convention routes it into every reporting configuration, incl.
        # the bell configs). The graphlike format carries exactly one observable action
        # per detector set, so the measured decision must REPLACE the nominal-class
        # flips the public boundary edge inherits — the nominal class describes the
        # background mechanism, not the seam excess, and the hidden per-site attribution
        # is not derivable from any public metadata or detector statistic (the P/N
        # flavors are mean-identical); only the measured joint decides it. Associations
        # elsewhere are pairwise shadows of non-graphlike mechanisms — adding them as
        # independent boundary edges overcorrects — so they are dropped.
        frame = meta["configs"][config]["frame_convention"]
        basis_of_gid = {d["id"]: d["basis"] for d in meta["superset_detectors"]}
        group_basis = {"m_zz": "Z", "m_xx": "X"}
        kept: list[dict] = []
        assoc_groups: dict[int, list[str]] = {}
        for m in mechs:
            if not m.pop("_assoc", False):
                kept.append(m)
                continue
            gid = columns[m["detectors"][0]]
            if anc_of_gid[gid] not in merged_only_anc:
                continue
            groups = sorted(
                {
                    g
                    for obs in m["flips_observables"]
                    for g in frame.get(obs, [])
                    if g in group_basis and group_basis[g] == basis_of_gid[gid]
                }
            )
            if groups:
                assoc_groups[gid] = groups
        for m in kept:
            if len(m["detectors"]) != 1:
                continue
            gid = columns[m["detectors"][0]]
            if anc_of_gid.get(gid) in merged_only_anc:
                m["flips_observables"] = assoc_groups.get(gid, [])
        per_config[config] = kept
    dem_cnot = build_cnot_superset_dem(per_config)

    # --- self-evaluation on held-out shots ---
    eval_memory_shots, eval_cnot_shots = _split_two_shots(eval_shots, label="eval_shots")
    if eval_cnot_shots < len(CNOT_BATTERY):
        raise ValueError(f"eval_shots must reserve at least {len(CNOT_BATTERY)} CNOT shots")
    m5 = D.build_matching(D.memory_mechanisms(dem_d5), dets5.shape[1])
    ev5, evo5 = collect_memory("d5", C.MEMORY_SCORING_ROUNDS, eval_memory_shots, rng)
    pred5 = D.decode_batch(m5, ev5, 1)[:, 0]
    eps5_total = float(np.mean(pred5 != evo5))
    eps5 = D.per_cycle(eps5_total, C.MEMORY_SCORING_ROUNDS)

    m3 = D.build_matching(D.memory_mechanisms(dem_d3), dets3.shape[1])
    pred3 = D.decode_batch(m3, dets3, 1)[:, 0]
    eps3 = D.per_cycle(float(np.mean(pred3 != obs3)), C.MEMORY_SCORING_ROUNDS)

    self_cnot, _self_cnot_entries = evaluate_cnot_battery(dem_cnot, eval_cnot_shots, rng)

    shots_mem = mem_shots + d3_shots + eval_memory_shots
    shots_surgery = cnot_shots + eval_cnot_shots

    return {
        "task_id": C.TASK_ID,
        "method": (
            "Bulk DEMs estimated from single-patch memory data (public-adjacency rates "
            "via two-point correlations, residual boundary rates); the CNOT superset DEM "
            "estimated directly from merged-layout data (z+x reporting configurations), "
            "including Bonferroni-discovered correlated pairs absent from the public "
            "adjacency (routing/merge-window leakage); MWPM with log((1-p)/p) weights."
        ),
        "stage_a_mode": "device_probe",
        "probed_spec": {"device_id": C.DEVICE_ID, "floors": [C.FLOOR_MEM, C.FLOOR_CNOT]},
        "stage_b_mode": "memory_characterization",
        "shots_for_memory_characterization_raw": shots_mem,
        "dem_memory_d5": dem_d5,
        "decoder_config_memory": {"decoder_type": "mwpm"},
        "self_measured_memory_logical_rate_raw": eps5,
        "lambda_d3_d5_raw": float(eps3 / max(eps5, 1e-9)),
        "memory_characterization_note": (
            "per-edge rates are strongly inhomogeneous; correlated pairs beyond the "
            "public adjacency detected under Bonferroni control"
        ),
        "stage_c_mode": "surgery_characterization",
        "shots_for_surgery_characterization_raw": shots_surgery,
        "dem_cnot_spacetime": dem_cnot,
        "decoder_config_cnot": {"decoder_type": "mwpm"},
        "self_measured_cnot_error_raw": self_cnot,
        "surgery_characterization_note": (
            "seam detectors fire well above bulk-transferred predictions; merged-layout "
            "two-point correlations reveal routing-region degradation and merge-window "
            "leakage pairs no memory experiment shows"
        ),
        "stage_d_mode": "submit_decoders",
        "dem_memory_d3": dem_d3,
    }


def _local_edges(meta: dict, config: str) -> list[dict]:
    """The config's nominal adjacency re-expressed in LOCAL detector columns."""
    columns = meta["configs"][config]["detector_columns"]
    local_of = {g: i for i, g in enumerate(columns)}
    out = []
    for e in meta["configs"][config]["matching_graph_nominal"]:
        if any(d not in local_of for d in e["detectors"]):
            continue
        out.append(
            {
                "detectors": [local_of[d] for d in e["detectors"]],
                "p_nominal": e["p_nominal"],
                "flips_observables": e["flips_observables"],
            }
        )
    return out


__all__ = [
    "build_cnot_superset_dem",
    "CNOT_BATTERY",
    "collect_cnot",
    "collect_memory",
    "evaluate_cnot_battery",
    "estimate_mechanisms",
    "solve",
]


def reference_answer() -> dict:
    """Entry point for the verifier smoke tooling: a full data-driven answer envelope."""
    return solve()
