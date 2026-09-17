"""Strategy-ladder anchors for ``logical_cnot_decoder_calibration``.

Floor placement is decided against an explicit AGENT-STRATEGY LADDER on the realized
instance, all scored by fresh-shot replay:

Memory (d=5):
- ``m_iid``     — the notebook recipe taken literally: independent depolarizing — only
  same-round spatial pairs, same-ancilla temporal pairs, and boundary singles at nominal
  uniform rates, no schedule-derived diagonal (hook) edges + MWPM (the memory FAIL
  anchor, ~1.5e-3 vs floor 1.0e-3).
- ``m_uniform`` — the FULL public matching-graph topology at NOMINAL uniform rates +
  MWPM (passes — competence-floor side, ~0.7e-3: the memory gate is a competence floor;
  the CNOT gate is the discriminator).
- ``m_exact``  — the exact hidden circuit-level Pauli DEM (perfect calibration upper
  bound; leakage is NOT in it — leakage is not schedule-derivable).
- ``m_est``    — the data-driven reference-solver DEM (rates + correlations incl.
  discovered leakage pairs; PASS side).

CNOT:
- ``c_uniform``  — nominal public adjacency (rung i, outer envelope).
- ``c_transfer`` — the exact circuit DEM under the BULK-TRANSFER config (perfect bulk
  calibration tiled onto the CNOT graph, routing = bulk means, no leakage; rung ii). It
  already contains every schedule-derivable merged-window mechanism (seam hooks,
  transition edges) at transferred rates, so it strictly dominates the rung
  (iii-b) hybrid.
- ``c_rescale``  — rung (iii): ``c_transfer`` with every seam-incident edge rescaled so
  the predicted single-detector means match the OBSERVED seam detector means (the
  correlation-free cheese).
- ``c_parity_guess_uniform`` / ``c_parity_guess_transfer`` and ``c_guess_rule_best`` —
  the zero-joint-data attribution-guess family (rung iii-c),
  agent-information-only: base DEM + observed-MEAN residual singles on the merged-only
  seam detectors with the observable attribution fixed by a deterministic public rule
  (basis label / no attribution / footprint-position heuristics). Since the
  Part-3 attribution hardening the hidden per-site P/N flavor mix defeats every such
  rule (each window and each footprint class mixes both parity classes, and the P/N
  flavor pairs are provably mean-identical), so the entire family sits on the FAIL
  side — the flip of the original counterexample, pinned by ``ordered_ok``.
- ``c_attrib_measured`` — the PASS-side twin: the same mean-residual construction with
  the MEASURED (true) attribution, i.e. what the taught detector x observable
  co-occurrence measurement achieves over the public nominal base.
- ``c_attrib_flip1_min`` — the luckiest 7-of-8-bits guess (minimum over all single-site
  attribution flips). Its ``max_entry`` must exceed the per-entry cap: the cap, not the
  average floor, is the lever that kills near-correct lucky guessing (pinned by
  ``ordered_ok``; a full 8-bit guess is ~0.4% likely and requires executing the whole
  residual pipeline).
- ``c_est``      — the data-driven reference-solver superset DEM (rung iv, PASS target).

The floors must sit so uniform/transfer/rescale and the whole attribution-guess family
fail robustly while the estimated/measured DEMs pass with margin; the materialization
test asserts the ordering. Attribution is measurable ONLY through detector x observable
joint statistics (co-occurrence, or an equivalent data-driven selection such as per-site
self-decoding) — those are genuine measurements and are meant to pass; the FAIL family
is exactly the zero-joint-data strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qiqcbench.qsim.hidden_dynamics.logical_cnot_decoder_calibration import construction as C
from qiqcbench.qsim.hidden_dynamics.logical_cnot_decoder_calibration import decoders as D
from qiqcbench.qsim.hidden_dynamics.logical_cnot_decoder_calibration import (
    reference_solver as R,
)
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery import circuits as CC
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery import layout as L
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery import metadata as MD
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery.noise_model import (
    exact_circuit_noise,
)


def _edges_to_dem(edges: list[dict]) -> list[dict]:
    return [
        {
            "detectors": list(e["detectors"]),
            "p": min(float(e["p_nominal"]), 0.4999),
            "flips_observables": list(e["flips_observables"]),
        }
        for e in edges
    ]


def dem_memory_uniform() -> list[dict]:
    meta = MD.memory_metadata("d5", C.MEMORY_SCORING_ROUNDS)
    return _edges_to_dem(meta["matching_graph_nominal"])


def dem_memory_iid() -> list[dict]:
    """The notebook recipe taken literally: independent depolarizing -> only the
    same-round spatial pairs, same-ancilla temporal pairs, and boundary singles at the
    nominal uniform rates; NO schedule-derived diagonal (hook-orientation) edges, no
    final-block cross structure."""
    meta = MD.memory_metadata("d5", C.MEMORY_SCORING_ROUNDS)
    dets = meta["detectors"]
    keep: list[dict] = []
    for e in meta["matching_graph_nominal"]:
        ds = e["detectors"]
        if len(ds) == 1:
            keep.append(e)
            continue
        a, b = dets[ds[0]], dets[ds[1]]
        same_round = a["round"] == b["round"]
        same_anc = a["ancilla"] == b["ancilla"] and abs(a["round"] - b["round"]) == 1
        if same_round or same_anc:
            keep.append(e)
    return _edges_to_dem(keep)


def dem_memory_exact() -> list[dict]:
    bundle = CC.memory_bundle("d5", C.MEMORY_SCORING_ROUNDS)
    noisy = CC.with_noise(bundle, exact_circuit_noise(C.build_hidden_config()))
    return _edges_to_dem(MD.dem_edges(noisy, [o.name for o in bundle.observables]))


def dem_cnot_uniform() -> list[dict]:
    meta = D._cnot_meta()
    per_config = {}
    for config in ("z", "x"):
        per_config[config] = [
            {
                "detectors": e["detectors"],
                "p": min(float(e["p_nominal"]), 0.4999),
                "flips_observables": e["flips_observables"],
            }
            for e in R._local_edges(meta, config)
        ]
    return R.build_cnot_superset_dem(per_config)


def dem_cnot_transfer() -> list[dict]:
    """Exact circuit DEM of the CNOT configurations under the bulk-transfer hidden config
    (routing = bulk means, no leakage) — perfect bulk calibration + trusting the notebook."""
    transfer = C.bulk_transfer_hidden_config()
    noise = exact_circuit_noise(transfer)
    per_config = {}
    for config in ("z", "x"):
        bundle = CC.cnot_bundle(config, 0, 0)
        noisy = CC.with_noise(bundle, noise)
        edges = MD.dem_edges(noisy, [o.name for o in bundle.observables])
        per_config[config] = [
            {
                "detectors": e["detectors"],
                "p": min(float(e["p_nominal"]), 0.4999),
                "flips_observables": e["flips_observables"],
            }
            for e in edges
        ]
    return R.build_cnot_superset_dem(per_config)


def _seam_global_detectors() -> set[int]:
    """Superset ids of detectors whose plaquette support touches a routing data qubit."""
    seam_pos = set()
    for p in L.MERGED_ZZ.plaquettes:
        if set(p.data) & set(L.ROUTING_ZZ):
            seam_pos.add(p.pos)
    for p in L.MERGED_XX.plaquettes:
        if set(p.data) & set(L.ROUTING_XX):
            seam_pos.add(p.pos)
    meta = D._cnot_meta()
    return {
        d["id"]
        for d in meta["superset_detectors"]
        if (d["ancilla"][0], d["ancilla"][1]) in seam_pos
    }


def dem_cnot_rescale(*, observe_shots: int = 100_000, seed: int = 424242) -> list[dict]:
    """Rung (iii): bulk transfer + seam-incident edge probabilities rescaled to match the
    OBSERVED seam single-detector means (cheap, correlation-free)."""
    dem = dem_cnot_transfer()
    seam = _seam_global_detectors()
    meta = D._cnot_meta()

    # observed + predicted single-detector means per superset id (z/x configs)
    obs_mean: dict[int, list[float]] = {}
    rng = np.random.default_rng(seed)
    for config in ("z", "x"):
        dets, _ = R.collect_cnot(config, observe_shots // 2, rng)
        columns = meta["configs"][config]["detector_columns"]
        mu = dets.mean(axis=0)
        for i, g in enumerate(columns):
            obs_mean.setdefault(g, []).append(float(mu[i]))
    x_obs = {g: float(np.mean(v)) for g, v in obs_mean.items()}

    log_pred = {g: 0.0 for g in x_obs}
    for m in dem:
        term = float(np.log(max(1.0 - 2.0 * m["p"], 1e-9)))
        for g in m["detectors"]:
            if g in log_pred:
                log_pred[g] += term
    lam: dict[int, float] = {}
    for g in seam:
        if g not in x_obs:
            continue
        target = float(np.log(max(1.0 - 2.0 * x_obs[g], 1e-9)))
        lam[g] = float(np.clip(target / min(log_pred[g], -1e-9), 0.2, 8.0))

    out = []
    for m in dem:
        factors = [lam[g] for g in m["detectors"] if g in lam]
        if factors:
            scale = float(np.exp(np.mean(np.log(factors))))
            p = 0.5 * (1.0 - max(1.0 - 2.0 * m["p"], 1e-9) ** scale)
            m = dict(m, p=min(max(p, 1e-9), 0.4999))
        out.append(m)
    return out


def _merged_only_seam_detector_basis() -> dict[int, str]:
    """Superset id -> public ``basis`` label for the merged-only seam detectors,
    derived from AGENT-AVAILABLE metadata only: the superset detectors whose ancilla
    coordinate appears in no single-patch layout of the public spec (8 seam ancilla
    sites carrying 16 superset detector ids, 8 per window basis)."""
    single_patch_ancillas = {
        (a["pos"][0], a["pos"][1])
        for layout in C.build_public_spec()["layouts"]
        if not layout["name"].startswith("merged_")
        for a in layout["ancillas"]
    }
    meta = D._cnot_meta()
    return {
        d["id"]: d["basis"]
        for d in meta["superset_detectors"]
        if (d["ancilla"][0], d["ancilla"][1]) not in single_patch_ancillas
    }


def _seam_site_detectors() -> dict[tuple[int, int], dict]:
    """Seam ancilla (x, y) -> {"d01": earlier-round superset id, "d12": later,
    "basis": window basis} — from agent-available metadata only (the public detector
    tables carry ancilla coordinates and rounds)."""
    meta = D._cnot_meta()
    seam_ids = set(_merged_only_seam_detector_basis())
    by_anc: dict[tuple[int, int], list[dict]] = {}
    for d in meta["superset_detectors"]:
        if d["id"] in seam_ids:
            by_anc.setdefault((d["ancilla"][0], d["ancilla"][1]), []).append(d)
    out = {}
    for anc, dets in by_anc.items():
        dets = sorted(dets, key=lambda d: d["round"])
        out[anc] = {"d01": dets[0]["id"], "d12": dets[1]["id"], "basis": dets[0]["basis"]}
    return out


def _observed_seam_residuals(
    base: str, observe_shots: int, seed: int
) -> tuple[list[dict], dict[int, float], dict[int, str]]:
    """Shared machinery of the attribution-guess rungs: base DEM + observed
    single-detector MEAN residuals over its prediction at the merged-only seam
    detectors (``1-2<x_i> = prod(1-2p_e)``; pure means, zero covariance computation;
    means averaged across the z/x reporting configurations; kept above 5e-3; clipped
    at 0.45). Attribution is left to the caller."""
    if base == "uniform":
        dem = dem_cnot_uniform()
    elif base == "transfer":
        dem = dem_cnot_transfer()
    else:
        raise ValueError(f"unknown attribution-guess base {base!r}")
    meta = D._cnot_meta()
    seam_basis = _merged_only_seam_detector_basis()

    obs_mean: dict[int, list[float]] = {}
    rng = np.random.default_rng(seed)
    for config in ("z", "x"):
        dets, _ = R.collect_cnot(config, observe_shots // 2, rng)
        columns = meta["configs"][config]["detector_columns"]
        mu = dets.mean(axis=0)
        for i, g in enumerate(columns):
            obs_mean.setdefault(g, []).append(float(mu[i]))

    log_pred = {g: 0.0 for g in seam_basis}
    for m in dem:
        term = float(np.log(max(1.0 - 2.0 * m["p"], 1e-9)))
        for g in m["detectors"]:
            if g in log_pred:
                log_pred[g] += term

    residuals: dict[int, float] = {}
    for g in sorted(seam_basis):
        if g not in obs_mean:
            continue
        x_obs = float(np.mean(obs_mean[g]))
        ratio = (1.0 - 2.0 * x_obs) / max(float(np.exp(log_pred[g])), 1e-9)
        p_res = 0.5 * (1.0 - ratio)
        if p_res > 5e-3:
            residuals[g] = float(min(p_res, 0.45))
    return dem, residuals, seam_basis


def _parity_group(basis: str) -> str:
    return "m_zz" if basis == "Z" else "m_xx"


ATTRIB_GUESS_RULES = ("all_parity", "none", "d01_parity", "d12_parity")


def dem_cnot_attrib_guess(
    *,
    base: str = "uniform",
    rule: str = "all_parity",
    observe_shots: int = 80_000,
    seed: int = 424243,
) -> list[dict]:
    """The zero-joint-data attribution-guess family, agent-information
    only: base DEM + one observed-MEAN residual single per merged-only seam detector,
    with the observable attribution fixed by a deterministic public rule instead of a
    measurement — ``all_parity`` (the original basis-label guess: Z-window detector ->
    ``m_zz``, X-window -> ``m_xx``), ``none`` (residual singles without attribution),
    ``d01_parity`` / ``d12_parity`` (footprint-position heuristics). Since the
    Part-3 model change every rule is wrong on >= 2 sites by construction (the realized
    flavor draw mixes parity classes in each window and each footprint class), so the
    whole family sits on the FAIL side of the ladder."""
    if rule not in ATTRIB_GUESS_RULES:
        raise ValueError(f"unknown attribution rule {rule!r}")
    dem, residuals, seam_basis = _observed_seam_residuals(base, observe_shots, seed)
    site_class: dict[int, str] = {}
    for info in _seam_site_detectors().values():
        site_class[info["d01"]] = "d01"
        site_class[info["d12"]] = "d12"

    def flips(g: int) -> list[str]:
        if rule == "none":
            return []
        if rule == "all_parity" or site_class.get(g) == rule.split("_")[0]:
            return [_parity_group(seam_basis[g])]
        return []

    out = list(dem)
    for g, p in sorted(residuals.items()):
        out.append({"detectors": [g], "p": p, "flips_observables": flips(g)})
    return out


def dem_cnot_parity_guess(
    *, base: str = "uniform", observe_shots: int = 80_000, seed: int = 424243
) -> list[dict]:
    """The original counterexample rung: the ``all_parity`` basis-label guess
    (see :func:`dem_cnot_attrib_guess`). FAIL side since the Part-3 model change."""
    return dem_cnot_attrib_guess(
        base=base, rule="all_parity", observe_shots=observe_shots, seed=seed
    )


def _true_seam_attribution() -> dict[int, list[str]]:
    """HIDDEN-TRUTH per-site attribution (maintainer-side, like the other exact-model
    anchor inputs): footprint superset id -> the parity groups its excess flips."""
    from qiqcbench.qsim.qtypes.surface_code_lattice_surgery import layout as L

    site_dets = _seam_site_detectors()
    hidden = C.build_hidden_config()
    out: dict[int, list[str]] = {}
    for site in hidden.leakage:
        if site.mode != "activation":
            continue
        anc = L.CHIP_COORDS[site.qubit]
        info = site_dets[(anc[0], anc[1])]
        footprint = "d01" if (site.delay, site.lifetime) in ((0, 1), (1, 40)) else "d12"
        parity = site.delay == 0
        out[info[footprint]] = [_parity_group(info["basis"])] if parity else []
    return out


def dem_cnot_attrib_oracle(
    *,
    wrong_sites: tuple[int, ...] = (),
    base: str = "uniform",
    observe_shots: int = 80_000,
    seed: int = 424243,
) -> list[dict]:
    """Mean-residual DEM with the TRUE per-site attribution, optionally flipped at
    ``wrong_sites`` (footprint superset ids). ``wrong_sites=()`` is the PASS-side
    "measured attribution" rung (what co-occurrence measurement achieves over the
    public base); ``len(wrong_sites)=1`` bounds the best 7-of-8-bits lucky guess — the
    per-entry cap must fail every such single-bit flip."""
    dem, residuals, seam_basis = _observed_seam_residuals(base, observe_shots, seed)
    truth = _true_seam_attribution()
    out = list(dem)
    for g, p in sorted(residuals.items()):
        flips = truth.get(g, [])
        if g in wrong_sites:
            flips = [] if flips else [_parity_group(seam_basis[g])]
        out.append({"detectors": [g], "p": p, "flips_observables": flips})
    return out


@dataclass
class AnchorReport:
    m_iid: float
    m_uniform: float
    m_exact: float
    m_est: float
    c_uniform: float
    c_transfer: float
    c_rescale: float
    c_est: float
    # the basis-label guess — FAIL side since the Part-3 attribution
    # hardening; part of ordered_ok.
    c_parity_guess_uniform: float
    c_parity_guess_transfer: float
    # worst (lowest-average) member of the remaining zero-joint-data rule family
    # (none / d01_parity / d12_parity on the uniform base) — FAIL side.
    c_guess_rule_best: float
    c_guess_rule_best_max_entry: float
    # PASS side: mean-residual DEM with the measured (true) attribution — what
    # co-occurrence measurement achieves over the public nominal base.
    c_attrib_measured: float
    c_attrib_measured_max_entry: float
    # the luckiest 7-of-8-bits guess: minimum over all single-site attribution flips.
    # The per-entry cap must fail every one of them (the average floor need not).
    c_attrib_flip1_min: float
    c_attrib_flip1_min_max_entry: float
    c_rescale_max_entry: float
    c_est_max_entry: float
    c_parity_guess_uniform_max_entry: float
    c_parity_guess_transfer_max_entry: float
    floor_mem: float
    floor_cnot: float

    @property
    def ordered_ok(self) -> bool:
        return (
            self.m_iid > self.floor_mem > self.m_est
            and self.c_uniform > self.floor_cnot
            and self.c_transfer > self.floor_cnot
            and self.c_rescale > self.floor_cnot
            and self.c_parity_guess_uniform > self.floor_cnot
            and self.c_parity_guess_transfer > self.floor_cnot
            and self.c_guess_rule_best > self.floor_cnot
            and self.c_attrib_flip1_min_max_entry > C.CNOT_PER_ENTRY_CAP
            and self.c_est < self.floor_cnot
            and self.c_est_max_entry < C.CNOT_PER_ENTRY_CAP
            and self.c_attrib_measured < self.floor_cnot
            and self.c_attrib_measured_max_entry < C.CNOT_PER_ENTRY_CAP
        )

    def __str__(self) -> str:
        return (
            f"memory/cycle: iid={self.m_iid:.3e} uniform={self.m_uniform:.3e} "
            f"exact={self.m_exact:.3e} "
            f"est={self.m_est:.3e} | floor={self.floor_mem:.3e}\n"
            f"cnot/op: uniform={self.c_uniform:.3e} transfer={self.c_transfer:.3e} "
            f"rescale={self.c_rescale:.3e} est={self.c_est:.3e} "
            f"(max entry rescale={self.c_rescale_max_entry:.3e} est={self.c_est_max_entry:.3e}) "
            f"| floor={self.floor_cnot:.3e}\n"
            f"cnot/op attribution guesses (FAIL side): "
            f"basis-label uniform={self.c_parity_guess_uniform:.3e} "
            f"transfer={self.c_parity_guess_transfer:.3e} "
            f"(max entry {self.c_parity_guess_uniform_max_entry:.3e}/"
            f"{self.c_parity_guess_transfer_max_entry:.3e}) "
            f"rule-best={self.c_guess_rule_best:.3e} "
            f"(max {self.c_guess_rule_best_max_entry:.3e})\n"
            f"cnot/op measured attribution (PASS side): "
            f"avg={self.c_attrib_measured:.3e} max={self.c_attrib_measured_max_entry:.3e} | "
            f"lucky flip1 min: avg={self.c_attrib_flip1_min:.3e} "
            f"max-entry={self.c_attrib_flip1_min_max_entry:.3e} "
            f"(cap {C.CNOT_PER_ENTRY_CAP:.3e})\n"
            f"ordered_ok={self.ordered_ok}"
        )


def compute_anchors(
    *,
    mem_shots: int = 150_000,
    cnot_shots_per_entry: int = 30_000,
    solver_kwargs: dict | None = None,
    seed: int = 999,
) -> AnchorReport:
    rng = np.random.default_rng(seed)
    noise = D.true_noise()

    def mem(dem: list[dict]) -> float:
        _, pc = D.replay_memory(
            dem,
            "mwpm",
            shots=mem_shots,
            rng=np.random.default_rng(rng.integers(1 << 30)),
            noise=noise,
        )
        return pc

    def cnot(dem: list[dict]) -> tuple[float, float]:
        rep = D.replay_cnot_battery(
            dem,
            "mwpm",
            shots_per_entry=cnot_shots_per_entry,
            rng=np.random.default_rng(rng.integers(1 << 30)),
            noise=noise,
        )
        return rep["average"], rep["max_entry"]

    answer = R.solve(**(solver_kwargs or {}))

    m_iid = mem(dem_memory_iid())
    m_uniform = mem(dem_memory_uniform())
    m_exact = mem(dem_memory_exact())
    m_est = mem(answer["dem_memory_d5"])
    c_uni, _ = cnot(dem_cnot_uniform())
    c_tr, _ = cnot(dem_cnot_transfer())
    c_re, c_re_max = cnot(dem_cnot_rescale())
    c_es, c_es_max = cnot(answer["dem_cnot_spacetime"])
    # appended AFTER the earlier rungs so their shared-rng replay draws are unchanged
    c_pg_uni, c_pg_uni_max = cnot(dem_cnot_parity_guess(base="uniform"))
    c_pg_tr, c_pg_tr_max = cnot(dem_cnot_parity_guess(base="transfer"))
    rule_reports = [
        cnot(dem_cnot_attrib_guess(base="uniform", rule=rule))
        for rule in ("none", "d01_parity", "d12_parity")
    ]
    best = min(range(len(rule_reports)), key=lambda k: rule_reports[k][0])
    c_meas, c_meas_max = cnot(dem_cnot_attrib_oracle())
    flip_reports = [
        cnot(dem_cnot_attrib_oracle(wrong_sites=(g,))) for g in sorted(_true_seam_attribution())
    ]

    return AnchorReport(
        m_iid=m_iid,
        m_uniform=m_uniform,
        m_exact=m_exact,
        m_est=m_est,
        c_uniform=c_uni,
        c_transfer=c_tr,
        c_rescale=c_re,
        c_est=c_es,
        c_parity_guess_uniform=c_pg_uni,
        c_parity_guess_transfer=c_pg_tr,
        c_guess_rule_best=rule_reports[best][0],
        c_guess_rule_best_max_entry=rule_reports[best][1],
        c_attrib_measured=c_meas,
        c_attrib_measured_max_entry=c_meas_max,
        c_attrib_flip1_min=min(r[0] for r in flip_reports),
        c_attrib_flip1_min_max_entry=min(r[1] for r in flip_reports),
        c_rescale_max_entry=c_re_max,
        c_est_max_entry=c_es_max,
        c_parity_guess_uniform_max_entry=c_pg_uni_max,
        c_parity_guess_transfer_max_entry=c_pg_tr_max,
        floor_mem=C.FLOOR_MEM,
        floor_cnot=C.FLOOR_CNOT,
    )


__all__ = [
    "ATTRIB_GUESS_RULES",
    "AnchorReport",
    "compute_anchors",
    "dem_cnot_attrib_guess",
    "dem_cnot_attrib_oracle",
    "dem_cnot_parity_guess",
    "dem_cnot_rescale",
    "dem_cnot_transfer",
    "dem_cnot_uniform",
    "dem_memory_exact",
    "dem_memory_uniform",
]
