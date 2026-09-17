"""Canonical hidden instance + public payloads for ``logical_cnot_decoder_calibration``.

The hidden circuit-level noise model is a pure function of ``SEED`` with a FROZEN draw
order (never reorder the draws after materialization — the YAML-parity test pins the
materialized configs to this construction). The engine consumes the materialized hidden
YAML; the verifier rebuilds the identical model from here, so the gate can never drift
from the device.

Floors and anchors are re-derived from the realized instance by ``anchors.py`` (the
strategy-ladder anchors); the shipped values below are recorded here and
asserted against the re-derived anchors by the slow materialization test.
"""

from __future__ import annotations

import numpy as np

from qiqcbench.qsim.qtypes.surface_code_lattice_surgery import circuits as CC
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery import layout as L
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery import metadata as MD
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery.device import (
    HiddenLatticeSurgeryConfig,
)

TASK_ID = "logical_cnot_decoder_calibration"
DEVICE_ID = "sc_lattice_surgery_114q_v0"
SEED = 47250811

CYCLE_TIME_US = 0.6  # fast-cycle transmon grid; keeps d=3 patches comfortably sub-threshold

# --- bulk hidden ranges (widened at materialization so the uniform
# notebook recipe fails robustly — MWPM is famously weight-robust, so the realized
# inhomogeneity must be strong) ---
T1_RANGE_US = (70.0, 180.0)
T2_RATIO_RANGE = (0.4, 0.9)  # T2 = r * 2 * T1
CZ_RANGE = (0.0003, 0.0035)
P_M01_RANGE = (0.001, 0.006)
P_M10_RANGE = (0.004, 0.012)
P_RESET_RANGE = (0.0005, 0.002)

# --- leakage (classical stuck-ancilla state machine; Pauli-approximated) ---
N_BULK_LEAK_SITES = 18  # leakage-prone MEASURE qubits drawn from the memory layouts
LEAK_P_ENTER_RANGE = (0.0015, 0.0035)
LEAK_LIFETIMES = (3, 4, 5)
LEAK_P_FLIP_ROUND = 0.75

# --- routing-region degradation (the lever; parked-qubit drift, deliberately exaggerated;
# the merged-only seam ancillas are ALL leakage-prone in ACTIVATION mode. Each site
# carries one hidden attribution FLAVOR drawn from ROUTING_LEAK_FLAVORS:
# a P-flavor leak's flip set contains the first-round record, so one transition detector
# fires AND the window's joint parity (m_zz/m_xx) flips; an N-flavor leak enters after
# round 0 and stays stuck until the window ends, so the SAME single transition detector
# fires with NO parity flip. Within each footprint class the P and N flavors differ
# exactly by the detector-silent all-run flip, so no detector mean, pair statistic, or
# raw-record statistic separates them at any window length or transition mode — the
# per-site attribution is measurable only through detector x observable joint statistics
# (co-occurrence, or an equivalent data-driven selection). A single-detector RESCALE of
# edge weights still cannot bridge the excess (the rescale anchor fails), and guessing
# the attribution from the public ``basis`` labels — that counterexample — now
# fails the floor because the realized draw mixes P and N sites in both windows (see the
# parity-guess anchor rung, FAIL side). ---
ROUTING_T1_FACTOR_RANGE = (0.35, 0.55)
ROUTING_CZ_FACTOR_RANGE = (1.8, 2.5)
ROUTING_LEAK_P_ENTER_RANGE = (0.08, 0.12)
ROUTING_LEAK_P_FLIP_ROUND = 1.0
# (delay, lifetime) per attribution flavor; lifetime 40 exceeds every legal merged run
# (max_rounds 32), i.e. the leak persists until the window ends. Cold round-0 runs never
# trigger activation leaks (a global boot is not an activation event) — this keeps the
# no-transitions merged window, whose first-round seam detectors are deterministic, from
# exposing the flavor through detector means.
ROUTING_LEAK_FLAVORS = (
    (0, 1),  # P@d01: activation transient, flips r0     -> fires d01, flips parity
    (0, 2),  # P@d12: activation transient, flips r0,r1  -> fires d12, flips parity
    (1, 40),  # N@d01: stuck from round 1, flips r1..end  -> fires d01, no parity flip
    (2, 40),  # N@d12: stuck from round 2, flips r2..end  -> fires d12, no parity flip
)

# --- scoring constants (floors re-derived by anchors.py; see the materialization test).
# Realized anchors (this seed, 20k/entry
# paired replays): memory/cycle iid=1.51e-3, uniform-public-topology=0.78e-3,
# exact=0.76e-3, data-estimated=0.49e-3; cnot/op uniform=0.269, bulk-transfer=0.268,
# seam-rescaled=0.268, data-estimated=0.144 (max entry 0.150), measured-attribution
# mean-residual=0.140 (max 0.150). The zero-joint-data attribution-guess family all
# FAILS: basis-label guess 0.219 (uniform base) / 0.229 (transfer), no-attribution
# 0.287, footprint rules 0.234-0.274; every single-site attribution flip (the
# luckiest 7-of-8-bits guess) has max entry >= 0.195 > cap 0.185 (min avg 0.161), and
# every 2-wrong assignment fails the average floor (>= 0.188). The per-entry cap is
# therefore the lucky-guess lever; the honest band (est/measured, <= 0.150 max) clears
# both gates with ~18-20% margin. ---
MEMORY_SCORING_ROUNDS = 12
FLOOR_MEM = 1.0e-3  # per cycle, d=5 memory (iid fails at 1.5x, calibrated passes at 0.7x)
FLOOR_CNOT = 1.75e-1  # per op, battery-averaged (see the ladder below)
# The per-entry cap is the lever that catches a LUCKY near-correct attribution guess
# (7 of 8 hidden bits): a single wrong bit concentrates ~p_enter of error onto one
# window's battery entries, pushing its worst entry past the cap while the average can
# stay under the floor. It is deliberately tight (~1.06x the floor, matching the honest
# band's natural max/avg entry spread) rather than the historical 2x.
CNOT_PER_ENTRY_CAP = 1.85e-1
REPLAY_SHOTS_MEMORY = 400_000
REPLAY_SHOTS_PER_ENTRY = 100_000
REPLAY_SHOTS_BELL = 100_000
SHOT_BUDGET = 2_000_000
SHOT_ROUNDS_BUDGET = 28_000_000
MAX_ROUNDS = 32
MAX_SHOTS_PER_CALL = 100_000

Z_BATTERY = ("00", "01", "10", "11")
X_BATTERY = ("00", "01", "10", "11")  # |±±> labels, 0 = |+>
CNOT_BATTERY = tuple(("z", entry) for entry in Z_BATTERY) + tuple(
    ("x", entry) for entry in X_BATTERY
)


def _seam_windows() -> dict[int, str]:
    """Seam ancilla qubit id -> merge window ('zz' | 'xx'), from the layout geometry."""
    old_zz = {(p.pos, p.basis) for part in (L.PATCH_C, L.PATCH_INT) for p in part.plaquettes}
    zz = {
        L.qubit_id(p.pos)
        for p in L.MERGED_ZZ.plaquettes
        if p.basis == "Z" and (p.pos, p.basis) not in old_zz
    }
    routing_sites = {q for q in L.routing_qubit_ids() if L.CHIP_COORDS[q][0] % 2 == 0}
    return {q: ("zz" if q in zz else "xx") for q in sorted(routing_sites)}


def _flavor_assignment_acceptable(sites: list[int], flavors) -> bool:
    """The realized attribution assignment must defeat every fixed footprint->parity
    rule: each merge window and each footprint class (d01 / d12) must contain BOTH
    parity classes, so any zero-joint-data attribution rule is wrong on >= 2 sites."""
    window_of = _seam_windows()
    parity_of = {0: "P", 1: "P", 2: "N", 3: "N"}  # ROUTING_LEAK_FLAVORS order
    footprint_of = {0: "d01", 1: "d12", 2: "d01", 3: "d12"}
    windows: dict[str, set[str]] = {"zz": set(), "xx": set()}
    classes: dict[str, set[str]] = {"d01": set(), "d12": set()}
    for q, f in zip(sites, flavors, strict=True):
        windows[window_of[q]].add(parity_of[int(f)])
        classes[footprint_of[int(f)]].add(parity_of[int(f)])
    return all(v == {"P", "N"} for v in (*windows.values(), *classes.values()))


def _all_cz_pairs() -> list[tuple[int, int]]:
    """Every two-qubit interaction pair used by ANY layout (union over bundles)."""
    pairs: set[tuple[int, int]] = set()
    for patch in list(L.MEMORY_PATCHES.values()) + [L.MERGED_ZZ, L.MERGED_XX]:
        for p in patch.plaquettes:
            a = L.qubit_id(p.pos)
            for d in p.data:
                q = L.qubit_id(d)
                pairs.add((min(a, q), max(a, q)))
    return sorted(pairs)


def build_hidden_config() -> HiddenLatticeSurgeryConfig:
    rng = np.random.default_rng(SEED)
    n = L.N_CHIP_QUBITS
    routing = L.routing_qubit_ids()

    # draw order is FROZEN: qubit params, cz params, routing degradation, leakage topology
    t1 = rng.uniform(*T1_RANGE_US, size=n)
    r2 = rng.uniform(*T2_RATIO_RANGE, size=n)
    p_m01 = rng.uniform(*P_M01_RANGE, size=n)
    p_m10 = rng.uniform(*P_M10_RANGE, size=n)
    p_reset = rng.uniform(*P_RESET_RANGE, size=n)

    pairs = _all_cz_pairs()
    p_cz = rng.uniform(*CZ_RANGE, size=len(pairs))

    t1_factor = rng.uniform(*ROUTING_T1_FACTOR_RANGE, size=n)
    cz_factor = rng.uniform(*ROUTING_CZ_FACTOR_RANGE, size=len(pairs))
    for q in routing:
        t1[q] *= t1_factor[q]
    for k, (a, b) in enumerate(pairs):
        if a in routing or b in routing:
            p_cz[k] *= cz_factor[k]

    # leakage topology (stuck-ancilla model): bulk sites drawn from the MEASURE qubits of
    # the memory layouts, plus EVERY merged-only seam ancilla at boosted rates (the
    # merge-window leakage mechanism no single-patch experiment can see).
    memory_ancillas = sorted(
        {L.qubit_id(p.pos) for patch in L.MEMORY_PATCHES.values() for p in patch.plaquettes}
    )
    bulk_sites = sorted(rng.choice(memory_ancillas, size=N_BULK_LEAK_SITES, replace=False))
    bulk_enter = rng.uniform(*LEAK_P_ENTER_RANGE, size=N_BULK_LEAK_SITES)
    bulk_life = rng.choice(LEAK_LIFETIMES, size=N_BULK_LEAK_SITES)
    routing_sites = sorted(q for q in routing if L.CHIP_COORDS[q][0] % 2 == 0)
    routing_enter = rng.uniform(*ROUTING_LEAK_P_ENTER_RANGE, size=len(routing_sites))
    # appended AFTER earlier draws so the bulk instance remains unchanged;
    # deterministic rejection loop — part of the frozen draw order — so the realized
    # assignment structurally defeats every fixed footprint->parity attribution rule.
    while True:
        routing_flavor = rng.integers(0, len(ROUTING_LEAK_FLAVORS), size=len(routing_sites))
        if _flavor_assignment_acceptable(routing_sites, routing_flavor):
            break

    # Seam ancillas get SYMMETRIC readout (deterministic mean of the drawn asymmetric
    # pair, no extra draws): an N-flavor leak inverts the actual record values for the
    # window tail, so a readout asymmetry would shift downstream classification rates
    # between the otherwise mean-identical P/N flavors (a second-order side channel).
    seam = set(routing_sites)
    qubits = {
        q: {
            "t1_us": round(float(t1[q]), 4),
            "t2_us": round(float(r2[q] * 2.0 * t1[q]), 4),
            "p_reset": round(float(p_reset[q]), 6),
            "p_m01": round(float(0.5 * (p_m01[q] + p_m10[q]) if q in seam else p_m01[q]), 6),
            "p_m10": round(float(0.5 * (p_m01[q] + p_m10[q]) if q in seam else p_m10[q]), 6),
        }
        for q in range(n)
    }
    cz_errors = [
        {"q1": a, "q2": b, "p": round(float(min(p_cz[k], 0.02)), 6)}
        for k, (a, b) in enumerate(pairs)
    ]
    leakage = [
        {
            "qubit": int(q),
            "p_enter": round(float(p), 6),
            "lifetime": int(life),
            "p_flip_round": LEAK_P_FLIP_ROUND,
        }
        for q, p, life in zip(bulk_sites, bulk_enter, bulk_life, strict=True)
    ] + [
        {
            "qubit": int(q),
            "p_enter": round(float(p), 6),
            "lifetime": int(ROUTING_LEAK_FLAVORS[f][1]),
            "p_flip_round": ROUTING_LEAK_P_FLIP_ROUND,
            "mode": "activation",
            "delay": int(ROUTING_LEAK_FLAVORS[f][0]),
        }
        for q, p, f in zip(routing_sites, routing_enter, routing_flavor, strict=True)
    ]

    return HiddenLatticeSurgeryConfig.model_validate(
        {
            "schema_version": 1,
            "device_id": DEVICE_ID,
            "qtype": "surface_code_lattice_surgery",
            "seed": SEED,
            "cycle_time_us": CYCLE_TIME_US,
            "qubits": qubits,
            "cz_errors": cz_errors,
            "leakage": leakage,
            "stale_lab_notebook": stale_notebook_payload(),
        }
    )


def bulk_transfer_hidden_config() -> HiddenLatticeSurgeryConfig:
    """The hidden config an agent would infer by PERFECT bulk calibration + trusting the
    notebook's routing claim: routing-region qubits/gates carry bulk-mean parameters and
    no leakage anywhere (leakage is not schedule-derivable). Anchor rung (ii)/(iii) input."""
    true = build_hidden_config()
    routing = L.routing_qubit_ids()
    bulk = [q for q in range(L.N_CHIP_QUBITS) if q not in routing]
    mean_t1 = float(np.mean([true.qubits[q].t1_us for q in bulk]))
    mean_t2 = float(np.mean([true.qubits[q].t2_us for q in bulk]))
    mean_reset = float(np.mean([true.qubits[q].p_reset for q in bulk]))
    mean_m01 = float(np.mean([true.qubits[q].p_m01 for q in bulk]))
    mean_m10 = float(np.mean([true.qubits[q].p_m10 for q in bulk]))
    bulk_cz = [e.p for e in true.cz_errors if e.q1 not in routing and e.q2 not in routing]
    mean_cz = float(np.mean(bulk_cz))

    payload = true.model_dump()
    for q in routing:
        payload["qubits"][q] = {
            "t1_us": mean_t1,
            "t2_us": mean_t2,
            "p_reset": mean_reset,
            "p_m01": mean_m01,
            "p_m10": mean_m10,
        }
    for e in payload["cz_errors"]:
        if e["q1"] in routing or e["q2"] in routing:
            e["p"] = mean_cz
    payload["leakage"] = []
    return HiddenLatticeSurgeryConfig.model_validate(payload)


# --------------------------------------------------------------------------- #
# Public payloads.
# --------------------------------------------------------------------------- #


def stale_notebook_payload() -> dict:
    return {
        "noise_model": (
            "the chip is uniform: all two-qubit gates ~0.35%, all T1 ~70 us; independent "
            "depolarizing noise is a fine model"
        ),
        "routing_region": (
            "routing-region qubits were matched to the bulk in the last calibration "
            "campaign; use bulk numbers for them"
        ),
        "surgery_advice": (
            "the merge windows introduce no new error mechanisms beyond the bulk — reuse "
            "your memory DEM on the CNOT spacetime graph; at most rescale to the seam "
            "detector rates"
        ),
        "decoder_advice": (
            "decode the two patches independently; the joint parities are reliable as raw outcomes"
        ),
        "correlation_advice": "treat detection events as independent",
        "budget_advice": "memory characterization is all you need; the CNOT will follow",
        "note": (
            "Uniform chip, trustworthy routing region, no new physics in the merges, "
            "independent detections, independent patches."
        ),
    }


def _layout_specs() -> list[dict]:
    specs = []
    for name, patch in list(L.MEMORY_PATCHES.items()) + [
        ("merged_zz", L.MERGED_ZZ),
        ("merged_xx", L.MERGED_XX),
    ]:
        specs.append(
            {
                "name": name,
                "data_qubits": sorted(L.qubit_id(c) for c in patch.data),
                "ancillas": [
                    {
                        "qubit": L.qubit_id(p.pos),
                        "pos": [p.pos[0], p.pos[1]],
                        "basis": p.basis,
                        "support": sorted(L.qubit_id(c) for c in p.data),
                    }
                    for p in sorted(patch.plaquettes, key=lambda p: (p.pos[1], p.pos[0]))
                ],
            }
        )
    return specs


def build_public_spec() -> dict:
    return {
        "schema_version": 1,
        "device_id": DEVICE_ID,
        "qtype": "surface_code_lattice_surgery",
        "task_id": TASK_ID,
        "code": "rotated_surface_code",
        "n_chip_qubits": L.N_CHIP_QUBITS,
        "chip_coords": [[c[0], c[1]] for c in L.CHIP_COORDS],
        "layouts": _layout_specs(),
        "routing_zz_data": sorted(L.qubit_id(c) for c in L.ROUTING_ZZ),
        "routing_xx_data": sorted(L.qubit_id(c) for c in L.ROUTING_XX),
        "cycle_time_us": CYCLE_TIME_US,
        "schedule_note": (
            "4 two-qubit interaction sub-steps per round; X-plaquettes traverse their "
            "data (dx,dy) in order (1,-1),(-1,-1),(1,1),(-1,1) and Z-plaquettes in order "
            "(1,-1),(1,1),(-1,-1),(-1,1). The sub-step order fixes the emergent hook "
            "orientations; the rates are the device's to be measured."
        ),
        "memory_scoring_rounds": MEMORY_SCORING_ROUNDS,
        "cnot_rounds": CC.CNOT_ROUNDS,
        "merge_rounds": CC.MERGE_ROUNDS,
        "transition_pre_rounds": CC.TRANSITION_PRE_ROUNDS,
        "transition_post_rounds": CC.TRANSITION_POST_ROUNDS,
        "decoder_types": ["mwpm", "belief_matching"],
        "nominal_cz_error": 0.0035,
        "nominal_t1_us": 70.0,
        "budgets": {
            "shot_budget": SHOT_BUDGET,
            "shot_rounds_budget": SHOT_ROUNDS_BUDGET,
            "max_rounds": MAX_ROUNDS,
            "max_shots_per_call": MAX_SHOTS_PER_CALL,
        },
        "floor_mem": FLOOR_MEM,
        "floor_cnot": FLOOR_CNOT,
        "cnot_per_entry_cap": CNOT_PER_ENTRY_CAP,
        "replay_shots_memory": REPLAY_SHOTS_MEMORY,
        "replay_shots_per_entry": REPLAY_SHOTS_PER_ENTRY,
        "frame_convention": MD.frame_convention_summary(),
        "observable_names": [
            "obs_logical",
            "obs_z_c_out",
            "obs_z_t_out",
            "obs_x_c_out",
            "obs_x_t_out",
            "obs_bell_zz",
            "obs_bell_xx",
            "m_zz",
            "m_xx",
            "m_z_int",
        ],
        "notes": (
            "Calibrate decoders for this processor: d=5 memory and the lattice-surgery "
            "CNOT. Submit DEMs + decoder configs; both are scored by replay on fresh "
            "shots. Machine-readable per-layout detector indexing and matching-graph "
            "adjacency files ship with the task materials."
        ),
    }


def sampler_noise():
    from qiqcbench.qsim.qtypes.surface_code_lattice_surgery.noise_model import (
        sampler_noise_from_hidden,
    )

    return sampler_noise_from_hidden(build_hidden_config())
