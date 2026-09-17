"""Instance construction for time_budgeted_shadow_surrogate_60q.

Builds the hidden 60-qubit bounded-gate circuit instance, the stationary
per-qubit measurement noise, the sealed target inputs, and the closed-form
hidden truth: one damped trigonometric polynomial per qubit pair,

    C_ij(x) = (1 - 2 e_i)(1 - 2 e_j) * ( <X_i X_j> + <Y_i Y_j> + <Z_i Z_j> ) / 3 .

Family (windowed parity network).  Twelve latent sites carry all eighteen
rotations (six angles, multiplicity three; one or two rotations per site about
x/y axes, four two-rotation sites repeating one axis so that their
Z-expectation is a two-term trigonometric function).  Every other qubit is a
rail whose computational bit is the parity of one or two latent bits
referenced inside a window of eight sites (exact CX routing through swap
ladders), plus three long-range references.  Then

    <Z_i Z_j> = prod_{s in S(i) xor S(j)} f_s(x),

which is nonconstant whenever the reference sets differ, never an exact zero,
differs from the product of marginals whenever the sets overlap, and lets
every angle reach most of the device.  The observable's X/Y sectors are empty
for this classical-parity structure, so the signal lives in the ZZ sector.

Everything is deterministic in the instance seed.  Target-blind structural
admission runs before target materialization and rejects roots that collapse
into constant, zero-heavy, low-rank, or over-sparse response families.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q.active_fixture import (
    INSTANCE_SEED,
)
from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.circuit import CircuitOp
from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.pauli_poly import (
    TrigPoly,
    TrigTerm,
    backpropagate_pauli,
)
from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.sim import (
    BondDimensionExceeded,
    build_mps,
)

TASK_ID = "time_budgeted_shadow_surrogate_60q"
DEVICE_ID = "boundedgate_60q_v0"
SCIENTIFIC_EDITION = "windowed_parity_network_v9"


class AdmissionRejectedError(RuntimeError):
    """A fully evaluated instance failed the frozen scientific admission policy."""


NormalFormKey = tuple[tuple[int, ...], tuple[int, ...]]


N_QUBITS = 60
INPUT_DIM = 6
ROTATION_GATES = 18
DEGREE_BOUND = 3
ANGLE_MULTIPLICITIES = (3, 3, 3, 3, 3, 3)  # sums to ROTATION_GATES

TOTAL_SHOT_BUDGET = 315_000
MAX_JOBS = 40
MAX_BLOCKS_PER_JOB = 64
MAX_SETTINGS_PER_JOB = 192
MAX_SHOTS_PER_SETTING = 4_000
# Exact-MPS runtime guard. The parity network's bond dimension is 32 on ~96 %
# of uniform inputs with a structural tail (plateau at 64, rare 72); a job that
# exceeds the cap fails after its shots were consumed, so the cap keeps a >3x
# margin over the observed tail and admission measures the tail directly.
MAX_BOND = 256
MPS_ADMISSION_PROBES = 64

NOISE_RANGE = (0.01, 0.05)
N_TARGETS = 40
TARGET_CHALLENGE_SCHEMA_VERSION = 1
NEAR_BOUNDARY_COORDINATE_COUNT = 2
NEAR_BOUNDARY_DELTA_RANGE = (0.01, 0.08)
NEAR_BOUNDARY_SAMPLING_RULE = (
    "start with six independent Uniform[-pi, pi] coordinates; choose two "
    "coordinates uniformly without replacement; independently for each selected "
    "coordinate k, draw s_k Uniform{-1, +1} and delta_k Uniform[0.01, 0.08], "
    "then set x_k = s_k * (pi - delta_k)"
)

# The challenge distribution is public even though its realized coordinates
# stay sealed until the measurement lock.  Counts match ``build_target_inputs``
# exactly; keeping this object construction-owned prevents the instruction,
# runtime receipt, and materializer from drifting apart.
TARGET_STRATA = (
    {
        "name": "uniform_torus",
        "count": 16,
        "sampling_rule": "six independent Uniform[-pi, pi] coordinates",
    },
    {
        "name": "near_boundary",
        "count": 8,
        "sampling_rule": NEAR_BOUNDARY_SAMPLING_RULE,
    },
    {
        "name": "diagonal_off_axis",
        "count": 8,
        "sampling_rule": (
            "draw magnitude Uniform[0.2, pi] and six independent uniform signs; "
            "return magnitude times the sign vector"
        ),
    },
    {
        "name": "sign_related_pairs",
        "count": 8,
        "sampling_rule": (
            "draw four bases with six independent Uniform[-pi, pi] coordinates; "
            "for each base return the base and its coordinatewise product with a "
            "uniform sign vector conditioned to contain at least one -1"
        ),
    },
)

# ---------- family parameters (windowed parity network) ----------
N_LATENT_SITES = 12
LATENT_MIN_SPACING = 2
MAX_ROTATIONS_PER_SITE = 2
SAME_AXIS_SITES = 4  # two-rotation sites whose two rotations share an axis
RAIL_WINDOW = 8
RAIL_REFERENCE_COUNTS = (1, 2, 2)
LONG_RANGE_LINKS = 3
LONG_RANGE_MAX_DISTANCE = 16

# The device is stationary in this edition: the readout flip probabilities are
# fixed for the whole run and the drift vector is identically zero.  The
# runtime keeps its drift code path (a later edition may switch it on), but
# admission requires zero drift here so no root ships an undisclosed nuisance.

# Machine-readable retirement authority.  Historical reconstruction may still
# call ``build_instance(seed)``; certification and emission must reject these
# roots before consuming measurement streams or writing artifacts.  The
# version-8 fixture root (four unentangled blocks, 26-feature dictionary,
# usage drift) is retired with the edition change.
RETIRED_DEVELOPMENT_INSTANCE_SEEDS = frozenset(
    {
        20260808,
        21571553166570421846271576071350939316512800085440374431521323458454583766309,
        # Retired fixture root: certified on a review run, then retired when
        # the public-class disclosure changed.
        103322207954960586000033493131670577167976149807677301449770386280643513044134,
        # Retired fixture root: certified on the physics-only public class,
        # retired when the lab notebook was neutralized.
        8876616530219385075218486464065381141127256010723743146254866215078514722951,
        # Retired fixture root: certified under certificate schema 4 / z-only
        # reference policy, then retired after the schema-5 pilot-axis
        # certification policy landed.
        10861105579994775327925206138977799888524792774571911403400440011292210173820,
    }
)

# ---------- target-blind admission policy ----------
ADMISSION_PROBE_INPUTS = 300
MIN_NONCONSTANT_PAIR_FRACTION = 0.95
MAX_EXACT_ZERO_PAIRS = 0
MIN_OUTPUT_SPAN_RANK = 150
MIN_ANGLE_PAIR_FRACTION = 0.35
MIN_MULTI_TERM_FRACTION = 0.5
MAX_ADMITTED_MEDIAN_BOND = 32  # typical (median over probes) bond: mesoscopic, not scrambled
MAX_ADMITTED_BOND = 128  # maximum over probes: half the runtime cap
MAX_RAW_TERMS_PER_PAIR = 64
# Hidden, root-independent construction rule: every scored pair is a linear
# combination of at most this many quotient-ring normal-form monomials
# ``prod_k cos(x_k)**a_k * sin(x_k)**b_k`` with ``a_k + b_k <= DEGREE_BOUND``
# and ``b_k in {0, 1}``.  This cap and representation are not public; the
# public, method-neutral Fourier cap below is the response-class contract.
ADMISSION_MAX_NORMAL_FORM_SUPPORT = 16
# Public, method-neutral response-class bound.  Coefficients are expressed in
# the standard complex Fourier basis exp(i m dot x), with one mode per distinct
# integer frequency vector m.  Conjugate vectors count separately.  The bound
# is root-independent and is enforced by admission for every shipped instance;
# no pair-specific support or coefficient is disclosed.
MAX_FOURIER_MODES_PER_RESPONSE = 64
NORMAL_FORM_POWERS_PER_ANGLE = tuple(
    (cos_power, sin_power)
    for cos_power in range(DEGREE_BOUND + 1)
    for sin_power in (0, 1)
    if cos_power + sin_power <= DEGREE_BOUND
)
PUBLIC_DICTIONARY_SIZE = len(NORMAL_FORM_POWERS_PER_ANGLE) ** INPUT_DIM
_NORMAL_FORM_ZERO_ATOL = 1e-10

# ---------- certification policy ----------
# Historical rosters stay machine-readable so no successor can reuse a stream
# that an earlier edition burned or froze.
ROUND7_BURNED_CERT_DESIGN_SEEDS = (12345, 22345, 32345, 42345, 52345)
ROUND7_UNCONSUMED_CERT_DESIGN_SEEDS = (62345, 72345)
V8_CERT_DESIGN_SEEDS = (14512273635978057133, 6156200627333752294, 16244851261566726927)
V8_CERT_ABLATION_DESIGN_SEED = 17648807810592046037
# Version-9 study 1 (shadow-surrogate-v9-a032dad865ea0c64672ae199, retired):
# slots 0 and 1 spent these streams before slot 2 failed on an OpenBLAS gesdd
# defect in the MPS engine.  The engine fix is a runtime change, so the panel
# is burned and study 2 draws a fresh one.
V9_STUDY1_BURNED_CERT_DESIGN_SEEDS = (
    4291424809592256914,
    6174911636838028758,
    14792142729430639323,
)
V9_STUDY1_BURNED_CERT_ABLATION_DESIGN_SEED = 13653007854223992727
# Version-9 study 2 (shadow-surrogate-v9-29cbce38657bf04bb2e0d56a, passed
# 3/3): its panel is consumed; the public-class disclosure changed afterwards
# (dictionary and support cap withdrawn), so study 3 draws panel 3.
V9_STUDY2_BURNED_CERT_DESIGN_SEEDS = (
    7187451298263061798,
    11712592192304217017,
    11408307049807792884,
)
V9_STUDY2_BURNED_CERT_ABLATION_DESIGN_SEED = 16211375326860284183
# Version-9 study 3 (shadow-surrogate-v9-aaf10c3c036ec7fe9f0121ff, passed
# 3/3): its panel is consumed; the lab notebook changed afterwards (hidden
# device config is study-bound), so study 4 draws panel 4.
V9_STUDY3_BURNED_CERT_DESIGN_SEEDS = (
    6377181486216928644,
    13238058527069378107,
    15151186964199920102,
)
V9_STUDY3_BURNED_CERT_ABLATION_DESIGN_SEED = 5381396907324197307
# Version-9 study 4 (shadow-surrogate-v9-2e865ea5561e84a2d0818d46, passed
# 3/3): its schema-4 / z-only panel is consumed; the current schema-5
# pilot-axis certification policy requires study 5.
V9_STUDY4_BURNED_CERT_DESIGN_SEEDS = (
    17601488130784265091,
    10469556057634076948,
    18291487717505621119,
)
V9_STUDY4_BURNED_CERT_ABLATION_DESIGN_SEED = 10924036213613070538
# Certificate roster: three fresh-stream reference runs plus one
# predeclared weak-baseline stream, derived as the leading 64 bits of
# SHA-256("qiqcbench/time_budgeted_shadow_surrogate_60q/edition-v9/certificate-panel-5/"
# + "global-0" | "global-1" | "global-2" | "ablation") so they are reproducible
# and disjoint from every earlier roster.
CERT_DESIGN_SEEDS = (
    17410548859490344397,
    507672031164323841,
    3565465318225523703,
)
CERT_ABLATION_DESIGN_SEED = 10595735473498428275
_HISTORICAL_CERT_STREAMS = frozenset(
    {
        *ROUND7_BURNED_CERT_DESIGN_SEEDS,
        *ROUND7_UNCONSUMED_CERT_DESIGN_SEEDS,
        *V8_CERT_DESIGN_SEEDS,
        V8_CERT_ABLATION_DESIGN_SEED,
        *V9_STUDY1_BURNED_CERT_DESIGN_SEEDS,
        V9_STUDY1_BURNED_CERT_ABLATION_DESIGN_SEED,
        *V9_STUDY2_BURNED_CERT_DESIGN_SEEDS,
        V9_STUDY2_BURNED_CERT_ABLATION_DESIGN_SEED,
        *V9_STUDY3_BURNED_CERT_DESIGN_SEEDS,
        V9_STUDY3_BURNED_CERT_ABLATION_DESIGN_SEED,
        *V9_STUDY4_BURNED_CERT_DESIGN_SEEDS,
        V9_STUDY4_BURNED_CERT_ABLATION_DESIGN_SEED,
    }
)
assert CERT_ABLATION_DESIGN_SEED not in _HISTORICAL_CERT_STREAMS | set(CERT_DESIGN_SEEDS)
assert not set(CERT_DESIGN_SEEDS) & _HISTORICAL_CERT_STREAMS
CERTIFICATE_SCHEMA_VERSION = 5
REFERENCE_POLICY_ID = "orthogonal_fourier_omp_pilot_axis_v2"
# Predeclared weak baselines and the gate each must fail on its own stream.
# These are feasibility checks (the gate separates the reference from a
# method that omits one ingredient), not causal ablations.
CERT_ABLATION_EXPECTED_GATE = {
    "even_xyz": "g1_accuracy",
    "fourier_single": "g1_accuracy",
    "additive": "g1_accuracy",
    "nearest_input": "g1_accuracy",
    "constant_mean": "g1_accuracy",
}
# Exact, target-independent structural evidence recorded in the certificate.
CERT_STRUCTURAL_DIAGNOSTIC_POLICY = {
    "output_span": {
        "kind": "exact_output_span_rank_v2",
        "target_independent": True,
        "minimums": {"nonconstant_output_span_rank": MIN_OUTPUT_SPAN_RANK},
    }
}
# Per-statistic certificate margins (owner decision, 2026-08-27).  The global
# RMSE keeps the 1.3x headroom; the two extreme statistics (worst target over
# 40 targets, maximum over 70,800 cells) are order statistics whose stream-to-
# stream spread is far wider than the mean's, so a uniform 1.3x would burn
# admissible roots for noise.  Measured over nine reference runs on two roots:
# worst target 0.0020-0.0025, maximum cell 0.020-0.033.
CERT_MIN_ACCURACY_MARGIN = 1.3
CERT_MIN_WORST_TARGET_MARGIN = 1.15
CERT_MIN_TAIL_MARGIN = 1.05
# Exact resource accounting for certificate rows.  The reference spends one
# pilot job (eight inputs, three global settings, 500 shots each: 12,000
# shots) and then 1,000 inputs in the pilot-selected global axis at 303 shots
# each in 16 jobs of at most 64 blocks: 315,000 shots in 17 jobs.  The
# even-allocation baseline spends 1,000 inputs x 3 settings x 105 shots in 16
# jobs.
CERT_GLOBAL_BUDGET = (315_000, 17)
CERT_EVEN_XYZ_BUDGET = (315_000, 16)


def certificate_expected_budget(protocol: str) -> tuple[int, int]:
    """Return exact ``(shots_used, jobs_used)`` required for certification."""
    if protocol == "even_xyz":
        return CERT_EVEN_XYZ_BUDGET
    if protocol == "global" or protocol in CERT_ABLATION_EXPECTED_GATE:
        return CERT_GLOBAL_BUDGET
    raise ValueError(f"protocol {protocol!r} is not covered by certificate policy")


# Certificate authentication: certify_edition HMAC-signs the certificate body
# with this operator-held key; emit_edition refuses any certificate whose
# signature does not verify under the key in force. Content re-validation
# rejects hand-written stat-free certificates; the signature additionally
# rejects a fully fabricated rich certificate (plausible fake stats) minted
# outside a real certify_edition run. The committed dev edition uses the
# well-known dev key ("00" * 32 — no secrecy claim, the dev edition is public
# anyway); scored private editions must be certified and emitted under a
# fresh private key.
CERT_SIGNING_KEY_ENV = "QIQCBENCH_CERT_SIGNING_KEY"
_HEX_256BIT = re.compile(r"[0-9a-f]{64}\Z")


def load_cert_signing_key() -> str:
    """Read + validate the operator certificate-signing key (fail closed)."""
    value = os.environ.get(CERT_SIGNING_KEY_ENV)
    if value is None:
        raise ValueError(
            f"{CERT_SIGNING_KEY_ENV} is not set: certification and emission both "
            "require the operator certificate-signing key (32 bytes, lowercase hex)"
        )
    if _HEX_256BIT.fullmatch(value) is None:
        raise ValueError(f"{CERT_SIGNING_KEY_ENV} must be 32 bytes encoded as 64 lowercase hex")
    return value


def certificate_signature(certificate: dict[str, Any], key_hex: str) -> str:
    """HMAC-SHA256 over the canonical certificate body (sans signature)."""
    body = {k: v for k, v in sorted(certificate.items()) if k != "signature"}
    return hmac.new(bytes.fromhex(key_hex), _json_bytes(body), hashlib.sha256).hexdigest()


def pair_order(n: int = N_QUBITS) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


# ---------- circuit construction ----------
#
# Physics that fixes the family shape (review 2026-08-27, demonstrated on
# prototypes): a two-point Pauli expectation of a Clifford-plus-rotations
# circuit on |0...0> is nonzero only when the back-propagated string lands on
# Z-type strings, so generic scrambling zeroes almost every pair; the
# isotropic observable sees a single Pauli sector of any classical-copy
# structure and per-qubit output frames turn cross-sector pairs into exact
# zeros; plain copies make same-source pairs constant.  Parities of one or
# two rotated latent bits inside overlapping windows avoid all three.


def _append_swap(ops: list[CircuitOp], a: int, b: int) -> None:
    ops.append(CircuitOp(gate="cx", q=a, q2=b))
    ops.append(CircuitOp(gate="cx", q=b, q2=a))
    ops.append(CircuitOp(gate="cx", q=a, q2=b))


def append_remote_cx(ops: list[CircuitOp], src: int, dst: int) -> None:
    """Exact CX src->dst on the nearest-neighbour chain for ANY intermediate
    state: the four-CX ladder over one intermediate site, otherwise swap
    routing (6d - 5 CX gates for distance d)."""
    if src == dst:
        raise ValueError("remote CX needs distinct sites")
    distance = abs(dst - src)
    if distance == 1:
        ops.append(CircuitOp(gate="cx", q=src, q2=dst))
        return
    step = 1 if dst > src else -1
    if distance == 2:
        middle = src + step
        for _ in range(2):
            ops.append(CircuitOp(gate="cx", q=src, q2=middle))
            ops.append(CircuitOp(gate="cx", q=middle, q2=dst))
        return
    position = src
    path: list[int] = []
    while abs(dst - position) > 1:
        _append_swap(ops, position, position + step)
        path.append(position)
        position += step
    ops.append(CircuitOp(gate="cx", q=position, q2=dst))
    for site in reversed(path):
        _append_swap(ops, site, site + step)


@dataclass(frozen=True)
class WindowedParityLayout:
    """Private layout record (never emitted into agent-visible material)."""

    latent_sites: tuple[int, ...]
    rotations: dict[int, tuple[tuple[str, int], ...]]
    rail_references: dict[int, tuple[int, ...]]


def build_windowed_parity_layout(rng: np.random.Generator) -> WindowedParityLayout:
    while True:
        latent = sorted(rng.choice(N_QUBITS, size=N_LATENT_SITES, replace=False).tolist())
        if all(b - a >= LATENT_MIN_SPACING for a, b in zip(latent[:-1], latent[1:], strict=True)):
            break
    rails = [q for q in range(N_QUBITS) if q not in latent]
    pool = [k for k in range(INPUT_DIM) for _ in range(ANGLE_MULTIPLICITIES[k])]
    rng.shuffle(pool)
    per_site: dict[int, list[int]] = {s: [] for s in latent}
    for site, angle in zip(latent, pool[:N_LATENT_SITES], strict=True):
        per_site[site].append(angle)
    for angle in pool[N_LATENT_SITES:]:
        for site in rng.permutation(latent):
            site = int(site)
            if len(per_site[site]) < MAX_ROTATIONS_PER_SITE and angle not in per_site[site]:
                per_site[site].append(angle)
                break
        else:  # pragma: no cover - multiplicity/site counts make this impossible
            raise RuntimeError("could not place every rotation on a latent site")
    two_rotation_sites = [s for s in latent if len(per_site[s]) == 2]
    same_axis = {
        int(s)
        for s in rng.choice(
            two_rotation_sites, size=min(SAME_AXIS_SITES, len(two_rotation_sites)), replace=False
        )
    }
    rotations: dict[int, tuple[tuple[str, int], ...]] = {}
    for site in latent:
        angles = per_site[site]
        first = str(rng.choice(["x", "y"]))
        if len(angles) == 1:
            axes = (first,)
        else:
            axes = (first, first if site in same_axis else ("x" if first == "y" else "y"))
        rotations[site] = tuple(
            (axis, int(angle)) for axis, angle in zip(axes, angles, strict=True)
        )
    references: dict[int, tuple[int, ...]] = {}
    used: set[tuple[int, ...]] = set()
    for rail in rails:
        near = [s for s in latent if abs(s - rail) <= RAIL_WINDOW]
        if not near:
            near = [min(latent, key=lambda s: abs(s - rail))]
        count = min(int(rng.choice(RAIL_REFERENCE_COUNTS)), len(near))
        for _ in range(40):
            ref = tuple(sorted(int(v) for v in rng.choice(near, size=count, replace=False)))
            if ref not in used:
                break
        used.add(ref)
        references[rail] = ref
    for _ in range(LONG_RANGE_LINKS):
        rail = int(rng.choice(rails))
        far = [
            s
            for s in latent
            if RAIL_WINDOW < abs(s - rail) <= LONG_RANGE_MAX_DISTANCE and s not in references[rail]
        ]
        if far:
            far_site = int(rng.choice(far))
            base = (
                list(references[rail])[:-1] if len(references[rail]) > 1 else list(references[rail])
            )
            references[rail] = tuple(sorted(base + [far_site]))
    return WindowedParityLayout(
        latent_sites=tuple(latent),
        rotations=rotations,
        rail_references=references,
    )


def layout_to_ops(layout: WindowedParityLayout) -> list[CircuitOp]:
    ops: list[CircuitOp] = []
    for site in layout.latent_sites:
        for axis, angle in layout.rotations[site]:
            ops.append(CircuitOp(gate="rot", q=site, axis=axis, angle=angle))
    for rail in sorted(layout.rail_references):
        for site in layout.rail_references[rail]:
            append_remote_cx(ops, site, rail)
    return ops


def build_circuit(rng: np.random.Generator) -> list[CircuitOp]:
    ops = layout_to_ops(build_windowed_parity_layout(rng))
    assert sum(op.gate == "rot" for op in ops) == ROTATION_GATES
    return ops


def build_noise(rng: np.random.Generator) -> list[float]:
    return [float(v) for v in rng.uniform(*NOISE_RANGE, size=N_QUBITS)]


def build_drift(rng: np.random.Generator, flip_prob: list[float]) -> list[float]:
    """Stationary edition: the drift vector is identically zero.

    The RNG argument is accepted (and left untouched) so the domain stream
    layout stays identical to earlier editions.
    """
    del rng
    return [0.0] * len(flip_prob)


def _draw_near_boundary_target(rng: np.random.Generator) -> np.ndarray:
    """Draw one boundary row with independent sign/delta per selected coordinate."""
    target = rng.uniform(-np.pi, np.pi, size=INPUT_DIM)
    indices = rng.choice(INPUT_DIM, size=NEAR_BOUNDARY_COORDINATE_COUNT, replace=False)
    # Preserve the sign-then-delta draw order for each coordinate. Besides
    # making the product law explicit, this keeps existing target bytes stable.
    for index in indices:
        sign = float(rng.choice([-1.0, 1.0]))
        delta = rng.uniform(*NEAR_BOUNDARY_DELTA_RANGE)
        target[index] = sign * (np.pi - delta)
    return target


def build_target_inputs(rng: np.random.Generator) -> np.ndarray:
    """40 hidden-test inputs: uniform, near-boundary, diagonal, sign-related pairs."""
    targets: list[np.ndarray] = []
    for _ in range(16):
        targets.append(rng.uniform(-np.pi, np.pi, size=INPUT_DIM))
    for _ in range(8):
        targets.append(_draw_near_boundary_target(rng))
    for _ in range(8):
        magnitude = rng.uniform(0.2, np.pi)
        signs = rng.choice([-1.0, 1.0], size=INPUT_DIM)
        targets.append(magnitude * signs)
    for _ in range(4):
        x = rng.uniform(-np.pi, np.pi, size=INPUT_DIM)
        # The flip must actually flip something: an all-(+1) draw would emit
        # the same row twice (this shipped 39 distinct targets once).
        while True:
            flip = rng.choice([-1.0, 1.0], size=INPUT_DIM)
            if np.any(flip < 0):
                break
        targets.append(x)
        targets.append(x * flip)
    out = np.array(targets)
    assert out.shape == (N_TARGETS, INPUT_DIM)
    assert len({tuple(row) for row in out.tolist()}) == N_TARGETS, "duplicate target rows"
    return out


# ---------- closed-form truth ----------


def _combine_polys(polys: list[TrigPoly], scales: list[float], d: int) -> TrigPoly:
    """Exact linear combination of raw trigonometric polynomials."""
    if len(polys) != len(scales):
        raise ValueError("polys/scales length mismatch")
    merged: dict[tuple[tuple[int, ...], tuple[int, ...]], float] = {}
    for poly, scale in zip(polys, scales, strict=True):
        for term in poly.terms:
            key = (term.cos_pow, term.sin_pow)
            merged[key] = merged.get(key, 0.0) + scale * term.coeff
    return TrigPoly(
        d,
        [
            TrigTerm(coeff=c, cos_pow=key[0], sin_pow=key[1])
            for key, c in sorted(merged.items())
            if abs(c) > 1e-14
        ],
    )


def build_pair_axis_polys(
    ops: list[CircuitOp], flip_prob: list[float]
) -> list[tuple[TrigPoly, TrigPoly, TrigPoly]]:
    """Per-axis contributions whose X/Y/Z sum equals each scored C_ij.

    Each component includes both the 1/3 observable coefficient and terminal
    readout damping.  Retaining this private decomposition lets admission test
    whether one measurement axis captures nearly all input-dependent signal.
    """
    out: list[tuple[TrigPoly, TrigPoly, TrigPoly]] = []
    for i, j in pair_order():
        damping = (1.0 - 2.0 * flip_prob[i]) * (1.0 - 2.0 * flip_prob[j]) / 3.0
        components_list = [
            _combine_polys(
                [backpropagate_pauli(ops, INPUT_DIM, {i: axis, j: axis})],
                [damping],
                INPUT_DIM,
            )
            for axis in ("x", "y", "z")
        ]
        out.append((components_list[0], components_list[1], components_list[2]))
    return out


def build_pair_polys(ops: list[CircuitOp], flip_prob: list[float]) -> list[TrigPoly]:
    """One damped C_ij trig polynomial per pair, in lexicographic pair order."""
    return [
        _combine_polys(list(components), [1.0, 1.0, 1.0], INPUT_DIM)
        for components in build_pair_axis_polys(ops, flip_prob)
    ]


def build_single_axis_polys(
    ops: list[CircuitOp], flip_prob: list[float]
) -> list[tuple[TrigPoly, TrigPoly, TrigPoly]]:
    """Measured one-body X/Y/Z expectations, including terminal damping."""
    out: list[tuple[TrigPoly, TrigPoly, TrigPoly]] = []
    for q in range(N_QUBITS):
        damping = 1.0 - 2.0 * flip_prob[q]
        components_list = [
            _combine_polys(
                [backpropagate_pauli(ops, INPUT_DIM, {q: axis})],
                [damping],
                INPUT_DIM,
            )
            for axis in ("x", "y", "z")
        ]
        out.append((components_list[0], components_list[1], components_list[2]))
    return out


def evaluate_pair_polys(polys: list[TrigPoly], xs: np.ndarray) -> np.ndarray:
    """Labels with shape (n_inputs, n_pairs)."""
    xs = np.atleast_2d(xs)
    out = np.empty((xs.shape[0], len(polys)))
    for col, poly in enumerate(polys):
        out[:, col] = poly.evaluate_batch(xs)
    return out


def _power_fourier(cos_power: int, sin_power: int) -> tuple[tuple[int, complex], ...]:
    """Canonical exp(i h x) coefficients for cos(x)^c sin(x)^s, c+s <= 3."""
    coeffs: dict[int, complex] = {0: 1.0 + 0.0j}
    factors = [({-1: 0.5, 1: 0.5})] * cos_power + [{-1: 0.5j, 1: -0.5j}] * sin_power
    for factor in factors:
        nxt: dict[int, complex] = {}
        for h0, c0 in coeffs.items():
            for h1, c1 in factor.items():
                nxt[h0 + h1] = nxt.get(h0 + h1, 0.0j) + c0 * c1
        coeffs = nxt
    return tuple(sorted((h, c) for h, c in coeffs.items() if abs(c) > 1e-14))


def canonical_fourier_coefficients(poly: TrigPoly) -> dict[tuple[int, ...], complex]:
    """Reduce raw sin/cos powers to the unique multivariate Fourier basis."""
    out: dict[tuple[int, ...], complex] = {}
    for term in poly.terms:
        partial: dict[tuple[int, ...], complex] = {(): complex(term.coeff)}
        for cos_power, sin_power in zip(term.cos_pow, term.sin_pow, strict=True):
            factor = _power_fourier(cos_power, sin_power)
            nxt: dict[tuple[int, ...], complex] = {}
            for prefix, c0 in partial.items():
                for harmonic, c1 in factor:
                    key = prefix + (harmonic,)
                    nxt[key] = nxt.get(key, 0.0j) + c0 * c1
            partial = nxt
        for key, value in partial.items():
            out[key] = out.get(key, 0.0j) + value
    return {key: value for key, value in out.items() if abs(value) > 1e-12}


def _canonical_trig_normal_form_unpruned(poly: TrigPoly) -> dict[NormalFormKey, float]:
    """Unique quotient-ring coefficients, retaining exact-zero cancellations.

    The reduction applies ``sin(x)^2 = 1 - cos(x)^2`` independently in every
    coordinate, then merges identical monomials.  Unlike the raw
    back-propagation representation, this cannot count algebraic aliases as
    independent structure.  Keeping zero-valued merged entries here lets the
    public-family admission distinguish exact cancellation from a small,
    numerically ambiguous coefficient instead of choosing support by a silent
    tolerance.
    """
    merged: dict[NormalFormKey, float] = {}
    for term in poly.terms:
        partial: dict[NormalFormKey, float] = {((0,) * poly.d, (0,) * poly.d): float(term.coeff)}
        for angle, (cos_power, sin_power) in enumerate(
            zip(term.cos_pow, term.sin_pow, strict=True)
        ):
            parity = sin_power % 2
            pairs = sin_power // 2
            factor = [
                (cos_power + 2 * extra, parity, float((-1) ** extra * math.comb(pairs, extra)))
                for extra in range(pairs + 1)
            ]
            nxt: dict[NormalFormKey, float] = {}
            for (cos_powers, sin_powers), coefficient in partial.items():
                for reduced_cos, reduced_sin, scale in factor:
                    new_cos = cos_powers[:angle] + (reduced_cos,) + cos_powers[angle + 1 :]
                    new_sin = sin_powers[:angle] + (reduced_sin,) + sin_powers[angle + 1 :]
                    key = (new_cos, new_sin)
                    nxt[key] = nxt.get(key, 0.0) + coefficient * scale
            partial = nxt
        for key, coefficient in partial.items():
            merged[key] = merged.get(key, 0.0) + coefficient
    return merged


def canonical_trig_normal_form(poly: TrigPoly) -> dict[NormalFormKey, float]:
    """Unique quotient-ring form with every sine exponent in ``{0, 1}``."""
    merged = _canonical_trig_normal_form_unpruned(poly)
    return {key: value for key, value in merged.items() if abs(value) > 1e-12}


# ---------- target-blind admission ----------


@dataclass(frozen=True)
class NormalFormSupportDiagnostics:
    """Aggregate public-class facts; no per-pair support or coefficient."""

    max_support: int
    p50_support: int
    p90_support: int
    over_cap_pairs: int
    multi_term_fraction: float
    max_terms_per_pair: int
    failures: list[str]


def normal_form_support_diagnostics(polys: list[TrigPoly]) -> NormalFormSupportDiagnostics:
    supports = np.array(
        [
            sum(
                1
                for value in _canonical_trig_normal_form_unpruned(poly).values()
                if abs(value) > _NORMAL_FORM_ZERO_ATOL
            )
            for poly in polys
        ]
    )
    raw_terms = max((len(poly.terms) for poly in polys), default=0)
    failures: list[str] = []
    over_cap = int((supports > ADMISSION_MAX_NORMAL_FORM_SUPPORT).sum())
    if over_cap:
        failures.append(
            f"{over_cap} pairs exceed the admission normal-form cap {ADMISSION_MAX_NORMAL_FORM_SUPPORT}"
        )
    multi_term = float(np.mean(supports >= 2)) if len(supports) else 0.0
    if multi_term < MIN_MULTI_TERM_FRACTION:
        failures.append(f"multi-term fraction {multi_term:.3f} < {MIN_MULTI_TERM_FRACTION}")
    if raw_terms > MAX_RAW_TERMS_PER_PAIR:
        failures.append(
            f"raw support {raw_terms} exceeds tractability cap {MAX_RAW_TERMS_PER_PAIR}"
        )
    for poly in polys:
        for term in poly.terms:
            if any(c + s > DEGREE_BOUND for c, s in zip(term.cos_pow, term.sin_pow, strict=True)):
                failures.append("a pair polynomial exceeds the public per-angle degree bound")
                break
        else:
            continue
        break
    return NormalFormSupportDiagnostics(
        max_support=int(supports.max()) if len(supports) else 0,
        p50_support=int(np.percentile(supports, 50)) if len(supports) else 0,
        p90_support=int(np.percentile(supports, 90)) if len(supports) else 0,
        over_cap_pairs=over_cap,
        multi_term_fraction=multi_term,
        max_terms_per_pair=int(raw_terms),
        failures=failures,
    )


def fourier_mode_supports(polys: list[TrigPoly]) -> np.ndarray:
    """Return the exact complex-Fourier mode count for each response."""
    return np.asarray([len(canonical_fourier_coefficients(poly)) for poly in polys], dtype=int)


@dataclass(frozen=True)
class AdmissionReport:
    nonconstant_fraction: float
    exact_zero_pairs: int
    nonconstant_output_span_rank: int
    angle_pair_fractions: list[float]
    min_angle_pair_fraction: float
    max_normal_form_support: int
    p50_normal_form_support: int
    p90_normal_form_support: int
    max_fourier_modes_per_response: int
    multi_term_fraction: float
    max_terms_per_pair: int
    max_mps_bond: int
    median_mps_bond: int
    rotation_count: int
    max_angle_multiplicity: int
    n_gates: int
    passed: bool
    failures: list[str]


def certificate_structural_diagnostics(
    admission: AdmissionReport,
) -> dict[str, dict[str, Any]]:
    """Exact target-independent certificate evidence taken from admission."""
    policy = CERT_STRUCTURAL_DIAGNOSTIC_POLICY["output_span"]
    metrics: dict[str, int | float] = {
        field_name: getattr(admission, field_name) for field_name in policy["minimums"]
    }
    passed = all(
        isinstance(metrics[name], (int, float))
        and math.isfinite(float(metrics[name]))
        and metrics[name] >= minimum
        for name, minimum in policy["minimums"].items()
    )
    return {
        "output_span": {
            "kind": policy["kind"],
            "target_independent": policy["target_independent"],
            "metrics": metrics,
            "minimums": dict(policy["minimums"]),
            "passed": passed,
        }
    }


def admission_report(
    ops: list[CircuitOp],
    polys: list[TrigPoly],
    *,
    rng: np.random.Generator,
    mps_rng: np.random.Generator,
    n_probe: int = ADMISSION_PROBE_INPUTS,
    flip_prob: list[float] | None = None,
    drift: list[float] | None = None,
) -> AdmissionReport:
    failures: list[str] = []
    del flip_prob  # noise enters through the damped polynomials already

    rotation_angles = [op.angle for op in ops if op.gate == "rot"]
    rotation_count = len(rotation_angles)
    multiplicity = max((rotation_angles.count(k) for k in range(INPUT_DIM)), default=0)
    if rotation_count > ROTATION_GATES:
        failures.append(
            f"rotation count {rotation_count} exceeds the public bound {ROTATION_GATES}"
        )
    if multiplicity > DEGREE_BOUND:
        failures.append(f"angle multiplicity {multiplicity} exceeds the public degree bound")
    if drift is not None and any(abs(value) > 0.0 for value in drift):
        failures.append("stationary edition requires an identically zero drift vector")

    support = normal_form_support_diagnostics(polys)
    failures.extend(support.failures)
    fourier_supports = fourier_mode_supports(polys)
    max_fourier_modes = int(fourier_supports.max()) if len(fourier_supports) else 0
    over_fourier_cap = int((fourier_supports > MAX_FOURIER_MODES_PER_RESPONSE).sum())
    if over_fourier_cap:
        failures.append(
            f"{over_fourier_cap} pairs exceed the public complex-Fourier mode cap "
            f"{MAX_FOURIER_MODES_PER_RESPONSE}"
        )

    probe = rng.uniform(-np.pi, np.pi, size=(n_probe, INPUT_DIM))
    labels = evaluate_pair_polys(polys, probe)
    variance = labels.var(axis=0)
    nonconstant = variance > 1e-8
    nonconstant_fraction = float(np.mean(nonconstant))
    if nonconstant_fraction < MIN_NONCONSTANT_PAIR_FRACTION:
        failures.append(
            f"nonconstant fraction {nonconstant_fraction:.3f} < {MIN_NONCONSTANT_PAIR_FRACTION}"
        )
    exact_zero_pairs = int(np.sum(np.abs(labels).max(axis=0) < 1e-12))
    if exact_zero_pairs > MAX_EXACT_ZERO_PAIRS:
        failures.append(f"{exact_zero_pairs} exact-zero pairs > {MAX_EXACT_ZERO_PAIRS}")
    centered = labels - labels.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    rank = int(np.sum(singular > 1e-9 * singular[0])) if singular.size and singular[0] > 0 else 0
    if rank < MIN_OUTPUT_SPAN_RANK:
        failures.append(f"nonconstant output span rank {rank} < {MIN_OUTPUT_SPAN_RANK}")

    angle_counts = np.zeros(INPUT_DIM)
    for poly in polys:
        touched = set()
        for term in poly.terms:
            for k in range(INPUT_DIM):
                if term.cos_pow[k] or term.sin_pow[k]:
                    touched.add(k)
        for k in touched:
            angle_counts[k] += 1
    angle_fractions = [float(v / max(len(polys), 1)) for v in angle_counts]
    min_angle_fraction = min(angle_fractions) if angle_fractions else 0.0
    if min_angle_fraction < MIN_ANGLE_PAIR_FRACTION:
        failures.append(
            f"weakest angle drives {min_angle_fraction:.3f} of pairs < {MIN_ANGLE_PAIR_FRACTION}"
        )

    probe_bonds: list[int] = []
    for probe_index, x in enumerate(
        mps_rng.uniform(-np.pi, np.pi, size=(MPS_ADMISSION_PROBES, INPUT_DIM))
    ):
        try:
            mps = build_mps(ops, x, N_QUBITS, max_bond=MAX_BOND)
        except BondDimensionExceeded as exc:
            failures.append(f"MPS admission probe {probe_index} exceeded max bond: {exc}")
            probe_bonds.append(MAX_BOND + 1)
            break
        probe_bonds.append(max(mps.bond_dims(), default=1))
    max_mps_bond = max(probe_bonds)
    median_mps_bond = int(np.median(probe_bonds))
    if max_mps_bond > MAX_ADMITTED_BOND:
        failures.append(f"maximum exact bond dimension {max_mps_bond} > {MAX_ADMITTED_BOND}")
    if median_mps_bond > MAX_ADMITTED_MEDIAN_BOND:
        failures.append(
            f"median exact bond dimension {median_mps_bond} > {MAX_ADMITTED_MEDIAN_BOND}"
        )

    return AdmissionReport(
        nonconstant_fraction=nonconstant_fraction,
        exact_zero_pairs=exact_zero_pairs,
        nonconstant_output_span_rank=rank,
        angle_pair_fractions=angle_fractions,
        min_angle_pair_fraction=float(min_angle_fraction),
        max_normal_form_support=support.max_support,
        p50_normal_form_support=support.p50_support,
        p90_normal_form_support=support.p90_support,
        max_fourier_modes_per_response=max_fourier_modes,
        multi_term_fraction=support.multi_term_fraction,
        max_terms_per_pair=support.max_terms_per_pair,
        max_mps_bond=max_mps_bond,
        median_mps_bond=median_mps_bond,
        rotation_count=rotation_count,
        max_angle_multiplicity=multiplicity,
        n_gates=len(ops),
        passed=not failures,
        failures=failures,
    )


def admission_summary(admission: AdmissionReport) -> dict[str, Any]:
    """Audit-trail dictionary written into the hidden truth."""
    return {
        "nonconstant_fraction": admission.nonconstant_fraction,
        "exact_zero_pairs": admission.exact_zero_pairs,
        "nonconstant_output_span_rank": admission.nonconstant_output_span_rank,
        "angle_pair_fractions": admission.angle_pair_fractions,
        "min_angle_pair_fraction": admission.min_angle_pair_fraction,
        "max_normal_form_support": admission.max_normal_form_support,
        "p50_normal_form_support": admission.p50_normal_form_support,
        "p90_normal_form_support": admission.p90_normal_form_support,
        "max_fourier_modes_per_response": admission.max_fourier_modes_per_response,
        "multi_term_fraction": admission.multi_term_fraction,
        "max_terms_per_pair": admission.max_terms_per_pair,
        "max_mps_bond": admission.max_mps_bond,
        "median_mps_bond": admission.median_mps_bond,
        "rotation_count": admission.rotation_count,
        "max_angle_multiplicity": admission.max_angle_multiplicity,
        "n_gates": admission.n_gates,
    }


@dataclass(frozen=True)
class ShadowSurrogateInstance:
    seed: int
    circuit: list[CircuitOp]
    flip_prob: list[float]
    drift: list[float]
    target_inputs: np.ndarray
    pair_polys: list[TrigPoly]
    labels: np.ndarray
    admission: AdmissionReport


@dataclass(frozen=True)
class ShadowSurrogateAdmissionCandidate:
    """Target-free construction state for structural admission decisions."""

    seed: int
    circuit: list[CircuitOp]
    flip_prob: list[float]
    drift: list[float]
    pair_polys: list[TrigPoly]
    admission: AdmissionReport


def new_private_seed() -> int:
    """256-bit root seed for a scored private edition.

    A date-like or otherwise enumerable seed is unsafe even when unpublished:
    the public target floats are a perfect seed-verification oracle (regenerate
    candidate seeds, compare targets, recover the hidden circuit). A 256-bit
    root makes enumeration impossible; domain separation in ``build_instance``
    additionally keeps the target stream from leaking the circuit/noise
    streams.
    """
    return secrets.randbits(256)


def _domain_rng(seed: int, domain: str) -> np.random.Generator:
    """Domain child stream via SHA-256 (a real KDF step, not SeedSequence's
    statistical mixing): given a secret root, one domain's outputs reveal
    nothing about another's derivation. Residual caveat, documented: PCG64
    output itself is not cryptographically hiding, so published target floats
    still expose PCG64 state of the TARGET stream only."""
    digest = hashlib.sha256(f"qiqcbench/{TASK_ID}/v2/{domain}/{seed}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


def build_admission_only(seed: int = INSTANCE_SEED) -> ShadowSurrogateAdmissionCandidate:
    """Build and structurally admit a candidate without sampling target rows."""
    circuit = build_circuit(_domain_rng(seed, "circuit"))
    flip_prob = build_noise(_domain_rng(seed, "noise"))
    drift = build_drift(_domain_rng(seed, "drift"), flip_prob)
    pair_axis_polys = build_pair_axis_polys(circuit, flip_prob)
    pair_polys = [
        _combine_polys(list(components), [1.0, 1.0, 1.0], INPUT_DIM)
        for components in pair_axis_polys
    ]
    admission = admission_report(
        circuit,
        pair_polys,
        rng=_domain_rng(seed, "admission"),
        mps_rng=_domain_rng(seed, "mps-admission"),
        flip_prob=flip_prob,
        drift=drift,
    )
    return ShadowSurrogateAdmissionCandidate(
        seed=seed,
        circuit=circuit,
        flip_prob=flip_prob,
        drift=drift,
        pair_polys=pair_polys,
        admission=admission,
    )


def materialize_admission_candidate(
    candidate: ShadowSurrogateAdmissionCandidate,
    *,
    check_admission: bool = True,
) -> ShadowSurrogateInstance:
    """Generate sealed target rows only after target-free structural admission."""
    if check_admission and not candidate.admission.passed:
        raise AdmissionRejectedError(
            f"instance seed {candidate.seed} fails admission: {candidate.admission.failures}"
        )
    target_inputs = build_target_inputs(_domain_rng(candidate.seed, "targets"))
    # Labels are the start-of-run truth (the device is stationary in this
    # edition, so they are also the whole-run truth).
    labels = evaluate_pair_polys(candidate.pair_polys, target_inputs)
    return ShadowSurrogateInstance(
        seed=candidate.seed,
        circuit=candidate.circuit,
        flip_prob=candidate.flip_prob,
        drift=candidate.drift,
        target_inputs=target_inputs,
        pair_polys=candidate.pair_polys,
        labels=labels,
        admission=candidate.admission,
    )


def build_instance(
    seed: int = INSTANCE_SEED, *, check_admission: bool = True
) -> ShadowSurrogateInstance:
    # Domain-separated child streams (circuit / noise / targets / admission):
    # the published target floats reveal nothing about the hidden circuit or
    # noise draws, and per-domain changes cannot silently shift the others.
    candidate = build_admission_only(seed)
    return materialize_admission_candidate(candidate, check_admission=check_admission)


def find_admissible_seed(start: int, *, max_tries: int = 50) -> int:
    """Development-only search over target-free structural admission.

    This helper must not be called after targets have been materialized or a
    reference/ablation certificate has been attempted. Scored roots are
    precommitted by the materializer; a post-target
    certificate failure retires that edition instead of selecting a new seed.
    """
    for offset in range(max_tries):
        seed = start + offset
        if build_admission_only(seed).admission.passed:
            return seed
    raise RuntimeError(f"no admissible seed in [{start}, {start + max_tries})")


# ---------- materials ----------


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _compact_json_bytes(payload: Any) -> bytes:
    """Canonical compact JSON used by the public sealed-target commitment."""
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def challenge_metadata(instance: ShadowSurrogateInstance) -> dict[str, Any]:
    """Exact identifiers/digests for the sealed ordered target matrix."""
    target_inputs = [[float(v) for v in row] for row in instance.target_inputs]
    target_inputs_sha256 = hashlib.sha256(
        b"qiqcbench/sealed-target-inputs/v1\0" + _compact_json_bytes(target_inputs)
    ).hexdigest()
    challenge_id = hashlib.sha256(
        f"qiqcbench/{TASK_ID}/challenge-id/v1/{instance.seed}".encode()
    ).hexdigest()
    commitment_nonce = hashlib.sha256(
        f"qiqcbench/{TASK_ID}/challenge-nonce/v1/{instance.seed}".encode()
    ).hexdigest()
    commitment_payload = {
        "challenge_id": challenge_id,
        "target_inputs_sha256": target_inputs_sha256,
        "commitment_nonce": commitment_nonce,
    }
    return {
        "schema_version": TARGET_CHALLENGE_SCHEMA_VERSION,
        "challenge_id": challenge_id,
        "commitment_scheme": "qiqcbench_sealed_target_sha256_v1",
        "target_inputs": target_inputs,
        "target_inputs_sha256": target_inputs_sha256,
        "target_commitment_sha256": hashlib.sha256(
            b"qiqcbench/sealed-target-commitment/v1\0" + _compact_json_bytes(commitment_payload)
        ).hexdigest(),
        "commitment_nonce": commitment_nonce,
    }


def build_public_target_challenge_from_metadata(challenge: dict[str, Any]) -> dict[str, Any]:
    """Public challenge contract from a trusted opening, never coordinates."""
    return {
        "schema_version": TARGET_CHALLENGE_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "device_id": DEVICE_ID,
        "challenge_id": challenge["challenge_id"],
        "commitment_scheme": challenge["commitment_scheme"],
        "input_dimension": INPUT_DIM,
        "target_count": N_TARGETS,
        "domain": {
            "lower_inclusive": -float(np.pi),
            "upper_inclusive": float(np.pi),
            "periodic": True,
        },
        "sampling_strata": [dict(stratum) for stratum in TARGET_STRATA],
        "target_commitment_sha256": challenge["target_commitment_sha256"],
        "commitment": {
            "scheme": challenge["commitment_scheme"],
            "target_digest_domain": "qiqcbench/sealed-target-inputs/v1\\0",
            "commitment_domain": "qiqcbench/sealed-target-commitment/v1\\0",
            "domain_suffix_interpretation": "\\0 denotes one NUL byte (0x00)",
            "target_digest_payload": "ordered target_inputs list",
            "opening_payload_fields": [
                "challenge_id",
                "target_inputs_sha256",
                "commitment_nonce",
            ],
            "canonicalization": (
                "Python 3.11 json.dumps(value, allow_nan=False, ensure_ascii=False, "
                "separators=(',', ':'), sort_keys=True), encoded as UTF-8"
            ),
        },
    }


def build_public_target_challenge(instance: ShadowSurrogateInstance) -> dict[str, Any]:
    """Public challenge contract: commitment and distribution, never coordinates."""
    return build_public_target_challenge_from_metadata(challenge_metadata(instance))


def build_public_observable_index() -> dict[str, Any]:
    return {
        "schema_version": 6,
        "task_id": TASK_ID,
        "n_qubits": N_QUBITS,
        "n_pairs": len(pair_order()),
        "observable": "(X_i X_j + Y_i Y_j + Z_i Z_j) / 3",
        "scored_quantity": (
            "as-measured expectation at the device's start-of-run condition, "
            "including the device's own readout errors; no mitigation"
        ),
        "hypothesis_class": {
            "max_rotation_gates": ROTATION_GATES,
            "max_harmonic_degree_per_angle": DEGREE_BOUND,
            "input_dimension": INPUT_DIM,
        },
        "response_class": {
            "schema_version": 2,
            "semantics": "bounded_gate_fourier_response_v2",
            "observable": "summed_C_ij",
            "physics": (
                "the hidden circuit contains at most max_rotation_gates parameterized "
                "non-Clifford rotations controlled by the input_dimension input angles"
            ),
            "harmonic_structure": (
                "for every scored pair, the dependence on each input angle is "
                "trigonometric with harmonic degree at most "
                "max_harmonic_degree_per_angle, so the responses belong to a "
                "finite-dimensional multivariate trigonometric function class"
            ),
            "fourier_representation": {
                "form": "sum_m a_m * exp(i * m dot x)",
                "frequency_vectors": (
                    "integer vectors m with -max_harmonic_degree_per_angle <= m_k <= "
                    "max_harmonic_degree_per_angle"
                ),
                "max_nonzero_modes_per_response": MAX_FOURIER_MODES_PER_RESPONSE,
                "mode_counting": (
                    "each distinct integer frequency vector counts once; conjugate "
                    "frequency vectors count separately"
                ),
            },
            "support_semantics": (
                "each scored response contains at most "
                "max_nonzero_modes_per_response nonzero complex Fourier modes; active "
                "frequency vectors and coefficients are hidden and may differ across pairs"
            ),
            "pair_supports_disclosed": False,
            "exact_zero_pairs_disclosed": False,
            "coefficients_disclosed": False,
        },
        "order": "lexicographic (i, j), 0 <= i < j < n_qubits",
        "pairs": [[i, j] for i, j in pair_order()],
    }


def build_hidden_truth(instance: ShadowSurrogateInstance) -> dict[str, Any]:
    challenge = challenge_metadata(instance)
    return {
        "task_id": TASK_ID,
        "device_id": DEVICE_ID,
        "seed": instance.seed,
        "challenge_id": challenge["challenge_id"],
        "target_inputs_sha256": challenge["target_inputs_sha256"],
        "target_commitment_sha256": challenge["target_commitment_sha256"],
        "commitment_nonce": challenge["commitment_nonce"],
        "target_challenge": dict(challenge),
        # Edition binding: the hidden truth commits to the exact public-material
        # bytes it describes, so the verifier can prove the inputs shown to the
        # agent are the inputs its labels were computed for.
        "public_materials_sha256": {
            "target_challenge.json": hashlib.sha256(
                _json_bytes(build_public_target_challenge(instance))
            ).hexdigest(),
            "observable_index.json": hashlib.sha256(
                _json_bytes(build_public_observable_index())
            ).hexdigest(),
        },
        "input_dimension": INPUT_DIM,
        "n_qubits": N_QUBITS,
        "depolarizing_flip_prob": instance.flip_prob,
        # Hidden nuisance, recorded for the audit trail only: the labels below
        # are the u = 0 values and do NOT depend on it.
        "readout_drift_per_qubit": instance.drift,
        "target_inputs": challenge["target_inputs"],
        "labels": [[float(v) for v in row] for row in instance.labels],
        "pair_polys": [poly.to_jsonable() for poly in instance.pair_polys],
        "admission": admission_summary(instance.admission),
    }


def instance_digest(instance: ShadowSurrogateInstance) -> str:
    """sha256 over the EXECUTABLE identity plus the truth payload.

    The circuit operations (and noise) must be inside the digest: the truth
    payload alone would let a mutated circuit paired with stale truth fields
    keep the same digest while runtime labels diverge (verified attack:
    changing one rotation's angle left the old digest valid while labels
    moved by RMSE 0.09). This is the identity a certify_edition certificate
    binds to and emit_edition verifies.
    """
    payload = {
        "hidden_truth": build_hidden_truth(instance),
        "target_challenge": challenge_metadata(instance),
        "circuit": [op.model_dump(exclude_none=True) for op in instance.circuit],
        "flip_prob": instance.flip_prob,
        # The drift is part of the executable device identity: two editions
        # with identical truth but different drift are different experiments.
        "readout_drift": instance.drift,
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def verify_instance_consistency(instance: ShadowSurrogateInstance) -> None:
    """Recompute the truth FROM THE CIRCUIT and require it to match.

    A digest binds bytes, not physics: an internally inconsistent instance
    (mutated circuit with stale pair_polys/labels) would digest fine, so
    emission recomputes the closed-form labels from the executable circuit
    and noise and compares them to the instance's stored labels.
    """
    recomputed_polys = build_pair_polys(instance.circuit, instance.flip_prob)
    recomputed = evaluate_pair_polys(recomputed_polys, instance.target_inputs)
    if not np.allclose(recomputed, instance.labels, atol=1e-12, rtol=0.0):
        worst = float(np.max(np.abs(recomputed - instance.labels)))
        raise ValueError(
            "instance labels are not the closed-form truth of its circuit/noise "
            f"(max deviation {worst:.6g}): refusing an internally inconsistent edition"
        )


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _validate_public_material_inventory(
    public_dir: Path, expected_names: set[str], *, require_complete: bool
) -> None:
    entries = {path.name: path for path in public_dir.iterdir()}
    entry_names = set(entries)
    invalid_types = sorted(
        name for name, path in entries.items() if path.is_symlink() or not path.is_file()
    )
    if (
        not entry_names <= expected_names
        or invalid_types
        or (require_complete and entry_names != expected_names)
    ):
        raise ValueError(
            "public material inventory must contain exactly the expected regular files; "
            f"expected {sorted(expected_names)}, found {sorted(entry_names)}, "
            f"non-regular entries {invalid_types}"
        )


def write_materials(
    dest_dir: Path, instance: ShadowSurrogateInstance | None = None
) -> dict[str, Path]:
    """Write public + hidden task materials; returns the written paths."""
    if instance is None:
        instance = build_instance()
    public_dir = dest_dir / "public"
    hidden_dir = dest_dir / "hidden"
    public_dir.mkdir(parents=True, exist_ok=True)
    # The preceding transductive edition wrote scored coordinates here. Remove
    # only that known legacy leak during the one-way migration; every other
    # unexpected entry remains a fail-closed inventory error below.
    (public_dir / "target_inputs.json").unlink(missing_ok=True)
    expected_public_inventory = {"observable_index.json", "target_challenge.json"}
    _validate_public_material_inventory(
        public_dir, expected_public_inventory, require_complete=False
    )
    hidden_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "target_challenge": public_dir / "target_challenge.json",
        "observable_index": public_dir / "observable_index.json",
        "hidden_truth": hidden_dir / "truth.json",
    }
    _write_atomic(paths["target_challenge"], _json_bytes(build_public_target_challenge(instance)))
    _write_atomic(paths["observable_index"], _json_bytes(build_public_observable_index()))
    _validate_public_material_inventory(
        public_dir, expected_public_inventory, require_complete=True
    )
    _write_atomic(paths["hidden_truth"], _json_bytes(build_hidden_truth(instance)))
    return paths
