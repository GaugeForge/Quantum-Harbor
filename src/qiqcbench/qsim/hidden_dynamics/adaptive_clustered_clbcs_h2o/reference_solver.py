"""Public-observation-only Track-B feasibility solver (maintainer-only).

This module tests whether the public C-LBCS task contract is achievable without
using hidden state information during design.  It consumes the public Hamiltonian
and a legal pilot transcript, estimates same-setting Pauli means/covariances, selects
control variates within the public cap against the estimated between-setting quadratic
form, and refines the scheme against the estimated full clustered variance.

The key lever is to optimize the 16-component C-LBCS
*mixture scheme* for the state-dependent variance, not the state-independent
``V_Haar``. ``V_within``'s diagonal is
``sum_j c_j^2 (1 - e_j^2) / h_j`` — a coverage objective with coefficients
reweighted by ``(1 - e_j^2)`` (near-eigenoperators drop out), estimable from the
pilot.  Compatible pairs measured in the same pilot setting supply the
off-diagonal covariance terms used by the refinement stage.

This is a MAINTAINER feasibility/reference tool, not an agent surface. It plays
both roles for the demonstration: the hidden state (``construction.build_hidden_state``)
acts only as the device that returns raw pilot outcomes.  Hidden means enter only
after the design is locked, when ``construction.private_cluster_variance`` performs
the same post-run certification as the separate-mode verifier.

Run the feasibility sweep on demand (NOT part of the default/slow pytest paths):

    uv run python -m qiqcbench.qsim.hidden_dynamics.adaptive_clustered_clbcs_h2o.reference_solver \
      50 --workers 8 --output \
      configs/task_materials/adaptive_clustered_clbcs_h2o/hidden/feasibility_certificate_v5.json

The release path has no seed option. It draws fresh OS entropy only after the
scientific contract digest is frozen, retains commitments rather than replayable
entropy, and refuses to replace a certificate for the same contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from qiqcbench.qsim.hidden_dynamics.adaptive_clustered_clbcs_h2o import construction as C

TASK = "adaptive_clustered_clbcs_h2o"
NQ = 14
_AX = {"X": 0, "Y": 1, "Z": 2}
_H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
_SDG = np.array([[1, 0], [0, -1j]], complex)
_ROT = {"Z": np.eye(2, dtype=complex), "X": _H, "Y": _H @ _SDG}

PILOT_MAX_SETTINGS = 96
PILOT_MAX_SHOTS = 24576
PILOT_SHOTS_PER_SETTING_MAX = 512
PILOT_SHOTS_MULTIPLE = 64
MAX_CONTROL_VARIATES = 192
PRODUCTION_SETTINGS = 192
PRODUCTION_SHOTS_PER_SETTING = 256
GATE = 0.044
NOMINAL_ONE_SIGMA_COVERAGE = 0.6826894921370859
TRACK_B_WILSON_LOWER_FLOOR = 0.80
_WILSON_Z_95 = 1.959963984540054
_STREAM_DOMAINS = (
    "pilot_outcomes",
    "optimizer_restarts",
    "production_basis",
    "production_outcomes",
)


@dataclass(frozen=True)
class PilotStatistics:
    """Sufficient statistics reconstructed from legal raw pilot rows."""

    means: np.ndarray
    shots_per_term: np.ndarray
    product_sums: dict[tuple[int, int], float]
    product_shots: dict[tuple[int, int], int]
    num_settings: int
    row_shots: tuple[int, ...]


@dataclass(frozen=True)
class WithinModel:
    """Pilot-estimated full within-setting objective.

    ``pair_q`` already contains the factor ``2 c_j c_l covariance(j,l)``.
    ``pair_union`` indexes ``supports``, whose first ``num_terms`` entries are
    the observable terms themselves.
    """

    supports: tuple[tuple[tuple[int, int], ...], ...]
    num_terms: int
    diagonal: np.ndarray
    pair_j: np.ndarray
    pair_l: np.ndarray
    pair_union: np.ndarray
    pair_q: np.ndarray


@dataclass(frozen=True)
class TrialEntropy:
    """Private entropy for one prospective feasibility trial."""

    trial_id: str
    pilot_entropy: int
    optimizer_entropy: int
    production_basis_entropy: int
    production_outcome_entropy: int

    @property
    def commitment(self) -> str:
        values = [
            self.pilot_entropy,
            self.optimizer_entropy,
            self.production_basis_entropy,
            self.production_outcome_entropy,
        ]
        payload = json.dumps(values, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ProspectivePanel:
    """One fresh, contract-bound release-validation panel."""

    panel_id: str
    contract_digest: str
    panel_entropy_commitment: str
    trials: tuple[TrialEntropy, ...]


def contract_digest(digests: dict[str, str]) -> str:
    payload = json.dumps(digests, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def draw_prospective_panel(
    n_trials: int,
    contract_digests: dict[str, str],
) -> ProspectivePanel:
    """Draw one fresh four-domain entropy tuple per trial from the OS CSPRNG."""
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    trials = tuple(
        TrialEntropy(
            trial_id=f"trial-{trial:03d}",
            pilot_entropy=secrets.randbits(128),
            optimizer_entropy=secrets.randbits(128),
            production_basis_entropy=secrets.randbits(128),
            production_outcome_entropy=secrets.randbits(128),
        )
        for trial in range(n_trials)
    )
    private_payload = [
        [
            trial.pilot_entropy,
            trial.optimizer_entropy,
            trial.production_basis_entropy,
            trial.production_outcome_entropy,
        ]
        for trial in trials
    ]
    commitment = hashlib.sha256(
        json.dumps(private_payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ProspectivePanel(
        panel_id=commitment[:32],
        contract_digest=contract_digest(contract_digests),
        panel_entropy_commitment=commitment,
        trials=trials,
    )


def ensure_certificate_slot(path: Path, current_contract_digest: str) -> None:
    """Prevent result-shopping by replacing a panel for an unchanged contract."""
    if not path.is_file():
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    existing_digest = existing.get("prospective_panel", {}).get("contract_digest")
    if existing_digest == current_contract_digest:
        raise ValueError(
            "this contract already has a prospective certificate; "
            "change the contract before drawing another release panel"
        )


# --------------------------------------------------------------------------- #
# Hamiltonian + support index
# --------------------------------------------------------------------------- #
def load_problem():
    terms, coeffs, term_indices, identity = C.load_hamiltonian(TASK)
    supp = [[(i, _AX[ch]) for i, ch in enumerate(p) if ch != "I"] for p in terms]
    return terms, np.asarray(coeffs, float), term_indices, supp


# --------------------------------------------------------------------------- #
# C-LBCS scheme optimizer: minimize sum_j w_j / h_j (diagonal V_within proxy)
# --------------------------------------------------------------------------- #
def _coverage_vec(weights, beta, supp):
    logb = np.log(np.clip(beta, 1e-300, None))
    K = weights.shape[0]
    g = np.ones((K, len(supp)))
    for j, s in enumerate(supp):
        if s:
            acc = np.zeros(K)
            for i, a in s:
                acc += logb[:, i, a]
            g[:, j] = np.exp(acc)
    return weights @ g, g


def _marginal_init(w_coef, supp, K, rng):
    base = np.full((NQ, 3), 1e-6)
    for j, s in enumerate(supp):
        for i, a in s:
            base[i, a] += w_coef[j]
    base = np.sqrt(base)
    base /= base.sum(axis=-1, keepdims=True)
    beta = np.empty((K, NQ, 3))
    for k in range(K):
        pert = base * rng.uniform(0.6, 1.4, size=(NQ, 3))
        beta[k] = pert / pert.sum(axis=-1, keepdims=True)
    return beta


def _opt_once(w_coef, supp, K, iters, eta0, seed):
    rng = np.random.default_rng(seed)
    beta = _marginal_init(w_coef, supp, K, rng)
    weights = np.ones(K) / K
    active = np.flatnonzero(w_coef > 0)
    best = (weights.copy(), beta.copy(), np.inf)
    for it in range(iters):
        eta = eta0 * (0.1 ** (it / iters))
        h, g = _coverage_vec(weights, beta, supp)
        h = np.clip(h, 1e-12, None)
        s = -w_coef / h**2
        gw = g @ s
        gw /= np.abs(gw).max() + 1e-30
        weights = weights * np.exp(-eta * gw)
        weights = np.clip(weights, 1e-12, None)
        weights /= weights.sum()
        gb = np.zeros((K, NQ, 3))
        for j in active:
            sj = s[j]
            for i, a in supp[j]:
                gb[:, i, a] += sj * weights * g[:, j]
        gb /= np.clip(beta, 1e-12, None)
        gb /= np.abs(gb).max() + 1e-30
        beta = beta * np.exp(-eta * gb)
        beta = np.clip(beta, 1e-12, None)
        beta /= beta.sum(axis=-1, keepdims=True)
        f = float(np.sum(w_coef / h))
        if f < best[2]:
            best = (weights.copy(), beta.copy(), f)
    return best


def optimize_scheme(
    w_coef,
    supp,
    *,
    K=16,
    iters=1500,
    eta=0.6,
    restarts=6,
    seed=0,
    score=None,
    score_top=3,
):
    """Restarted diagonal optimizer with an optional public estimated scorer."""
    cands = [_opt_once(w_coef, supp, K, iters, eta, seed + r) for r in range(restarts)]
    cands.sort(key=lambda c: c[2])
    if score is None:
        return cands[0][0], cands[0][1]
    best = min(cands[:score_top], key=lambda c: score(c[0], c[1]))
    return best[0], best[1]


# --------------------------------------------------------------------------- #
# Pilot (joint-sampled product-Pauli measurements on the hidden state)
# --------------------------------------------------------------------------- #
def _sample_basis(state, basis, shots, rng):
    t = state.reshape([2] * NQ)
    for q in range(NQ):
        if basis[q] != "Z":
            t = np.moveaxis(np.tensordot(_ROT[basis[q]], t, axes=([1], [q])), 0, q)
    p = np.abs(t.reshape(-1)) ** 2
    p = p / p.sum()
    draws = rng.choice(p.size, size=shots, p=p)
    return ((draws[:, None] >> np.arange(NQ - 1, -1, -1)) & 1).astype(np.int8)


def _qwc_groups(terms, order):
    groups = []
    for j in order:
        p = terms[j]
        for grp in groups:
            if all(p[i] == "I" or grp["ax"].get(i, p[i]) == p[i] for i in range(NQ)):
                for i in range(NQ):
                    if p[i] != "I":
                        grp["ax"][i] = p[i]
                grp["members"].append(j)
                break
        else:
            groups.append({"ax": {i: p[i] for i in range(NQ) if p[i] != "I"}, "members": [j]})
    return [("".join(g["ax"].get(i, "Z") for i in range(NQ)), g["members"]) for g in groups]


def _basis_covers(pauli: str, basis: str) -> bool:
    return all(p == "I" or p == q for p, q in zip(pauli, basis, strict=True))


def run_pilot(
    state,
    terms,
    coeffs,
    supp,
    rng,
    *,
    max_settings=PILOT_MAX_SETTINGS,
    max_shots=PILOT_MAX_SHOTS,
):
    """Collect public Hamiltonian-driven pilot statistics within every public cap."""
    order = sorted(range(len(terms)), key=lambda j: (-abs(coeffs[j]), -len(supp[j]), j))
    first_count = min(48, max_settings)
    first_groups = _qwc_groups(terms, order)[:first_count]
    first_shots = min(256, PILOT_SHOTS_PER_SETTING_MAX)
    num = np.zeros(len(terms))
    den = np.zeros(len(terms))
    product_sums: dict[tuple[int, int], float] = {}
    product_shots: dict[tuple[int, int], int] = {}
    row_shots: list[int] = []

    def collect(basis, shots):
        bits = _sample_basis(state, basis, shots, rng)
        covered = [j for j, pauli in enumerate(terms) if _basis_covers(pauli, basis)]
        signs = np.empty((shots, len(covered)), dtype=float)
        for col, j in enumerate(covered):
            par = np.zeros(bits.shape[0], dtype=np.int64)
            for i, _a in supp[j]:
                par ^= bits[:, i]
            signs[:, col] = 1.0 - 2.0 * par
            num[j] += float(signs[:, col].sum())
            den[j] += shots
        cross = signs.T @ signs
        for a, j in enumerate(covered):
            for b in range(a + 1, len(covered)):
                ell = covered[b]
                key = (j, ell)
                product_sums[key] = product_sums.get(key, 0.0) + float(cross[a, b])
                product_shots[key] = product_shots.get(key, 0) + shots
        row_shots.append(shots)

    for basis, _members in first_groups:
        collect(basis, first_shots)

    # The second stage is selected only from first-stage observations.  It focuses
    # the remaining shots on apparently nonzero Pauli means and their QWC products.
    interim = np.zeros(len(terms))
    observed = den > 0
    interim[observed] = num[observed] / den[observed]
    significance = np.zeros(len(terms))
    significance[observed] = np.abs(interim[observed]) * np.sqrt(den[observed])
    candidates = np.flatnonzero(significance >= 2.0)
    if len(candidates) == 0:
        candidates = np.flatnonzero(observed)
    adaptive_order = sorted(
        candidates,
        key=lambda j: (-abs(coeffs[j] * interim[j]), -len(supp[j]), int(j)),
    )
    adaptive_groups = _qwc_groups(terms, adaptive_order)
    remaining_shots = max_shots - sum(row_shots)
    second_shots = min(PILOT_SHOTS_PER_SETTING_MAX, remaining_shots)
    second_shots = (second_shots // PILOT_SHOTS_MULTIPLE) * PILOT_SHOTS_MULTIPLE
    second_count = 0
    if second_shots > 0:
        second_count = min(max_settings - len(row_shots), remaining_shots // second_shots)
    for row in range(second_count):
        basis, _members = adaptive_groups[row % len(adaptive_groups)]
        collect(basis, second_shots)

    mean = np.zeros(len(terms))
    nz = den > 0
    mean[nz] = num[nz] / den[nz]
    return PilotStatistics(
        means=mean,
        shots_per_term=den,
        product_sums=product_sums,
        product_shots=product_shots,
        num_settings=len(row_shots),
        row_shots=tuple(row_shots),
    )


def _shrink(ehat, neff, active):
    out = ehat.copy()
    for j in active:
        if neff[j] > 0:
            var = max(1e-6, (1.0 - ehat[j] ** 2) / neff[j])
            out[j] = ehat[j] * (ehat[j] ** 2) / (ehat[j] ** 2 + var)
    return out


def run_production(
    state,
    terms,
    coeffs,
    identity,
    supp,
    control,
    weights,
    beta,
    basis_rng,
    outcome_rng,
):
    """Execute and analyze the pinned estimator from raw clustered outcomes."""
    h = np.array([C.coverage(pauli, weights, beta) for pauli in terms])
    constant = identity + float(np.dot(coeffs, control))
    setting_estimates = np.empty(PRODUCTION_SETTINGS, dtype=float)
    axes = np.array(["X", "Y", "Z"])
    for row in range(len(setting_estimates)):
        component = int(basis_rng.choice(len(weights), p=weights))
        basis = "".join(
            axes[int(basis_rng.choice(3, p=beta[component, qubit]))] for qubit in range(NQ)
        )
        bits = _sample_basis(state, basis, PRODUCTION_SHOTS_PER_SETTING, outcome_rng)
        estimate = constant
        for j, pauli in enumerate(terms):
            if not _basis_covers(pauli, basis):
                continue
            parity = np.zeros(bits.shape[0], dtype=np.int64)
            for qubit, _axis in supp[j]:
                parity ^= bits[:, qubit]
            observed = float((1.0 - 2.0 * parity).mean())
            estimate += coeffs[j] * (observed - control[j]) / h[j]
        setting_estimates[row] = estimate
    energy = float(setting_estimates.mean())
    cluster_se = float(setting_estimates.std(ddof=1) / np.sqrt(len(setting_estimates)))
    return energy, cluster_se


def solve_track_a(*, refinement_iters=100):
    """Refine the public stale-notebook warm start against the exact public objective."""
    terms, coeffs, _term_indices, supp = load_problem()
    notebook_path = C.hamiltonian_csv_path(TASK).with_name("stale_notebook.json")
    warm_start = json.loads(notebook_path.read_text(encoding="utf-8"))["track_a_warm_start"]
    model = WithinModel(
        supports=tuple(tuple(s) for s in supp),
        num_terms=len(terms),
        diagonal=coeffs**2,
        pair_j=np.array([], dtype=int),
        pair_l=np.array([], dtype=int),
        pair_union=np.array([], dtype=int),
        pair_q=np.array([], dtype=float),
    )
    weights, beta = refine_scheme_with_covariance(
        np.asarray(warm_start["mixture_weights"], dtype=float),
        np.asarray(warm_start["local_basis_probabilities_xyz"], dtype=float),
        model,
        iters=refinement_iters,
        eta0=0.1,
    )
    return C.track_a_variance(terms, coeffs, weights, beta)


def _between_matrix(terms, coeffs, weights, beta, indices):
    """Quadratic matrix ``Q`` such that ``V_between = residual.T @ Q @ residual``."""
    indices = np.asarray(indices, dtype=int)
    h = np.array([C.coverage(terms[j], weights, beta) for j in indices])
    q = np.empty((len(indices), len(indices)), dtype=float)
    for a, j in enumerate(indices):
        q[a, a] = coeffs[j] ** 2 * (1.0 / h[a] - 1.0)
        for b in range(a + 1, len(indices)):
            ell = indices[b]
            up = C.qwc_union_product(terms[j], terms[ell])
            ratio = 0.0
            if up is not None:
                ratio = C.coverage(up[0], weights, beta) / (h[a] * h[b])
            q[a, b] = q[b, a] = coeffs[j] * coeffs[ell] * (ratio - 1.0)
    return q


def select_control_variates(
    terms,
    coeffs,
    estimated_means,
    measured,
    weights,
    beta,
    *,
    cap=MAX_CONTROL_VARIATES,
):
    """Choose a sparse mean vector by minimizing estimated basis-selection noise.

    Reverse greedy starts with no residuals and chooses the terms that are cheapest
    to leave uncontrolled.  This directly targets the between-setting quadratic;
    ranking only by ``|c_j e_j|`` performs poorly once the cap genuinely binds.
    """
    candidates = np.flatnonzero(measured & (np.abs(estimated_means) > 1e-12))
    if len(candidates) <= cap:
        control = np.zeros(len(terms))
        control[candidates] = estimated_means[candidates]
        return control
    q = _between_matrix(terms, coeffs, weights, beta, candidates)
    residual = np.zeros(len(candidates), dtype=float)
    uncontrolled: list[int] = []
    for _ in range(len(candidates) - cap):
        qd = q @ residual
        increase = (
            2.0 * estimated_means[candidates] * qd + np.diag(q) * estimated_means[candidates] ** 2
        )
        if uncontrolled:
            increase[np.asarray(uncontrolled, dtype=int)] = np.inf
        chosen = int(np.argmin(increase))
        residual[chosen] = estimated_means[candidates[chosen]]
        uncontrolled.append(chosen)
    keep = np.ones(len(candidates), dtype=bool)
    uncontrolled_array = np.asarray(uncontrolled, dtype=int)
    keep[uncontrolled_array] = False
    selected = np.flatnonzero(keep)
    residual_uncontrolled = estimated_means[candidates[uncontrolled_array]]
    q_selected = q[np.ix_(selected, selected)]
    rhs = -q[np.ix_(selected, uncontrolled_array)] @ residual_uncontrolled
    regularizer = max(1e-12, 1e-10 * float(np.max(np.diag(q_selected))))
    residual_selected = np.linalg.lstsq(
        q_selected + regularizer * np.eye(len(selected)),
        rhs,
        rcond=1e-10,
    )[0]
    optimized_constants = np.clip(
        estimated_means[candidates[selected]] - residual_selected,
        -1.0,
        1.0,
    )
    control = np.zeros(len(terms))
    control[candidates[selected]] = optimized_constants
    return control


def build_clustered_model(
    terms,
    coeffs,
    supp,
    estimated_means,
    pilot,
    *,
    control_means=None,
    production_shots=256,
):
    """Build the scheme-dependent part of ``V_between + V_within / R``.

    The omitted ``-(sum_j c_j d_j)^2`` term is constant with respect to the
    measurement scheme, so it does not affect candidate ranking or gradients.
    """
    residual = (
        np.zeros(len(terms), dtype=float)
        if control_means is None
        else estimated_means - np.asarray(control_means, dtype=float)
    )
    diagonal = coeffs**2 * (
        residual**2 + (1.0 - np.clip(estimated_means**2, 0.0, 1.0)) / production_shots
    )
    supports: list[tuple[tuple[int, int], ...]] = [tuple(s) for s in supp]
    support_index = {s: i for i, s in enumerate(supports)}
    covariance: dict[tuple[int, int], float] = {}
    for (j, ell), total in pilot.product_sums.items():
        shots = pilot.product_shots[(j, ell)]
        product_mean = total / shots
        value = product_mean - estimated_means[j] * estimated_means[ell]
        covariance[(j, ell)] = value * shots / (shots + 64.0)

    pair_j: list[int] = []
    pair_l: list[int] = []
    pair_union: list[int] = []
    pair_q: list[float] = []
    for j in range(len(terms)):
        for ell in range(j + 1, len(terms)):
            cov = covariance.get((j, ell), 0.0) / production_shots
            between = residual[j] * residual[ell]
            if cov == 0.0 and between == 0.0:
                continue
            q = 2.0 * coeffs[j] * coeffs[ell] * (cov + between)
            if abs(q) < 1e-10:
                continue
            up = C.qwc_union_product(terms[j], terms[ell])
            if up is None:
                continue
            union_supp = tuple((i, _AX[ch]) for i, ch in enumerate(up[0]) if ch != "I")
            u = support_index.get(union_supp)
            if u is None:
                u = len(supports)
                supports.append(union_supp)
                support_index[union_supp] = u
            pair_j.append(j)
            pair_l.append(ell)
            pair_union.append(u)
            pair_q.append(q)
    return WithinModel(
        supports=tuple(supports),
        num_terms=len(terms),
        diagonal=diagonal,
        pair_j=np.asarray(pair_j, dtype=int),
        pair_l=np.asarray(pair_l, dtype=int),
        pair_union=np.asarray(pair_union, dtype=int),
        pair_q=np.asarray(pair_q, dtype=float),
    )


def build_within_model(terms, coeffs, supp, estimated_means, pilot):
    """Compatibility wrapper for the pilot-estimated ``V_within / R`` model."""
    return build_clustered_model(terms, coeffs, supp, estimated_means, pilot)


def within_model_value_and_derivative(weights, beta, model):
    """Return the estimated full within objective and dF/d(coverage)."""
    h, g = _coverage_vec(weights, beta, model.supports)
    h = np.clip(h, 1e-12, None)
    ht = h[: model.num_terms]
    value = float(np.sum(model.diagonal / ht))
    derivative = np.zeros(len(model.supports), dtype=float)
    derivative[: model.num_terms] = -model.diagonal / ht**2
    if len(model.pair_q):
        hj = ht[model.pair_j]
        hl = ht[model.pair_l]
        hu = h[model.pair_union]
        pair_value = model.pair_q * hu / (hj * hl)
        value += float(np.sum(pair_value))
        np.add.at(derivative, model.pair_j, -pair_value / hj)
        np.add.at(derivative, model.pair_l, -pair_value / hl)
        np.add.at(derivative, model.pair_union, model.pair_q / (hj * hl))
    return value, derivative, g


def refine_scheme_with_covariance(
    weights,
    beta,
    model,
    *,
    iters=250,
    eta0=0.2,
):
    """Exponentiated-gradient refinement against pilot-estimated full covariance."""
    weights = np.asarray(weights, dtype=float).copy()
    beta = np.asarray(beta, dtype=float).copy()
    best = (weights.copy(), beta.copy(), np.inf)
    for it in range(iters):
        value, derivative, g = within_model_value_and_derivative(weights, beta, model)
        if value < best[2]:
            best = (weights.copy(), beta.copy(), value)
        eta = eta0 * (0.1 ** (it / max(iters, 1)))
        grad_w = g @ derivative
        grad_w /= np.max(np.abs(grad_w)) + 1e-30
        weights *= np.exp(-eta * grad_w)
        weights = np.clip(weights, 1e-12, None)
        weights /= weights.sum()

        grad_b = np.zeros_like(beta)
        active = np.flatnonzero(np.abs(derivative) > 1e-18)
        for s in active:
            for i, axis in model.supports[s]:
                grad_b[:, i, axis] += derivative[s] * weights * g[:, s]
        grad_b /= np.clip(beta, 1e-12, None)
        grad_b /= np.max(np.abs(grad_b)) + 1e-30
        beta *= np.exp(-eta * grad_b)
        beta = np.clip(beta, 1e-12, None)
        beta /= beta.sum(axis=-1, keepdims=True)
    return best[0], best[1]


# --------------------------------------------------------------------------- #
# Full solve for one prospective trial
# --------------------------------------------------------------------------- #
def solve_track_b(
    trial: TrialEntropy,
    *,
    restarts=4,
    iters=500,
    refinement_iters=100,
):
    started = time.perf_counter()
    terms, coeffs, term_indices, supp = load_problem()
    _, _, _, identity = C.load_hamiltonian(TASK)
    state = C.build_hidden_state()

    pilot_rng = np.random.default_rng(trial.pilot_entropy)
    pilot = run_pilot(state, terms, coeffs, supp, pilot_rng)
    measured = pilot.shots_per_term > 0
    reliable = measured & (
        np.abs(pilot.means) * np.sqrt(np.maximum(pilot.shots_per_term, 1.0)) >= 2.0
    )
    eshr = _shrink(pilot.means, pilot.shots_per_term, np.flatnonzero(measured))

    w_coef = coeffs**2 * (1.0 - np.clip(eshr**2, 0, 1))
    within_model = build_within_model(terms, coeffs, supp, eshr, pilot)

    def estimated_within(w, b):
        return within_model_value_and_derivative(w, b, within_model)[0]

    w, b = optimize_scheme(
        w_coef,
        supp,
        iters=iters,
        restarts=restarts,
        seed=trial.optimizer_entropy,
        score=estimated_within,
    )
    w, b = refine_scheme_with_covariance(
        w,
        b,
        within_model,
        iters=refinement_iters,
    )
    cv = select_control_variates(
        terms,
        coeffs,
        eshr,
        reliable,
        w,
        b,
        cap=MAX_CONTROL_VARIATES,
    )
    for _ in range(2):
        clustered_model = build_clustered_model(
            terms,
            coeffs,
            supp,
            eshr,
            pilot,
            control_means=cv,
        )
        w, b = refine_scheme_with_covariance(
            w,
            b,
            clustered_model,
            iters=refinement_iters,
        )
        cv = select_control_variates(
            terms,
            coeffs,
            eshr,
            reliable,
            w,
            b,
            cap=MAX_CONTROL_VARIATES,
        )

    # Mirror the public lock boundary before hidden truth is consulted.  The
    # feasibility certificate must evaluate the same canonical probability law
    # and rounded controls that qsim would execute, not the optimizer's raw
    # floating-point arrays.
    control_entries = [
        {"term_index": int(term_indices[j]), "mean": float(mean)}
        for j, mean in enumerate(cv)
        if mean != 0.0
    ]
    canonical = C.canonical_request(
        w,
        b,
        control_entries,
        TASK,
        C.hamiltonian_sha256(TASK),
    )
    w, b, locked_controls = C.locked_request_from_canonical(canonical)
    by_term_index = {int(term_index): j for j, term_index in enumerate(term_indices)}
    cv = np.zeros(len(terms), dtype=float)
    for entry in locked_controls:
        cv[by_term_index[entry["term_index"]]] = entry["mean"]

    # Hidden truth is used only now, after the public-data-derived design is locked.
    expect, _ = C.make_sparse_expectation()
    true_means = np.array([expect(p) for p in terms])
    res = C.private_cluster_variance(terms, coeffs, true_means, cv, expect, w, b)
    true_energy = identity + float(np.dot(coeffs, true_means))
    energy, cluster_se = run_production(
        state,
        terms,
        coeffs,
        identity,
        supp,
        cv,
        w,
        b,
        np.random.default_rng(trial.production_basis_entropy),
        np.random.default_rng(trial.production_outcome_entropy),
    )
    energy_error = abs(energy - true_energy)
    return {
        "trial_id": trial.trial_id,
        "entropy_commitment": trial.commitment,
        "sigma_private": res["sigma"],
        "within": res["within"],
        "between": res["between"],
        "pilot_settings": pilot.num_settings,
        "pilot_shots_per_setting": sorted(set(pilot.row_shots)),
        "pilot_setting_shot_counts": {
            str(shots): count for shots, count in sorted(Counter(pilot.row_shots).items())
        },
        "pilot_total_shots": sum(pilot.row_shots),
        "evaluator_calls": 0,
        "nonzero_control_variates": int(np.sum(np.abs(cv) > 0)),
        "production_settings": PRODUCTION_SETTINGS,
        "production_shots_per_setting": PRODUCTION_SHOTS_PER_SETTING,
        "production_energy_hartree": energy,
        "production_energy_error_hartree": energy_error,
        "production_cluster_se_hartree_1sigma": cluster_se,
        "production_1sigma_covers": energy_error <= cluster_se,
        "wall_clock_s": time.perf_counter() - started,
    }


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires at least one trial")
    p = successes / total
    z2 = _WILSON_Z_95**2
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    radius = _WILSON_Z_95 * np.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total**2)) / denominator
    return float(center - radius), float(center + radius)


def summarize_feasibility(track_a_v_haar, results):
    """Evaluate the complete release gate."""
    ordered = sorted(results, key=lambda result: result["trial_id"])
    n_trials = len(ordered)
    sig = np.array([result["sigma_private"] for result in ordered])
    energy_errors = np.array([result["production_energy_error_hartree"] for result in ordered])
    coverage_count = int(sum(result["production_1sigma_covers"] for result in ordered))
    coverage = coverage_count / n_trials
    coverage_interval = _wilson_interval(coverage_count, n_trials)
    median_wall_clock = float(np.median([result["wall_clock_s"] for result in ordered]))
    npass = int((np.round(sig, 4) <= GATE).sum())
    pass_rate = npass / n_trials
    pass_interval = _wilson_interval(npass, n_trials)
    release_ready = (
        n_trials >= 50
        and round(track_a_v_haar, 2) <= 600.0
        and pass_rate >= 0.90
        and pass_interval[0] >= TRACK_B_WILSON_LOWER_FLOOR
        and coverage_interval[0] <= NOMINAL_ONE_SIGMA_COVERAGE <= coverage_interval[1]
        and median_wall_clock < 1800.0
    )
    return {
        "n_trials": n_trials,
        "track_a_v_haar": float(track_a_v_haar),
        "track_a_passed": round(track_a_v_haar, 2) <= 600.0,
        "track_b_pass_count": npass,
        "track_b_pass_rate": pass_rate,
        "track_b_pass_rate_wilson_95": list(pass_interval),
        "track_b_wilson_lower_floor": TRACK_B_WILSON_LOWER_FLOOR,
        "track_b_sigma_median": float(np.median(sig)),
        "track_b_sigma_mean": float(sig.mean()),
        "track_b_sigma_min": float(sig.min()),
        "track_b_sigma_max": float(sig.max()),
        "production_median_energy_error_hartree": float(np.median(energy_errors)),
        "production_1sigma_coverage": coverage,
        "production_1sigma_coverage_wilson_95": list(coverage_interval),
        "production_1sigma_nominal": NOMINAL_ONE_SIGMA_COVERAGE,
        "median_wall_clock_s": median_wall_clock,
        "release_ready": release_ready,
    }


def _print_result(result):
    ok = "PASS" if round(result["sigma_private"], 4) <= GATE else "fail"
    print(
        f"  {result['trial_id']}: sigma={result['sigma_private']:.6f} "
        f"within={result['within']:.1f} between={result['between']:.4f} "
        f"settings={result['pilot_settings']} shots/set={result['pilot_shots_per_setting']} "
        f"CVs={result['nonzero_control_variates']} "
        f"energy_err={result['production_energy_error_hartree']:.4f} "
        f"covered={result['production_1sigma_covers']} {ok}",
        flush=True,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract_digests() -> dict[str, str]:
    repo = C.hamiltonian_csv_path(TASK).parents[4]
    qtype_dir = repo / "src" / "qiqcbench" / "qsim" / "qtypes" / "composite_measurement_estimation"
    return {
        "hamiltonian": C.hamiltonian_sha256(TASK),
        "public_device_spec": _sha256(
            repo / "configs" / "devices" / "clbcs_h2o_14q_v0.public.yaml"
        ),
        "hidden_device_config": _sha256(
            repo / "configs" / "devices" / "clbcs_h2o_14q_v0.hidden.example.yaml"
        ),
        "instruction": _sha256(repo / "harbor_tasks" / TASK / "instruction.md"),
        "hidden_scorer": _sha256(
            repo / "configs" / "task_materials" / TASK / "hidden" / "hidden_scorer.yaml"
        ),
        "hidden_construction": _sha256(Path(C.__file__)),
        "scientific_scorer": _sha256(
            repo / "src" / "qiqcbench" / "qsim" / "hidden_dynamics" / TASK / "scorer.py"
        ),
        "device_schema": _sha256(qtype_dir / "device.py"),
        "simulator_backend": _sha256(qtype_dir / "backend.py"),
        "measurement_engine": _sha256(qtype_dir / "engine.py"),
        "task_config": _sha256(repo / "configs" / "tasks" / f"{TASK}.yaml"),
        "harbor_task": _sha256(repo / "harbor_tasks" / TASK / "task.toml"),
        "harbor_compose": _sha256(
            repo / "harbor_tasks" / TASK / "environment" / "docker-compose.yaml"
        ),
        "task_edition_v4": _sha256(
            repo / "configs" / "evaluation" / "task_editions" / TASK / "v4.yaml"
        ),
        "task_edition_v5": _sha256(
            repo / "configs" / "evaluation" / "task_editions" / TASK / "v5.yaml"
        ),
        "verifier_revision_v9": _sha256(
            repo / "configs" / "evaluation" / "verifier_revisions" / TASK / "v9.yaml"
        ),
        "verifier_revision_v14": _sha256(
            repo / "configs" / "evaluation" / "verifier_revisions" / TASK / "v14.yaml"
        ),
        "reference_solver": _sha256(Path(__file__)),
    }


def _write_certificate(
    path: Path,
    track_a_v_haar: float,
    results: list[dict],
    summary: dict,
    panel: ProspectivePanel,
    frozen_contract_digests: dict[str, str],
) -> None:
    current_contract_digests = _contract_digests()
    if current_contract_digests != frozen_contract_digests:
        raise ValueError("the scientific contract changed while the prospective panel was running")
    if panel.contract_digest != contract_digest(current_contract_digests):
        raise ValueError("prospective panel is not bound to the current scientific contract")
    if len(panel.trials) < 50:
        raise ValueError("a prospective release certificate requires at least 50 trials")
    ensure_certificate_slot(path, panel.contract_digest)

    ordered_results = sorted(results, key=lambda item: item["trial_id"])
    expected_commitments = {trial.trial_id: trial.commitment for trial in panel.trials}
    observed_commitments = {
        result["trial_id"]: result["entropy_commitment"] for result in ordered_results
    }
    if len(ordered_results) != len(panel.trials) or observed_commitments != expected_commitments:
        raise ValueError("prospective result set does not match the complete drawn panel")
    if summary != summarize_feasibility(track_a_v_haar, ordered_results):
        raise ValueError("prospective summary does not match the complete result set")

    certificate = {
        "schema_version": 3,
        "task_id": TASK,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "solver_contract": "public_observation_only_full_clustered_v3",
        "prospective_panel": {
            "purpose": "prospective_release_validation",
            "selection_rule": "fresh_os_entropy_after_contract_freeze_v1",
            "contract_digest": panel.contract_digest,
            "panel_id": panel.panel_id,
            "panel_entropy_commitment": panel.panel_entropy_commitment,
            "entropy_bits_per_stream": 128,
            "stream_domains": list(_STREAM_DOMAINS),
            "raw_entropy_retained": False,
            "outcome_dependent_filtering": False,
            "reusable_after_contract_change": False,
        },
        "contract_digests": current_contract_digests,
        "track_a_v_haar": float(track_a_v_haar),
        "results": ordered_results,
        "summary": summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n_trials", nargs="?", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.n_trials <= 0:
        parser.error("n_trials must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.output is not None and args.n_trials < 50:
        parser.error("a prospective release certificate requires at least 50 trials")

    frozen_contract_digests = _contract_digests()
    frozen_contract_digest = contract_digest(frozen_contract_digests)
    if args.output is not None:
        ensure_certificate_slot(args.output, frozen_contract_digest)
    panel = draw_prospective_panel(args.n_trials, frozen_contract_digests)
    print(
        f"Prospective panel {panel.panel_id} bound to contract {panel.contract_digest[:16]}...",
        flush=True,
    )

    track_a_v_haar = solve_track_a()
    print(f"Track A public warm-start refinement: V_Haar={track_a_v_haar:.6f}")
    results = []
    if args.workers == 1:
        for trial in panel.trials:
            result = solve_track_b(trial)
            results.append(result)
            _print_result(result)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(solve_track_b, trial): trial.trial_id for trial in panel.trials
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                _print_result(result)

    summary = summarize_feasibility(track_a_v_haar, results)
    print("\n=== " + json.dumps(summary, sort_keys=True) + " ===")
    if args.output is not None:
        _write_certificate(
            args.output,
            track_a_v_haar,
            results,
            summary,
            panel,
            frozen_contract_digests,
        )
        print(f"wrote {args.output}")
    release_ready = bool(summary["release_ready"])
    return 0 if args.n_trials < 50 or release_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
