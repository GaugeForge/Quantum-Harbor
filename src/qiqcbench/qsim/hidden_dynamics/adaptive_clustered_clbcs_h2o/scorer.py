"""Hidden scorer for ``adaptive_clustered_clbcs_h2o``.

The verifier RECOMPUTES every scored quantity (it does not trust agent-reported
scalars):

* Track A ``V_Haar`` from the submitted scheme + the public Hamiltonian;
* Track B ``sigma_private`` from the LOCKED scheme + control variates + the
  reconstructed hidden state;
* the production energy and cluster standard error from the returned raw counts.

It then assigns diagnostic tiers/points and returns a binary reward from the
direct conjunction of the five scientific/evidence checks.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from ruamel.yaml import YAML

from qiqcbench.qsim.hidden_dynamics.adaptive_clustered_clbcs_h2o import construction as C
from qiqcbench.qsim.hidden_dynamics.adaptive_clustered_clbcs_h2o.schema import (
    AdaptiveClbcsHiddenScorer,
    TierThresholds,
)

_yaml = YAML(typ="safe")


class DigestMismatch(ValueError):
    """Raised when the public Hamiltonian does not match the recorded digest."""


class SubmissionValidationError(ValueError):
    """Raised only when model-owned final-answer fields cannot be normalized."""


def _answer_section(answer: dict[str, Any], name: str) -> dict[str, Any]:
    value = answer.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SubmissionValidationError(f"answer.{name} must be an object")
    return dict(value)


def _finite_reported_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SubmissionValidationError(f"{field_name} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SubmissionValidationError(f"{field_name} must be a finite number") from exc
    if not np.isfinite(parsed):
        raise SubmissionValidationError(f"{field_name} must be a finite number")
    return parsed


def _numeric_array(value: object, *, field_name: str) -> np.ndarray:
    def validate_tree(item: object, path: str) -> None:
        if isinstance(item, list):
            for index, child in enumerate(item):
                validate_tree(child, f"{path}[{index}]")
            return
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise SubmissionValidationError(f"{path} must be a JSON number")

    if not isinstance(value, list):
        raise SubmissionValidationError(f"{field_name} must be an array")
    validate_tree(value, field_name)
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SubmissionValidationError(f"{field_name} must be numeric") from exc
    if not np.all(np.isfinite(result)):
        raise SubmissionValidationError(f"{field_name} must contain finite numbers")
    return result


def _normalize_submission(answer: dict[str, Any], scorer: AdaptiveClbcsHiddenScorer) -> dict:
    """Normalize only agent-controlled fields before hidden scientific recomputation."""

    if not isinstance(answer, dict):
        raise SubmissionValidationError("final answer must be an object")
    normalized = dict(answer)
    track_a = _answer_section(answer, "track_a")
    track_b = _answer_section(answer, "track_b")
    track_b_result = _answer_section(answer, "track_b_result")

    if "mixture_weights" in track_a:
        if "local_basis_probabilities_xyz" not in track_a:
            raise SubmissionValidationError(
                "answer.track_a.local_basis_probabilities_xyz is required with mixture_weights"
            )
        weights = _numeric_array(
            track_a["mixture_weights"],
            field_name="answer.track_a.mixture_weights",
        )
        beta = _numeric_array(
            track_a["local_basis_probabilities_xyz"],
            field_name="answer.track_a.local_basis_probabilities_xyz",
        )
        try:
            C.validate_scheme(weights, beta, num_components=scorer.num_components)
        except ValueError as exc:
            raise SubmissionValidationError(f"answer.track_a scheme is invalid: {exc}") from exc
        track_a["mixture_weights"] = weights.tolist()
        track_a["local_basis_probabilities_xyz"] = beta.tolist()

    for field_name in (
        "energy_raw_hartree",
        "uncertainty_hartree_1sigma_cluster",
    ):
        if track_b_result.get(field_name) is not None:
            track_b_result[field_name] = _finite_reported_float(
                track_b_result[field_name],
                field_name=f"answer.track_b_result.{field_name}",
            )

    normalized["track_a"] = track_a
    normalized["track_b"] = track_b
    normalized["track_b_result"] = track_b_result
    return normalized


def load_hidden_scorer(path: Path) -> AdaptiveClbcsHiddenScorer:
    raw = _yaml.load(Path(path).read_text(encoding="utf-8"))
    return AdaptiveClbcsHiddenScorer.model_validate(raw)


def verify_hamiltonian_digest(scorer: AdaptiveClbcsHiddenScorer, public_dir: Path) -> None:
    path = Path(public_dir) / C.HAMILTONIAN_FILENAME
    if not path.is_file():
        raise DigestMismatch(f"public Hamiltonian missing under {public_dir}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != scorer.public_hamiltonian_sha256:
        raise DigestMismatch(
            f"public Hamiltonian digest mismatch: {actual} != {scorer.public_hamiltonian_sha256}"
        )


def _tier(value: float, thr: TierThresholds) -> str:
    if value <= thr.gold:
        return "gold"
    if value <= thr.silver:
        return "silver"
    if value <= thr.bronze:
        return "bronze"
    if value <= thr.pass_:
        return "pass"
    return "none"


def _as_scheme(scheme: dict) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(scheme["mixture_weights"], dtype=float)
    beta = np.asarray(scheme["local_basis_probabilities_xyz"], dtype=float)
    return weights, beta


def _control_vector(control_entries: list[dict], term_indices: tuple[int, ...]) -> np.ndarray:
    by_index = {int(i): j for j, i in enumerate(term_indices)}
    control = np.zeros(len(term_indices), dtype=float)
    for entry in control_entries:
        ti = int(entry["term_index"])
        if ti not in by_index:
            raise ValueError(f"control variate term_index {ti} not a non-identity term")
        control[by_index[ti]] = float(entry["mean"])
    return control


# --------------------------------------------------------------------------- #
# Track A / Track B recompute
# --------------------------------------------------------------------------- #
def score_track_a(scheme: dict, scorer: AdaptiveClbcsHiddenScorer) -> dict:
    terms, coeffs, _, _ = C.load_hamiltonian(scorer.task_id)
    weights, beta = _as_scheme(scheme)
    weights, beta = C.normalize_scheme(weights, beta, num_components=scorer.num_components)
    _, hmin = C.coverage_floor(terms, weights, beta)
    if hmin < scorer.coverage_floor:
        return {"valid": False, "reason": f"coverage floor violated ({hmin:.2e})"}
    v_haar = C.track_a_variance(terms, coeffs, weights, beta)
    v_cmp = round(v_haar, scorer.track_a_comparison_decimals)
    tier = _tier(v_cmp, scorer.track_a_thresholds)
    return {
        "valid": True,
        "v_haar": v_haar,
        "v_haar_compared": v_cmp,
        "tier": tier,
        "passed": v_cmp <= scorer.track_a_thresholds.pass_,
    }


def score_track_b(
    scheme: dict, control_entries: list[dict], scorer: AdaptiveClbcsHiddenScorer
) -> dict:
    terms, coeffs, term_indices, _ = C.load_hamiltonian(scorer.task_id)
    weights, beta = _as_scheme(scheme)
    weights, beta = C.normalize_scheme(weights, beta, num_components=scorer.num_components)
    _, hmin = C.coverage_floor(terms, weights, beta)
    if hmin < scorer.coverage_floor:
        return {"valid": False, "reason": f"coverage floor violated ({hmin:.2e})"}
    control = _control_vector(control_entries, term_indices)
    expectation, _ = C.make_sparse_expectation()
    true_means = np.array([expectation(p) for p in terms])
    res = C.private_cluster_variance(
        terms,
        coeffs,
        true_means,
        control,
        expectation,
        weights,
        beta,
        production_settings=scorer.production_settings,
        production_shots=scorer.production_shots,
    )
    sigma_cmp = round(res["sigma"], scorer.track_b_comparison_decimals)
    tier = _tier(sigma_cmp, scorer.track_b_thresholds)
    return {
        "valid": True,
        "sigma_private": res["sigma"],
        "sigma_private_compared": sigma_cmp,
        "between": res["between"],
        "within": res["within"],
        "tier": tier,
        "passed": sigma_cmp <= scorer.track_b_thresholds.pass_,
        "nonzero_control_variates": int(np.sum(np.abs(control) > 0)),
    }


# --------------------------------------------------------------------------- #
# Energy + cluster standard error from raw counts
# --------------------------------------------------------------------------- #
def _term_supports(terms: tuple[str, ...]) -> list[tuple[tuple[int, ...], str]]:
    """For each term, ((positions...), axes_string) over non-identity qubits."""
    out = []
    for p in terms:
        positions = tuple(i for i, ch in enumerate(p) if ch != "I")
        axes = "".join(p[i] for i in positions)
        out.append((positions, axes))
    return out


def recompute_energy_and_cluster_se(
    scheme: dict,
    control_entries: list[dict],
    production_rows: list[dict],
    scorer: AdaptiveClbcsHiddenScorer,
) -> dict:
    """Pinned inverse-coverage control-variate estimator from raw counts (Section 3b)."""
    terms, coeffs, term_indices, identity_coeff = C.load_hamiltonian(scorer.task_id)
    weights, beta = _as_scheme(scheme)
    weights, beta = C.normalize_scheme(weights, beta, num_components=scorer.num_components)
    h = np.array([C.coverage(p, weights, beta) for p in terms])
    control = _control_vector(control_entries, term_indices)
    supports = _term_supports(terms)
    const = float(np.dot(coeffs, control))  # sum_j c_j m_j

    z = np.empty(len(production_rows), dtype=float)
    for b, row in enumerate(production_rows):
        basis = row["basis"]
        counts: dict[str, int] = row["counts"]
        bitstrings = list(counts.keys())
        cvec = np.array([counts[bs] for bs in bitstrings], dtype=float)
        r_total = float(cvec.sum())
        # bits matrix: (n_distinct, NQ), bit at qubit i = int(bitstring[i])
        bits = np.array([[int(ch) for ch in bs] for bs in bitstrings], dtype=np.int8)
        corr = 0.0
        for j, (positions, axes) in enumerate(supports):
            # covered iff basis matches every non-identity axis of P_j
            if any(basis[pos] != ax for pos, ax in zip(positions, axes, strict=True)):
                continue
            parity = bits[:, list(positions)].sum(axis=1) & 1
            sign = 1 - 2 * parity  # +1 / -1
            mu_avg = float(np.dot(cvec, sign) / r_total)
            corr += (coeffs[j] / h[j]) * (mu_avg - control[j])
        z[b] = const + corr

    energy = identity_coeff + float(z.mean())
    b_count = len(z)
    if b_count >= 2:
        sigma_cluster = float(np.sqrt(np.sum((z - z.mean()) ** 2) / (b_count * (b_count - 1))))
    else:  # pragma: no cover
        sigma_cluster = float("nan")
    return {"energy": energy, "sigma_cluster": sigma_cluster, "exact_energy": _exact_energy(scorer)}


def _exact_energy(scorer: AdaptiveClbcsHiddenScorer) -> float:
    terms, coeffs, _, identity_coeff = C.load_hamiltonian(scorer.task_id)
    expectation, _ = C.make_sparse_expectation()
    means = np.array([expectation(p) for p in terms])
    return identity_coeff + float(np.dot(coeffs, means))


# --------------------------------------------------------------------------- #
# Top-level scoring
# --------------------------------------------------------------------------- #
def score_submission(
    *,
    answer: dict,
    locked_scheme: dict,
    locked_control_entries: list[dict],
    production_rows: list[dict],
    request_digest: str | None,
    scorer: AdaptiveClbcsHiddenScorer,
    public_dir: Path,
) -> dict:
    """Return a score report dict including ``reward`` (0/1) and per-component tiers."""
    verify_hamiltonian_digest(scorer, public_dir)
    answer = _normalize_submission(answer, scorer)

    penalties: list[str] = []
    if answer.get("track_b_result", {}).get("energy_raw_hartree") is None:
        penalties.append("missing track_b_result.energy_raw_hartree")

    # Track A is the one scientific object supplied in the final answer. Track
    # B's scheme and control variates are supplied once, at the qsim lock, and
    # arrive here only through verifier-validated qsim evidence.
    track_a_scheme = answer.get("track_a", {})
    if "mixture_weights" not in track_a_scheme:
        penalties.append("missing Track-A scheme")

    # The sole answer-to-production binding. The driver has already recomputed
    # this digest from the locked qsim evidence; no duplicate design is trusted
    # or required here.
    tb = answer.get("track_b", {})
    reported_digest = tb.get("request_digest")
    digest_matches = bool(request_digest is not None and reported_digest == request_digest)
    if not digest_matches:
        penalties.append(
            f"final-answer request_digest {reported_digest!r} != locked {request_digest!r}"
        )

    # --- Track A ---
    track_a = (
        score_track_a(track_a_scheme, scorer)
        if "mixture_weights" in track_a_scheme
        else {
            "valid": False,
            "reason": "no scheme",
        }
    )
    if not track_a.get("valid"):
        penalties.append(f"Track-A invalid: {track_a.get('reason')}")

    # --- Track B (primary) ---
    track_b = score_track_b(locked_scheme, locked_control_entries, scorer)
    if not track_b.get("valid"):
        raise RuntimeError(
            "verifier-validated locked Track-B evidence became scientifically invalid: "
            f"{track_b.get('reason')}"
        )

    # --- energy + uncertainty from raw counts ---
    energy_block: dict[str, Any] = {}
    uncertainty_block: dict[str, Any] = {}
    if production_rows:
        rc = recompute_energy_and_cluster_se(
            locked_scheme, locked_control_entries, production_rows, scorer
        )
        e_hat, sigma_cluster, e_star = rc["energy"], rc["sigma_cluster"], rc["exact_energy"]
        reported_e = answer.get("track_b_result", {}).get("energy_raw_hartree")
        reported_error = abs(float(reported_e) - e_hat) if reported_e is not None else None
        reported_matches = bool(
            reported_error is not None
            and reported_error <= scorer.energy_report_abs_tolerance_hartree
        )
        hidden_truth_error = abs(e_hat - e_star)
        energy_block = {
            "energy_recomputed": e_hat,
            "reported": reported_e,
            "reported_abs_error": reported_error,
            "report_tolerance": scorer.energy_report_abs_tolerance_hartree,
            "reported_matches_recompute": reported_matches,
            "exact_energy": e_star,
            # Keep ``abs_error`` for historical report readers; from scorer v2
            # onward it is explicitly a hidden-truth diagnostic, not a gate.
            "abs_error": hidden_truth_error,
            "hidden_truth_abs_error": hidden_truth_error,
            "passed": reported_matches,
        }
        if reported_e is not None and not reported_matches:
            penalties.append(
                "reported energy does not reproduce the pinned estimator on the "
                f"recorded counts: |delta|={reported_error:.6g} > "
                f"{scorer.energy_report_abs_tolerance_hartree:.6g} Hartree"
            )

        # Reported uncertainty must reproduce the canonical same-data cluster SE.
        reported_u = answer.get("track_b_result", {}).get("uncertainty_hartree_1sigma_cluster")
        sigma_private = track_b["sigma_private"]
        u_block: dict[str, Any] = {
            "sigma_cluster_ref": sigma_cluster,
            "sigma_private": sigma_private,
            "reported": reported_u,
            "report_tolerance": scorer.uncertainty_report_abs_tolerance_hartree,
        }
        if reported_u is None:
            penalties.append("missing reported cluster uncertainty")
            u_block["passed"] = False
        else:
            reported_u = float(reported_u)
            reported_error = abs(reported_u - sigma_cluster)
            reported_matches = bool(
                reported_error <= scorer.uncertainty_report_abs_tolerance_hartree
            )
            u_block["reported_abs_error"] = reported_error
            u_block["reported_matches_recompute"] = reported_matches
            if not reported_matches:
                penalties.append(
                    "reported cluster uncertainty does not reproduce the canonical "
                    f"same-count setting-cluster SE: |delta|={reported_error:.6g} > "
                    f"{scorer.uncertainty_report_abs_tolerance_hartree:.6g} Hartree"
                )
        u_block["passed"] = bool(u_block.get("reported_matches_recompute", False))
        uncertainty_block = u_block

    # Diagnostic component score. Binary admission below is deliberately not
    # inferred from this score, its points, or any hidden cap state.
    total = 0
    if not penalties:
        total += scorer.points_validity
    total += _tier_points(track_a.get("tier"), scorer.points_track_a)
    total += _tier_points(track_b.get("tier"), scorer.points_track_b)
    if energy_block.get("passed"):
        total += scorer.points_energy
    if uncertainty_block.get("passed"):
        total += scorer.points_uncertainty

    admission_checks = {
        "track_a_quality": bool(track_a.get("valid") and track_a.get("passed")),
        "track_b_quality": bool(track_b.get("valid") and track_b.get("passed")),
        "locked_request_binding": digest_matches,
        "energy_readback": bool(energy_block.get("passed")),
        "cluster_uncertainty_readback": bool(uncertainty_block.get("passed")),
    }
    reward = int(all(admission_checks.values()))

    return {
        "task_id": scorer.task_id,
        "reward": reward,
        "total_score": total,
        "penalties": penalties,
        "admission_checks": admission_checks,
        "track_a": track_a,
        "track_b": track_b,
        "energy": energy_block,
        "uncertainty": uncertainty_block,
        "request_digest": request_digest,
    }


def _tier_points(tier: str | None, max_points: int) -> int:
    return {
        "gold": max_points,
        "silver": int(round(max_points * 0.85)),
        "bronze": int(round(max_points * 0.70)),
        "pass": int(round(max_points * 0.50)),
    }.get(tier or "none", 0)
