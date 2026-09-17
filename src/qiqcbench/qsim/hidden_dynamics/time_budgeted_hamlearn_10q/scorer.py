"""Scoring authority for ``time_budgeted_hamlearn_10q``.

Loads the hidden scorer YAML, fails closed if the public materials do not match
their recorded digests, then scores the submitted answer:

* L∞ and relative L2 over the 111-term vector (raw or refined pair,
  §1 pair-atomic rule);
* support-recovery and single-realization uncertainty diagnostics (§9b-§9d);
* **evidence-owned** budget (§9e): the authoritative ``T_used`` is summed from
  accepted-probe evidence and the structured verifiable activity claim is
  cross-checked;
* descriptive finite-time compatibility diagnostics between the submitted
  Hamiltonian and the realized raw outcomes;
* §9f binary-pass conditions and §9g failure diagnostics, emitted as a ScoreReport.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

from ruamel.yaml import YAML

from qiqcbench.qsim.hidden_dynamics.time_budgeted_hamlearn_10q.construction import (
    build_stale_vector,
)
from qiqcbench.qsim.hidden_dynamics.time_budgeted_hamlearn_10q.dictionary import (
    N_TERMS,
    term_to_index,
)
from qiqcbench.qsim.hidden_dynamics.time_budgeted_hamlearn_10q.evidence import certify_evidence
from qiqcbench.qsim.hidden_dynamics.time_budgeted_hamlearn_10q.oracle import ProbeEvidence
from qiqcbench.qsim.hidden_dynamics.time_budgeted_hamlearn_10q.schema import (
    HamLearnAnswer,
    HamLearnHiddenScorer,
)

__all__ = [
    "DigestMismatch",
    "ScoreReport",
    "linf",
    "relative_l2",
    "load_hidden_scorer",
    "score",
    "score_from_materials",
    "verify_public_material_digests",
]

_yaml = YAML()
_BUDGET_TOL_US = 1e-6
_REPORT_TOL_US = 1e-6


class DigestMismatch(ValueError):
    """Raised when a public material file does not match its recorded digest."""


def load_hidden_scorer(path: Path) -> HamLearnHiddenScorer:
    """Load and validate the hidden scorer YAML at ``path``."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Hidden scorer YAML missing at {path}")
    raw = _yaml.load(Path(path).read_text(encoding="utf-8"))
    return HamLearnHiddenScorer.model_validate(raw)


def verify_public_material_digests(scorer: HamLearnHiddenScorer, public_dir: Path) -> None:
    """Confirm every public material matches the scorer's digest (fails closed)."""
    for filename, expected in scorer.public_material_digests.items():
        path = Path(public_dir) / filename
        if not path.is_file():
            raise DigestMismatch(f"Public material {filename!r} missing under {public_dir}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise DigestMismatch(
                f"Public material {filename!r} digest mismatch: expected {expected}, got {actual}"
            )


def linf(omega_hat: list[float], omega_star: list[float]) -> float:
    """Worst coefficient-wise absolute error."""
    return max(abs(a - b) for a, b in zip(omega_hat, omega_star, strict=True))


def relative_l2(omega_hat: list[float], omega_star: list[float]) -> float:
    """Relative coefficient L2 error, equal to relative Hamiltonian Frobenius error."""
    denominator = math.sqrt(sum(value * value for value in omega_star))
    if denominator == 0.0:
        raise ValueError("relative L2 error is undefined for an all-zero hidden vector")
    numerator = math.sqrt(
        sum((hat - star) ** 2 for hat, star in zip(omega_hat, omega_star, strict=True))
    )
    return numerator / denominator


@dataclass
class ScoreReport:
    """Mirrors the VQE ``score_report.json`` shape."""

    task_id: str
    linf: float
    relative_l2: float
    relative_l2_within_threshold: bool
    level: str  # gold | silver | bronze | pass | fail
    scored_vector_kind: str  # raw | refined
    support_true_total: int
    support_detected_true: int
    support_false_positives: int
    uncertainty_covered: int
    uncertainty_total: int
    uncertainty_median_1sigma: float
    budget_us: float
    budget_used_us: float
    activity_summary_evolution_time_us: float
    within_budget: bool
    accepted_row_count: int
    ran_nonzero_time_probe: bool
    evidence_min_evolve_time_us: float | None
    evidence_max_evolve_time_us: float | None
    verifiable_activity_consistent: bool
    evidence_mean_binomial_deviance: float | None
    evidence_p90_binomial_deviance: float | None
    evidence_data_compatible: bool
    binary_pass: bool
    reward_binary: int
    penalties: list[str] = field(default_factory=list)


def _level(value: float, scorer: HamLearnHiddenScorer) -> str:
    if value <= scorer.linf_gold:
        return "gold"
    if value <= scorer.linf_silver:
        return "silver"
    if value <= scorer.linf_bronze:
        return "bronze"
    if value <= scorer.linf_pass:
        return "pass"
    return "fail"


def _select_scored_vector(answer: HamLearnAnswer) -> tuple[list[float], list[float], str]:
    """Pair-atomic refined rule (§1): refined iff both vector AND sigma present."""
    if (
        answer.omega_vector_refined_rad_per_us is not None
        and answer.omega_vector_refined_rad_per_us_1sigma is not None
    ):
        return (
            answer.omega_vector_refined_rad_per_us,
            answer.omega_vector_refined_rad_per_us_1sigma,
            "refined",
        )
    return (
        answer.omega_vector_raw_rad_per_us,
        answer.omega_vector_raw_rad_per_us_1sigma,
        "raw",
    )


def _support_diagnostic(omega_hat: list[float], scorer: HamLearnHiddenScorer) -> tuple[int, int]:
    """Thresholded strong-term detections and true-zero false positives (§9b/§9c)."""
    tau = scorer.support_tau
    true_support = set(scorer.support_terms)
    detected_true = sum(1 for t in true_support if abs(omega_hat[term_to_index(t)]) > tau)
    false_positives = sum(
        1 for i in range(N_TERMS) if scorer.omega_star[i] == 0.0 and abs(omega_hat[i]) > tau
    )
    return detected_true, false_positives


def _uncertainty_coverage(
    omega_hat: list[float], sigma: list[float], omega_star: list[float]
) -> int:
    """Count of coefficients satisfying ``|err| <= max(2 sigma, 0.025)`` (§9d)."""
    return sum(
        1
        for hat, sig, star in zip(omega_hat, sigma, omega_star, strict=True)
        if abs(hat - star) <= max(2.0 * sig, 0.025)
    )


def score(
    answer: HamLearnAnswer,
    scorer: HamLearnHiddenScorer,
    evidence: ProbeEvidence,
) -> ScoreReport:
    """Score a validated answer against the hidden scorer + oracle evidence."""
    omega_star = scorer.omega_star
    omega_hat, sigma, kind = _select_scored_vector(answer)

    value = linf(omega_hat, omega_star)
    relative_l2_value = relative_l2(omega_hat, omega_star)
    relative_l2_within_threshold = relative_l2_value <= scorer.relative_l2_pass
    level = _level(value, scorer)

    detected_true, false_positives = _support_diagnostic(omega_hat, scorer)
    covered = _uncertainty_coverage(omega_hat, sigma, omega_star)
    median_sigma = sorted(sigma)[len(sigma) // 2]

    # Evidence-owned budget: authoritative T_used is the oracle's accepted sum.
    budget_used = evidence.budget_used_us
    activity = answer.activity_summary.verifiable
    self_report = activity.total_accepted_evolution_time_us
    within_budget = budget_used <= scorer.budget_us + _BUDGET_TOL_US
    accepted = evidence.accepted_row_count
    ran_probe = evidence.ran_nonzero_time_probe
    accepted_times = [float(row["evolve_time_us"]) for row in evidence.accepted_rows]
    evidence_min_time = min(accepted_times) if accepted_times else None
    evidence_max_time = max(accepted_times) if accepted_times else None
    certification = certify_evidence(omega_hat, evidence.accepted_rows)

    penalties: list[str] = []
    report_consistent = True
    if abs(budget_used - self_report) > _REPORT_TOL_US:
        report_consistent = False
        penalties.append(
            "activity_summary.verifiable total evolution time "
            f"{self_report:.4f} us != evidence {budget_used:.4f} us"
        )
    if activity.accepted_probe_rows != accepted:
        report_consistent = False
        penalties.append(
            "activity_summary.verifiable accepted_probe_rows "
            f"{activity.accepted_probe_rows} != evidence {accepted}"
        )
    reported_min = activity.min_accepted_evolve_time_us
    reported_max = activity.max_accepted_evolve_time_us
    if (
        evidence_min_time is None
        or evidence_max_time is None
        or abs(reported_min - evidence_min_time) > _REPORT_TOL_US
        or abs(reported_max - evidence_max_time) > _REPORT_TOL_US
    ):
        report_consistent = False
        penalties.append(
            "activity_summary.verifiable probe-time range "
            f"[{reported_min:.4f}, {reported_max:.4f}] us != evidence "
            f"[{evidence_min_time!r}, {evidence_max_time!r}] us"
        )
    if not within_budget:
        penalties.append(f"over budget: used {budget_used:.4f} us > {scorer.budget_us} us")
    if accepted > scorer.max_probe_rows:
        penalties.append(f"too many accepted probe rows: {accepted} > {scorer.max_probe_rows}")
    if not ran_probe:
        penalties.append("no accepted nonzero-time probe ran")
    # Stale-only diagnostic (the L∞ metric already fails it; flag for clarity).
    if linf(omega_hat, build_stale_vector().tolist()) < 0.02 and value > scorer.linf_pass:
        penalties.append("submitted vector matches the stale notebook model")

    binary_pass = bool(
        value <= scorer.linf_pass
        and relative_l2_within_threshold
        and within_budget
        and accepted <= scorer.max_probe_rows
        and ran_probe
        and len(omega_hat) == N_TERMS
        and report_consistent
    )

    return ScoreReport(
        task_id="time_budgeted_hamlearn_10q",
        linf=value,
        relative_l2=relative_l2_value,
        relative_l2_within_threshold=relative_l2_within_threshold,
        level=level,
        scored_vector_kind=kind,
        support_true_total=len(scorer.support_terms),
        support_detected_true=detected_true,
        support_false_positives=false_positives,
        uncertainty_covered=covered,
        uncertainty_total=N_TERMS,
        uncertainty_median_1sigma=median_sigma,
        budget_us=scorer.budget_us,
        budget_used_us=budget_used,
        activity_summary_evolution_time_us=self_report,
        within_budget=within_budget,
        accepted_row_count=accepted,
        ran_nonzero_time_probe=ran_probe,
        evidence_min_evolve_time_us=evidence_min_time,
        evidence_max_evolve_time_us=evidence_max_time,
        verifiable_activity_consistent=report_consistent,
        evidence_mean_binomial_deviance=certification.mean_binomial_deviance,
        evidence_p90_binomial_deviance=certification.p90_binomial_deviance,
        evidence_data_compatible=certification.data_compatible,
        binary_pass=binary_pass,
        reward_binary=int(binary_pass),
        penalties=penalties,
    )


def score_from_materials(
    answer: HamLearnAnswer,
    public_dir: Path,
    hidden_scorer_path: Path,
    evidence: ProbeEvidence,
) -> ScoreReport:
    """Load the hidden scorer, **verify public digests**, then score.

    The integrated entry point so callers (the future Harbor verifier) cannot
    forget the fail-closed digest check that binds scoring to the exact public
    materials. Raises :class:`DigestMismatch` before scoring on any tamper.
    """
    scorer = load_hidden_scorer(hidden_scorer_path)
    verify_public_material_digests(scorer, public_dir)
    return score(answer, scorer, evidence)
