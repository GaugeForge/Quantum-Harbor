"""Retired aggregate-fit utilities for historical g-f development evidence.

Neither shipped g-f verifier scores this estimator. Both current tasks evaluate
deterministic hidden pre-final-readout endpoints and use explicitly cited raw
records only as execution evidence. These pure helpers remain importable for
old analysis notebooks; they are not a public task contract or scoring authority.

The historical estimator had three load-bearing properties.

**Cadence-bound selection.** The code-space bit-flip lifetime genuinely depends on
the erasure-detection cadence (per cycle, the double-decay bit-flip grows as
``t_cycle**2``, undetected seepage as ``t_cycle`` and the DD pulse error is
constant, so the per-microsecond error rate is ``a t + b + c/t``). A recompute is
therefore only meaningful against the cadence the submission declares; selecting
the widest sweep globally silently tests an answer derived at one cadence against
another.

**One pooled population, and it must be complete.** Every qualifying record at the
declared cadence is pooled — the two preparations of the arm (``0L``/``1L`` for Z,
``+X``/``-X`` for X) and every job at that cadence. Nothing is chosen for being
favourable, and nothing may be left out either: a fit to one preparation alone
carries in full the readout bias that pooling exists to cancel, so both
preparations must actually reach the fit. Presence in the log is not enough — a
token one-shot record makes a preparation present while contributing no point the
fitter can use, which is the same biased single-preparation estimate. Completeness
is therefore measured on ``point_signal``, the fitter's own per-point eligibility,
and reported as ``contributing_prep_states`` against ``missing_prep_states`` (never
run) and ``starved_prep_states`` (run, but no usable point).

**Code-space denominator.** The post-selected population is the shots that were
never erasure-flagged *and* whose final qutrit readout is inside the logical code
space ``{|g>, |f>}``. A final ``|e>`` is leakage, not a logical bit flip, so
counting it in the denominator is a category error. It is also the property that
makes the estimator SPAM-robust: writing the readout matrix as ``P(m|t)`` and the
code-space-restricted polarization as ``y = 2p - 1``, the two preparations have
residuals at zero polarization of

    ``0L: (P(g|g) + P(g|f) - P(f|g) - P(f|f)) / (P(g|g) + P(f|g) + P(g|f) + P(f|f))``
    ``1L: -1 x the same quantity``

which cancel exactly when the two arms are pooled, for *any* confusion matrix.
Keeping ``|e>`` in the denominator instead leaves an uncancelled residual of
``-(P(e|g) + P(e|f)) / 2`` that biases the fitted lifetime short.
"""

from __future__ import annotations

import math
from typing import Any

EVIDENCE_ACTION = "logical_memory_evidence"
EVIDENCE_TOOLS = ("run_logical_memory_sweep", "run_logical_memory")
PROTOCOL_DD = "xy4"
CODE_SPACE_DENOM = "n_no_flag_codespace"

__all__ = [
    "CODE_SPACE_DENOM",
    "EVIDENCE_ACTION",
    "EVIDENCE_TOOLS",
    "PROTOCOL_DD",
    "fit_lifetime_us",
    "incomplete_preparation_note",
    "missing_denominator_note",
    "point_signal",
    "select_pooled_evidence",
]


def point_signal(n_post: int, n_correct: int, *, min_survivors: int) -> float | None:
    """The fitter's per-point eligibility, in one place so nothing can drift from it.

    Returns the polarization ``y = 2p - 1`` when the point enters the fit, ``None``
    when it does not. A point is dropped for two unrelated reasons: too few
    post-selected survivors to estimate ``p`` from, or a polarization that has
    already decayed into the noise — the second is ordinary physics at long times.
    """
    if n_post < min_survivors:
        return None
    y = 2.0 * (n_correct / n_post) - 1.0
    if y <= 0.02:
        return None
    return y


def incomplete_preparation_note(manifest: dict[str, Any]) -> str:
    """Name the preparation(s) that did not reach the fit; empty when both did.

    Distinguishes the two ways an arm ends up single-preparation, because they call
    for different things from the agent: one preparation was never run, or it ran
    but every one of its points was dropped by the fitter's own eligibility test.

    **This gate is load-bearing by necessity, not by convenience — do not simplify
    it away in favour of a tolerance.** The single-preparation readout bias is
    antisymmetric, so it lives in the *difference* between the two preparations and
    is invisible in the pooled value the verifier grades. Three replacements were
    measured against real qsim data and all three failed: a tightened band on the
    value (honest T1_Z spans 512-664 us at one cadence from ordinary agent choices,
    while `1L`-alone lands at 637 inside that range, and honest pooled T1_Z varies
    2.1x across the legal cadence range, so `0L`-alone at 3.52 us equals an honest
    pooled fit at 5.0 us to 0.5%); a lower-variance estimator; and an
    evidence-adaptive z-score gate (a single-preparation fit uses half the data, so
    its sigma inflates by ~sqrt(2) and absorbs the bias — closest single-preparation
    |z| 0.01-0.13 against worst honest |z| 1.98-3.01). Improving the measurement
    shrinks the honest spread and the bias together, so no gate on the value can
    separate them.

    **The estimator is at the information limit — "just fit it better" is not
    available.** Variance-correct GLS weights for ``ln(2p - 1)`` and a
    binomial-likelihood MLE of the same model both reproduce the shipped scatter
    (the log never diverges here: ``y`` falls only 0.89 to 0.72 over the reference
    window, so the shipped ``n_post`` weight is already within ~2x of optimal), a
    straight-line fit leaves residuals within +/-1.7 sigma, and each fit is
    consistent with its own error bars: chi2/dof 1.65 and RMS(z) 1.25 over the
    honest configuration family, 1.02 and 0.98 over the wider evidence family. The
    spread is what the evidence supports, not slack in the fit.

    Known and accepted residual: a well-sampled arm plus one 30-survivor point of
    the other preparation clears this gate while the token point carries ~0.04% of
    the fit weight, leaving the fit effectively single-preparation. Closing that
    needs a rule on preparation *weight*, which was measured to cost more than it
    buys at every non-arbitrary threshold.
    """
    missing = manifest.get("missing_prep_states") or []
    starved = manifest.get("starved_prep_states") or []
    if not missing and not starved:
        return ""
    causes = []
    if missing:
        causes.append(f"{missing} never ran")
    if starved:
        causes.append(
            f"{starved} ran but contributed no usable point (too few post-selected "
            "survivors, or no polarization left to fit)"
        )
    basis = str(manifest.get("measure_basis", "")).upper()
    return (
        f"the {basis}-basis arm at the declared cadence "
        f"{manifest.get('declared_cycle_time_us')} us fitted only preparation(s) "
        f"{manifest.get('contributing_prep_states') or []}: " + "; ".join(causes) + ". The "
        "three-state readout is asymmetric, so a fit to one preparation alone is biased in "
        "a fixed direction and that bias cancels only when both preparations of the arm are "
        "pooled — both must contribute"
    )


def missing_denominator_note(manifest: dict[str, Any]) -> str:
    """Name the stale-aggregate case, which otherwise reads as 'no evidence'."""
    n = manifest.get("n_records_missing_code_space_denominator", 0)
    if not n:
        return ""
    return (
        f"; {n} otherwise-qualifying record(s) carry no {CODE_SPACE_DENOM!r} count and "
        "were dropped — that aggregate predates the current evidence contract, so the "
        "qsim image that produced this log is older than this verifier"
    )


def fit_lifetime_us(points: list[tuple[float, int, int]], *, min_survivors: int) -> float | None:
    """Weighted linear fit of ``ln(2p - 1) = c + s t``; returns ``T = -1/s``.

    ``points`` is ``[(total_evolution_us, n_post_selected, n_correct), ...]``. The
    intercept absorbs the multiplicative readout contrast; the additive readout
    offset is what the pooled code-space denominator cancels.
    """
    xs: list[float] = []
    ys: list[float] = []
    ws: list[float] = []
    for t, n_post, n_correct in points:
        y = point_signal(n_post, n_correct, min_survivors=min_survivors)
        if y is None:
            continue
        xs.append(float(t))
        ys.append(math.log(y))
        ws.append(float(n_post))
    if len(xs) < 3:
        return None
    sw = sum(ws)
    swx = sum(w * x for w, x in zip(ws, xs, strict=True))
    swy = sum(w * y for w, y in zip(ws, ys, strict=True))
    swxx = sum(w * x * x for w, x in zip(ws, xs, strict=True))
    swxy = sum(w * x * y for w, x, y in zip(ws, xs, ys, strict=True))
    denom = sw * swxx - swx * swx
    if abs(denom) < 1e-12:
        return None
    slope = (sw * swxy - swx * swy) / denom
    if slope >= -1e-9:
        return None
    return -1.0 / slope


def select_pooled_evidence(
    events: list[dict],
    *,
    prep_states: set[str],
    measure_basis: str,
    declared_cycle_time_us: float,
    cadence_rel_tol: float,
    min_survivors: int,
) -> tuple[list[tuple[float, int, int]], dict[str, Any]]:
    """Pool every ``dd=xy4`` record of one arm at the declared cadence.

    Returns ``(fit_points, manifest)``. The manifest is the single evidence
    binding reported by the verifier: which cadence was demanded, which records
    and completed jobs were admitted, and the pooled population it fitted. It is
    a deterministic function of the declared cadence and the log — the verifier
    never picks among candidate sweeps.

    ``min_survivors`` is the fitter's own threshold, taken here so that
    ``contributing_prep_states`` is the set of preparations that actually reach the
    fit rather than the set that merely appears in the log. A one-shot record makes
    a preparation *present* while contributing nothing, which is the same biased
    single-preparation estimate as never running it.
    """
    manifest: dict[str, Any] = {
        "declared_cycle_time_us": round(float(declared_cycle_time_us), 4),
        "cadence_rel_tol": cadence_rel_tol,
        "measure_basis": measure_basis,
        "dd": PROTOCOL_DD,
        "denominator": CODE_SPACE_DENOM,
        "min_survivors": min_survivors,
        "matched_cycle_times_us": [],
        "prep_states": [],
        "contributing_prep_states": [],
        "missing_prep_states": sorted(prep_states),
        "starved_prep_states": [],
        "job_ids": [],
        "n_records": 0,
        "n_points": 0,
        "span_us": 0.0,
        "n_records_at_other_cadence": 0,
        "n_records_missing_code_space_denominator": 0,
    }
    if declared_cycle_time_us <= 0:
        return [], manifest

    arm = [
        e
        for e in events
        if isinstance(e, dict)
        and e.get("action") == EVIDENCE_ACTION
        and e.get("tool") in EVIDENCE_TOOLS
        and e.get("measure_basis") == measure_basis
        and e.get("prep_state") in prep_states
        and e.get("dd") == PROTOCOL_DD
        and (e.get("points") or [])
    ]

    fit_points: list[tuple[float, int, int]] = []
    times: list[float] = []
    cadences: set[float] = set()
    preps: set[str] = set()
    contributing: set[str] = set()
    jobs: set[str] = set()
    for ev in arm:
        cyc = float(ev.get("cycle_time_us", 0.0))
        if abs(cyc - declared_cycle_time_us) > cadence_rel_tol * declared_cycle_time_us:
            manifest["n_records_at_other_cadence"] += 1
            continue
        pts = ev["points"]
        if any(CODE_SPACE_DENOM not in p for p in pts):
            manifest["n_records_missing_code_space_denominator"] += 1
            continue
        prep = str(ev.get("prep_state"))
        cadences.add(round(cyc, 4))
        preps.add(prep)
        if ev.get("job_id"):
            jobs.add(str(ev["job_id"]))
        manifest["n_records"] += 1
        for p in pts:
            t = float(p["total_evolution_us"])
            n_post, n_correct = int(p[CODE_SPACE_DENOM]), int(p["n_no_flag_correct"])
            fit_points.append((t, n_post, n_correct))
            times.append(t)
            if point_signal(n_post, n_correct, min_survivors=min_survivors) is not None:
                contributing.add(prep)

    manifest["matched_cycle_times_us"] = sorted(cadences)
    manifest["prep_states"] = sorted(preps)
    manifest["contributing_prep_states"] = sorted(contributing)
    manifest["missing_prep_states"] = sorted(set(prep_states) - preps)
    manifest["starved_prep_states"] = sorted(preps - contributing)
    manifest["job_ids"] = sorted(jobs)
    manifest["n_points"] = len(fit_points)
    manifest["span_us"] = round(max(times) - min(times), 2) if times else 0.0
    return fit_points, manifest
