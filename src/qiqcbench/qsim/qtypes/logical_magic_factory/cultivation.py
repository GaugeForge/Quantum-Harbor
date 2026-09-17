"""Cultivation-factory effective model (logical_magic_factory qtype).

Logical-effective model of a magic-state *cultivation* line (inject -> H_L
kickback cultivation rounds -> interleaved d=3 QEC cycles -> escape/graft ->
d=5 grafted cycles -> terminal logical measurement), after
arXiv:2512.13908 / arXiv:2409.17595. No physical-qubit simulation: the
per-shot state is the scalar pair (orthogonal population ``b``, coherent
misalignment ``m``) plus a slow AR(1) drift phase, and every stage is a
parameterized channel. The three faithfulness requirements this module
implements:

1. Detector flags are *correlated with that shot's error events* (flags catch
   events with per-stage probability ``c``); post-selecting on a flag group
   therefore genuinely removes caught errors from the delivered ensemble.
2. The escape timing has a real interior optimum: the graft transient
   ``dg * zeta**(n-1)`` rings down with extra d=5 cycles while each cycle adds
   weakly-detected (``c_q5`` small) incoherent error.
3. The slow drift phase alpha is invisible to every detector and enters only
   the delivered state / terminal measurement -- the coherent component that
   a prep-axis measurement counts per shot.

Error-event contributions are *independent across stages* (a stage's flag
probability depends only on that stage's own event, plus a deterministic
coherent term for cultivation rounds), so the post-selected delivered
infidelity is exactly computable (:func:`true_delivered_infidelity`) from the
stages' small product distribution, used by both the hidden scorer and
materialization -- no Monte Carlo in the truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TARGET_THETA = math.pi / 4  # Bloch polar angle of |H> about y from |0>

DETECTOR_GROUPS = ("injection", "cultivation", "qec", "graft")


@dataclass(frozen=True)
class CultivationParams:
    """Hidden per-stage parameters (see HiddenCultivationParams for the YAML side)."""

    lam_inj: float
    d_inj: float
    c_inj: float
    delta_inj_rad: float
    lam_cult: float
    d_cult: float
    c_cult: float
    kappa_cult: float
    g_cult: float
    s_cult_good: float
    s_cult_bad: float
    c_cult_bad_scale: float
    lam_q3: float
    d_q3: float
    c_q3: float
    lam_ext: float
    d_ext: float
    c_ext: float
    graft_transient_rad: float
    graft_ringdown: float
    lam_q5: float
    d_q5: float
    c_q5: float
    drift_sigma_rad: float
    drift_ar_rho: float
    readout_p00: float
    readout_p11: float


@dataclass(frozen=True)
class Schedule:
    """One cultivation schedule (the agent's control surface)."""

    injection_theta_rad: float
    cultivation_rounds: int
    qec_cycles_per_round: int
    escape_cycle_n: int


def _stage_table(p: CultivationParams, sch: Schedule) -> list[tuple[str, float, float, float, int]]:
    """(group, lam, d_base, c_catch, cultivation_rounds_after) per noise stage."""
    rounds, qpr = sch.cultivation_rounds, sch.qec_cycles_per_round
    c_cult = p.c_cult * (1.0 if qpr > 0 else p.c_cult_bad_scale)
    st: list[tuple[str, float, float, float, int]] = [
        ("injection", p.lam_inj, p.d_inj, p.c_inj, rounds)
    ]
    for r in range(rounds):
        after = rounds - r - 1
        st.append(("cultivation", p.lam_cult, p.d_cult, c_cult, after))
        for _ in range(qpr):
            st.append(("qec", p.lam_q3, p.d_q3, p.c_q3, after))
    st.append(("graft", p.lam_ext, p.d_ext, p.c_ext, 0))
    for _ in range(max(0, sch.escape_cycle_n - 1)):
        st.append(("graft", p.lam_q5, p.d_q5, p.c_q5, 0))
    return st


def _survival(p: CultivationParams, sch: Schedule) -> float:
    return p.s_cult_good if sch.qec_cycles_per_round > 0 else p.s_cult_bad


def _coherent_misalignment(p: CultivationParams, sch: Schedule) -> float:
    """Deterministic residual coherent angle magnitude at delivery (excludes
    drift).

    The residual injection-angle error (device offset + the agent's theta
    error, damped by cultivation) and the graft flux transient live on
    DIFFERENT rotation axes, so their magnitudes combine in quadrature and
    cannot cancel each other. (Edition 3.1: the earlier collinear sum allowed
    an unphysical theta=0 bypass that cancelled the transient — flagged by
    the 2026-08-05 sol-wave audit, exploited by no trial.)
    """
    d = p.delta_inj_rad + (sch.injection_theta_rad - TARGET_THETA)
    for _ in range(sch.cultivation_rounds):
        d *= p.g_cult
    transient = p.graft_transient_rad * (p.graft_ringdown ** max(0, sch.escape_cycle_n - 1))
    return math.hypot(d, transient)


def _cultivation_extras(p: CultivationParams, sch: Schedule) -> list[float]:
    """Deterministic coherent term added to each cultivation round's flag prob."""
    d = p.delta_inj_rad + (sch.injection_theta_rad - TARGET_THETA)
    out = []
    for _ in range(sch.cultivation_rounds):
        out.append(p.kappa_cult * math.sin(d / 2.0) ** 2)
        d *= p.g_cult
    return out


def _mean_capped_orthogonal_population(q_by_after: dict[int, list[float]], surv: float) -> float:
    """``E[min(b, 0.5)]`` over the engine's per-shot orthogonal population.

    Each stage independently contributes ``0.5 * surv**after`` when it holds an
    uncaught error event, and :meth:`CultivationEngine.run_point` caps the
    per-shot total at 0.5 BEFORE delivering the state. The expectation
    therefore has to be taken over the exact discrete distribution of ``b``
    (a small product of Poisson-binomials, one per ``after`` level), not of an
    uncapped mean. Levels are folded largest-weight-first and everything at or
    past the cap collapses into one bucket, so the state stays tiny.
    """
    dist = {0.0: 1.0}
    at_cap = 0.0
    for after in sorted(q_by_after):
        weight = 0.5 * (surv**after)
        pmf = [1.0]
        for q in q_by_after[after]:
            nxt = [0.0] * (len(pmf) + 1)
            for k, pk in enumerate(pmf):
                nxt[k] += pk * (1.0 - q)
                nxt[k + 1] += pk * q
            pmf = nxt
        folded: dict[float, float] = {}
        for b, pb in dist.items():
            for k, pk in enumerate(pmf):
                if pk == 0.0:
                    continue
                nb = b + k * weight
                if nb >= 0.5:
                    at_cap += pb * pk
                else:
                    folded[nb] = folded.get(nb, 0.0) + pb * pk
        dist = folded
    return sum(b * pb for b, pb in dist.items()) + 0.5 * at_cap


def true_delivered_infidelity(
    p: CultivationParams, sch: Schedule, postselect_mask: frozenset[str] | set[str]
) -> tuple[float, float]:
    """Exact (eps_true, retention) for a schedule under a post-selection mask.

    ``eps_true`` is the infidelity of the delivered (mask-accepted) logical
    state to the ideal |H>; ``retention`` is the acceptance probability. Both
    are exact expectations over :meth:`CultivationEngine.run_point`'s own
    per-shot sampler under the independent-event model.
    """
    mask = set(postselect_mask)
    surv = _survival(p, sch)
    extras = _cultivation_extras(p, sch)
    ci = 0
    p_accept = 1.0
    q_by_after: dict[int, list[float]] = {}
    for group, lam, d_base, c_catch, after in _stage_table(p, sch):
        extra = 0.0
        if group == "cultivation":
            extra = extras[ci]
            ci += 1
        d_eff = min(1.0, d_base + extra)
        # The engine flags an error shot with probability min(1, d_eff+c) and
        # credits the error to b ONLY on a shot that stayed unflagged, so the
        # spurious detector rate d_eff removes the event as surely as the
        # correlated catch does. Leaving d_eff out of the unmasked branch
        # overstated eps by up to 17.6% and inverted the sign of the
        # post-selection effect (2026-08-28 design audit).
        p_event_uncaught = lam * max(0.0, 1.0 - min(1.0, d_eff + c_catch))
        if group in mask:
            p_no_event = (1.0 - lam) * (1.0 - d_eff)
            p_stage_accept = p_no_event + p_event_uncaught
            q = p_event_uncaught / p_stage_accept if p_stage_accept > 0.0 else 0.0
        else:
            p_stage_accept = 1.0
            q = p_event_uncaught
        p_accept *= p_stage_accept
        q_by_after.setdefault(after, []).append(q)
    b = _mean_capped_orthogonal_population(q_by_after, surv)
    # The drift phase rotates the delivered Bloch vector about z (:func:`_bloch`),
    # so it damps only the x component of the projection onto |H> -- damping
    # all of cos(m) understates the delivered fidelity.
    pol = TARGET_THETA + _coherent_misalignment(p, sch)
    damp = math.exp(-(p.drift_sigma_rad**2) / 2.0)
    proj = math.sin(pol) * math.sin(TARGET_THETA) * damp + math.cos(pol) * math.cos(TARGET_THETA)
    eps = (1.0 - (1.0 - 2.0 * b) * proj) / 2.0
    return float(eps), float(p_accept)


def injection_only_infidelity(p: CultivationParams) -> float:
    """Baseline: infidelity of a raw injected copy (no cultivation, no graft)."""
    b = p.lam_inj * 0.5
    cos_eff = math.cos(p.delta_inj_rad) * math.exp(-(p.drift_sigma_rad**2) / 2.0)
    return float(b + (1.0 - 2.0 * b) * (1.0 - cos_eff) / 2.0)


_AXES = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def _bloch(theta: float, m: float, alpha: float, b: float) -> np.ndarray:
    """Delivered Bloch vector: shrink by orthogonal pop, rotate by m about y
    (through the prep great circle) then by drift alpha about z."""
    pol = theta + m
    v = np.array([math.sin(pol), 0.0, math.cos(pol)])
    ca, sa = math.cos(alpha), math.sin(alpha)
    v = np.array([ca * v[0] - sa * v[1], sa * v[0] + ca * v[1], v[2]])
    return (1.0 - 2.0 * b) * v


class CultivationEngine:
    """Stateful run-long engine: budget ledger + AR(1) drift across shots.

    RNG is injected (engine contract); the backend owns exactly one instance
    per run so the injection budget and drift trajectory persist.
    """

    def __init__(self, params: CultivationParams, budget: int, rng: np.random.Generator):
        self._p = params
        self._budget = budget
        self._rng = rng
        self._alpha = 0.0
        self.injections_consumed = 0

    @property
    def budget_remaining(self) -> int:
        return max(0, self._budget - self.injections_consumed)

    def run_point(
        self, sch: Schedule, measure_axis: str, shots: int
    ) -> dict[str, list[int]] | None:
        """Sample ``shots`` injection attempts. Returns per-shot arrays, or
        None if the point would exceed the remaining budget (not charged)."""
        if self.injections_consumed + shots > self._budget:
            return None
        p = self._p
        rng = self._rng
        stages = _stage_table(p, sch)
        extras = _cultivation_extras(p, sch)
        surv = _survival(p, sch)
        n_h = _bloch(TARGET_THETA, 0.0, 0.0, 0.0)
        m_det = _coherent_misalignment(p, sch)
        flags_out = {g: np.zeros(shots, dtype=np.uint8) for g in DETECTOR_GROUPS}
        outcomes = np.zeros(shots, dtype=np.uint8)
        q = math.sqrt(1.0 - p.drift_ar_rho**2) * p.drift_sigma_rad
        for k in range(shots):
            self._alpha = p.drift_ar_rho * self._alpha + q * rng.normal()
            ci = 0
            b = 0.0
            for group, lam, d_base, c_catch, after in stages:
                extra = 0.0
                if group == "cultivation":
                    extra = extras[ci]
                    ci += 1
                event = rng.random() < lam
                p_flag = min(1.0, d_base + extra + (c_catch if event else 0.0))
                if rng.random() < p_flag:
                    flags_out[group][k] = 1
                elif event:
                    b += 0.5 * (surv**after)
            # theta error is already inside m_det; deliver in the target frame
            v = _bloch(TARGET_THETA, m_det, self._alpha, min(b, 0.5))
            if measure_axis == "target_axis":
                p1 = (1.0 - float(np.dot(v, n_h))) / 2.0
            else:
                p1 = (1.0 - float(np.dot(v, _AXES[measure_axis]))) / 2.0
            p1_obs = p1 * p.readout_p11 + (1.0 - p1) * (1.0 - p.readout_p00)
            outcomes[k] = 1 if rng.random() < p1_obs else 0
        self.injections_consumed += shots
        return {
            "outcomes": outcomes.tolist(),
            **{f"flag_{g}": flags_out[g].tolist() for g in DETECTOR_GROUPS},
        }


__all__ = [
    "DETECTOR_GROUPS",
    "TARGET_THETA",
    "CultivationEngine",
    "CultivationParams",
    "Schedule",
    "injection_only_infidelity",
    "true_delivered_infidelity",
]
