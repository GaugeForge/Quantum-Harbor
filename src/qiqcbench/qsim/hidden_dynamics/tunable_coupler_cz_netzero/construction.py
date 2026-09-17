"""Materializer + reference solver for tunable_coupler_cz_netzero.

The reference designs a net-zero flux trajectory hitting phi_2Q=pi with low
leakage, predistorts it so the programmed DAC waveform realizes the intent, parks
the coupler where J and zeta both vanish, and reports the single-qubit virtual-Z.
``build_hidden_scorer`` freezes the truth + thresholds.

**What is measured and what is not.** Exactly one quantity is recovered through
the agent surface: the flux-line short-settling parameters.
``measure_flux_settling`` runs a square-pulse cryoscope on the engine through
the same ``run_flux_pulse`` path an agent uses, fits the pole bank from the
cumulative Ramsey phase, and the reference predistorts with *that* estimate. A
reference that predistorts with the exact hidden line cannot demonstrate that
``predistortion_residual`` is reachable by measurement -- its margin would be an
artifact of consuming the answer. Keep it that way: never pass the hidden
settling truth into ``ct.predistort`` here.

Everything else is read from hidden truth rather than characterized: the
anharmonicity and crossing flux come straight from ``HiddenTunableCouplerConfig``,
the idle/gate coupler biases are root-found on the hidden ``J_eff``/``zeta``
curves, ``_CRYO_COUPLER_FLUX`` is a hardcoded hidden-config null, and
``_calibrate_delta_hold`` closes its phase loop against the hidden physics replay
using the true line.

That is a known deviation from a reference that uses
only public tools and never reads hidden device/line truth. What this module
delivers is a diagnostic smoke test that the
conjunctive gate set is jointly satisfiable, not a performance baseline and not a
model of how an agent solves the task. "Honest reference" is therefore scoped to
the settling parameters and must not be stated more broadly.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML

from qiqcbench.qsim.devices import configs_root, load_hidden_config, load_public_spec
from qiqcbench.qsim.hidden_dynamics.tunable_coupler_cz_netzero.schema import CzHiddenScorer
from qiqcbench.qsim.qtypes.transmon_pair_tunable_coupler import control as ct
from qiqcbench.qsim.qtypes.transmon_pair_tunable_coupler import physics as ph
from qiqcbench.qsim.qtypes.transmon_pair_tunable_coupler.device import (
    HiddenTunableCouplerConfig,
    PublicTunableCouplerSpec,
)

DEVICE_ID = "tunable_coupler_cz_v0"
_DT = 0.4
_T_RAMP, _T_HOLD, _DELTA_HOLD = 2.0, 22.0, 0.015  # validated reference operating point
# Execution-evidence floor: minimum completed run_flux_pulse jobs
# before any scoring.
#
# This is an ANTI-GAMING BACKSTOP against the purely-analytic route (waveform
# designed offline, device never meaningfully driven), NOT a discriminator. The
# physics gates discriminate on their own: an early weak run cleared the floor and
# still failed 7 of 13 physics gates, which is a far more informative outcome than
# an evidence rejection.
#
# Observed distribution (so a future editor can re-derive this choice). The gate
# counts COMPLETED jobs; the analytic row was recorded in run_flux_pulse CALLS
# and its completed count was never measured. Completed <= calls, so that run is
# below 15 either way, but the two are not the same unit and the row is marked:
#     4   completed jobs -- submitted 28 experiments but polled only 4 to
# completion, then answered anyway
#   ~10   CALLS (completed unknown, <= 10) -- purely-analytic submission
#    17   completed jobs -- shallow-but-real calibration
#    25   completed jobs -- weak-but-real calibration (it ran chevrons, coupler
#         scans, and a cryoscope that recovered tau)
# 1650+ completed jobs -- strong run
#  2777   completed jobs -- historical genuine pass
#
# The floor was first set to 25 and observation showed that was too high. The
# decisive case is the 17-job run: under a floor of 25 it would have been
# rejected on evidence, discarding NINE valid physics verdicts from a trial that
# genuinely drove the device. (A second run landing at exactly 25 first showed
# the number was mis-set; the 17-job run showed what leaving it there would
# cost.) Analytic (<=10 calls) and real-but-shallow (17 completed) are less than
# 2x apart, so a scalar floor cannot cleanly separate those classes -- 15 is a
# backstop placed in that gap, not a boundary that means anything physical.
#
# The more principled fix -- binding the Stage-A claims (alpha_2, J, settling) to
# the specific completed jobs that support them, as the design also proposes --
# needs a per-claim evidence contract in the answer schema (job-id citations) and
# is therefore an agent-visible task-contract change. It is deferred to a future
# edition rather than folded into this one.
_MIN_COMPLETED_FLUX_JOBS = 15


def _phys(h: HiddenTunableCouplerConfig) -> ph.TransmonPairParams:
    return ph.TransmonPairParams(
        omega_1_max=h.omega_1_max_ghz,
        omega_2=h.omega_2_ghz,
        alpha_1=h.alpha_1_ghz,
        alpha_2=h.alpha_2_ghz,
        arc_asym=h.arc_asym,
        t1_us=h.t1_us,
        t2_us=h.t2_us,
    )


def _coupler(h: HiddenTunableCouplerConfig) -> ph.CouplerParams:
    return ph.CouplerParams(
        omega_c_max=h.omega_c_max_ghz,
        g12=h.g12_ghz,
        gg=h.gg_ghz2,
        omega_1_gate=h.omega_2_ghz + h.alpha_2_ghz,
        omega_1_idle=h.omega_1_max_ghz,
        omega_2=h.omega_2_ghz,
        zeta_scale=h.zeta_scale,
    )


def _true_line(h: HiddenTunableCouplerConfig) -> ct.FluxLineModel:
    return ct.FluxLineModel(
        amplitudes=tuple(h.flux_settle_amplitudes), taus_ns=tuple(h.flux_settle_taus_ns)
    )


def _solve_idle_flux(cp: ph.CouplerParams) -> float:
    """Coupler flux where idle J_eff = 0 (root-find on a grid + refine)."""
    grid = np.linspace(0.20, 0.40, 401)
    vals = [ph.j_eff(x, cp, cp.omega_1_idle) for x in grid]
    for i in range(len(grid) - 1):
        if vals[i] * vals[i + 1] < 0:
            t = abs(vals[i]) / (abs(vals[i]) + abs(vals[i + 1]))
            return float(grid[i] + t * (grid[i + 1] - grid[i]))
    return 0.266


def _solve_gate_flux(cp: ph.CouplerParams, target_mhz: float = -10.0) -> float:
    grid = np.linspace(0.30, 0.42, 241)
    best = min(grid, key=lambda x: abs(ph.j_eff(x, cp) * 1e3 - target_mhz))
    return float(best)


# ---------- honest in-situ cryoscope (the reference's own Stage-A measurement) ----------
#
# Protocol: prep q1 in |+> (ry(pi/2)), program a SQUARE flux step of N samples,
# read out both quadratures, for N = 1..n_max. Convention, derived from the
# engine's rotation and phase conventions:
#     post ry(pi/2) -> P1 = (1 + cos theta)/2      (cos quadrature)
#     post rx(pi/2) -> P1 = (1 + sin theta)/2      (sin quadrature)
# Hidden readout error is a common affine map on both quadratures, so atan2
# recovers theta once the common offset is removed; the offset is estimated from
# the sweep itself, which spans many phase cycles.
#
# Estimator: separable non-linear least squares on the
# CUMULATIVE phase. Only ~tau_1/dt samples carry the fast transient, so
# differencing is noise-dominated; and a 2K-dimensional pole bank cannot be
# grid-searched the way the retired scalar estimator was. The outer optimiser
# runs bounded multi-start least-squares over (A_k, tau_k); the inner exact
# lstsq removes the lab-frame alias slope and phase origin, with the model
# weight PINNED to 1. Pinning is load-bearing: the transient enters the
# cumulative phase linearly in A_k, so a free model coefficient is degenerate
# with a common scale on the amplitudes -- consistent but with variance far too
# large to use (it measured ~3x the pinned spread at this shot budget).
# The ~0.2% systematic pinning absorbs is the hidden arc_asym evaluated at 0 in
# the model; the estimator needs no hidden truth.
# Hardcoded from the hidden config's gate-frame J_eff null (|J| = 7.5e-6 MHz), not
# measured: it keeps the cryoscope Ramsey single-qubit. An agent has to find this
# bias by scanning; the reference is not modelling that search.
_CRYO_COUPLER_FLUX = 0.309755
_CRYO_AMP = 0.20
# ~100 samples (40 ns) covers ~2.9 slow-pole time constants; measured full-gate
# feasibility saturates here, and the cryoscope's submit+poll job count -- not
# engine time -- is what a real budget pays for.
_CRYO_N_MAX = 100
_CRYO_SHOTS = 4096
# Fit bounds and multi-starts. Wide relative to the truth on purpose: the
# reference is a smoke test, not an oracle, so the search must not encode the
# hidden values beyond "short-time settling".
_FIT_BOUNDS_LO = (0.02, 0.02, 0.4, 4.0)
_FIT_BOUNDS_HI = (0.70, 0.70, 6.0, 40.0)
_FIT_STARTS = (
    (0.30, 0.08, 1.2, 10.0),
    (0.45, 0.15, 2.5, 20.0),
    (0.25, 0.05, 0.8, 6.0),
    (0.55, 0.20, 3.0, 30.0),
)


def _cryo_point(engine, n_samples: int, amp: float, shots: int, axis: str) -> float:
    from qiqcbench.qsim.qtypes.transmon_pair_tunable_coupler.wire import FluxPulseRequest

    req = FluxPulseRequest(
        shots=shots,
        coupler_flux=_CRYO_COUPLER_FLUX,
        programmed_flux_q1=[amp] * n_samples,
        sample_dt_ns=_DT,
        prep_ops=[{"qubit": 1, "axis": "y", "angle_rad": math.pi / 2}],
        post_ops=[{"qubit": 1, "axis": axis, "angle_rad": math.pi / 2}],
        measure_qubits=[1],
    )
    outs = engine.run(req, DEVICE_ID, "cryo").data.outcomes[0]
    return sum(1 for o in outs if o[0] == "1") / len(outs)


def _model_cumulative_phase(
    amps: np.ndarray, taus: np.ndarray, amp: float, n_max: int, omega_max: float
) -> np.ndarray:
    """Model cumulative Ramsey phase for a square step through the pole bank.

    Evaluated on the symmetric arc (asym = 0): the hidden ``arc_asym`` is a
    ~0.2% effect the pinned-weight fit absorbs, not an input.
    """
    n = np.arange(1, n_max + 1, dtype=float)
    settled = np.ones(n_max)
    for a_k, t_k in zip(amps, taus, strict=True):
        beta = 1.0 - _DT / (t_k + _DT)
        settled = settled - a_k * beta**n
    realized = amp * settled
    omega = omega_max * np.sqrt(np.abs(np.cos(np.pi * realized)))
    return 2 * math.pi * np.cumsum(omega) * _DT


def _fit_settling(
    theta: np.ndarray,
    amp: float,
    n_max: int,
    omega_max: float,
    *,
    _pin_model_weight: bool = True,
) -> ct.FluxLineModel:
    """Separable NLS for the pole bank; rejects non-minimum-phase estimates.

    ``_pin_model_weight=False`` exists only so tests can guard the pinned
    design against regression to the free-weight solve.
    """
    from scipy.optimize import least_squares

    k = np.arange(n_max, dtype=float)
    tail = np.column_stack([k, np.ones(n_max)])

    def projected_residual(x: np.ndarray) -> np.ndarray:
        m = _model_cumulative_phase(x[:2], x[2:], amp, n_max, omega_max)
        if _pin_model_weight:
            design, target = tail, theta - m
        else:
            design, target = np.column_stack([m, tail]), theta
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        return target - design @ coef

    def model_scale(x: np.ndarray) -> float:
        if _pin_model_weight:
            return 1.0
        m = _model_cumulative_phase(x[:2], x[2:], amp, n_max, omega_max)
        coef, *_ = np.linalg.lstsq(np.column_stack([m, tail]), theta, rcond=None)
        return float(coef[0])

    candidates: list[tuple[float, ct.FluxLineModel]] = []
    for x0 in _FIT_STARTS:
        fit = least_squares(
            projected_residual,
            x0=np.array(x0),
            bounds=(np.array(_FIT_BOUNDS_LO), np.array(_FIT_BOUNDS_HI)),
            method="trf",
        )
        scale = model_scale(fit.x)
        amps = np.abs(fit.x[:2] * scale)
        taus = fit.x[2:]
        order = np.argsort(taus)
        model = ct.FluxLineModel(
            amplitudes=tuple(float(a) for a in amps[order]),
            taus_ns=tuple(float(t) for t in taus[order]),
        )
        # A fitted b-polynomial with zeros outside the unit circle makes the
        # predistortion recursion diverge; such fits are rejected, not used.
        if ct.inverse_is_stable(_DT, model):
            candidates.append((float(fit.cost), model))
    if not candidates:
        raise RuntimeError("cryoscope fit found no minimum-phase settling estimate")
    return min(candidates, key=lambda c: c[0])[1]


def measure_flux_settling(
    hidden: HiddenTunableCouplerConfig,
    rng: np.random.Generator | None = None,
    *,
    amp: float = _CRYO_AMP,
    n_max: int = _CRYO_N_MAX,
    shots: int = _CRYO_SHOTS,
    _pin_model_weight: bool = True,
) -> ct.FluxLineModel:
    """Recover the settling pole bank in situ from engine shot data + the PUBLIC arc.

    Reads no hidden field other than by driving the engine, exactly as an agent
    would through ``run_flux_pulse``. Returns the fitted line, taus ascending.
    """
    from qiqcbench.qsim.qtypes.transmon_pair_tunable_coupler.engine import TunableCouplerEngine

    engine = TunableCouplerEngine(hidden, rng or np.random.default_rng(20260817))
    omega_max = load_public_spec(DEVICE_ID).omega_1_max_ghz  # public

    ns = range(1, n_max + 1)
    p_cos = np.array([_cryo_point(engine, n, amp, shots, "y") for n in ns])
    p_sin = np.array([_cryo_point(engine, n, amp, shots, "x") for n in ns])
    off = 0.5 * (p_cos.mean() + p_sin.mean())
    theta = np.unwrap(np.arctan2(p_sin - off, p_cos - off))
    return _fit_settling(theta, amp, n_max, omega_max, _pin_model_weight=_pin_model_weight)


def _calibrate_delta_hold(
    h: HiddenTunableCouplerConfig, line_measured: ct.FluxLineModel, awg: ct.AwgSpec
) -> float:
    """Trim the hold detuning so the REALIZED pulse lands the conditional phase on pi.

    A fixed operating point is not a fair reference: the phase is steep in the
    predistortion, so a small change in the measured settling moves ``phi_2Q``
    out of a 0.05 rad budget. Any real agent closes this loop by scanning the
    pulse and measuring the conditional phase (a strong observed run scanned amplitude and duration for exactly this).

    This loop is NOT that scan. It evaluates ``phase_err`` by calling the hidden
    physics replay with the true line, so it is an oracle calibration against
    the waveform the agent's measured settling produces -- cheaper than a
    shot-based scan and adequate for a feasibility smoke test, but it is not
    evidence that the phase loop is closable through the surface. Only
    ``measure_flux_settling`` carries that claim.
    """
    p, cp = _phys(h), _coupler(h)
    line_true = _true_line(h)
    j_gate = ph.j_eff(_solve_gate_flux(cp), cp)

    def phase_err(delta_hold: float) -> float:
        intended = ph.reference_netzero_flux(_T_RAMP, _T_HOLD, delta_hold, _DT, p, cp)
        programmed = ct.predistort(intended, _DT, line_measured)
        realized = ct.realized_flux(programmed, _DT, awg, line_true)
        phi2q = ph.conditional_phase_and_leakage(ph.cz_unitary(realized, j_gate, _DT, p))[0]
        return abs((phi2q - math.pi + math.pi) % (2 * math.pi) - math.pi)

    grid = np.linspace(_DELTA_HOLD * 0.7, _DELTA_HOLD * 1.3, 25)
    best = min(grid, key=phase_err)
    fine = np.linspace(best - 0.0008, best + 0.0008, 33)
    return float(min(fine, key=phase_err))


def reference_flux_pair(
    h: HiddenTunableCouplerConfig, line_measured: ct.FluxLineModel | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """(intended net-zero flux, predistorted programmed flux).

    ``line_measured`` is the reference's OWN cryoscope estimate. It is measured
    here when not supplied; the hidden line is never used for predistortion.
    """
    p, cp = _phys(h), _coupler(h)
    if line_measured is None:
        line_measured = measure_flux_settling(h)
    delta_hold = _calibrate_delta_hold(h, line_measured, ct.AwgSpec(full_scale_phi0=0.70))
    intended = ph.reference_netzero_flux(_T_RAMP, _T_HOLD, delta_hold, _DT, p, cp)
    programmed = ct.predistort(intended, _DT, line_measured)
    return intended, programmed


def reference_answer(
    h: HiddenTunableCouplerConfig, line_measured: ct.FluxLineModel | None = None
) -> dict:
    p, cp = _phys(h), _coupler(h)
    awg = ct.AwgSpec(full_scale_phi0=0.70)
    if line_measured is None:
        line_measured = measure_flux_settling(h)
    idle = _solve_idle_flux(cp)
    gate = _solve_gate_flux(cp)
    intended, programmed = reference_flux_pair(h, line_measured)
    # The realized flux is what the TRUE line does to the programmed waveform; the
    # reference only controls what it programs, using its measured settling.
    realized = ct.realized_flux(programmed, _DT, awg, _true_line(h))
    u = ph.cz_unitary(realized, ph.j_eff(gate, cp), _DT, p)
    phi2q, l1, phi01, phi10 = ph.conditional_phase_and_leakage(u)
    gate_t = len(intended) * _DT
    return {
        "task_id": "tunable_coupler_cz_netzero",
        "method": "characterized crossing/J/coupler curves + cryoscope settling; net-zero "
        "fast-adiabatic flux pulse; predistorted to the recovered settling; coupler parked "
        "where J=0 and zeta~0; virtual-Z from the single-qubit phases.",
        "stage_a": {
            "alpha_2_mhz_raw": h.alpha_2_ghz * 1e3,
            "alpha_2_mhz_1sigma": 2.0,
            "coupling_j_mhz_raw": abs(ph.j_eff(gate, cp)) * 1e3,
            "coupling_j_mhz_1sigma": 0.5,
            "avoided_crossing_gap_mhz_raw": 2 * math.sqrt(2) * abs(ph.j_eff(gate, cp)) * 1e3,
            "crossing_flux_q1_raw": ph.freq_to_flux(
                h.omega_2_ghz + h.alpha_2_ghz, h.omega_1_max_ghz
            ),
            "coupler_decoupling_flux_raw": idle,
            "idle_zz_khz_raw": abs(ph.zeta_idle(idle, cp)) * 1e6,
            "idle_zz_khz_1sigma": 0.5,
            "short_settling_amplitudes_raw": [float(x) for x in line_measured.amplitudes],
            "short_settling_tau_ns_raw": [float(x) for x in line_measured.taus_ns],
            "characterization_method": "chevron_coupler_sweep_plus_cryoscope",
        },
        "stage_b": {
            "coupler_idle_flux": idle,
            "coupler_gate_flux": gate,
            "pulse_family": "net_zero_smooth",
            "intended_flux_samples": [float(x) for x in intended],
            "programmed_dac_samples": [float(x) for x in programmed],
            "dac_bits": 16,
            "sample_dt_ns": _DT,
            "gate_time_ns": gate_t,
            "net_zero_satisfied": True,
            "virtual_z_q1_rad": float(phi10),
            "virtual_z_q2_rad": float(phi01),
            "predicted_conditional_phase_rad": float(phi2q),
            "predicted_leakage_raw": float(l1),
            "idle_zz_handling": "coupler_null",
        },
        "stage_c": {
            "measured_conditional_phase_rad_raw": float(phi2q),
            "measured_conditional_phase_rad_1sigma": 0.01,
            "measured_leakage_raw": float(l1),
            "measured_leakage_1sigma": 1e-3,
            "gate_infidelity_raw": float(ph.cz_gate_infidelity(u, phi01, phi10)),
            "gate_infidelity_1sigma": 1e-3,
            "predistortion_residual_raw": float(
                np.max(np.abs(realized[: len(intended)] - intended))
            ),
            "measured_idle_zz_khz_raw": abs(ph.zeta_idle(idle, cp)) * 1e6,
            "measured_idle_zz_khz_1sigma": 0.5,
            "verification_protocol": "conditional_phase_ramsey",
            "leakage_input_set": "four_comp_basis",
        },
    }


# ---------- trap answers (for tests) ----------


def trap_answers(
    h: HiddenTunableCouplerConfig, line_measured: ct.FluxLineModel | None = None
) -> dict[str, dict]:
    if line_measured is None:
        line_measured = measure_flux_settling(h)
    base = reference_answer(h, line_measured)
    intended, programmed = reference_flux_pair(h, line_measured)
    traps: dict[str, dict] = {}

    def clone(mut):
        import copy

        a = copy.deepcopy(base)
        mut(a)
        return a

    # 1. no predistortion: program the intended trajectory directly
    traps["no_predistortion"] = clone(
        lambda a: a["stage_b"].__setitem__("programmed_dac_samples", [float(x) for x in intended])
    )
    # 2. stale anharmonicity
    traps["stale_alpha2"] = clone(lambda a: a["stage_a"].__setitem__("alpha_2_mhz_raw", -300.0))
    # 3. stale idle coupler flux (residual ZZ)
    traps["stale_idle"] = clone(lambda a: a["stage_b"].__setitem__("coupler_idle_flux", 0.30))

    # 4. no virtual-Z
    def _novz(a):
        a["stage_b"]["virtual_z_q1_rad"] = 0.0
        a["stage_b"]["virtual_z_q2_rad"] = 0.0

    traps["no_virtual_z"] = clone(_novz)

    # 5. unipolar pulse (loses net-zero advantage)
    def _uni(a):
        uni = [abs(x) for x in intended]
        a["stage_b"]["intended_flux_samples"] = uni
        a["stage_b"]["programmed_dac_samples"] = [
            float(x) for x in ct.predistort(np.abs(intended), _DT, line_measured)
        ]
        a["stage_b"]["net_zero_satisfied"] = False

    traps["unipolar"] = clone(_uni)
    return traps


# ---------- materialization ----------


def build_hidden_scorer(
    public: PublicTunableCouplerSpec, hidden: HiddenTunableCouplerConfig
) -> CzHiddenScorer:
    cp = _coupler(hidden)
    gate = _solve_gate_flux(cp)
    idle = _solve_idle_flux(cp)
    return CzHiddenScorer(
        device_id=hidden.device_id,
        omega_1_max_ghz=hidden.omega_1_max_ghz,
        omega_2_ghz=hidden.omega_2_ghz,
        omega_c_max_ghz=hidden.omega_c_max_ghz,
        alpha_1_ghz=hidden.alpha_1_ghz,
        alpha_2_ghz=hidden.alpha_2_ghz,
        arc_asym=hidden.arc_asym,
        g12_ghz=hidden.g12_ghz,
        gg_ghz2=hidden.gg_ghz2,
        zeta_scale=hidden.zeta_scale,
        t1_us=hidden.t1_us,
        t2_us=hidden.t2_us,
        flux_settle_amplitudes=tuple(hidden.flux_settle_amplitudes),
        flux_settle_taus_ns=tuple(hidden.flux_settle_taus_ns),
        robustness_offset_phi0=hidden.robustness_offset_phi0,
        min_completed_flux_jobs=_MIN_COMPLETED_FLUX_JOBS,
        readout_q1=(hidden.readout_q1.p_0_to_1, hidden.readout_q1.p_1_to_0),
        readout_q2=(hidden.readout_q2.p_0_to_1, hidden.readout_q2.p_1_to_0),
        j_gate_mhz=abs(ph.j_eff(gate, cp)) * 1e3,
        coupler_idle_flux=idle,
        coupler_gate_flux=gate,
        crossing_flux_q1=ph.freq_to_flux(
            hidden.omega_2_ghz + hidden.alpha_2_ghz, hidden.omega_1_max_ghz
        ),
    )


def hidden_scorer_path(root: Path | None = None) -> Path:
    base = configs_root() if root is None else root / "configs"
    return base / "task_materials" / "tunable_coupler_cz_netzero" / "hidden" / "scorer.example.yaml"


def materialize(root: Path | None = None) -> Path:
    base = configs_root() if root is None else root / "configs"
    hidden_path = base / "devices" / f"{DEVICE_ID}.hidden.example.yaml"
    public = load_public_spec(DEVICE_ID)
    hidden = load_hidden_config(hidden_path)
    assert isinstance(public, PublicTunableCouplerSpec)
    assert isinstance(hidden, HiddenTunableCouplerConfig)
    scorer = build_hidden_scorer(public, hidden)
    out = hidden_scorer_path(root)
    out.parent.mkdir(parents=True, exist_ok=True)
    yaml = YAML()
    yaml.default_flow_style = False
    with out.open("w") as f:
        yaml.dump(scorer.model_dump(), f)
    return out


__all__ = [
    "build_hidden_scorer",
    "hidden_scorer_path",
    "materialize",
    "measure_flux_settling",
    "reference_answer",
    "reference_flux_pair",
    "trap_answers",
]
