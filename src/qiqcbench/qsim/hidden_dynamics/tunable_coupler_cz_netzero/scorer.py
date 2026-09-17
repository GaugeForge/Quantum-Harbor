"""Authoritative scorer for tunable_coupler_cz_netzero.

Replays the agent's PROGRAMMED q1 flux waveform through the control stack
(distortion + DAC) to the realized flux, then through the two-transmon CZ model,
and binds the Stage-C quantities (conditional phase after the reported virtual-Z,
leakage, fidelity, net-zero 1/f advantage, idle ZZ, predistortion residual). The
agent's self-reported ``measured_*`` values are recorded for human review only;
every graded quantity is recomputed here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from ruamel.yaml import YAML

from qiqcbench.qsim.hidden_dynamics.tunable_coupler_cz_netzero.schema import CzHiddenScorer
from qiqcbench.qsim.qtypes.transmon_pair_tunable_coupler import control as ct
from qiqcbench.qsim.qtypes.transmon_pair_tunable_coupler import physics as ph

TASK_ID = "tunable_coupler_cz_netzero"
_TWO_PI = 2.0 * math.pi


class ScorerError(ValueError):
    """Raised when a submission cannot be scored on physics grounds."""


class InfrastructureError(RuntimeError):
    """Qsim/verifier artifact failure that must never become model reward zero.

    qsim owns ``experiment_log.jsonl``: the agent cannot create, delete or edit
    it (the qsim_logs volume is mounted read-only into the agent container). A
    missing or structurally broken log is therefore an infrastructure fault, and
    scoring it as "zero completed jobs" would charge the execution-evidence
    floor to the model for something it did not do.

    Infrastructure faults use reserved verifier exit 2 and no reward file,
    but deliberately not strict number parsing. See the comment in
    ``count_flux_evidence``: an agent can steer a non-finite float into a logged
    field, so strict parsing would let it void its own losing run. The boundary
    here is structural damage only.
    """


def count_flux_evidence(log_path: str | Path) -> dict[str, int]:
    """Count run_flux_pulse submissions and completed polls in the qsim log.

    Raises ``InfrastructureError`` when the log is absent or does not parse; an
    empty-but-valid log is a legitimate (analytic) model outcome and returns
    zeros for the evidence floor to judge.
    """
    source = Path(log_path)
    if not source.is_file():
        raise InfrastructureError("experiment_log.jsonl is missing")
    # Decode inside the handler: qsim writes this file, so undecodable bytes or an
    # unreadable file are its fault, not the agent's. Left outside, a
    # UnicodeDecodeError escaped as an uncaught exception, Python exited 1, and the
    # shell read exit 1 as a scored rejection -- charging a qsim fault to the model.
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise InfrastructureError(f"experiment_log.jsonl is unreadable: {exc}") from exc
    flux_jobs: set[str] = set()
    runs = 0
    completed: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        # LOAD-BEARING: this is a LENIENT json.loads, deliberately unlike the
        # strict parsing. Do NOT add ``parse_constant=_reject_constant``;
        # that hands the agent a
        # one-call self-void:
        #
        #   1. FluxPulseRequest.coupler_flux is an unbounded float and pydantic v2
        #      accepts non-finite values.
        #   2. `1e400` is VALID JSON and Python parses it to inf, so no unusual
        #      wire encoding is needed to get one in.
        #   3. actions/transmon_pair_tunable_coupler.submit_flux_pulse logs
        #      coupler_flux verbatim and unconditionally AT SUBMIT TIME, before the
        #      job runs, so the poisoned line lands whether or not the job
        #      completes (it fails: "math domain error").
        #   4. state.log writes it with json.dumps, which emits a bare `Infinity`
        #      token -- invalid STRICT JSON, accepted by a default json.loads.
        #
        # Under strict parsing, one run_flux_pulse with coupler_flux=1e400 makes the
        # whole log "corrupt", and a losing run converts itself into an
        # infrastructure no-score at the agent's option -- deleting the verdict
        # instead of merely mis-assigning it. Non-finite numbers here are agent
        # data, not evidence damage; only structural damage (absent file, truncated
        # or non-object line) is infrastructure.
        try:
            event = json.loads(line)
        except (ValueError, json.JSONDecodeError) as exc:
            raise InfrastructureError(
                f"experiment log line {line_number} is invalid JSON: {exc}"
            ) from exc
        if not isinstance(event, dict):
            raise InfrastructureError(f"experiment log line {line_number} is not an object")
        if event.get("tool") == "run_flux_pulse" and event.get("job_id"):
            runs += 1
            flux_jobs.add(event["job_id"])
        if event.get("tool") == "get_job_result" and event.get("status") == "complete":
            completed.add(event.get("job_id", ""))
    return {
        "run_flux_pulse_calls": runs,
        "completed_jobs": len(completed),
        "completed_flux_jobs": len(flux_jobs & completed),
    }


@dataclass(frozen=True)
class ScoreResult:
    gate_passed: bool
    checks: dict[str, bool]
    metrics: dict[str, float]
    penalties: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _phys(s: CzHiddenScorer) -> ph.TransmonPairParams:
    return ph.TransmonPairParams(
        omega_1_max=s.omega_1_max_ghz,
        omega_2=s.omega_2_ghz,
        alpha_1=s.alpha_1_ghz,
        alpha_2=s.alpha_2_ghz,
        arc_asym=s.arc_asym,
        t1_us=s.t1_us,
        t2_us=s.t2_us,
    )


def _coupler(s: CzHiddenScorer) -> ph.CouplerParams:
    return ph.CouplerParams(
        omega_c_max=s.omega_c_max_ghz,
        g12=s.g12_ghz,
        gg=s.gg_ghz2,
        omega_1_gate=s.omega_2_ghz + s.alpha_2_ghz,
        omega_1_idle=s.omega_1_max_ghz,
        omega_2=s.omega_2_ghz,
        zeta_scale=s.zeta_scale,
    )


def _floats(v: Any) -> np.ndarray | None:
    if not isinstance(v, list) or not v:
        return None
    try:
        # OverflowError: JSON admits arbitrary-precision integers, and
        # float(10**400) raises it -- outside the ValueError family, so left
        # uncaught it would ride the driver's reserved infrastructure exit
        # (a one-field self-void).
        return np.array([float(x) for x in v], dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None


def _finite(name: str, value: Any) -> float:
    """Parse one reported numeric, fail-closed on non-finite values.

    Reported ``NaN``/``inf`` survive the JSON round-trip and
    poison downstream comparisons in the agent's favour (``max(0.0, 1.0 - nan)``
    is ``0.0``). Every numeric read from the answer goes through here.
    """
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        # OverflowError: a huge JSON integer is agent content, never
        # infrastructure.
        raise ScorerError(f"{name} is not a finite number") from exc
    if not math.isfinite(v):
        raise ScorerError(f"{name} must be finite")
    return v


def _finite_vector(name: str, v: Any) -> np.ndarray:
    arr = _floats(v)
    if arr is None:
        raise ScorerError(f"missing/malformed {name}")
    if not np.isfinite(arr).all():
        raise ScorerError(f"{name} must be finite")
    return arr


def _settling_within_band(
    amps: np.ndarray, taus: np.ndarray, s: CzHiddenScorer
) -> tuple[bool, float]:
    """Grade the reported settling vectors against the hidden pole bank.

    Compared tau-ascending on both sides, so reporting the same physics in the
    other pole order must not fail (the parameters are identified only up to
    permutation). A report with the wrong number of poles fails on shape — a
    scientific miss (the line was not recovered), not a malformed submission.
    Returns (within_band, worst relative error) for the metrics record.
    """
    true_order = np.argsort(s.flux_settle_taus_ns)
    true_a = np.asarray(s.flux_settle_amplitudes, dtype=float)[true_order]
    true_t = np.asarray(s.flux_settle_taus_ns, dtype=float)[true_order]
    if len(amps) != len(true_a):
        return False, float("inf")
    order = np.argsort(taus)
    rel = np.concatenate(
        [
            np.abs(amps[order] - true_a) / np.abs(true_a),
            np.abs(taus[order] - true_t) / np.abs(true_t),
        ]
    )
    worst = float(np.max(rel))
    return worst <= s.settling_rel_tol, worst


def score_answer(answer: dict, s: CzHiddenScorer) -> ScoreResult:
    # Type-validate the containers before reading any field:
    # a non-dict stage_a/stage_b would otherwise surface as AttributeError,
    # which the driver's typed exit-code handler must never have to classify.
    if not isinstance(answer, dict):
        raise ScorerError("answer must be a JSON object")
    a = answer.get("stage_a") or {}
    b = answer.get("stage_b") or {}
    if not isinstance(a, dict):
        raise ScorerError("stage_a must be a JSON object")
    if not isinstance(b, dict):
        raise ScorerError("stage_b must be a JSON object")

    p, cp = _phys(s), _coupler(s)
    awg = ct.AwgSpec(full_scale_phi0=0.70)
    line = ct.FluxLineModel(
        amplitudes=tuple(s.flux_settle_amplitudes), taus_ns=tuple(s.flux_settle_taus_ns)
    )
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    pen: list[str] = []
    notes: list[str] = []

    # The agent picks its own gate/idle coupler biases (instruction Stage B).
    # Parse them up front: Stage-A truth recomputation and the Stage-B/C replay
    # must use the same biases. Both fields are required by instruction.md, with
    # no hidden-truth fallback: under the former
    # ``b.get(field, hidden_default)`` an agent that never located the null was
    # scored as if it had, so omission beat honest reporting.
    if b.get("coupler_gate_flux") is None:
        raise ScorerError("missing stage_b.coupler_gate_flux")
    if b.get("coupler_idle_flux") is None:
        raise ScorerError("missing stage_b.coupler_idle_flux")
    cgate = _finite("stage_b.coupler_gate_flux", b["coupler_gate_flux"])
    cidle = _finite("stage_b.coupler_idle_flux", b["coupler_idle_flux"])

    # ---------- Stage A: device + line characterization ----------
    if a.get("alpha_2_mhz_raw") is None:
        raise ScorerError("missing stage_a.alpha_2_mhz_raw")
    alpha2 = _finite("stage_a.alpha_2_mhz_raw", a["alpha_2_mhz_raw"])
    metrics["alpha_2_mhz"] = alpha2
    true_a2 = s.alpha_2_ghz * 1e3
    checks["alpha_2"] = (
        s.alpha_2_band_mhz[0] <= alpha2 <= s.alpha_2_band_mhz[1]
        and abs(alpha2 - true_a2) <= abs(true_a2) * s.alpha_2_rel_tol
        and abs(alpha2 - s.alpha_2_stale_reject_mhz) > 5.0
    )
    if not checks["alpha_2"]:
        pen.append(f"alpha_2_mhz_raw={alpha2} off truth {true_a2} (or stale)")

    if a.get("coupling_j_mhz_raw") is None:
        raise ScorerError("missing stage_a.coupling_j_mhz_raw")
    j = _finite("stage_a.coupling_j_mhz_raw", a["coupling_j_mhz_raw"])
    metrics["coupling_j_mhz"] = j
    # Truth is |J_eff| at the AGENT's declared gate bias -- the same bias the
    # Stage-B/C replay uses. The instruction tells the agent to pick the gate
    # bias, so grading against the hidden reference's own bias would fail a
    # correct measurement made anywhere else on the tuning curve.
    # instruction.md states this grading basis explicitly: the value is graded
    # at the agent's declared bias without changing the public contract,
    # and the 2026-08-17 strong run quoted an accurate J from a different bias
    # while saying so only in prose. The tolerance stays relative because the
    # curve is steep by design; the absolute floor below only keeps the gate
    # satisfiable where |J_eff| passes through its null.
    # The stale-notebook reject applies only where it discriminates (truth well
    # separated from the stale value); the static plausibility band is
    # superseded by the relative check against the recomputed truth.
    jg_true_mhz = abs(ph.j_eff(cgate, cp)) * 1e3
    metrics["coupling_j_true_mhz"] = jg_true_mhz
    j_tol_mhz = max(jg_true_mhz * s.j_rel_tol, s.j_abs_tol_mhz)
    j_rel_ok = abs(abs(j) - jg_true_mhz) <= j_tol_mhz
    stale_discriminates = abs(jg_true_mhz - s.j_stale_reject_mhz) > 2.0
    j_stale_ok = abs(abs(j) - s.j_stale_reject_mhz) > 1.0 if stale_discriminates else True
    checks["coupling_j"] = j_rel_ok and j_stale_ok
    if not checks["coupling_j"]:
        pen.append(
            f"coupling_j_mhz_raw={j} off truth {jg_true_mhz:.3f} at "
            f"coupler_gate_flux={cgate} (or stale)"
        )

    dec = a.get("coupler_decoupling_flux_raw")
    if dec is not None:
        dec = _finite("stage_a.coupler_decoupling_flux_raw", dec)
    checks["decoupling_flux"] = (
        dec is not None and abs(dec - s.coupler_idle_flux) <= s.decoupling_flux_tol
    )
    # Settling recovery: the reported quantity is
    # the vector pole bank, graded per parameter. A missing report is a scored
    # miss (nothing was substituted for it); a malformed one — non-list, empty,
    # non-finite, or amplitude/tau length mismatch — is fail-closed ScorerError.
    amps_raw = a.get("short_settling_amplitudes_raw")
    taus_raw = a.get("short_settling_tau_ns_raw")
    if amps_raw is None or taus_raw is None:
        checks["settling_recovered"] = False
    else:
        amps = _finite_vector("stage_a.short_settling_amplitudes_raw", amps_raw)
        taus = _finite_vector("stage_a.short_settling_tau_ns_raw", taus_raw)
        if len(amps) != len(taus):
            raise ScorerError("short_settling amplitudes/taus report different pole counts")
        checks["settling_recovered"], worst_rel = _settling_within_band(amps, taus, s)
        metrics["settling_worst_rel_err"] = worst_rel if math.isfinite(worst_rel) else 999.0
    if not checks["settling_recovered"]:
        pen.append("flux-line settling not recovered (cryoscope) — required for predistortion")
    # Reported-for-the-record uncertainties: notes only, never scored. Parse
    # defensively -- this loop must not be able to raise on any payload shape
    # (an unguarded float() here was an OverflowError self-void route).
    for f1 in ("alpha_2_mhz_1sigma", "coupling_j_mhz_1sigma", "idle_zz_khz_1sigma"):
        try:
            ok1 = a.get(f1) is not None and float(a.get(f1, 0)) > 0
        except (TypeError, ValueError, OverflowError):
            ok1 = False
        if not ok1:
            notes.append(f"missing/non-positive {f1}")

    # ---------- Stage B/C: replay the programmed waveform ----------
    intended = _finite_vector("stage_b.intended_flux_samples", b.get("intended_flux_samples"))
    programmed = _finite_vector("stage_b.programmed_dac_samples", b.get("programmed_dac_samples"))
    # The replay grades the FULL programmed array while the
    # gate window is measured on ``intended``, so unequal lengths bought an
    # unscrutinised physics tail, and the min-length residual truncation made a
    # 1-sample programmed array bank the headline gate. Equal length is stated
    # in instruction.md and costless (predistortion is length-preserving); the
    # sample cap is a redundant compute backstop; the AWG grid is fixed public
    # hardware, so a free-floating dt must not decouple gate time from the
    # sample count.
    if len(programmed) != len(intended):
        raise ScorerError(
            f"programmed_dac_samples has {len(programmed)} samples but "
            f"intended_flux_samples has {len(intended)}; the lengths must be equal"
        )
    if len(programmed) > s.max_programmed_samples:
        raise ScorerError(f"programmed_dac_samples exceeds {s.max_programmed_samples} samples")
    # The 0.4 default on omission is stated publicly in instruction.md: it is
    # the public AWG grid — the field's only legal value — so defaulting
    # discloses nothing and adds no rejection surface.
    dt = _finite("stage_b.sample_dt_ns", b.get("sample_dt_ns", 0.4))
    if abs(dt - 0.4) > 1e-9:
        raise ScorerError("sample_dt_ns must be the public 0.4 ns AWG grid")
    gate_t = len(intended) * dt
    metrics["gate_time_ns"] = gate_t
    checks["gate_window"] = s.gate_window_ns[0] <= gate_t <= s.gate_window_ns[1]
    vz1 = b.get("virtual_z_q1_rad")
    vz2 = b.get("virtual_z_q2_rad")
    checks["virtual_z_reported"] = vz1 is not None and vz2 is not None
    if not checks["virtual_z_reported"]:
        pen.append("single-qubit virtual-Z corrections not reported (headline trap)")
    vz1 = _finite("stage_b.virtual_z_q1_rad", vz1) if vz1 is not None else 0.0
    vz2 = _finite("stage_b.virtual_z_q2_rad", vz2) if vz2 is not None else 0.0

    # net-zero (intended trajectory)
    integral = float(abs(np.sum(intended) * dt))
    metrics["netzero_integral_phi0_ns"] = integral
    checks["net_zero"] = integral <= s.netzero_integral_max_phi0_ns
    if not checks["net_zero"]:
        pen.append("intended trajectory is not net-zero")

    # predistortion: realized must match intended (equal lengths enforced above,
    # so the whole replayed waveform is graded — no truncation window)
    realized = ct.realized_flux(programmed, dt, awg, line)
    residual = float(np.max(np.abs(realized - intended)))
    metrics["predistortion_residual"] = residual
    checks["predistortion_residual"] = residual <= s.predistortion_residual_max
    if not checks["predistortion_residual"]:
        pen.append(
            f"predistortion residual {residual:.2e} > {s.predistortion_residual_max} (program != realized)"
        )

    # replay the gate
    jg = ph.j_eff(cgate, cp)
    u = ph.cz_unitary(realized, jg, dt, p)
    phi2q, l1, phi01, phi10 = ph.conditional_phase_and_leakage(u)
    phi2q_err = abs((phi2q - math.pi + math.pi) % _TWO_PI - math.pi)
    metrics["phi_2Q_rad"] = phi2q
    metrics["phi_2Q_err"] = phi2q_err
    metrics["leakage"] = l1
    checks["conditional_phase"] = phi2q_err <= s.phi2q_tol_rad
    checks["leakage"] = l1 <= s.leakage_max
    # infidelity uses the AGENT's reported virtual-Z (no/wrong vZ -> high infidelity).
    # cz_gate_infidelity(u, phi_01, phi_10): phi_01 corrects |01> (q2), phi_10 |10> (q1).
    # "Report the virtual-Z corrections" admits two sign conventions -- the
    # accumulated single-qubit phase, or the Z you would APPLY to cancel it
    # (-phi). Accept both by scoring the better one: a wrong pair
    # still fails on physics; only a global convention flip is forgiven.
    coh = min(
        ph.cz_gate_infidelity(u, vz2, vz1),
        ph.cz_gate_infidelity(u, -vz2, -vz1),
    )
    infid = coh + ph.decoherence_infidelity(gate_t, p)
    metrics["gate_infidelity"] = infid
    checks["gate_fidelity"] = infid <= s.infidelity_max
    if not checks["conditional_phase"]:
        pen.append(f"phi_2Q err {phi2q_err:.3f} > {s.phi2q_tol_rad} rad")
    if not checks["leakage"]:
        pen.append(f"leakage {l1:.2e} > {s.leakage_max}")
    if not checks["gate_fidelity"]:
        pen.append(f"infidelity {infid:.2e} > {s.infidelity_max} (check virtual-Z, calibration)")

    # net-zero robustness advantage under the injected flux offset
    dphi = s.robustness_offset_phi0
    base = ph.conditional_phase_and_leakage(ph.cz_unitary(realized, jg, dt, p))[0]
    off_nz = ph.conditional_phase_and_leakage(ph.cz_unitary(realized + dphi, jg, dt, p))[0]
    uni = np.abs(realized)
    base_u = ph.conditional_phase_and_leakage(ph.cz_unitary(uni, jg, dt, p))[0]
    off_u = ph.conditional_phase_and_leakage(ph.cz_unitary(uni + dphi, jg, dt, p))[0]
    drift_nz = abs((off_nz - base + math.pi) % _TWO_PI - math.pi)
    drift_u = abs((off_u - base_u + math.pi) % _TWO_PI - math.pi)
    adv = drift_u / drift_nz if drift_nz > 1e-9 else float("inf")
    metrics["netzero_drift"] = drift_nz
    metrics["unipolar_drift"] = drift_u
    metrics["netzero_advantage"] = adv if adv != float("inf") else 999.0
    checks["netzero_advantage"] = adv >= s.netzero_advantage_min
    if not checks["netzero_advantage"]:
        pen.append(f"net-zero advantage {adv:.1f}x < {s.netzero_advantage_min}x")

    # idle ZZ at the chosen idle coupler flux
    idle_zz_khz = abs(ph.zeta_idle(cidle, cp)) * 1e6
    metrics["idle_zz_khz"] = idle_zz_khz
    checks["idle_zz"] = idle_zz_khz <= s.idle_zz_max_khz
    if not checks["idle_zz"]:
        pen.append(f"idle ZZ {idle_zz_khz:.1f} kHz > {s.idle_zz_max_khz} kHz (coupler idle bias)")

    gate_passed = all(checks.values())
    return ScoreResult(
        gate_passed=gate_passed, checks=checks, metrics=metrics, penalties=pen, notes=notes
    )


def load_hidden_scorer(path: str | Path) -> CzHiddenScorer:
    yaml = YAML(typ="safe")
    with Path(path).open(encoding="utf-8") as f:
        return CzHiddenScorer.model_validate(yaml.load(f))


__all__ = [
    "InfrastructureError",
    "ScoreResult",
    "ScorerError",
    "TASK_ID",
    "count_flux_evidence",
    "load_hidden_scorer",
    "score_answer",
]
