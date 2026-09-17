"""Verifier scoring for ``logical_cnot_decoder_calibration``.

Binary pass/fail, ``rating_mode: rubric``. The verifier reconstructs BOTH submitted DEMs +
decoder configs and replays standard decoders on FRESH hidden-noise shots (independent
RNG):

1. well-formed submission (both DEMs valid over their public detector sets, full
   single-detector coverage, mechanism-count caps, whitelisted decoders, fields in band);
2. qsim budget evidence (experiments were actually run; shot + shot-rounds caps hold);
3. memory gate: replayed d=5 logical error rate per cycle < ``floor_mem``;
4. CNOT gate: battery-averaged replayed error < ``floor_cnot`` AND every entry
   < ``2 x floor_cnot``.

Ungated derived figures reported in the score report: ``Lambda(3->5)`` (recomputed by
replay; gross disagreement with the submitted ``lambda_d3_d5_raw`` is flagged, never
failed) and the logical Bell-fidelity witness lower bound from the bell reporting
configurations decoded with the submitted CNOT DEM. Self-reported
rates are consistency-checked, never trusted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from qiqcbench.qsim.hidden_dynamics.logical_cnot_decoder_calibration import construction as C
from qiqcbench.qsim.hidden_dynamics.logical_cnot_decoder_calibration import decoders as D

MECHANISM_CAP_FACTOR = 20
MIN_EVIDENCE_SHOTS_PER_CONFIGURATION = 500
_CNOT_EVIDENCE_KEYS = {f"{basis}:{logical_input}" for basis, logical_input in C.CNOT_BATTERY}
_EXPERIMENT_RESULT_ACTIONS = {
    "run_memory_experiment_result",
    "run_merged_memory_result",
    "run_lattice_surgery_cnot_result",
}
_STAGE_MODES = {
    "stage_a_mode": "device_probe",
    "stage_b_mode": "memory_characterization",
    "stage_c_mode": "surgery_characterization",
    "stage_d_mode": "submit_decoders",
}
_NOTE_FIELDS = ("memory_characterization_note", "surgery_characterization_note")


class EvidenceInfrastructureError(RuntimeError):
    """Qsim-owned evidence is missing, corrupt, or internally inconsistent."""


@dataclass
class ScoreReport:
    task_id: str
    reward: int
    passed: bool
    memory_gate_passed: bool | None
    cnot_gate_passed: bool | None
    replayed_memory_rate_per_cycle: float | None
    replayed_cnot_error_average: float | None
    replayed_cnot_entries: dict | None
    floor_mem: float
    floor_cnot: float
    per_entry_cap: float
    lambda_replayed: float | None = None
    f_bell_lower_bound: float | None = None
    decoder_memory: str | None = None
    decoder_cnot: str | None = None
    n_mechanisms_memory: int | None = None
    n_mechanisms_cnot: int | None = None
    self_report_consistent: bool | None = None
    penalties: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "disposition": (
                "scored"
                if self.cnot_gate_passed is not None
                and self.replayed_cnot_error_average is not None
                else "model_failure"
            ),
            "reward": self.reward,
            "passed": self.passed,
            "memory_gate_passed": self.memory_gate_passed,
            "cnot_gate_passed": self.cnot_gate_passed,
            "replayed_memory_rate_per_cycle": self.replayed_memory_rate_per_cycle,
            "replayed_cnot_error_average": self.replayed_cnot_error_average,
            "replayed_cnot_entries": self.replayed_cnot_entries,
            "floor_mem": self.floor_mem,
            "floor_cnot": self.floor_cnot,
            "per_entry_cap": self.per_entry_cap,
            "lambda_replayed": self.lambda_replayed,
            "f_bell_lower_bound": self.f_bell_lower_bound,
            "decoder_memory": self.decoder_memory,
            "decoder_cnot": self.decoder_cnot,
            "n_mechanisms_memory": self.n_mechanisms_memory,
            "n_mechanisms_cnot": self.n_mechanisms_cnot,
            "self_report_consistent": self.self_report_consistent,
            "penalties": self.penalties,
            "flags": self.flags,
        }


def _fail(reason: str, **kw) -> ScoreReport:
    return ScoreReport(
        task_id=C.TASK_ID,
        reward=0,
        passed=False,
        memory_gate_passed=kw.get("memory_gate_passed"),
        cnot_gate_passed=kw.get("cnot_gate_passed"),
        replayed_memory_rate_per_cycle=kw.get("mem_rate"),
        replayed_cnot_error_average=kw.get("cnot_avg"),
        replayed_cnot_entries=kw.get("cnot_entries"),
        floor_mem=C.FLOOR_MEM,
        floor_cnot=C.FLOOR_CNOT,
        per_entry_cap=C.CNOT_PER_ENTRY_CAP,
        decoder_memory=kw.get("decoder_memory"),
        decoder_cnot=kw.get("decoder_cnot"),
        n_mechanisms_memory=kw.get("n_mem"),
        n_mechanisms_cnot=kw.get("n_cnot"),
        penalties=[reason],
    )


def model_failure_report(reason: str) -> ScoreReport:
    """Build the task's complete structured summary for an invalid/missing submission."""

    return _fail(reason)


def _validate_dem(
    dem: object, n_detectors: int, active_sets: list[set[int]], label: str
) -> list[dict]:
    if not isinstance(dem, list) or not dem:
        raise ValueError(f"{label} must be a non-empty list of mechanisms")
    if len(dem) > MECHANISM_CAP_FACTOR * n_detectors:
        raise ValueError(
            f"{label} has {len(dem)} mechanisms; cap is {MECHANISM_CAP_FACTOR}x the "
            f"detector count ({MECHANISM_CAP_FACTOR * n_detectors})"
        )
    covered: set[int] = set()
    for mech in dem:
        dets, _p, _names = D.normalize_mechanism(mech)
        for d in dets:
            if not (0 <= d < n_detectors):
                raise ValueError(f"{label}: detector id {d} out of range [0, {n_detectors})")
            covered.add(d)
    for k, active in enumerate(active_sets):
        missing = active - covered
        if missing:
            raise ValueError(
                f"{label}: {len(missing)} active detectors have no incident mechanism "
                f"(e.g. id {min(missing)}); every active detector needs coverage"
            )
        del k
    return dem


def _in_band(value: object, lo: float, hi: float) -> bool:
    try:
        return lo < float(value) < hi
    except (TypeError, ValueError):
        return False


def _positive_int(answer: dict, key: str) -> int:
    value = answer.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _strict_nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer (bool is not accepted)")
    return value


def _strict_positive_int(value: object, label: str) -> int:
    parsed = _strict_nonnegative_int(value, label)
    if parsed == 0:
        raise ValueError(f"{label} must be a positive integer")
    return parsed


def validate_submission(answer: dict) -> tuple[list[dict], str, list[dict], str]:
    # The FinalAnswer envelope's task_id is validated at submission time and again by
    # the verifier; the inner answer.task_id is optional and only checked when present,
    # so an otherwise-correct submission is never rejected over the duplicated field.
    if "task_id" in answer and answer["task_id"] != C.TASK_ID:
        raise ValueError(f"task_id, when present, must be {C.TASK_ID!r}")
    if not isinstance(answer.get("method"), str) or not answer["method"].strip():
        raise ValueError("method must be a non-empty string")
    for key, expected in _STAGE_MODES.items():
        if answer.get(key) != expected:
            raise ValueError(f"{key} must be {expected!r}")
    if not isinstance(answer.get("probed_spec"), dict):
        raise ValueError("probed_spec must be an object")
    for key in _NOTE_FIELDS:
        if not isinstance(answer.get(key), str) or not answer[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    memory_shots = _positive_int(answer, "shots_for_memory_characterization_raw")
    surgery_shots = _positive_int(answer, "shots_for_surgery_characterization_raw")
    if memory_shots + surgery_shots > C.SHOT_BUDGET:
        raise ValueError(
            "reported characterization shots exceed the public shot budget: "
            f"{memory_shots + surgery_shots} > {C.SHOT_BUDGET}"
        )
    if not _in_band(answer.get("self_measured_memory_logical_rate_raw"), 0.0, 0.5):
        raise ValueError("self_measured_memory_logical_rate_raw must be in (0, 0.5)")
    if not _in_band(answer.get("self_measured_cnot_error_raw"), 0.0, 0.5):
        raise ValueError("self_measured_cnot_error_raw must be in (0, 0.5)")
    if not _in_band(answer.get("lambda_d3_d5_raw"), 0.5, 20.0):
        raise ValueError("lambda_d3_d5_raw must be in (0.5, 20)")
    fb = answer.get("f_bell_logical_raw")
    if fb is not None and not _in_band(fb, -1.0, 1.0 + 1e-9):
        raise ValueError("f_bell_logical_raw must lie in (-1, 1]")

    from qiqcbench.qsim.qtypes.surface_code_lattice_surgery import circuits as CC

    mem_bundle = CC.memory_bundle("d5", C.MEMORY_SCORING_ROUNDS)
    dem_mem = _validate_dem(
        answer.get("dem_memory_d5"),
        mem_bundle.n_detectors,
        [set(range(mem_bundle.n_detectors))],
        "dem_memory_d5",
    )
    meta = D._cnot_meta()
    n_super = meta["n_superset_detectors"]
    active_sets = [set(cfg["detector_columns"]) for cfg in meta["configs"].values()]
    dem_cnot = _validate_dem(
        answer.get("dem_cnot_spacetime"), n_super, active_sets, "dem_cnot_spacetime"
    )

    def decoder_of(key: str) -> str:
        cfg = answer.get(key) or {}
        dt = cfg.get("decoder_type") if isinstance(cfg, dict) else None
        if dt not in D.SUPPORTED_DECODERS:
            raise ValueError(
                f"{key}.decoder_type must be one of {D.SUPPORTED_DECODERS}, got {dt!r}"
            )
        options = cfg.get("options", {})
        if not isinstance(options, dict):
            raise ValueError(f"{key}.options must be an object")
        if options:
            raise ValueError(f"{key}.options has no supported v1 entries; use an empty object")
        return dt

    return dem_mem, decoder_of("decoder_config_memory"), dem_cnot, decoder_of("decoder_config_cnot")


def reported_shot_count_flag(answer: dict, qsim_shots_used: int) -> str:
    reported = int(answer["shots_for_memory_characterization_raw"]) + int(
        answer["shots_for_surgery_characterization_raw"]
    )
    if reported == qsim_shots_used:
        return ""
    return (
        f"reported characterization shots ({reported}) differ from the qsim budget meter "
        f"({qsim_shots_used}); qsim evidence is authoritative"
    )


def _normalized_answer(payload: dict) -> dict:
    nested = payload.get("answer")
    return nested if isinstance(nested, dict) else payload


def recover_experiment_evidence(log_path, final_payload: dict) -> dict:
    """Join causally ordered evidence and aggregate chunks by scientific configuration."""
    if not isinstance(final_payload, dict):
        raise EvidenceInfrastructureError("persisted final-answer artifact is not an object")
    path = Path(log_path)
    summary = {
        "shots_used": 0,
        "shot_rounds_used": 0,
        "evidence_reasons": [],
        "d5_memory_job_id": None,
        "d5_memory_job_ids": [],
        "d5_memory_shots": 0,
        "cnot_job_ids": {},
        "cnot_shots_by_entry": {},
        "qualified_evidence_shots": 0,
        "qualified_evidence_shot_rounds": 0,
        "final_event_index": None,
    }
    if not path.is_file():
        raise EvidenceInfrastructureError("qsim experiment log is missing")

    events: list[tuple[int, dict]] = []
    invalid_records: list[tuple[int, str]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise EvidenceInfrastructureError("qsim experiment log cannot be read") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            invalid_records.append(
                (line_number, f"experiment log line {line_number} is not valid JSON")
            )
            continue
        if not isinstance(event, dict):
            invalid_records.append(
                (line_number, f"experiment log line {line_number} is not an object")
            )
            continue
        events.append((line_number, event))

    final_answer = _normalized_answer(final_payload)
    task_final_indices = [
        index
        for index, (_, event) in enumerate(events)
        if event.get("action", event.get("tool")) == "submit_final_answer"
        and event.get("task_id") == C.TASK_ID
    ]
    if not task_final_indices:
        detail = (
            invalid_records[0][1]
            if invalid_records
            else ("no final-answer submission for this task in qsim experiment log")
        )
        raise EvidenceInfrastructureError(detail)
    cutoff = task_final_indices[-1]
    cutoff_line = events[cutoff][0]
    invalid_before_final = [
        reason for line_number, reason in invalid_records if line_number < cutoff_line
    ]
    if invalid_before_final:
        raise EvidenceInfrastructureError(invalid_before_final[0])
    summary["final_event_index"] = cutoff
    if events[cutoff][1].get("answer") != final_answer:
        raise EvidenceInfrastructureError(
            "persisted final answer does not match the latest task submission"
        )
    before_final = [event for _, event in events[:cutoff]]

    submissions: dict[str, list[tuple[int, dict]]] = {}
    results: dict[str, list[tuple[int, dict]]] = {}
    polls: dict[str, list[tuple[int, dict]]] = {}
    for event_index, event in enumerate(before_final):
        action = event.get("action", event.get("tool"))
        job_id = event.get("job_id")
        if isinstance(action, str) and action.startswith("submit_run_"):
            if isinstance(job_id, str):
                submissions.setdefault(job_id, []).append((event_index, event))
        elif action == "get_job_result":
            if isinstance(job_id, str):
                polls.setdefault(job_id, []).append((event_index, event))
            elif event.get("status") == "complete":
                summary["evidence_reasons"].append(
                    "orphan completed poll before final submission has no valid job_id"
                )
            continue
        elif action in _EXPERIMENT_RESULT_ACTIONS:
            if isinstance(job_id, str):
                results.setdefault(job_id, []).append((event_index, event))
            else:
                summary["evidence_reasons"].append(
                    f"{action} before final submission has no valid job_id"
                )
            budget = event.get("budget")
            if not isinstance(budget, dict):
                summary["evidence_reasons"].append(
                    f"{action} for {job_id!r} has no qsim budget meter"
                )
                continue
            try:
                shots_used = _strict_nonnegative_int(budget.get("shots_used"), "budget.shots_used")
                shot_rounds_used = _strict_nonnegative_int(
                    budget.get("shot_rounds_used"), "budget.shot_rounds_used"
                )
            except ValueError as exc:
                summary["evidence_reasons"].append(f"{action} for {job_id!r}: {exc}")
                continue
            summary["shots_used"] = max(summary["shots_used"], shots_used)
            summary["shot_rounds_used"] = max(summary["shot_rounds_used"], shot_rounds_used)

    # Results may legitimately remain unpolled, and submitted jobs may fail or remain
    # running.  A known result without one earlier submission, or a *completed* poll
    # without the full earlier submit/result chain, cannot be emitted by qsim and makes
    # the official evidence log corrupt rather than merely incomplete.
    for job_id, job_results in results.items():
        job_submissions = submissions.get(job_id, [])
        for result_index, result in job_results:
            if len(job_submissions) != 1 or job_submissions[0][0] >= result_index:
                summary["evidence_reasons"].append(
                    f"orphan or out-of-order {result.get('action')} for job {job_id!r}"
                )
    for job_id, job_polls in polls.items():
        job_submissions = submissions.get(job_id, [])
        job_results = results.get(job_id, [])
        for poll_index, poll in job_polls:
            if poll.get("status") != "complete":
                continue
            if (
                len(job_submissions) != 1
                or len(job_results) != 1
                or not job_submissions[0][0] < job_results[0][0] < poll_index
            ):
                summary["evidence_reasons"].append(
                    f"orphan or out-of-order completed poll for job {job_id!r}"
                )

    def joined_job(job_id: str, result_action: str) -> tuple[int, dict, dict] | None:
        job_submissions = submissions.get(job_id, [])
        job_results = results.get(job_id, [])
        if len(job_submissions) != 1 or len(job_results) != 1:
            return None
        submit_index, submission = job_submissions[0]
        result_index, result = job_results[0]
        if result.get("action", result.get("tool")) != result_action:
            return None
        completed_poll_indices = [
            poll_index
            for poll_index, poll in polls.get(job_id, [])
            if poll.get("status") == "complete"
        ]
        if not any(
            submit_index < result_index < poll_index for poll_index in completed_poll_indices
        ):
            return None
        return submit_index, submission, result

    from qiqcbench.qsim.qtypes.surface_code_lattice_surgery import circuits as CC

    qualified_jobs: dict[str, tuple[int, int]] = {}
    d5_chunks: list[tuple[int, str, int]] = []
    cnot_chunks: dict[str, list[tuple[int, str, int]]] = {
        key: [] for key in sorted(_CNOT_EVIDENCE_KEYS)
    }
    for job_id, job_submissions in submissions.items():
        if len(job_submissions) != 1:
            continue
        event = job_submissions[0][1]
        request = event.get("request")
        if not isinstance(request, dict):
            continue
        action = event.get("action")
        if action == "submit_run_memory_experiment" and request.get("layout") == "d5":
            joined = joined_job(job_id, "run_memory_experiment_result")
            if joined is None:
                continue
            submit_index, _submission, result = joined
            try:
                submitted_shots = _strict_positive_int(request.get("shots"), "request.shots")
                result_shots = _strict_positive_int(result.get("shots"), "result.shots")
                request_rounds = _strict_positive_int(request.get("rounds"), "request.rounds")
                result_rounds = _strict_positive_int(result.get("rounds"), "result.rounds")
            except ValueError:
                continue
            if (
                result.get("experiment") != "memory"
                or result.get("layout") != "d5"
                or result_shots != submitted_shots
                or request_rounds != C.MEMORY_SCORING_ROUNDS
                or result_rounds != C.MEMORY_SCORING_ROUNDS
            ):
                continue
            d5_chunks.append((submit_index, job_id, submitted_shots))
            qualified_jobs[job_id] = (submitted_shots, submitted_shots * request_rounds)
        elif action == "submit_run_lattice_surgery_cnot":
            config = request.get("config")
            logical_input = request.get("logical_input")
            key = f"{config}:{logical_input}"
            if key not in _CNOT_EVIDENCE_KEYS:
                continue
            joined = joined_job(job_id, "run_lattice_surgery_cnot_result")
            if joined is None:
                continue
            submit_index, _submission, result = joined
            try:
                submitted_shots = _strict_positive_int(request.get("shots"), "request.shots")
                result_shots = _strict_positive_int(result.get("shots"), "result.shots")
                result_rounds = _strict_positive_int(result.get("rounds"), "result.rounds")
            except ValueError:
                continue
            if (
                result.get("experiment") != "cnot"
                or result.get("layout") != config
                or result_shots != submitted_shots
                or result_rounds != CC.CNOT_ROUNDS
            ):
                continue
            cnot_chunks[key].append((submit_index, job_id, submitted_shots))
            qualified_jobs[job_id] = (
                submitted_shots,
                submitted_shots * CC.CNOT_ROUNDS,
            )

    d5_chunks.sort()
    d5_shots = sum(shots for _, _, shots in d5_chunks)
    summary["d5_memory_shots"] = d5_shots
    if d5_shots < MIN_EVIDENCE_SHOTS_PER_CONFIGURATION:
        summary["evidence_reasons"].append(
            f"missing completed, observed fixed {C.MEMORY_SCORING_ROUNDS}-round d5 memory "
            "evidence in ordered submit -> result -> completed poll before final submission: "
            f"{d5_shots} fully joined shots; at least "
            f"{MIN_EVIDENCE_SHOTS_PER_CONFIGURATION} required"
        )
    else:
        d5_job_ids = [job_id for _, job_id, _ in d5_chunks]
        summary["d5_memory_job_ids"] = d5_job_ids
        summary["d5_memory_job_id"] = d5_job_ids[-1]

    for key, chunks in cnot_chunks.items():
        chunks.sort()
        shots = sum(chunk_shots for _, _, chunk_shots in chunks)
        summary["cnot_shots_by_entry"][key] = shots
        if shots < MIN_EVIDENCE_SHOTS_PER_CONFIGURATION:
            summary["evidence_reasons"].append(
                f"missing completed, observed CNOT battery entry {key} in ordered submit -> "
                f"result -> completed poll before final submission: {shots} fully joined shots; "
                f"at least {MIN_EVIDENCE_SHOTS_PER_CONFIGURATION} required"
            )
        else:
            summary["cnot_job_ids"][key] = [job_id for _, job_id, _ in chunks]

    qualified_shots = sum(shots for shots, _ in qualified_jobs.values())
    qualified_shot_rounds = sum(shot_rounds for _, shot_rounds in qualified_jobs.values())
    summary["qualified_evidence_shots"] = qualified_shots
    summary["qualified_evidence_shot_rounds"] = qualified_shot_rounds
    if summary["shots_used"] < qualified_shots:
        summary["evidence_reasons"].append(
            "qsim cumulative shot meter is below the cost of qualified evidence jobs: "
            f"{summary['shots_used']} < {qualified_shots}"
        )
    if summary["shot_rounds_used"] < qualified_shot_rounds:
        summary["evidence_reasons"].append(
            "qsim cumulative shot-rounds meter is below the cost of qualified evidence jobs: "
            f"{summary['shot_rounds_used']} < {qualified_shot_rounds}"
        )
    return summary


def score(
    answer: dict,
    *,
    budget_evidence: dict | None = None,
    replay_seed: int = C.SEED + 1,
    replay_shots_memory: int = C.REPLAY_SHOTS_MEMORY,
    replay_shots_per_entry: int = C.REPLAY_SHOTS_PER_ENTRY,
    replay_shots_bell: int = C.REPLAY_SHOTS_BELL,
) -> ScoreReport:
    if not isinstance(answer, dict):
        return _fail("invalid submission: top-level value must be an object")
    if isinstance(answer.get("answer"), dict):
        answer = answer["answer"]

    try:
        dem_mem, dec_mem, dem_cnot, dec_cnot = validate_submission(answer)
    except (ValueError, KeyError, TypeError) as exc:
        return _fail(f"invalid submission: {exc}")

    kw = {
        "decoder_memory": dec_mem,
        "decoder_cnot": dec_cnot,
        "n_mem": len(dem_mem),
        "n_cnot": len(dem_cnot),
    }

    # Budget gate: qsim-owned evidence that experiments were run under the finite budgets.
    if budget_evidence is None:
        return _fail(
            "no qsim experiment evidence found: you must run experiments under the shot "
            "budget before submitting",
            **kw,
        )
    evidence_reasons = list(budget_evidence.get("evidence_reasons", []))
    if not budget_evidence.get("d5_memory_job_id"):
        evidence_reasons.append("missing completed, observed d5 memory evidence")
    missing_cnot = _CNOT_EVIDENCE_KEYS - set(budget_evidence.get("cnot_job_ids", {}))
    if missing_cnot:
        evidence_reasons.append(
            f"missing completed, observed CNOT battery evidence: {sorted(missing_cnot)}"
        )
    if evidence_reasons:
        return _fail("incomplete qsim experiment evidence: " + "; ".join(evidence_reasons), **kw)
    try:
        shots_used = _strict_nonnegative_int(budget_evidence.get("shots_used"), "shots_used")
        shot_rounds_used = _strict_nonnegative_int(
            budget_evidence.get("shot_rounds_used"), "shot_rounds_used"
        )
    except ValueError as exc:
        return _fail(f"invalid qsim budget evidence: {exc}", **kw)
    if shots_used <= 0:
        return _fail("no shots used: the DEMs must be estimated from experiment data", **kw)
    if shots_used > C.SHOT_BUDGET:
        return _fail(f"shot budget exceeded: {shots_used} > {C.SHOT_BUDGET}", **kw)
    if shot_rounds_used > C.SHOT_ROUNDS_BUDGET:
        return _fail(
            f"shot-rounds budget exceeded: {shot_rounds_used} > {C.SHOT_ROUNDS_BUDGET}", **kw
        )

    penalties: list[str] = []
    flags: list[str] = []
    shot_count_flag = reported_shot_count_flag(answer, shots_used)
    if shot_count_flag:
        flags.append(shot_count_flag)
    noise = D.true_noise()
    rng = np.random.default_rng(replay_seed)

    # --- memory gate ---
    try:
        _, mem_rate = D.replay_memory(
            dem_mem, dec_mem, shots=replay_shots_memory, rng=rng, noise=noise
        )
    except (ValueError, KeyError, TypeError) as exc:
        return _fail(f"memory replay failed: {exc}", **kw)
    memory_ok = mem_rate < C.FLOOR_MEM

    # --- CNOT gate ---
    try:
        battery = D.replay_cnot_battery(
            dem_cnot, dec_cnot, shots_per_entry=replay_shots_per_entry, rng=rng, noise=noise
        )
    except (ValueError, KeyError, TypeError) as exc:
        return _fail(
            f"cnot replay failed: {exc}",
            memory_gate_passed=memory_ok,
            mem_rate=mem_rate,
            **kw,
        )
    cnot_ok = battery["average"] < C.FLOOR_CNOT and battery["max_entry"] < C.CNOT_PER_ENTRY_CAP

    passed = memory_ok and cnot_ok
    if not memory_ok:
        penalties.append(
            f"memory gate failed: replayed {mem_rate:.3e}/cycle >= floor {C.FLOOR_MEM:.3e}"
        )
    if not cnot_ok:
        penalties.append(
            f"cnot gate failed: replayed avg {battery['average']:.3e} (floor "
            f"{C.FLOOR_CNOT:.3e}), max entry {battery['max_entry']:.3e} (cap "
            f"{C.CNOT_PER_ENTRY_CAP:.3e})"
        )

    # --- ungated derived figures ---
    lam = None
    try:
        d3_dem = answer.get("dem_memory_d3")
        if isinstance(d3_dem, list) and d3_dem:
            d3_mechs = d3_dem
        else:
            d3_mechs = D.reference_d3_dem()
        _, d3_rate = D.replay_memory(
            d3_mechs,
            dec_mem,
            layout="d3_control",
            shots=min(replay_shots_memory, 200_000),
            rng=rng,
            noise=noise,
        )
        lam = float(d3_rate / max(mem_rate, 1e-9))
        sub_lam = float(answer.get("lambda_d3_d5_raw"))
        if lam > 0 and (sub_lam / lam > 3.0 or lam / sub_lam > 3.0):
            flags.append(f"lambda_d3_d5_raw {sub_lam:.2f} disagrees >3x with replayed {lam:.2f}")
    except (ValueError, KeyError, TypeError) as exc:  # ungated: never fail the run
        flags.append(f"lambda recompute skipped: {exc}")

    f_bell = None
    try:
        bell = D.replay_bell_witness(
            dem_cnot, dec_cnot, shots=replay_shots_bell, rng=rng, noise=noise
        )
        f_bell = bell["f_bell_lower_bound"]
        fb_sub = answer.get("f_bell_logical_raw")
        if fb_sub is not None:
            inf_sub, inf_rep = 1.0 - float(fb_sub), 1.0 - f_bell
            if inf_rep > 1e-6 and (inf_sub / inf_rep > 3.0 or inf_rep / max(inf_sub, 1e-9) > 3.0):
                flags.append(
                    f"f_bell_logical_raw {fb_sub:.3f} infidelity disagrees >3x with "
                    f"derived {f_bell:.3f}"
                )
    except (ValueError, KeyError, TypeError) as exc:
        flags.append(f"bell witness skipped: {exc}")

    # self-report consistency (never trusted, only flagged)
    self_ok = True
    try:
        if abs(float(answer["self_measured_memory_logical_rate_raw"]) - mem_rate) > max(
            3.0e-3, 1.5 * mem_rate
        ):
            self_ok = False
            flags.append("self-reported memory rate far from replayed value")
        if abs(float(answer["self_measured_cnot_error_raw"]) - battery["average"]) > max(
            5.0e-2, 1.0 * battery["average"]
        ):
            self_ok = False
            flags.append("self-reported cnot error far from replayed value")
    except (TypeError, ValueError, KeyError):
        self_ok = None

    return ScoreReport(
        task_id=C.TASK_ID,
        reward=1 if passed else 0,
        passed=passed,
        memory_gate_passed=memory_ok,
        cnot_gate_passed=cnot_ok,
        replayed_memory_rate_per_cycle=mem_rate,
        replayed_cnot_error_average=battery["average"],
        replayed_cnot_entries=battery["entries"],
        floor_mem=C.FLOOR_MEM,
        floor_cnot=C.FLOOR_CNOT,
        per_entry_cap=C.CNOT_PER_ENTRY_CAP,
        lambda_replayed=lam,
        f_bell_lower_bound=f_bell,
        decoder_memory=dec_mem,
        decoder_cnot=dec_cnot,
        n_mechanisms_memory=len(dem_mem),
        n_mechanisms_cnot=len(dem_cnot),
        self_report_consistent=self_ok,
        penalties=penalties,
        flags=flags,
    )


def recover_budget_evidence(log_path) -> dict | None:
    """Return legacy budget-only diagnostics from the qsim experiment log.

    This compatibility helper is not sufficient evidence for ``score()``. Production
    verifiers should use ``recover_experiment_evidence`` to bind budgets to completed,
    observed jobs before the matching final submission.
    """
    import json
    from pathlib import Path

    path = Path(log_path)
    if not path.exists():
        return None
    shots = rounds = 0
    seen = False
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        b = ev.get("budget")
        if isinstance(b, dict) and "shots_used" in b:
            try:
                event_shots = _strict_nonnegative_int(b.get("shots_used"), "budget.shots_used")
                event_shot_rounds = _strict_nonnegative_int(
                    b.get("shot_rounds_used"), "budget.shot_rounds_used"
                )
            except ValueError:
                return None
            seen = True
            shots = max(shots, event_shots)
            rounds = max(rounds, event_shot_rounds)
    return {"shots_used": shots, "shot_rounds_used": rounds} if seen else None


__all__ = [
    "EvidenceInfrastructureError",
    "MIN_EVIDENCE_SHOTS_PER_CONFIGURATION",
    "ScoreReport",
    "model_failure_report",
    "recover_budget_evidence",
    "recover_experiment_evidence",
    "reported_shot_count_flag",
    "score",
    "validate_submission",
]
