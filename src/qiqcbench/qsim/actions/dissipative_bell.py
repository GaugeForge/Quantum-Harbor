"""Action seam for the driven-dissipative array qtype.

Capability-gated (``local_reservoir_stabilization``) submission of stabilize-and-
measure jobs. Beyond the usual submission log, the completed-job closure writes a
compact ``stabilization_evidence`` integrity summary (per-duration outcome
counts, plus the pair / reservoir settings / measurement bases) into the private
qsim log. The verifier cross-checks that summary but recomputes the scored raw
estimator from the digest-bound per-shot bitstrings published by
``get_job_result``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.actions.common import release_job_admission, reserve_job_admission
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.jobs import new_job_id
from qiqcbench.qsim.qtypes.driven_dissipative_transmon_array.wire import (
    StabilizeRequest,
    StabilizeSweepRequest,
)

_CAPABILITY = "local_reservoir_stabilization"
_EXPERIMENT_JOBS_BUDGET_KEY = "driven_dissipative_transmon_array:experiment_jobs"
_RAW_SHOT_RECORDS_BUDGET_KEY = "driven_dissipative_transmon_array:raw_shot_records"
_JOB_RESULT_POLLS_BUDGET_KEY = "driven_dissipative_transmon_array:job_result_polls"


def _require_capability(ctx: ActionContext, action: str) -> None:
    if _CAPABILITY not in ctx.state.active_capabilities:
        raise ValueError(f"{_CAPABILITY} capability is not active for this run; cannot {action}")


def _preflight_raw_records(
    ctx: ActionContext,
    *,
    shots: int,
    point_count: int,
    sweep: bool,
) -> int:
    """Bound known work/result shape before a queue slot or budget is consumed."""

    public = ctx.state.public
    budget = public.budget
    if shots > public.max_shots:
        raise ValueError(f"shots exceeds public max_shots={public.max_shots}")
    if sweep and point_count > budget.max_sweep_points_per_call:
        raise ValueError(
            f"duration_grid_us has {point_count} points; public per-call limit is "
            f"max_sweep_points_per_call={budget.max_sweep_points_per_call}"
        )
    raw_records = point_count * shots
    if raw_records > budget.max_raw_shot_records_per_call:
        raise ValueError(
            f"duration points x shots requests {raw_records} raw shot records; public "
            "per-call limit is max_raw_shot_records_per_call="
            f"{budget.max_raw_shot_records_per_call}"
        )
    return raw_records


def _rollback_submission_resources(
    ctx: ActionContext,
    *,
    job_id: str,
    raw_records: int,
    budgets_reserved: bool,
) -> None:
    unregister = getattr(ctx.state, "unregister_job_result_poll_budget", None)
    if callable(unregister):
        unregister(job_id=job_id)
    if budgets_reserved:
        ctx.state.release_action_budgets(
            reservations={
                _EXPERIMENT_JOBS_BUDGET_KEY: 1,
                _RAW_SHOT_RECORDS_BUDGET_KEY: raw_records,
            }
        )
    release_job_admission(ctx.state, job_id)


def _reserve_submission_resources(
    ctx: ActionContext,
    *,
    job_id: str,
    raw_records: int,
) -> dict[str, int]:
    """Atomically bind queue, cumulative evidence, and poll admission."""

    budget = ctx.state.public.budget
    reserve_job_admission(ctx.state, job_id)
    budgets_reserved = False
    try:
        evidence = ctx.state.reserve_action_budgets(
            reservations={
                _EXPERIMENT_JOBS_BUDGET_KEY: (1, budget.max_experiment_jobs),
                _RAW_SHOT_RECORDS_BUDGET_KEY: (
                    raw_records,
                    budget.max_raw_shot_records_total,
                ),
            }
        )
        budgets_reserved = True
        ctx.state.register_job_result_poll_budget(
            job_id=job_id,
            budget_key=_JOB_RESULT_POLLS_BUDGET_KEY,
            limit=budget.max_job_result_polls,
        )
    except Exception:
        _rollback_submission_resources(
            ctx,
            job_id=job_id,
            raw_records=raw_records,
            budgets_reserved=budgets_reserved,
        )
        raise
    return {
        "experiment_jobs_used": evidence[_EXPERIMENT_JOBS_BUDGET_KEY][1],
        "raw_shot_records_used": evidence[_RAW_SHOT_RECORDS_BUDGET_KEY][1],
    }


def _best_effort_submission_failure(
    ctx: ActionContext,
    *,
    tool: str,
    job_id: str,
    failure_stage: str,
) -> None:
    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": "stabilization_submission_failure",
                "tool": tool,
                "job_id": job_id,
                "failure_kind": "qsim_internal",
                "failure_stage": failure_stage,
            }
        )
    except Exception:
        return


def _log_evidence(
    ctx: ActionContext,
    *,
    tool: str,
    job_id: str,
    request: StabilizeRequest | StabilizeSweepRequest,
    result: JobResult,
) -> None:
    """Write per-duration outcome counts for a completed stabilization job.

    The verifier cross-checks this qsim-owned summary against the raw artifact;
    it never substitutes these counts for the per-shot evidence.
    """
    if result.status != "complete" or result.data is None:
        return
    durations = (result.metadata.sweep_coords or {}).get("duration_us", [])
    bases = [m.basis for m in request.measure]
    records = []
    for i, per_shot in enumerate(result.data.bitstrings):
        records.append(
            {
                "duration_us": float(durations[i]) if i < len(durations) else None,
                "counts": dict(Counter(per_shot)),
            }
        )
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "stabilization_evidence",
            "tool": tool,
            "job_id": job_id,
            "pair": request.pair,
            "initial_state": request.initial_state,
            "delta_s_mhz": request.delta_s_mhz,
            "delta_d_mhz": request.delta_d_mhz,
            "g_s_mhz": request.g_s_mhz,
            "g_d_mhz": request.g_d_mhz,
            "measured_sites": list(result.data.measured_qubits),
            "bases": bases,
            "records": records,
        }
    )


def submit_stabilization(
    ctx: ActionContext,
    *,
    device_id: str,
    pair: str,
    initial_state: str,
    delta_s_mhz: float,
    delta_d_mhz: float,
    g_s_mhz: float,
    g_d_mhz: float,
    duration_us: float,
    measure: list[dict[str, Any]],
    shots: int,
) -> dict[str, Any]:
    """Submit a single prepare-stabilize-measure request through the active backend."""
    _require_capability(ctx, "submit stabilization")
    request = StabilizeRequest.model_validate(
        {
            "shots": shots,
            "pair": pair,
            "initial_state": initial_state,
            "delta_s_mhz": delta_s_mhz,
            "delta_d_mhz": delta_d_mhz,
            "g_s_mhz": g_s_mhz,
            "g_d_mhz": g_d_mhz,
            "duration_us": duration_us,
            "measure": measure,
        }
    )
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    raw_records = _preflight_raw_records(ctx, shots=request.shots, point_count=1, sweep=False)
    job_id = new_job_id()
    resource_usage = _reserve_submission_resources(
        ctx,
        job_id=job_id,
        raw_records=raw_records,
    )
    try:
        salt = ctx.state.next_salt()
    except Exception:
        _rollback_submission_resources(
            ctx,
            job_id=job_id,
            raw_records=raw_records,
            budgets_reserved=True,
        )
        raise RuntimeError("qsim stabilization entropy allocation failed") from None

    def runner(job_id: str) -> JobResult:
        result = ctx.state.backend.run_stabilization(request, job_id, salt)
        _log_evidence(ctx, tool="run_stabilization", job_id=job_id, request=request, result=result)
        return result

    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": "submit_stabilization",
                "tool": "run_stabilization",
                "device_id": device_id,
                "shots": shots,
                "pair": pair,
                "initial_state": initial_state,
                "delta_s_mhz": delta_s_mhz,
                "delta_d_mhz": delta_d_mhz,
                "g_s_mhz": g_s_mhz,
                "g_d_mhz": g_d_mhz,
                "duration_us": duration_us,
                # The VALIDATED request, not the raw keyword argument: qsim
                # executes the normalized MeasureSetting, so the submission
                # event must record the canonical types it actually ran. Logging
                # the raw dicts let a JSON slip qsim accepted (site 0.0) reach
                # the verifier's exact-type binding and void a scoreable trial.
                "measure": [m.model_dump() for m in request.measure],
                "job_id": job_id,
                "raw_shot_records_this_call": raw_records,
                "max_raw_shot_records_per_call": (
                    ctx.state.public.budget.max_raw_shot_records_per_call
                ),
                "max_raw_shot_records_total": ctx.state.public.budget.max_raw_shot_records_total,
                "max_experiment_jobs": ctx.state.public.budget.max_experiment_jobs,
                **resource_usage,
            }
        )
    except Exception:
        _rollback_submission_resources(
            ctx,
            job_id=job_id,
            raw_records=raw_records,
            budgets_reserved=True,
        )
        _best_effort_submission_failure(
            ctx,
            tool="run_stabilization",
            job_id=job_id,
            failure_stage="submission_log_publication",
        )
        raise RuntimeError("qsim stabilization submission publication failed") from None
    try:
        ctx.state.jobs.submit(runner, job_id=job_id)
    except Exception:
        _rollback_submission_resources(
            ctx,
            job_id=job_id,
            raw_records=raw_records,
            budgets_reserved=True,
        )
        _best_effort_submission_failure(
            ctx,
            tool="run_stabilization",
            job_id=job_id,
            failure_stage="job_enqueue",
        )
        raise RuntimeError("qsim stabilization enqueue failed") from None
    return {"job_id": job_id, "status": "queued"}


def submit_stabilization_sweep(
    ctx: ActionContext,
    *,
    device_id: str,
    pair: str,
    initial_state: str,
    delta_s_mhz: float,
    delta_d_mhz: float,
    g_s_mhz: float,
    g_d_mhz: float,
    duration_grid_us: list[float],
    measure: list[dict[str, Any]],
    shots: int,
) -> dict[str, Any]:
    """Submit a stabilize-and-measure sweep over a grid of stabilization durations."""
    _require_capability(ctx, "submit stabilization sweep")
    request = StabilizeSweepRequest.model_validate(
        {
            "shots": shots,
            "pair": pair,
            "initial_state": initial_state,
            "delta_s_mhz": delta_s_mhz,
            "delta_d_mhz": delta_d_mhz,
            "g_s_mhz": g_s_mhz,
            "g_d_mhz": g_d_mhz,
            "duration_grid_us": duration_grid_us,
            "measure": measure,
        }
    )
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    raw_records = _preflight_raw_records(
        ctx,
        shots=request.shots,
        point_count=len(request.duration_grid_us),
        sweep=True,
    )
    job_id = new_job_id()
    resource_usage = _reserve_submission_resources(
        ctx,
        job_id=job_id,
        raw_records=raw_records,
    )
    try:
        salt = ctx.state.next_salt()
    except Exception:
        _rollback_submission_resources(
            ctx,
            job_id=job_id,
            raw_records=raw_records,
            budgets_reserved=True,
        )
        raise RuntimeError("qsim stabilization-sweep entropy allocation failed") from None

    def runner(job_id: str) -> JobResult:
        result = ctx.state.backend.run_stabilization_sweep(request, job_id, salt)
        _log_evidence(
            ctx, tool="run_stabilization_sweep", job_id=job_id, request=request, result=result
        )
        return result

    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": "submit_stabilization_sweep",
                "tool": "run_stabilization_sweep",
                "device_id": device_id,
                "shots": shots,
                "pair": pair,
                "initial_state": initial_state,
                "delta_s_mhz": delta_s_mhz,
                "delta_d_mhz": delta_d_mhz,
                "g_s_mhz": g_s_mhz,
                "g_d_mhz": g_d_mhz,
                "n_points": len(duration_grid_us),
                "duration_grid_us": [float(t) for t in duration_grid_us],
                # The VALIDATED request, not the raw keyword argument: qsim
                # executes the normalized MeasureSetting, so the submission
                # event must record the canonical types it actually ran. Logging
                # the raw dicts let a JSON slip qsim accepted (site 0.0) reach
                # the verifier's exact-type binding and void a scoreable trial.
                "measure": [m.model_dump() for m in request.measure],
                "job_id": job_id,
                "raw_shot_records_this_call": raw_records,
                "max_raw_shot_records_per_call": (
                    ctx.state.public.budget.max_raw_shot_records_per_call
                ),
                "max_raw_shot_records_total": ctx.state.public.budget.max_raw_shot_records_total,
                "max_experiment_jobs": ctx.state.public.budget.max_experiment_jobs,
                **resource_usage,
            }
        )
    except Exception:
        _rollback_submission_resources(
            ctx,
            job_id=job_id,
            raw_records=raw_records,
            budgets_reserved=True,
        )
        _best_effort_submission_failure(
            ctx,
            tool="run_stabilization_sweep",
            job_id=job_id,
            failure_stage="submission_log_publication",
        )
        raise RuntimeError("qsim stabilization-sweep submission publication failed") from None
    try:
        ctx.state.jobs.submit(runner, job_id=job_id)
    except Exception:
        _rollback_submission_resources(
            ctx,
            job_id=job_id,
            raw_records=raw_records,
            budgets_reserved=True,
        )
        _best_effort_submission_failure(
            ctx,
            tool="run_stabilization_sweep",
            job_id=job_id,
            failure_stage="job_enqueue",
        )
        raise RuntimeError("qsim stabilization-sweep enqueue failed") from None
    return {"job_id": job_id, "status": "queued"}
