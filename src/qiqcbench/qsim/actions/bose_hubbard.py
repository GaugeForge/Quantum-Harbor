"""Surface-neutral Bose-Hubbard evolution actions and evidence lifecycle."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.actions.common import release_job_admission, reserve_job_admission
from qiqcbench.qsim.backends.provider_artifacts import canonical_request_hash
from qiqcbench.qsim.core.wire import (
    BoseHubbardEvolveRequest,
    BoseHubbardEvolveSweepRequest,
    JobResult,
)
from qiqcbench.qsim.jobs import new_job_id

_CAPABILITY = "bose_hubbard_spectroscopy"
_QTYPE = "bose_hubbard_chain"
_EXPERIMENT_JOBS_BUDGET_KEY = "bose_hubbard_chain:experiment_jobs"
_RAW_SHOT_RECORDS_BUDGET_KEY = "bose_hubbard_chain:raw_shot_records"
_JOB_RESULT_POLLS_BUDGET_KEY = "bose_hubbard_chain:job_result_polls"


def _require_capability(ctx: ActionContext, action: str) -> None:
    if _CAPABILITY not in ctx.state.active_capabilities:
        raise ValueError(f"{_CAPABILITY} capability is not active for this run; cannot {action}")
    if getattr(ctx.state.hidden, "qtype", None) != _QTYPE:
        raise ValueError(f"{action} requires qtype={_QTYPE!r}")


def _preflight_request(
    ctx: ActionContext,
    request: BoseHubbardEvolveRequest | BoseHubbardEvolveSweepRequest,
) -> int:
    """Validate public controls and bound result shape before any side effect."""

    public = ctx.state.public
    budget = public.budget
    if request.shots > public.max_shots:
        raise ValueError(f"shots exceeds public max_shots={public.max_shots}")

    site_ids = {site.id for site in public.sites}
    init_sites = list(request.init_excited_sites)
    measured_sites = [setting.site for setting in request.measure]
    if len(init_sites) != len(set(init_sites)):
        raise ValueError("init_excited_sites must be distinct")
    if len(init_sites) > public.max_excitations:
        raise ValueError(f"at most {public.max_excitations} excited sites may be prepared")
    if any(site not in site_ids for site in init_sites):
        raise ValueError(f"init_excited_sites must be among {sorted(site_ids)!r}")
    if len(measured_sites) != len(set(measured_sites)):
        raise ValueError("measure sites must be distinct")
    if any(site not in site_ids for site in measured_sites):
        raise ValueError(f"measure sites must be among {sorted(site_ids)!r}")

    is_sweep = isinstance(request, BoseHubbardEvolveSweepRequest)
    times = list(request.time_grid_ns) if is_sweep else [float(request.evolution_time_ns)]
    if is_sweep and len(times) > budget.max_sweep_points_per_call:
        raise ValueError(
            f"time_grid_ns has {len(times)} points; public per-call limit is "
            f"max_sweep_points_per_call={budget.max_sweep_points_per_call}"
        )
    if any(
        not math.isfinite(time_ns) or time_ns < 0 or time_ns > public.max_evolution_time_ns
        for time_ns in times
    ):
        raise ValueError(
            f"evolution time must be finite and in [0, {public.max_evolution_time_ns}] ns"
        )
    resolution = public.time_resolution_ns
    for time_ns in times:
        coordinate = time_ns / resolution
        if not math.isclose(coordinate, round(coordinate), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"evolution times must align to public time_resolution_ns={resolution:g}"
            )

    raw_records = len(times) * request.shots
    if raw_records > budget.max_raw_shot_records_per_call:
        raise ValueError(
            f"time points x shots requests {raw_records} raw shot records; public per-call "
            "limit is max_raw_shot_records_per_call="
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
                "action": "bose_hubbard_submission_infrastructure_failure",
                "tool": tool,
                "job_id": job_id,
                "failure_kind": "qsim_internal",
                "failure_stage": failure_stage,
            }
        )
    except Exception:
        return


def _canonical_data_digest(data: Any) -> str:
    raw = json.dumps(
        data.model_dump(mode="json"),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _log_result_evidence(
    ctx: ActionContext,
    *,
    tool: str,
    job_id: str,
    request: BoseHubbardEvolveRequest | BoseHubbardEvolveSweepRequest,
    result: JobResult,
) -> None:
    if result.status != "complete" or result.data is None:
        return
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "bose_hubbard_result_evidence",
            "tool": tool,
            "evidence_schema_version": 1,
            "job_id": job_id,
            "request_digest": canonical_request_hash(request),
            "raw_data_sha256": _canonical_data_digest(result.data),
            "shots": result.shots,
            "result_kind": result.data.kind,
        }
    )


def _submit(
    ctx: ActionContext,
    *,
    device_id: str,
    tool: str,
    submission_action: str,
    request: BoseHubbardEvolveRequest | BoseHubbardEvolveSweepRequest,
    raw_records: int,
    runner_factory: Any,
) -> dict[str, Any]:
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
        raise RuntimeError("qsim Bose-Hubbard entropy allocation failed") from None

    def runner(allocated_job_id: str) -> JobResult:
        result = runner_factory(allocated_job_id, salt)
        _log_result_evidence(
            ctx,
            tool=tool,
            job_id=allocated_job_id,
            request=request,
            result=result,
        )
        return result

    budget = ctx.state.public.budget
    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": submission_action,
                "tool": tool,
                "device_id": device_id,
                "task_id": ctx.state.task_id,
                "job_id": job_id,
                "request": request.model_dump(mode="json"),
                "request_digest": canonical_request_hash(request),
                "raw_shot_records_this_call": raw_records,
                "max_raw_shot_records_total": budget.max_raw_shot_records_total,
                "max_experiment_jobs": budget.max_experiment_jobs,
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
            tool=tool,
            job_id=job_id,
            failure_stage="submission_log_publication",
        )
        raise RuntimeError("qsim Bose-Hubbard submission publication failed") from None
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
            tool=tool,
            job_id=job_id,
            failure_stage="job_enqueue",
        )
        raise RuntimeError("qsim Bose-Hubbard enqueue failed") from None
    return {"job_id": job_id, "status": "queued"}


def submit_evolution(
    ctx: ActionContext,
    *,
    device_id: str,
    init_excited_sites: list[int],
    evolution_time_ns: float,
    measure: list[dict[str, Any]],
    shots: int,
) -> dict[str, Any]:
    """Submit one prepare-evolve-measure request through the active backend."""

    _require_capability(ctx, "submit evolution")
    request = BoseHubbardEvolveRequest.model_validate(
        {
            "shots": shots,
            "init_excited_sites": init_excited_sites,
            "evolution_time_ns": evolution_time_ns,
            "measure": measure,
        }
    )
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    raw_records = _preflight_request(ctx, request)

    def runner_factory(job_id: str, salt: int) -> JobResult:
        return ctx.state.backend.run_evolution(request, job_id, salt)

    return _submit(
        ctx,
        device_id=device_id,
        tool="run_evolution",
        submission_action="submit_evolution",
        request=request,
        raw_records=raw_records,
        runner_factory=runner_factory,
    )


def submit_evolution_sweep(
    ctx: ActionContext,
    *,
    device_id: str,
    init_excited_sites: list[int],
    time_grid_ns: list[float],
    measure: list[dict[str, Any]],
    shots: int,
) -> dict[str, Any]:
    """Submit an evolve-and-measure sweep over a grid of evolution times."""

    _require_capability(ctx, "submit evolution sweep")
    request = BoseHubbardEvolveSweepRequest.model_validate(
        {
            "shots": shots,
            "init_excited_sites": init_excited_sites,
            "time_grid_ns": time_grid_ns,
            "measure": measure,
        }
    )
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    raw_records = _preflight_request(ctx, request)

    def runner_factory(job_id: str, salt: int) -> JobResult:
        return ctx.state.backend.run_evolution_sweep(request, job_id, salt)

    return _submit(
        ctx,
        device_id=device_id,
        tool="run_evolution_sweep",
        submission_action="submit_evolution_sweep",
        request=request,
        raw_records=raw_records,
        runner_factory=runner_factory,
    )


__all__ = ["submit_evolution", "submit_evolution_sweep"]
