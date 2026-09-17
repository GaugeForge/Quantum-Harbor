"""Action seam for the g-f erasure-qutrit qtype.

Gates the ``run_logical_memory`` / ``run_logical_memory_sweep`` tools on the
``mid_circuit_erasure_detection`` capability (fail-closed at the action layer,
not only at surface registration) and logs compact per-point summaries on job
completion. The memory verifier binds those summaries to cited raw artifacts;
it evaluates the pre-final-readout lifetime independently from hidden dynamics.
"""

from __future__ import annotations

from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.actions.common import release_job_admission, reserve_job_admission
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.jobs import new_job_id
from qiqcbench.qsim.qtypes.transmon_erasure_qutrit.wire import (
    LogicalMemoryRequest,
    LogicalMemorySweepRequest,
)

_CAPABILITY = "mid_circuit_erasure_detection"
_EXPERIMENT_JOBS_BUDGET_KEY = "transmon_erasure_qutrit:experiment_jobs"
_FINAL_ASSIGNMENTS_BUDGET_KEY = "transmon_erasure_qutrit:final_assignments"
_SYNDROME_BITS_BUDGET_KEY = "transmon_erasure_qutrit:syndrome_bits"
_JOB_RESULT_POLLS_BUDGET_KEY = "transmon_erasure_qutrit:job_result_polls"

# Expected (no-error) final outcome per prepared logical state: 0=g, 2=f.
_EXPECTED_OUTCOME = {"0L": 0, "1L": 2, "+X": 0, "-X": 2}
# Final qutrit readout outside the logical code space {|g>, |f>} is leakage.
_LEAK_OUTCOME = 1


def _require_capability(ctx: ActionContext, action: str) -> None:
    if _CAPABILITY not in ctx.state.active_capabilities:
        raise ValueError(f"{_CAPABILITY} capability is not active for this run; cannot {action}")


def _point_aggregate(point: Any, expected: int) -> dict[str, Any]:
    if hasattr(point, "mid_circuit_ancilla_post_readout_bitstrings"):
        flags = [int("1" in bits) for bits in point.mid_circuit_ancilla_post_readout_bitstrings]
        outcomes = list(point.final_qutrit_post_readout_assignments)
    else:
        flags = list(point.erasure_flags)
        outcomes = list(point.final_outcomes)
    n_shots = len(flags)
    n_no_flag = 0
    n_no_flag_codespace = 0
    n_no_flag_correct = 0
    n_leak = 0
    for flag, out in zip(flags, outcomes, strict=True):
        if out == _LEAK_OUTCOME:
            n_leak += 1
        if flag == 0:
            n_no_flag += 1
            if out != _LEAK_OUTCOME:
                n_no_flag_codespace += 1
            if out == expected:
                n_no_flag_correct += 1
    return {
        "n_rounds": point.n_rounds,
        "total_evolution_us": point.total_evolution_us,
        "n_shots": n_shots,
        "n_flag": n_shots - n_no_flag,
        "n_no_flag": n_no_flag,
        # Post-selected code-space population: never flagged AND final readout in
        # {|g>, |f>}. A final |e> is leakage, not a logical bit flip.
        "n_no_flag_codespace": n_no_flag_codespace,
        "n_no_flag_correct": n_no_flag_correct,
        "n_leak_outcome": n_leak,
    }


def _log_evidence(ctx: ActionContext, *, tool: str, request: Any, result: JobResult) -> None:
    if result.status != "complete" or result.data is None:
        return
    data = result.data
    if getattr(data, "kind", None) not in {
        "erasure_memory",
        "erasure_memory_round_resolved",
    }:
        return
    expected = _EXPECTED_OUTCOME.get(request.prep_state, -1)
    points = [_point_aggregate(p, expected) for p in data.points]
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "logical_memory_evidence",
            "tool": tool,
            "device_id": ctx.state.hidden.device_id,
            "job_id": result.job_id,
            "result_kind": data.kind,
            "prep_state": request.prep_state,
            "prep_basis": request.prep_basis,
            "measure_basis": request.measure_basis,
            "cycle_time_us": float(request.cycle_time_us),
            "dd": request.dd,
            "points": points,
        }
    )


def _preflight_raw_records(
    ctx: ActionContext,
    *,
    shots: int,
    n_rounds_grid: list[int],
) -> tuple[int, int]:
    """Validate bounded raw-result growth before allocating a job."""

    public = ctx.state.public
    budget = public.budget
    point_count = len(n_rounds_grid)
    if shots > public.max_shots:
        raise ValueError(f"shots exceeds public max_shots={public.max_shots}")
    if point_count > budget.max_sweep_points_per_call:
        raise ValueError(
            f"n_rounds_grid has {point_count} points; public per-call limit is "
            f"max_sweep_points_per_call={budget.max_sweep_points_per_call}"
        )
    final_assignments = shots * point_count
    if final_assignments > budget.max_final_assignments_per_call:
        raise ValueError(
            f"points x shots requests {final_assignments} final assignments; public "
            "per-call limit is max_final_assignments_per_call="
            f"{budget.max_final_assignments_per_call}"
        )
    syndrome_bits = shots * sum(n_rounds_grid)
    if syndrome_bits > budget.max_syndrome_bits_per_call:
        raise ValueError(
            f"sum(n_rounds_grid) x shots requests {syndrome_bits} syndrome bits; public "
            f"per-call limit is max_syndrome_bits_per_call={budget.max_syndrome_bits_per_call}"
        )
    return final_assignments, syndrome_bits


def _rollback_submission_resources(
    ctx: ActionContext,
    *,
    job_id: str,
    final_assignments: int,
    syndrome_bits: int,
    budgets_reserved: bool,
) -> None:
    unregister = getattr(ctx.state, "unregister_job_result_poll_budget", None)
    if callable(unregister):
        unregister(job_id=job_id)
    if budgets_reserved:
        ctx.state.release_action_budgets(
            reservations={
                _EXPERIMENT_JOBS_BUDGET_KEY: 1,
                _FINAL_ASSIGNMENTS_BUDGET_KEY: final_assignments,
                _SYNDROME_BITS_BUDGET_KEY: syndrome_bits,
            }
        )
    release_job_admission(ctx.state, job_id)


def _reserve_submission_resources(
    ctx: ActionContext,
    *,
    job_id: str,
    final_assignments: int,
    syndrome_bits: int,
) -> dict[str, int]:
    budget = ctx.state.public.budget
    reserve_job_admission(ctx.state, job_id)
    budgets_reserved = False
    try:
        evidence = ctx.state.reserve_action_budgets(
            reservations={
                _EXPERIMENT_JOBS_BUDGET_KEY: (1, budget.max_experiment_jobs),
                _FINAL_ASSIGNMENTS_BUDGET_KEY: (
                    final_assignments,
                    budget.max_final_assignments_total,
                ),
                _SYNDROME_BITS_BUDGET_KEY: (
                    syndrome_bits,
                    budget.max_syndrome_bits_total,
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
            final_assignments=final_assignments,
            syndrome_bits=syndrome_bits,
            budgets_reserved=budgets_reserved,
        )
        raise
    return {
        "experiment_jobs_used": evidence[_EXPERIMENT_JOBS_BUDGET_KEY][1],
        "final_assignments_used": evidence[_FINAL_ASSIGNMENTS_BUDGET_KEY][1],
        "syndrome_bits_used": evidence[_SYNDROME_BITS_BUDGET_KEY][1],
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
                "action": "logical_memory_submission_failure",
                "tool": tool,
                "job_id": job_id,
                "failure_kind": "qsim_internal",
                "failure_stage": failure_stage,
            }
        )
    except Exception:
        return


def submit_logical_memory(
    ctx: ActionContext,
    *,
    device_id: str,
    prep_basis: str,
    prep_state: str,
    measure_basis: str,
    n_rounds: int,
    cycle_time_us: float,
    dd: str,
    shots: int,
) -> dict[str, Any]:
    """Submit a single logical-memory experiment at a fixed number of rounds."""
    _require_capability(ctx, "submit logical memory")
    request = LogicalMemoryRequest.model_validate(
        {
            "shots": shots,
            "prep_basis": prep_basis,
            "prep_state": prep_state,
            "measure_basis": measure_basis,
            "n_rounds": n_rounds,
            "cycle_time_us": cycle_time_us,
            "dd": dd,
        }
    )
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    final_assignments, syndrome_bits = _preflight_raw_records(
        ctx,
        shots=request.shots,
        n_rounds_grid=[request.n_rounds],
    )
    job_id = new_job_id()
    resource_usage = _reserve_submission_resources(
        ctx,
        job_id=job_id,
        final_assignments=final_assignments,
        syndrome_bits=syndrome_bits,
    )
    try:
        salt = ctx.state.next_salt()
    except Exception:
        _rollback_submission_resources(
            ctx,
            job_id=job_id,
            final_assignments=final_assignments,
            syndrome_bits=syndrome_bits,
            budgets_reserved=True,
        )
        raise RuntimeError("qsim logical-memory entropy allocation failed") from None

    def runner(job_id: str) -> JobResult:
        result = ctx.state.backend.run_logical_memory(request, job_id, salt)
        _log_evidence(ctx, tool="run_logical_memory", request=request, result=result)
        return result

    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": "submit_logical_memory",
                "tool": "run_logical_memory",
                "device_id": device_id,
                "shots": shots,
                "prep_basis": prep_basis,
                "prep_state": prep_state,
                "measure_basis": measure_basis,
                "n_rounds": n_rounds,
                "cycle_time_us": float(cycle_time_us),
                "dd": dd,
                "job_id": job_id,
                "final_assignments_this_call": final_assignments,
                "syndrome_bits_this_call": syndrome_bits,
                **resource_usage,
            }
        )
    except Exception:
        _rollback_submission_resources(
            ctx,
            job_id=job_id,
            final_assignments=final_assignments,
            syndrome_bits=syndrome_bits,
            budgets_reserved=True,
        )
        _best_effort_submission_failure(
            ctx,
            tool="run_logical_memory",
            job_id=job_id,
            failure_stage="submission_log_publication",
        )
        raise RuntimeError("qsim logical-memory submission publication failed") from None
    try:
        ctx.state.jobs.submit(runner, job_id=job_id)
    except Exception:
        _rollback_submission_resources(
            ctx,
            job_id=job_id,
            final_assignments=final_assignments,
            syndrome_bits=syndrome_bits,
            budgets_reserved=True,
        )
        _best_effort_submission_failure(
            ctx,
            tool="run_logical_memory",
            job_id=job_id,
            failure_stage="job_enqueue",
        )
        raise RuntimeError("qsim logical-memory enqueue failed") from None
    return {"job_id": job_id, "status": "queued"}


def submit_logical_memory_sweep(
    ctx: ActionContext,
    *,
    device_id: str,
    prep_basis: str,
    prep_state: str,
    measure_basis: str,
    n_rounds_grid: list[int],
    cycle_time_us: float,
    dd: str,
    shots: int,
) -> dict[str, Any]:
    """Submit a logical-memory sweep over a grid of round counts."""
    _require_capability(ctx, "submit logical memory sweep")
    request = LogicalMemorySweepRequest.model_validate(
        {
            "shots": shots,
            "prep_basis": prep_basis,
            "prep_state": prep_state,
            "measure_basis": measure_basis,
            "n_rounds_grid": n_rounds_grid,
            "cycle_time_us": cycle_time_us,
            "dd": dd,
        }
    )
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    rounds = [int(n) for n in request.n_rounds_grid]
    final_assignments, syndrome_bits = _preflight_raw_records(
        ctx,
        shots=request.shots,
        n_rounds_grid=rounds,
    )
    job_id = new_job_id()
    resource_usage = _reserve_submission_resources(
        ctx,
        job_id=job_id,
        final_assignments=final_assignments,
        syndrome_bits=syndrome_bits,
    )
    try:
        salt = ctx.state.next_salt()
    except Exception:
        _rollback_submission_resources(
            ctx,
            job_id=job_id,
            final_assignments=final_assignments,
            syndrome_bits=syndrome_bits,
            budgets_reserved=True,
        )
        raise RuntimeError("qsim logical-memory-sweep entropy allocation failed") from None

    def runner(job_id: str) -> JobResult:
        result = ctx.state.backend.run_logical_memory_sweep(request, job_id, salt)
        _log_evidence(ctx, tool="run_logical_memory_sweep", request=request, result=result)
        return result

    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": "submit_logical_memory_sweep",
                "tool": "run_logical_memory_sweep",
                "device_id": device_id,
                "shots": shots,
                "prep_basis": prep_basis,
                "prep_state": prep_state,
                "measure_basis": measure_basis,
                "n_rounds_grid": rounds,
                "cycle_time_us": float(cycle_time_us),
                "dd": dd,
                "job_id": job_id,
                "final_assignments_this_call": final_assignments,
                "syndrome_bits_this_call": syndrome_bits,
                **resource_usage,
            }
        )
    except Exception:
        _rollback_submission_resources(
            ctx,
            job_id=job_id,
            final_assignments=final_assignments,
            syndrome_bits=syndrome_bits,
            budgets_reserved=True,
        )
        _best_effort_submission_failure(
            ctx,
            tool="run_logical_memory_sweep",
            job_id=job_id,
            failure_stage="submission_log_publication",
        )
        raise RuntimeError("qsim logical-memory-sweep submission publication failed") from None
    try:
        ctx.state.jobs.submit(runner, job_id=job_id)
    except Exception:
        _rollback_submission_resources(
            ctx,
            job_id=job_id,
            final_assignments=final_assignments,
            syndrome_bits=syndrome_bits,
            budgets_reserved=True,
        )
        _best_effort_submission_failure(
            ctx,
            tool="run_logical_memory_sweep",
            job_id=job_id,
            failure_stage="job_enqueue",
        )
        raise RuntimeError("qsim logical-memory-sweep enqueue failed") from None
    return {"job_id": job_id, "status": "queued"}
