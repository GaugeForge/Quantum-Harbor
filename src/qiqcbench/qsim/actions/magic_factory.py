"""Action seam for the logical_magic_factory qtype.

Gates the ``run_magic_benchmark_batch`` tool on the
``logical_magic_benchmarking`` capability (fail-closed at the action layer,
not only at surface registration) and logs **server-side** per-point
aggregates on job completion so the verifier can recompute the twirled,
post-selected two-copy Bell-measurement estimate of epsilon from the agent's
own evidence (the agent cannot fake post-selection or the twirl/basis choice).
"""

from __future__ import annotations

from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.logical_magic_factory.wire import MagicBenchmarkBatchRequest

_CAPABILITY = "logical_magic_benchmarking"


def _require_capability(ctx: ActionContext, action: str) -> None:
    if _CAPABILITY not in ctx.state.active_capabilities:
        raise ValueError(f"{_CAPABILITY} capability is not active for this run; cannot {action}")


def _point_aggregate(point: Any) -> dict[str, Any]:
    base = {
        "n_copies": point.n_copies,
        "twirl": point.twirl,
        "basis": point.basis,
        "shots": point.shots,
        "rejected": point.rejected,
        "reject_reason": point.reject_reason,
        "magic_states_consumed_this_point": point.magic_states_consumed_this_point,
        "magic_states_consumed_total": point.magic_states_consumed_total,
        "budget_remaining": point.budget_remaining,
    }
    if point.rejected:
        return base

    if point.n_copies == 1:
        flagged = list(point.flagged)
        outcomes = list(point.outcomes)
        n_clean = sum(1 for f in flagged if f == 0)
        n_clean_one = sum(1 for f, o in zip(flagged, outcomes, strict=True) if f == 0 and o == 1)
        base.update({"n_clean": n_clean, "n_clean_outcome1": n_clean_one})
        return base

    flagged1 = list(point.flagged)
    flagged2 = list(point.flagged2)
    outcomes1 = list(point.outcomes)
    outcomes2 = list(point.outcomes2)
    n_both_clean = 0
    n11_postselected = 0
    n11_all = 0
    for f1, f2, o1, o2 in zip(flagged1, flagged2, outcomes1, outcomes2, strict=True):
        is11 = o1 == 1 and o2 == 1
        if is11:
            n11_all += 1
        if f1 == 0 and f2 == 0:
            n_both_clean += 1
            if is11:
                n11_postselected += 1
    base.update(
        {
            "n_both_clean": n_both_clean,
            "n11_postselected": n11_postselected,
            "n11_all": n11_all,
        }
    )
    return base


def _log_evidence(ctx: ActionContext, *, tool: str, result: JobResult) -> None:
    if result.status != "complete" or result.data is None:
        return
    data = result.data
    if getattr(data, "kind", None) != "magic_benchmark":
        return
    points = [_point_aggregate(p) for p in data.points]
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "magic_benchmark_evidence",
            "tool": tool,
            "device_id": ctx.state.hidden.device_id,
            "job_id": result.job_id,
            "points": points,
        }
    )


def submit_magic_benchmark_batch(
    ctx: ActionContext,
    *,
    device_id: str,
    points: list[dict[str, Any]],
) -> dict[str, Any]:
    """Submit a batch of magic-benchmark points against the run-long factory."""
    _require_capability(ctx, "submit magic benchmark batch")
    request = MagicBenchmarkBatchRequest.model_validate({"points": points})
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        result = ctx.state.backend.run_magic_benchmark_batch(request, job_id, salt)
        _log_evidence(ctx, tool="run_magic_benchmark_batch", result=result)
        return result

    job_id = ctx.state.jobs.submit(runner)
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "submit_magic_benchmark_batch",
            "tool": "run_magic_benchmark_batch",
            "device_id": device_id,
            "n_points": len(points),
            "job_id": job_id,
        }
    )
    return {"job_id": job_id, "status": "queued"}


_CULTIVATION_CAPABILITY = "magic_state_cultivation"


def _cultivation_point_aggregate(point: Any) -> dict[str, Any]:
    """Server-side per-point aggregate the agent cannot fake: the schedule,
    consumption ledger, and the full 32-cell contingency of (four group
    flags) x (terminal outcome bit)."""
    base = {
        "injection_theta_rad": point.injection_theta_rad,
        "cultivation_rounds": point.cultivation_rounds,
        "qec_cycles_per_round": point.qec_cycles_per_round,
        "escape_cycle_n": point.escape_cycle_n,
        "measure_axis": point.measure_axis,
        "shots": point.shots,
        "rejected": point.rejected,
        "reject_reason": point.reject_reason,
        "injections_consumed_this_point": point.injections_consumed_this_point,
        "injections_consumed_total": point.injections_consumed_total,
        "budget_remaining": point.budget_remaining,
    }
    if point.rejected:
        return base
    cells: dict[str, int] = {}
    flags = (point.flag_injection, point.flag_cultivation, point.flag_qec, point.flag_graft)
    for s, out in enumerate(point.outcomes):
        key = "".join(str(f[s]) for f in flags) + f"_{out}"
        cells[key] = cells.get(key, 0) + 1
    base["cells"] = cells
    return base


def _log_cultivation_evidence(ctx: ActionContext, *, tool: str, result: JobResult) -> None:
    if result.status != "complete" or result.data is None:
        return
    data = result.data
    if getattr(data, "kind", None) != "cultivation":
        return
    points = [_cultivation_point_aggregate(p) for p in data.points]
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "cultivation_evidence",
            "tool": tool,
            "device_id": ctx.state.hidden.device_id,
            "job_id": result.job_id,
            "points": points,
        }
    )


def submit_cultivation_batch(
    ctx: ActionContext,
    *,
    device_id: str,
    points: list[dict[str, Any]],
) -> dict[str, Any]:
    """Submit a batch of cultivation points against the run-long line."""
    if _CULTIVATION_CAPABILITY not in ctx.state.active_capabilities:
        raise ValueError(
            f"{_CULTIVATION_CAPABILITY} capability is not active for this run; "
            "cannot submit cultivation batch"
        )
    from qiqcbench.qsim.qtypes.logical_magic_factory.wire import CultivationBatchRequest

    request = CultivationBatchRequest.model_validate({"points": points})
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")

    def runner(job_id: str) -> JobResult:
        result = ctx.state.backend.run_cultivation_batch(request, job_id, 0)
        _log_cultivation_evidence(ctx, tool="run_cultivation_batch", result=result)
        return result

    job_id = ctx.state.jobs.submit(runner)
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "submit_cultivation_batch",
            "tool": "run_cultivation_batch",
            "device_id": device_id,
            "n_points": len(points),
            "job_id": job_id,
        }
    )
    return {"job_id": job_id, "status": "queued"}


__all__ = ["submit_cultivation_batch", "submit_magic_benchmark_batch"]
