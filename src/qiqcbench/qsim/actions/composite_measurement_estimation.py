"""Action seam for the ``composite_measurement_estimation`` qtype.

Two synchronous calculators (``get_observable_spec``, ``evaluate_composite_scheme``)
and two async experiments (``run_pilot_measurements``, ``execute_locked_composite_scheme``).
Capability-gated and fail-closed here (defense in depth alongside MCP registration).
The production action logs the verifier evidence — per-setting basis + raw counts,
the canonical locked scheme + control variates, and the request digest — which the
separate-mode verifier reads back to recompute the score. The sampled per-setting
component ID is deliberately STRIPPED from the logged payload: ``/qsim_logs`` is
mounted read-only into the agent container, and the verifier scores from basis +
counts only, so the component ID would be verifier-only metadata that must not leak.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.actions.common import (
    QSIM_INTERNAL_FAILURE_KIND,
    release_job_admission,
    reserve_job_admission,
)
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.jobs import JobQueueFullError, new_job_id
from qiqcbench.qsim.qtypes.composite_measurement_estimation.wire import (
    JobPilotCountsData,
    LockedProductionRequest,
    PilotBatchRequest,
)

_CAP = "composite_measurement"
_SUBMISSION_FAILURE_ACTION = "composite_measurement_submission_infrastructure_failure"
_SUBMISSION_FAILURE_ERROR = "qsim composite-measurement job submission failed"


def _require(ctx: ActionContext, device_id: str) -> None:
    if _CAP not in ctx.state.active_capabilities:
        raise ValueError(f"{_CAP} capability is not active for this run")
    if device_id != ctx.state.public.device_id:
        raise ValueError(f"Unknown device {device_id!r}")


def _best_effort_submission_failure(
    ctx: ActionContext,
    *,
    tool: str,
    device_id: str,
    job_id: str,
    failure_stage: str,
) -> None:
    """Record a sanitized qsim-owned submission failure when logging still works."""

    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": _SUBMISSION_FAILURE_ACTION,
                "tool": tool,
                "device_id": device_id,
                "job_id": job_id,
                "failure_kind": QSIM_INTERNAL_FAILURE_KIND,
                "failure_stage": failure_stage,
            }
        )
    except Exception:
        return


def _submit_async_job(
    ctx: ActionContext,
    *,
    tool: str,
    device_id: str,
    submission_action: str,
    submission_fields: dict[str, Any],
    runner: Callable[[str], JobResult],
) -> dict[str, Any]:
    """Publish submission evidence before a worker can publish its result."""

    job_id = new_job_id()
    try:
        reserve_job_admission(ctx.state, job_id)
    except JobQueueFullError:
        # Typed queue backpressure is agent-actionable and has no accepted job.
        raise
    except Exception:
        _best_effort_submission_failure(
            ctx,
            tool=tool,
            device_id=device_id,
            job_id=job_id,
            failure_stage="job_admission",
        )
        raise RuntimeError(_SUBMISSION_FAILURE_ERROR) from None

    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": submission_action,
                "tool": tool,
                "device_id": device_id,
                "job_id": job_id,
                **submission_fields,
            }
        )
    except Exception:
        release_job_admission(ctx.state, job_id)
        _best_effort_submission_failure(
            ctx,
            tool=tool,
            device_id=device_id,
            job_id=job_id,
            failure_stage="submission_log_publication",
        )
        raise RuntimeError(_SUBMISSION_FAILURE_ERROR) from None

    try:
        ctx.state.jobs.submit(runner, job_id=job_id)
    except JobQueueFullError:
        # A real JobManager rejects at reserve_job_admission before evidence.
        # Preserve the model-owned type for compatible/test job managers too.
        release_job_admission(ctx.state, job_id)
        raise
    except Exception:
        release_job_admission(ctx.state, job_id)
        _best_effort_submission_failure(
            ctx,
            tool=tool,
            device_id=device_id,
            job_id=job_id,
            failure_stage="job_enqueue",
        )
        raise RuntimeError(_SUBMISSION_FAILURE_ERROR) from None
    return {"job_id": job_id, "status": "queued"}


def get_observable_spec(ctx: ActionContext, *, device_id: str) -> dict[str, Any]:
    _require(ctx, device_id)
    payload = ctx.state.backend.observable_spec()
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "get_observable_spec",
            "tool": "get_observable_spec",
            "device_id": device_id,
        }
    )
    return payload


def evaluate_composite_scheme(
    ctx: ActionContext,
    *,
    device_id: str,
    mixture_weights: list[float],
    local_basis_probabilities_xyz: list[list[list[float]]],
) -> dict[str, Any]:
    _require(ctx, device_id)
    result = ctx.state.backend.evaluate_scheme(
        mixture_weights=mixture_weights,
        local_basis_probabilities_xyz=local_basis_probabilities_xyz,
    )
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "evaluate_composite_scheme",
            "tool": "evaluate_composite_scheme",
            "device_id": device_id,
            "accepted": result.get("accepted"),
            "valid": result.get("valid"),
            "v_haar": result.get("average_one_shot_variance"),
            "evaluator_calls_used": result.get("evaluator_calls_used"),
        }
    )
    return result


def run_pilot_measurements(
    ctx: ActionContext, *, device_id: str, settings: list[dict[str, Any]]
) -> dict[str, Any]:
    _require(ctx, device_id)
    request = PilotBatchRequest.model_validate({"rows": settings})

    def runner(job_id: str) -> JobResult:
        result = ctx.state.backend.run_pilot_measurements(request, job_id, ctx.state.next_salt())
        data = result.data
        if isinstance(data, JobPilotCountsData):
            ctx.state.log(
                {
                    "surface": ctx.surface,
                    "action": "pilot_result",
                    "tool": "run_pilot_measurements",
                    "device_id": device_id,
                    "job_id": job_id,
                    "n_settings": len(data.rows),
                    "budget": data.budget.model_dump(),
                }
            )
        return result

    return _submit_async_job(
        ctx,
        tool="run_pilot_measurements",
        device_id=device_id,
        submission_action="submit_pilot_measurements",
        submission_fields={"n_settings": len(settings)},
        runner=runner,
    )


def execute_locked_composite_scheme(
    ctx: ActionContext,
    *,
    device_id: str,
    mixture_weights: list[float],
    local_basis_probabilities_xyz: list[list[list[float]]],
    control_variate_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    _require(ctx, device_id)
    request = LockedProductionRequest.model_validate(
        {
            "mixture_weights": mixture_weights,
            "local_basis_probabilities_xyz": local_basis_probabilities_xyz,
            "control_variate_entries": control_variate_entries,
        }
    )

    def runner(job_id: str) -> JobResult:
        result, evidence = ctx.state.backend.execute_locked_composite_scheme(
            request, job_id, ctx.state.next_salt()
        )
        if evidence:
            ctx.state.log(
                {
                    "surface": ctx.surface,
                    "action": "locked_production_result",
                    "tool": "execute_locked_composite_scheme",
                    "device_id": device_id,
                    "job_id": job_id,
                    "request_digest": evidence["request_digest"],
                    "canonical": evidence["canonical"],
                    "locked_weights": evidence["locked_weights"],
                    "locked_beta": evidence["locked_beta"],
                    "locked_control_entries": evidence["locked_control_entries"],
                    "production_settings": evidence["production_settings"],
                    "production_shots_per_setting": evidence["production_shots_per_setting"],
                    # Strip the sampled component_id: qsim_logs is mounted read-only
                    # into the agent container, so logging it would leak verifier-only
                    # metadata. The verifier scores from basis + counts only.
                    "production_rows": [
                        {"basis": r["basis"], "counts": r["counts"]} for r in evidence["rows"]
                    ],
                }
            )
        else:
            ctx.state.log(
                {
                    "surface": ctx.surface,
                    "action": "locked_production_failed",
                    "tool": "execute_locked_composite_scheme",
                    "device_id": device_id,
                    "job_id": job_id,
                    "error": result.error,
                }
            )
        return result

    return _submit_async_job(
        ctx,
        tool="execute_locked_composite_scheme",
        device_id=device_id,
        submission_action="submit_locked_composite_scheme",
        submission_fields={"n_control_variates": len(control_variate_entries)},
        runner=runner,
    )
