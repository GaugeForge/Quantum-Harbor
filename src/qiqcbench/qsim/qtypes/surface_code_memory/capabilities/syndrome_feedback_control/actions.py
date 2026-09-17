"""Task-owned actions for causal syndrome-feedback controller programs."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.actions.common import release_job_admission, reserve_job_admission
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.jobs import JobQueueFullError, new_job_id
from qiqcbench.qsim.qtypes.surface_code_memory.capabilities.syndrome_feedback_control.runtime import (
    FEEDBACK_EVIDENCE_SCHEMA_VERSION,
    instance_evidence_commitment,
)
from qiqcbench.qsim.qtypes.surface_code_memory.wire import (
    JobSyndromeControllerProgramData,
    JobSyndromeControlProbeDataV3,
    SyndromeControllerProgramRequest,
    SyndromeControlProbeRequest,
    _ControllerBudgetView,
)

_CAPABILITY = "syndrome_feedback_control"
_TRAJECTORY_BUDGET_KEY = "syndrome_feedback:experiment_trajectories"
_QEC_CYCLE_BUDGET_KEY = "syndrome_feedback:experiment_qec_cycles"


def _record_infrastructure_failure(
    ctx: ActionContext, *, device_id: str, job_id: str, tool: str, failure_stage: str
) -> None:
    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": "syndrome_feedback_infrastructure_failure",
                "tool": tool,
                "device_id": device_id,
                "job_id": job_id,
                "failure_stage": failure_stage,
            }
        )
    except Exception:
        return


def _require_capability(ctx: ActionContext, device_id: str) -> None:
    if _CAPABILITY not in ctx.state.active_capabilities:
        raise ValueError(f"{_CAPABILITY} capability is not active for this run")
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _evidence_summary(
    data: JobSyndromeControlProbeDataV3 | JobSyndromeControllerProgramData,
) -> dict[str, Any]:
    """Log artifact commitments and dimensions, never derived scientific rates."""

    common: dict[str, Any] = {
        "kind": data.kind,
        "control_data_schema_version": data.control_data_schema_version,
        "trajectories": data.trajectories,
        "epochs_per_trajectory": data.epochs_per_trajectory,
        "control_epoch_cycles": data.control_epoch_cycles,
        "detector_shape": data.detector_shape,
        "decoded_logical_shape": data.decoded_logical_shape,
        "raw_detector_sha256": hashlib.sha256(data.detector_events_b64.encode("ascii")).hexdigest(),
        "raw_decoded_logical_sha256": hashlib.sha256(
            data.decoded_logical_bits_b64.encode("ascii")
        ).hexdigest(),
        "applied_trim_history_sha256": _canonical_sha256(data.applied_trim_history),
        "budget": data.budget.model_dump(mode="json"),
    }
    if isinstance(data, JobSyndromeControlProbeDataV3):
        common["constant_trim"] = data.constant_trim
    else:
        common.update(
            {
                "controller_manifest_sha256": data.controller_manifest_sha256,
                "controller_abi_version": data.controller_abi_version,
                "requested_trim_history_sha256": _canonical_sha256(data.requested_trim_history),
                "actuator_limited_shape": data.actuator_limited_shape,
                "actuator_limited_sha256": hashlib.sha256(
                    data.actuator_limited_bits_b64.encode("ascii")
                ).hexdigest(),
            }
        )
    return common


def _evidence_provenance(ctx: ActionContext) -> dict[str, Any]:
    return {
        "evidence_schema_version": FEEDBACK_EVIDENCE_SCHEMA_VERSION,
        "instance_commitment": instance_evidence_commitment(
            ctx.state.hidden,
            task_id=ctx.state.task_id,
            instance_seed=ctx.state.backend.feedback_instance_seed(),
        ),
    }


def _feedback_spec(ctx: ActionContext):
    spec = getattr(ctx.state.public, "syndrome_feedback", None)
    if spec is None:
        raise RuntimeError("active syndrome-feedback capability has no public contract")
    return spec


def _validate_trajectory_request(ctx: ActionContext, trajectories: int) -> None:
    spec = _feedback_spec(ctx)
    if trajectories > spec.max_trajectories_per_call:
        raise ValueError(
            "trajectories exceeds public max_trajectories_per_call="
            f"{spec.max_trajectories_per_call}"
        )


def _reservation_amounts(ctx: ActionContext, trajectories: int) -> dict[str, int]:
    spec = _feedback_spec(ctx)
    return {
        _TRAJECTORY_BUDGET_KEY: trajectories,
        _QEC_CYCLE_BUDGET_KEY: (
            trajectories * spec.experiment_epochs_per_trajectory * spec.control_epoch_cycles
        ),
    }


def _reserve_experiment_budget(
    ctx: ActionContext,
    *,
    trajectories: int,
) -> _ControllerBudgetView:
    spec = _feedback_spec(ctx)
    reserve = getattr(ctx.state, "reserve_action_budgets", None)
    if not callable(reserve):
        raise RuntimeError("active qsim state cannot enforce syndrome-feedback budgets")
    amounts = _reservation_amounts(ctx, trajectories)
    qec_cycle_cap = (
        spec.experiment_trajectory_budget
        * spec.experiment_epochs_per_trajectory
        * spec.control_epoch_cycles
    )
    evidence = reserve(
        reservations={
            _TRAJECTORY_BUDGET_KEY: (
                amounts[_TRAJECTORY_BUDGET_KEY],
                spec.experiment_trajectory_budget,
            ),
            _QEC_CYCLE_BUDGET_KEY: (
                amounts[_QEC_CYCLE_BUDGET_KEY],
                qec_cycle_cap,
            ),
        }
    )
    return _ControllerBudgetView(
        trajectories_used=evidence[_TRAJECTORY_BUDGET_KEY][1],
        trajectories_cap=spec.experiment_trajectory_budget,
        qec_cycles_used=evidence[_QEC_CYCLE_BUDGET_KEY][1],
        qec_cycles_cap=qec_cycle_cap,
    )


def _rollback_submission_resources(
    ctx: ActionContext,
    *,
    job_id: str,
    trajectories: int,
) -> None:
    ctx.state.release_action_budgets(reservations=_reservation_amounts(ctx, trajectories))
    release_job_admission(ctx.state, job_id)


def _submit_async(
    ctx: ActionContext,
    *,
    device_id: str,
    tool: str,
    submit_action: str,
    submit_fields: dict[str, Any],
    trajectories: int,
    runner,
) -> dict[str, Any]:
    budget = _reserve_experiment_budget(ctx, trajectories=trajectories)
    job_id = new_job_id()
    if not callable(getattr(getattr(ctx.state, "jobs", None), "reserve_admission", None)):
        _rollback_submission_resources(ctx, job_id=job_id, trajectories=trajectories)
        raise RuntimeError("qsim cannot enforce atomic syndrome-feedback job admission")
    try:
        reserve_job_admission(ctx.state, job_id)
    except JobQueueFullError:
        ctx.state.release_action_budgets(reservations=_reservation_amounts(ctx, trajectories))
        raise
    except Exception:
        ctx.state.release_action_budgets(reservations=_reservation_amounts(ctx, trajectories))
        _record_infrastructure_failure(
            ctx,
            device_id=device_id,
            job_id=job_id,
            tool=tool,
            failure_stage="job_admission",
        )
        raise RuntimeError(f"qsim {tool} job admission failed") from None
    try:
        salt = ctx.state.next_salt()
    except Exception:
        _rollback_submission_resources(ctx, job_id=job_id, trajectories=trajectories)
        _record_infrastructure_failure(
            ctx,
            device_id=device_id,
            job_id=job_id,
            tool=tool,
            failure_stage="salt_allocation",
        )
        raise RuntimeError(f"qsim {tool} salt allocation failed") from None
    budget_payload = budget.model_dump(mode="json")
    try:
        ctx.state.log(
            {
                "action": submit_action,
                "job_id": job_id,
                "surface": ctx.surface,
                "device_id": device_id,
                **submit_fields,
                "budget": budget_payload,
            }
        )
    except Exception:
        _rollback_submission_resources(ctx, job_id=job_id, trajectories=trajectories)
        _record_infrastructure_failure(
            ctx,
            device_id=device_id,
            job_id=job_id,
            tool=tool,
            failure_stage="submission_log_publication",
        )
        raise RuntimeError(f"qsim {tool} submission publication failed") from None
    try:
        ctx.state.jobs.submit(
            lambda accepted_job_id: runner(accepted_job_id, salt, budget),
            job_id=job_id,
        )
    except Exception:
        _rollback_submission_resources(ctx, job_id=job_id, trajectories=trajectories)
        _record_infrastructure_failure(
            ctx,
            device_id=device_id,
            job_id=job_id,
            tool=tool,
            failure_stage="job_enqueue",
        )
        raise RuntimeError(f"qsim {tool} enqueue failed") from None
    return {
        "job_id": job_id,
        "status": "queued",
        **submit_fields,
        "budget": budget_payload,
    }


def submit_syndrome_control_probe(
    ctx: ActionContext,
    *,
    device_id: str,
    constant_trim: list[int],
    trajectories: int,
) -> dict[str, Any]:
    """Submit an async constant-trim raw-data probe."""

    _require_capability(ctx, device_id)
    request = SyndromeControlProbeRequest.model_validate(
        {"constant_trim": constant_trim, "trajectories": trajectories}
    )
    _validate_trajectory_request(ctx, request.trajectories)
    spec = _feedback_spec(ctx)
    if any(value < spec.trim_min or value > spec.trim_max for value in request.constant_trim):
        raise ValueError("constant_trim lies outside the public trim range")

    def runner(job_id: str, salt: int, budget: _ControllerBudgetView) -> JobResult:
        try:
            result = ctx.state.backend.run_syndrome_control_probe(
                request,
                job_id,
                salt,
                budget,
            )
            if isinstance(result.data, JobSyndromeControlProbeDataV3):
                ctx.state.log(
                    {
                        "surface": ctx.surface,
                        "action": "syndrome_control_probe_result",
                        "tool": "run_syndrome_control_probe",
                        "device_id": device_id,
                        "job_id": job_id,
                        **_evidence_provenance(ctx),
                        "data": _evidence_summary(result.data),
                    }
                )
            return result
        except Exception:
            _record_infrastructure_failure(
                ctx,
                device_id=device_id,
                job_id=job_id,
                tool="run_syndrome_control_probe",
                failure_stage="backend_or_evidence_publication",
            )
            raise

    return _submit_async(
        ctx,
        device_id=device_id,
        tool="run_syndrome_control_probe",
        submit_action="submit_syndrome_control_probe",
        submit_fields={
            "constant_trim": request.constant_trim,
            "trajectories": request.trajectories,
        },
        trajectories=request.trajectories,
        runner=runner,
    )


def validate_syndrome_controller_program(ctx: ActionContext, *, device_id: str) -> dict[str, Any]:
    """Validate and bind the staged bundle; never evaluate scientific quality."""

    _require_capability(ctx, device_id)
    return ctx.state.backend.validate_syndrome_controller_program()


def submit_syndrome_controller_program(
    ctx: ActionContext,
    *,
    device_id: str,
    controller_manifest_sha256: str,
    trajectories: int,
) -> dict[str, Any]:
    """Submit an async batch of causal controller trajectories."""

    _require_capability(ctx, device_id)
    request = SyndromeControllerProgramRequest.model_validate(
        {
            "controller_manifest_sha256": controller_manifest_sha256,
            "trajectories": trajectories,
        }
    )
    _validate_trajectory_request(ctx, request.trajectories)

    def runner(job_id: str, salt: int, budget: _ControllerBudgetView) -> JobResult:
        try:
            result = ctx.state.backend.run_syndrome_controller_program(
                request,
                job_id,
                salt,
                budget,
            )
            if isinstance(result.data, JobSyndromeControllerProgramData):
                ctx.state.log(
                    {
                        "surface": ctx.surface,
                        "action": "syndrome_controller_program_result",
                        "tool": "run_syndrome_controller_program",
                        "device_id": device_id,
                        "job_id": job_id,
                        **_evidence_provenance(ctx),
                        "data": _evidence_summary(result.data),
                    }
                )
            return result
        except Exception:
            _record_infrastructure_failure(
                ctx,
                device_id=device_id,
                job_id=job_id,
                tool="run_syndrome_controller_program",
                failure_stage="backend_or_evidence_publication",
            )
            raise

    return _submit_async(
        ctx,
        device_id=device_id,
        tool="run_syndrome_controller_program",
        submit_action="submit_syndrome_controller_program",
        submit_fields={
            "controller_manifest_sha256": request.controller_manifest_sha256,
            "trajectories": request.trajectories,
        },
        trajectories=request.trajectories,
        runner=runner,
    )


__all__ = [
    "submit_syndrome_control_probe",
    "submit_syndrome_controller_program",
    "validate_syndrome_controller_program",
]
