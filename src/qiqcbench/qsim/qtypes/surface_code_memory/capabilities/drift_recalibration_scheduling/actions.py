"""Task-owned action seam for the drift-recalibration-scheduling capability.

Completed stream jobs expose only a bounded pointer and digests in the qsim
JSONL log. The lossless detector, downtime, and logical-failure bit arrays live
in a poll-published raw record and remain the sole scientific evidence.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.actions.common import release_job_admission, reserve_job_admission
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.evidence import persist_public_raw_record
from qiqcbench.qsim.jobs import JobQueueFullError, new_job_id
from qiqcbench.qsim.qtypes.surface_code_memory.capabilities.drift_recalibration_scheduling.runtime import (
    DRIFT_EVIDENCE_SCHEMA_VERSION,
    canonical_policy,
    canonical_policy_digest,
    canonical_risk_model,
    instance_evidence_commitment,
)
from qiqcbench.qsim.qtypes.surface_code_memory.wire import (
    DriftControlRequest,
    DriftPolicy,
    DriftRiskModel,
    DriftStreamRequest,
    JobDriftControlData,
    JobDriftStreamData,
)

_CAPABILITY = "drift_recalibration_scheduling"


def _record_infrastructure_failure(
    ctx: ActionContext, *, device_id: str, job_id: str, tool: str, failure_stage: str
) -> None:
    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": "drift_recalibration_infrastructure_failure",
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


def _evidence_provenance(ctx: ActionContext) -> dict[str, Any]:
    """Versioned opaque binding of evidence to the graded run inputs."""
    return {
        "evidence_schema_version": DRIFT_EVIDENCE_SCHEMA_VERSION,
        "instance_commitment": instance_evidence_commitment(
            ctx.state.hidden,
            task_id=ctx.state.task_id,
            instance_seed=ctx.state.backend.drift_instance_seed(),
        ),
    }


def _stream_evidence(data: JobDriftStreamData) -> dict[str, Any]:
    """Return the bounded log record; never derive scientific aggregates."""
    return data.model_dump(
        exclude={"detector_events_b64", "down_mask_b64", "logical_failure_bits_b64"}
    )


def _persist_stream_data(
    ctx: ActionContext, *, job_id: str, data: JobDriftStreamData
) -> JobDriftStreamData:
    log_dir = getattr(ctx.state, "log_dir", None)
    relative = persist_public_raw_record(log_dir, job_id=job_id, record=data.model_dump())
    if relative is None:
        return data
    digest = hashlib.sha256((Path(log_dir) / relative).read_bytes()).hexdigest()
    return data.model_copy(
        update={
            "detector_events_b64": None,
            "down_mask_b64": None,
            "logical_failure_bits_b64": None,
            "raw_data_file": relative,
            "raw_data_sha256": digest,
        }
    )


def submit_drift_syndrome_stream(
    ctx: ActionContext, *, device_id: str, windows: int
) -> dict[str, Any]:
    """Submit an async syndrome-stream batch (advances device time)."""
    _require_capability(ctx, device_id)
    request = DriftStreamRequest.model_validate({"windows": windows})
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        try:
            result = ctx.state.backend.run_drift_syndrome_stream(request, job_id, salt)
            if isinstance(result.data, JobDriftStreamData):
                result.data = _persist_stream_data(ctx, job_id=job_id, data=result.data)
                ctx.state.log(
                    {
                        "surface": ctx.surface,
                        "action": "drift_stream_result",
                        "tool": "run_drift_syndrome_stream",
                        "device_id": device_id,
                        "job_id": job_id,
                        **_evidence_provenance(ctx),
                        "data": _stream_evidence(result.data),
                    }
                )
            return result
        except Exception:
            _record_infrastructure_failure(
                ctx,
                device_id=device_id,
                job_id=job_id,
                tool="run_drift_syndrome_stream",
                failure_stage="backend_or_evidence_publication",
            )
            raise

    job_id = new_job_id()
    reserve_job_admission(ctx.state, job_id)
    try:
        ctx.state.log(
            {
                "action": "submit_drift_syndrome_stream",
                "tool": "run_drift_syndrome_stream",
                "job_id": job_id,
                "surface": ctx.surface,
                "device_id": device_id,
                "request": request.model_dump(),
            }
        )
    except Exception:
        release_job_admission(ctx.state, job_id)
        _record_infrastructure_failure(
            ctx,
            device_id=device_id,
            job_id=job_id,
            tool="run_drift_syndrome_stream",
            failure_stage="submission_log_publication",
        )
        raise RuntimeError("qsim drift-stream submission publication failed") from None
    try:
        ctx.state.jobs.submit(runner, job_id=job_id)
    except JobQueueFullError:
        release_job_admission(ctx.state, job_id)
        raise
    except Exception:
        release_job_admission(ctx.state, job_id)
        _record_infrastructure_failure(
            ctx,
            device_id=device_id,
            job_id=job_id,
            tool="run_drift_syndrome_stream",
            failure_stage="job_enqueue",
        )
        raise RuntimeError("qsim drift-stream enqueue failed") from None
    return {"job_id": job_id, "status": "queued"}


def submit_drift_control(ctx: ActionContext, *, device_id: str, action: str) -> dict[str, Any]:
    """Submit an async recalibration/relocation control action."""
    _require_capability(ctx, device_id)
    request = DriftControlRequest.model_validate({"action": action})
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        try:
            result = ctx.state.backend.run_drift_control(request, job_id, salt)
            if isinstance(result.data, JobDriftControlData):
                ctx.state.log(
                    {
                        "surface": ctx.surface,
                        "action": "drift_control_result",
                        "tool": "run_drift_control",
                        "device_id": device_id,
                        "job_id": job_id,
                        **_evidence_provenance(ctx),
                        "data": result.data.model_dump(),
                    }
                )
            return result
        except Exception:
            _record_infrastructure_failure(
                ctx,
                device_id=device_id,
                job_id=job_id,
                tool="run_drift_control",
                failure_stage="backend_or_evidence_publication",
            )
            raise

    job_id = new_job_id()
    reserve_job_admission(ctx.state, job_id)
    try:
        ctx.state.log(
            {
                "action": "submit_drift_control",
                "tool": "run_drift_control",
                "job_id": job_id,
                "surface": ctx.surface,
                "device_id": device_id,
                "request": request.model_dump(),
            }
        )
    except Exception:
        release_job_admission(ctx.state, job_id)
        _record_infrastructure_failure(
            ctx,
            device_id=device_id,
            job_id=job_id,
            tool="run_drift_control",
            failure_stage="submission_log_publication",
        )
        raise RuntimeError("qsim drift-control submission publication failed") from None
    try:
        ctx.state.jobs.submit(runner, job_id=job_id)
    except JobQueueFullError:
        release_job_admission(ctx.state, job_id)
        raise
    except Exception:
        release_job_admission(ctx.state, job_id)
        _record_infrastructure_failure(
            ctx,
            device_id=device_id,
            job_id=job_id,
            tool="run_drift_control",
            failure_stage="job_enqueue",
        )
        raise RuntimeError("qsim drift-control enqueue failed") from None
    return {"job_id": job_id, "status": "queued"}


def validate_drift_policy(
    ctx: ActionContext,
    *,
    device_id: str,
    risk_model: dict[str, Any],
    maintenance_policy: dict[str, Any],
) -> dict[str, Any]:
    """Validate only grammar/digest; never evaluate scientific quality."""
    _require_capability(ctx, device_id)
    parsed_risk = DriftRiskModel.model_validate(risk_model)
    parsed_policy = DriftPolicy.model_validate(maintenance_policy)
    return {
        "valid": True,
        "submission_digest": canonical_policy_digest(parsed_risk, parsed_policy, ctx.state.public),
        "canonical_risk_model": canonical_risk_model(parsed_risk),
        "canonical_maintenance_policy": canonical_policy(parsed_policy, ctx.state.public),
        "performance": "not evaluated; the verifier replays the submitted policy",
    }


__all__ = [
    "submit_drift_syndrome_stream",
    "submit_drift_control",
    "validate_drift_policy",
]
