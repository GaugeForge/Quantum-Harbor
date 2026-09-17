"""Action and lossless-evidence seam for structured control experiments."""

from __future__ import annotations

import hashlib
import threading
from collections import Counter
from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.core.wire import JobBitstringData, JobResult
from qiqcbench.qsim.evidence import persist_public_raw_record
from qiqcbench.qsim.hidden_commitment import (
    hidden_device_model_commitment,
    public_device_model_commitment,
)
from qiqcbench.qsim.jobs import JobQueueFullError, new_job_id
from qiqcbench.qsim.qtypes.spin_chain_control.backend import ControlReservation
from qiqcbench.qsim.qtypes.spin_chain_control.wire import (
    ControlReadoutReferenceRequest,
    ControlTransferRequest,
)

CAPABILITY = "structured_control_probe"
EVIDENCE_CONTRACT = "spin_chain_structured_control_evidence_v1"
INFRASTRUCTURE_FAILURE = "QIQCBENCH_STRUCTURED_CONTROL_INFRASTRUCTURE_FAILURE"


def _require(ctx: ActionContext, device_id: str) -> None:
    if CAPABILITY not in ctx.state.active_capabilities:
        raise ValueError(f"{CAPABILITY} capability is not active for this run")
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")


def _identity(ctx: ActionContext) -> dict[str, Any]:
    task_id = getattr(ctx.state, "task_id", None) or ctx.state.public.task_id
    identity: dict[str, Any] = {
        "evidence_contract": EVIDENCE_CONTRACT,
        "evidence_schema_version": 1,
        "public_device_model_commitment": public_device_model_commitment(
            ctx.state.public, task_id=task_id
        ),
    }
    if getattr(ctx.state, "execution_context_id", None) is not None:
        secret = getattr(ctx.state, "hidden_commitment_secret", None)
        if secret is None:
            raise RuntimeError("execution-bound control evidence requires a secret")
        identity["hidden_device_model_commitment"] = hidden_device_model_commitment(
            ctx.state.hidden,
            task_id=task_id,
            commitment_secret=secret,
        )
    return identity


def _infrastructure_failure(
    ctx: ActionContext,
    *,
    job_id: str,
    device_id: str,
    tool: str,
    reservation: ControlReservation,
    identity: dict[str, Any],
    stage: str,
) -> JobResult:
    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": "structured_control_infrastructure_failure",
                "tool": tool,
                "device_id": device_id,
                "job_id": job_id,
                "submission_index": reservation.submission_index,
                "request_digest": reservation.request_digest,
                "failure_class": INFRASTRUCTURE_FAILURE,
                "failure_stage": stage,
                **identity,
            }
        )
    except Exception:
        pass
    return JobResult(
        job_id=job_id,
        device_id=device_id,
        status="failed",
        error=INFRASTRUCTURE_FAILURE,
    )


def _persist_result(
    ctx: ActionContext,
    *,
    job_id: str,
    tool: str,
    request: ControlTransferRequest | ControlReadoutReferenceRequest,
    reservation: ControlReservation,
    data: JobBitstringData,
    identity: dict[str, Any],
) -> dict[str, Any] | None:
    record = {
        "schema_version": 1,
        "evidence_contract": EVIDENCE_CONTRACT,
        "job_id": job_id,
        "device_id": ctx.state.public.device_id,
        "tool": tool,
        "kind": reservation.kind,
        "submission_index": reservation.submission_index,
        "request_digest": reservation.request_digest,
        "public_device_model_commitment": identity["public_device_model_commitment"],
        "request": request.model_dump(mode="json"),
        "result": data.model_dump(mode="json"),
    }
    relative = persist_public_raw_record(
        getattr(ctx.state, "log_dir", None), job_id=job_id, record=record
    )
    if relative is None:
        return None
    payload = (ctx.state.log_dir / relative).read_bytes()
    return {
        "path": relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _submit(
    ctx: ActionContext,
    *,
    device_id: str,
    tool: str,
    request: ControlTransferRequest | ControlReadoutReferenceRequest,
) -> dict[str, Any]:
    _require(ctx, device_id)
    identity = _identity(ctx)
    if isinstance(request, ControlTransferRequest):
        reservation = ctx.state.backend.reserve_transfer(request)
        backend_method = ctx.state.backend.run_control_transfer
        submit_action = "submit_control_transfer"
        result_action = "control_transfer_result"
    else:
        reservation = ctx.state.backend.reserve_readout_reference(request)
        backend_method = ctx.state.backend.run_control_readout_reference
        submit_action = "submit_control_readout_reference"
        result_action = "control_readout_reference_result"
    salt = ctx.state.next_salt()
    submission_logged = threading.Event()

    def runner(job_id: str) -> JobResult:
        submission_logged.wait()
        try:
            result = backend_method(request, job_id, salt, reservation=reservation)
        except Exception:
            return _infrastructure_failure(
                ctx,
                job_id=job_id,
                device_id=device_id,
                tool=tool,
                reservation=reservation,
                identity=identity,
                stage="backend_execution",
            )
        data = result.data
        if isinstance(data, JobBitstringData):
            try:
                artifact = _persist_result(
                    ctx,
                    job_id=job_id,
                    tool=tool,
                    request=request,
                    reservation=reservation,
                    data=data,
                    identity=identity,
                )
                counts = dict(sorted(Counter(data.bitstrings[0]).items()))
                event = {
                    "surface": ctx.surface,
                    "action": result_action,
                    "tool": tool,
                    "device_id": device_id,
                    "job_id": job_id,
                    "status": result.status,
                    "submission_index": reservation.submission_index,
                    "request_digest": reservation.request_digest,
                    "request": request.model_dump(mode="json"),
                    "shots": request.shots,
                    "measured_qubits": data.measured_qubits,
                    "counts": counts,
                    **identity,
                }
                if artifact is not None:
                    event["raw_result_artifact"] = artifact
                ctx.state.log(event)
            except Exception:
                return _infrastructure_failure(
                    ctx,
                    job_id=job_id,
                    device_id=device_id,
                    tool=tool,
                    reservation=reservation,
                    identity=identity,
                    stage="evidence_recording",
                )
        return result

    job_id = new_job_id()
    try:
        ctx.state.jobs.submit(runner, job_id=job_id)
    except JobQueueFullError:
        # Typed admission backpressure is agent-actionable model behavior, not
        # qsim infrastructure; misclassifying it would let a queue-flooding
        # agent void its own trial.
        ctx.state.backend.discard_reservation(reservation)
        raise
    except Exception:
        ctx.state.backend.discard_reservation(reservation)
        _infrastructure_failure(
            ctx,
            job_id=job_id,
            device_id=device_id,
            tool=tool,
            reservation=reservation,
            identity=identity,
            stage="job_enqueue",
        )
        raise RuntimeError("qsim structured-control job enqueue failed") from None
    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": submit_action,
                "tool": tool,
                "device_id": device_id,
                "job_id": job_id,
                "submission_index": reservation.submission_index,
                "request_digest": reservation.request_digest,
                "request": request.model_dump(mode="json"),
                "shots": request.shots,
                **identity,
            }
        )
    except Exception:
        _infrastructure_failure(
            ctx,
            job_id=job_id,
            device_id=device_id,
            tool=tool,
            reservation=reservation,
            identity=identity,
            stage="submission_log_publication",
        )
        raise RuntimeError("qsim structured-control submission publication failed") from None
    finally:
        submission_logged.set()
    return {"job_id": job_id, "status": "queued"}


def submit_control_transfer(
    ctx: ActionContext,
    *,
    device_id: str,
    controller_id: str,
    x_drive_scale_fraction: float,
    shots: int,
) -> dict[str, Any]:
    request = ControlTransferRequest.model_validate(
        {
            "controller_id": controller_id,
            "x_drive_scale_fraction": x_drive_scale_fraction,
            "shots": shots,
        }
    )
    return _submit(
        ctx,
        device_id=device_id,
        tool="run_control_transfer",
        request=request,
    )


def submit_control_readout_reference(
    ctx: ActionContext,
    *,
    device_id: str,
    prepared_bitstring: str,
    shots: int,
) -> dict[str, Any]:
    request = ControlReadoutReferenceRequest.model_validate(
        {"prepared_bitstring": prepared_bitstring, "shots": shots}
    )
    return _submit(
        ctx,
        device_id=device_id,
        tool="run_control_readout_reference",
        request=request,
    )


__all__ = [
    "CAPABILITY",
    "EVIDENCE_CONTRACT",
    "INFRASTRUCTURE_FAILURE",
    "submit_control_readout_reference",
    "submit_control_transfer",
]
