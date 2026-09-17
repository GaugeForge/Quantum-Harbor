"""Action/evidence seam for scheduled compiled-candidate experiments."""

from __future__ import annotations

import hashlib
import threading
from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.actions.common import release_job_admission, reserve_job_admission
from qiqcbench.qsim.core.wire import JobBitstringData, JobResult
from qiqcbench.qsim.evidence import persist_public_raw_record
from qiqcbench.qsim.hidden_commitment import (
    hidden_device_model_commitment,
    public_device_model_commitment,
)
from qiqcbench.qsim.jobs import JobQueueFullError, new_job_id
from qiqcbench.qsim.qtypes.scheduled_transmon_gate_model.backend import (
    CompilationReservation,
)
from qiqcbench.qsim.qtypes.scheduled_transmon_gate_model.engine import bitstring_counts
from qiqcbench.qsim.qtypes.scheduled_transmon_gate_model.wire import (
    CompilationMirrorRequest,
    CompilationReadoutReferenceRequest,
)

CAPABILITY = "compiled_candidate_evaluation"
EVIDENCE_CONTRACT = "scheduled_transmon_compilation_evidence_v1"
INFRASTRUCTURE_FAILURE = "QIQCBENCH_COMPILATION_INFRASTRUCTURE_FAILURE"


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
        "candidate_bank_sha256": ctx.state.public.candidate_bank_sha256,
        "public_device_model_commitment": public_device_model_commitment(
            ctx.state.public, task_id=task_id
        ),
    }
    if getattr(ctx.state, "execution_context_id", None) is not None:
        secret = getattr(ctx.state, "hidden_commitment_secret", None)
        if secret is None:
            raise RuntimeError("execution-bound compilation evidence requires a secret")
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
    reservation: CompilationReservation,
    identity: dict[str, Any],
    stage: str,
) -> JobResult:
    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": "compilation_infrastructure_failure",
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
    request: CompilationMirrorRequest | CompilationReadoutReferenceRequest,
    reservation: CompilationReservation,
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
        "candidate_bank_sha256": ctx.state.public.candidate_bank_sha256,
        "request": request.model_dump(mode="json"),
        "result": data.model_dump(mode="json"),
    }
    relative = persist_public_raw_record(
        getattr(ctx.state, "log_dir", None), job_id=job_id, record=record
    )
    if relative is None:
        return None
    source = ctx.state.log_dir / relative
    payload = source.read_bytes()
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
    request: CompilationMirrorRequest | CompilationReadoutReferenceRequest,
) -> dict[str, Any]:
    _require(ctx, device_id)
    identity = _identity(ctx)
    job_id = new_job_id()
    # Queue admission is the first side effect. A typed queue-full rejection is
    # public/model-actionable and, by contract, must not consume a physical
    # submission index, shot allocation, or RNG salt. Reserving the slot here
    # closes the race between the capacity check and JobManager.submit().
    reserve_job_admission(ctx.state, job_id)
    reservation: CompilationReservation | None = None
    try:
        if isinstance(request, CompilationMirrorRequest):
            reservation = ctx.state.backend.reserve_mirror(request)
            backend_method = ctx.state.backend.run_compilation_mirror
            submit_action = "submit_compilation_mirror"
            result_action = "compilation_mirror_result"
        else:
            reservation = ctx.state.backend.reserve_readout_reference(request)
            backend_method = ctx.state.backend.run_compilation_readout_reference
            submit_action = "submit_compilation_readout_reference"
            result_action = "compilation_readout_reference_result"
        salt = ctx.state.next_salt()
    except Exception:
        if reservation is not None:
            ctx.state.backend.discard_reservation(reservation)
        release_job_admission(ctx.state, job_id)
        raise
    submission_logged = threading.Event()

    def runner(job_id: str) -> JobResult:
        submission_logged.wait()
        try:
            result = backend_method(
                request,
                job_id,
                salt,
                reservation=reservation,
            )
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
                    "counts": bitstring_counts(data),
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

    try:
        ctx.state.jobs.submit(runner, job_id=job_id)
    except JobQueueFullError:
        # Typed admission backpressure is agent-actionable model behavior, not
        # qsim infrastructure; misclassifying it would let a queue-flooding
        # agent void its own trial.
        ctx.state.backend.discard_reservation(reservation)
        release_job_admission(ctx.state, job_id)
        raise
    except Exception:
        ctx.state.backend.discard_reservation(reservation)
        release_job_admission(ctx.state, job_id)
        _infrastructure_failure(
            ctx,
            job_id=job_id,
            device_id=device_id,
            tool=tool,
            reservation=reservation,
            identity=identity,
            stage="job_enqueue",
        )
        raise RuntimeError("qsim compilation job enqueue failed") from None
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
        raise RuntimeError("qsim compilation submission publication failed") from None
    finally:
        submission_logged.set()
    return {"job_id": job_id, "status": "queued"}


def submit_compilation_mirror(
    ctx: ActionContext,
    *,
    device_id: str,
    candidate_id: str,
    mirror_seed: int,
    shots: int,
) -> dict[str, Any]:
    request = CompilationMirrorRequest.model_validate(
        {"candidate_id": candidate_id, "mirror_seed": mirror_seed, "shots": shots}
    )
    return _submit(
        ctx,
        device_id=device_id,
        tool="run_compilation_mirror",
        request=request,
    )


def submit_compilation_readout_reference(
    ctx: ActionContext,
    *,
    device_id: str,
    prepared_bitstring: str,
    shots: int,
) -> dict[str, Any]:
    request = CompilationReadoutReferenceRequest.model_validate(
        {"prepared_bitstring": prepared_bitstring, "shots": shots}
    )
    return _submit(
        ctx,
        device_id=device_id,
        tool="run_compilation_readout_reference",
        request=request,
    )


__all__ = [
    "CAPABILITY",
    "EVIDENCE_CONTRACT",
    "INFRASTRUCTURE_FAILURE",
    "submit_compilation_mirror",
    "submit_compilation_readout_reference",
]
