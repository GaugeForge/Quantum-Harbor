"""Surface-neutral action seam for bounded-depth Rydberg trajectory jobs."""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.evidence import persist_public_raw_record
from qiqcbench.qsim.jobs import JobQueueFullError, new_job_id
from qiqcbench.qsim.qtypes.neutral_atom_manybody_simulator.wire import (
    JobRydbergTrajectoryData,
    RydbergTrajectoryRequest,
)

_CAPABILITY = "stroboscopic_rydberg_evolution"
_INFRASTRUCTURE_FAILURE = "QIQCBENCH_RYDBERG_TRAJECTORY_INFRASTRUCTURE_FAILURE"


def _infrastructure_failure(
    ctx: ActionContext, *, job_id: str, device_id: str, stage: str
) -> JobResult:
    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": "rydberg_trajectory_infrastructure_failure",
                "tool": "run_stroboscopic_rydberg_evolution",
                "device_id": device_id,
                "job_id": job_id,
                "failure_class": _INFRASTRUCTURE_FAILURE,
                "failure_stage": stage,
            }
        )
    except Exception:
        pass
    return JobResult(
        job_id=job_id,
        device_id=device_id,
        status="failed",
        error=_INFRASTRUCTURE_FAILURE,
    )


def submit_stroboscopic_rydberg_evolution(
    ctx: ActionContext, *, device_id: str, protocol_id: str, system_size: int, shots: int
) -> dict[str, Any]:
    if _CAPABILITY not in ctx.state.active_capabilities:
        raise ValueError(f"{_CAPABILITY} capability is not active for this run")
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    request = RydbergTrajectoryRequest.model_validate(
        {"protocol_id": protocol_id, "system_size": system_size, "shots": shots}
    )
    digest = hashlib.sha256(json.dumps(request.model_dump(), sort_keys=True).encode()).hexdigest()
    salt = ctx.state.next_salt()
    submission_logged = threading.Event()

    def runner(job_id: str) -> JobResult:
        submission_logged.wait()
        try:
            result = ctx.state.backend.run_stroboscopic_rydberg_evolution(request, job_id, salt)
        except Exception:
            return _infrastructure_failure(
                ctx,
                job_id=job_id,
                device_id=device_id,
                stage="backend_execution",
            )
        if isinstance(result.data, JobRydbergTrajectoryData):
            try:
                raw_data_file = persist_public_raw_record(
                    ctx.state.log_dir, job_id=job_id, record=result.data.model_dump()
                )
                if raw_data_file is not None:
                    result.data = result.data.model_copy(
                        update={
                            "load_occupancy_b64": None,
                            "midsequence_occupancy_b64": None,
                            "final_occupancy_b64": None,
                            "target_z_readout_b64": None,
                            "raw_data_file": raw_data_file,
                        }
                    )
                ctx.state.log(
                    {
                        "surface": ctx.surface,
                        "action": "rydberg_trajectory_result",
                        "tool": "run_stroboscopic_rydberg_evolution",
                        "device_id": device_id,
                        "job_id": job_id,
                        "request_digest": digest,
                        "data": result.data.model_dump(),
                    }
                )
            except Exception:
                return _infrastructure_failure(
                    ctx,
                    job_id=job_id,
                    device_id=device_id,
                    stage="evidence_publication",
                )
        elif result.status == "complete":
            return _infrastructure_failure(
                ctx,
                job_id=job_id,
                device_id=device_id,
                stage="invalid_backend_result",
            )
        return result

    job_id = new_job_id()
    try:
        ctx.state.jobs.submit(runner, job_id=job_id)
    except JobQueueFullError:
        # Typed admission backpressure is agent-actionable model behavior, not
        # qsim infrastructure; misclassifying it would let a queue-flooding
        # agent void its own trial.
        raise
    except Exception:
        _infrastructure_failure(
            ctx,
            job_id=job_id,
            device_id=device_id,
            stage="job_enqueue",
        )
        raise RuntimeError("qsim Rydberg trajectory enqueue failed") from None
    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": "submit_rydberg_trajectory",
                "tool": "run_stroboscopic_rydberg_evolution",
                "device_id": device_id,
                "job_id": job_id,
                "request": request.model_dump(),
                "request_digest": digest,
            }
        )
    except Exception:
        _infrastructure_failure(
            ctx,
            job_id=job_id,
            device_id=device_id,
            stage="submission_log_publication",
        )
        raise RuntimeError("qsim Rydberg submission publication failed") from None
    finally:
        submission_logged.set()
    return {"job_id": job_id, "status": "queued"}


__all__ = ["submit_stroboscopic_rydberg_evolution"]
