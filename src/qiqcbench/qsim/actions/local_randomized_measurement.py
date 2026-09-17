"""Surface-neutral action for qsim-owned local randomized measurements."""

from __future__ import annotations

import hashlib

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.jobs import new_job_id
from qiqcbench.qsim.qtypes.digital_gate_model.capabilities.randomized_measurement.materials import (
    load_protocol,
)
from qiqcbench.qsim.qtypes.digital_gate_model.capabilities.randomized_measurement.runtime import (
    run_local_randomized_measurement,
)
from qiqcbench.qsim.qtypes.digital_gate_model.capabilities.randomized_measurement.schemas import (
    JobLocalRandomizedMeasurementData,
)
from qiqcbench.qsim.task_materials import public_task_material_dir

CAPABILITY = "local_randomized_measurement"
EVIDENCE_CONTRACT = "local_randomized_measurement_v1"
_INTERNAL_ERROR = "qsim local randomized-measurement execution failed"


def _validate_device_contract(ctx: ActionContext, protocol) -> None:
    public_ids = tuple(qubit.id for qubit in ctx.state.public.qubits)
    if public_ids != tuple(range(protocol.n_qubits)):
        raise ValueError("randomized-measurement protocol does not match the active device")
    connectivity = {
        tuple(sorted((int(edge[0]), int(edge[1])))) for edge in ctx.state.public.connectivity
    }
    required = {
        tuple(edge)
        for edge in (
            *protocol.depth_cycle.even_cnot_edges,
            *protocol.depth_cycle.odd_cnot_edges,
        )
    }
    if not required <= connectivity:
        raise ValueError("randomized-measurement protocol uses unavailable device edges")


def submit_local_randomized_measurement(
    ctx: ActionContext,
    *,
    depth: int,
) -> dict[str, object]:
    """Validate, log, and enqueue one fresh randomized-measurement job."""

    if CAPABILITY not in ctx.state.active_capabilities:
        raise ValueError(f"{CAPABILITY} capability is not active for this run")
    if ctx.state.backend_mode != "simulator":
        raise ValueError("local randomized measurement currently supports simulator mode only")
    task_id = ctx.state.task_id
    if not task_id:
        raise ValueError("local randomized measurement requires an active task")
    public_dir = public_task_material_dir(task_id)
    protocol = load_protocol(public_dir)
    if protocol.task_id != task_id:
        raise ValueError("randomized-measurement protocol task_id does not match the active task")
    _validate_device_contract(ctx, protocol)
    if isinstance(depth, bool) or not isinstance(depth, int):
        raise ValueError("depth must be an integer")
    if not protocol.depth_limits.minimum <= depth <= protocol.depth_limits.maximum:
        raise ValueError(
            f"depth must lie in [{protocol.depth_limits.minimum}, {protocol.depth_limits.maximum}]"
        )

    protocol_path = public_dir / "measurement_protocol.json"
    protocol_sha256 = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    budget_key = f"{CAPABILITY}:{task_id}:jobs"
    ctx.state.reserve_action_budgets(reservations={budget_key: (1, protocol.max_experiment_jobs)})
    salt = ctx.state.next_salt()
    job_id = new_job_id()

    submission = {
        "surface": ctx.surface,
        "action": "submit_local_randomized_measurement",
        "tool": "run_local_randomized_measurement",
        "evidence_contract": EVIDENCE_CONTRACT,
        "evidence_schema_version": 1,
        "task_id": task_id,
        "device_id": ctx.state.hidden.device_id,
        "job_id": job_id,
        "depth": depth,
        "protocol_sha256": protocol_sha256,
    }
    ctx.state.log(submission)

    def runner(accepted_job_id: str) -> JobResult:
        unexpected_failure = False
        try:
            result = run_local_randomized_measurement(
                depth=depth,
                hidden=ctx.state.hidden,
                protocol=protocol,
                job_id=accepted_job_id,
                salt=salt,
                run_entropy=getattr(ctx.state.backend, "run_entropy", None),
            )
            if result.status != "complete" or not isinstance(
                result.data, JobLocalRandomizedMeasurementData
            ):
                unexpected_failure = True
                result = JobResult(
                    job_id=accepted_job_id,
                    device_id=ctx.state.hidden.device_id,
                    status="failed",
                    error=_INTERNAL_ERROR,
                )
        except Exception:
            unexpected_failure = True
            result = JobResult(
                job_id=accepted_job_id,
                device_id=ctx.state.hidden.device_id,
                status="failed",
                error=_INTERNAL_ERROR,
            )
        terminal = {
            "surface": ctx.surface,
            "action": "local_randomized_measurement_result",
            "tool": "run_local_randomized_measurement",
            "evidence_contract": EVIDENCE_CONTRACT,
            "evidence_schema_version": 1,
            "task_id": task_id,
            "device_id": ctx.state.hidden.device_id,
            "job_id": accepted_job_id,
            "depth": depth,
            "protocol_sha256": protocol_sha256,
            "status": result.status,
        }
        if unexpected_failure:
            terminal["failure_kind"] = "qsim_internal"
            terminal["error"] = _INTERNAL_ERROR
        else:
            terminal.update(
                {
                    "n_random_circuits": protocol.sampling.n_random_circuits,
                    "n_measurement_bases": protocol.sampling.n_measurement_bases,
                    "shots_per_basis": protocol.sampling.shots_per_basis,
                    "readout_calibration_shots_per_preparation": (
                        protocol.sampling.readout_calibration_shots_per_preparation
                    ),
                }
            )
        ctx.state.log(terminal)
        return result

    try:
        ctx.state.jobs.submit(runner, job_id=job_id)
    except Exception:
        ctx.state.log(
            {
                **submission,
                "action": "local_randomized_measurement_submission_failure",
                "status": "failed",
                "failure_kind": "qsim_internal",
                "error": _INTERNAL_ERROR,
            }
        )
        raise RuntimeError(_INTERNAL_ERROR) from None
    return {"job_id": job_id, "status": "queued"}


__all__ = ["CAPABILITY", "EVIDENCE_CONTRACT", "submit_local_randomized_measurement"]
