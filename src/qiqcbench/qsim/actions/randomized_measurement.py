"""Surface-neutral, evidence-bound ion-circuit and randomized-measurement actions."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.actions.common import release_job_admission, reserve_job_admission
from qiqcbench.qsim.backends.provider_artifacts import canonical_request_hash
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.jobs import new_job_id
from qiqcbench.qsim.qtypes.ion_trap_gate_model.wire import (
    IonCircuitRequest,
    RandomizedMeasurementBatchRequestV2,
)

__all__ = ["submit_ion_circuit", "submit_randomized_measurement_batch"]

_CAPABILITY = "randomized_measurement"
_QTYPE = "ion_trap_gate_model"
_EXPERIMENT_JOBS_BUDGET_KEY = "ion_trap_gate_model:experiment_jobs"
_RAW_SHOT_RECORDS_BUDGET_KEY = "ion_trap_gate_model:raw_shot_records"
_JOB_RESULT_POLLS_BUDGET_KEY = "ion_trap_gate_model:job_result_polls"


def _validate_state(ctx: ActionContext, *, action: str) -> str:
    state = ctx.state
    if _CAPABILITY not in state.active_capabilities:
        raise ValueError(
            f"{_CAPABILITY} capability is not active for this run; cannot submit {action}"
        )
    qtype = getattr(state.hidden, "qtype", None)
    if qtype != _QTYPE:
        raise ValueError(f"{action} requires qtype={_QTYPE!r}; got qtype={qtype!r}.")
    if not state.task_id:
        raise ValueError(f"{action} requires ctx.state.task_id")
    return str(state.hidden.device_id)


def _materials(ctx: ActionContext):
    from qiqcbench.qsim.qtypes.ion_trap_gate_model.capabilities.randomized_measurement import (
        load_randomized_measurement_materials,
    )
    from qiqcbench.qsim.task_materials import public_task_material_dir

    return load_randomized_measurement_materials(public_task_material_dir(ctx.state.task_id))


def _rollback_submission_resources(
    ctx: ActionContext,
    *,
    job_id: str,
    raw_records: int,
    budgets_reserved: bool,
) -> None:
    ctx.state.unregister_job_result_poll_budget(job_id=job_id)
    if budgets_reserved:
        ctx.state.release_action_budgets(
            reservations={
                _EXPERIMENT_JOBS_BUDGET_KEY: 1,
                _RAW_SHOT_RECORDS_BUDGET_KEY: raw_records,
            }
        )
    release_job_admission(ctx.state, job_id)


def _reserve_submission_resources(
    ctx: ActionContext, *, job_id: str, raw_records: int
) -> dict[str, int]:
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
    ctx: ActionContext, *, tool: str, job_id: str, failure_stage: str
) -> None:
    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": "randomized_measurement_submission_infrastructure_failure",
                "tool": tool,
                "job_id": job_id,
                "failure_kind": "qsim_internal",
                "failure_stage": failure_stage,
            }
        )
    except Exception:
        return


def _raw_data_digest(data: Any) -> str:
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
    action: str,
    tool: str,
    job_id: str,
    request: IonCircuitRequest | RandomizedMeasurementBatchRequestV2,
    result: JobResult,
) -> None:
    if result.status != "complete" or result.data is None:
        return
    event: dict[str, Any] = {
        "surface": ctx.surface,
        "action": action,
        "tool": tool,
        "job_id": job_id,
        "request_digest": canonical_request_hash(request),
        "raw_data_sha256": _raw_data_digest(result.data),
        "shots": result.shots,
        "result_kind": result.data.kind,
    }
    if isinstance(request, RandomizedMeasurementBatchRequestV2):
        event.update(
            {
                "subsystem_qubits": list(request.subsystem_qubits),
                "n_unitaries": request.n_unitaries,
                "shots_per_unitary": request.shots_per_unitary,
                "experiment_tag": (
                    None
                    if request.experiment_tag is None
                    else request.experiment_tag.model_dump(mode="json")
                ),
            }
        )
    ctx.state.log(event)


def _submit(
    ctx: ActionContext,
    *,
    device_id: str,
    tool: str,
    submission_action: str,
    result_action: str,
    request: IonCircuitRequest | RandomizedMeasurementBatchRequestV2,
    raw_records: int,
    runner_factory,
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
        raise RuntimeError("qsim ion-trap job allocation failed") from None

    def runner(allocated_job_id: str) -> JobResult:
        result = runner_factory(allocated_job_id, salt)
        _log_result_evidence(
            ctx,
            action=result_action,
            tool=tool,
            job_id=allocated_job_id,
            request=request,
            result=result,
        )
        return result

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
                "max_raw_shot_records_total": (ctx.state.public.budget.max_raw_shot_records_total),
                "max_experiment_jobs": ctx.state.public.budget.max_experiment_jobs,
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
        raise RuntimeError("qsim ion-trap submission publication failed") from None
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
        raise RuntimeError("qsim ion-trap enqueue failed") from None
    return {"job_id": job_id, "status": "queued"}


def submit_ion_circuit(ctx: ActionContext, request: IonCircuitRequest) -> dict[str, Any]:
    device_id = _validate_state(ctx, action="submit_ion_circuit")
    materials = _materials(ctx)

    from qiqcbench.qsim.qtypes.ion_trap_gate_model.capabilities.randomized_measurement import (
        run_ion_circuit_request,
        validate_ion_circuit_request,
    )

    validate_ion_circuit_request(request, ctx.state.public)

    def runner_factory(job_id: str, salt: int) -> JobResult:
        return run_ion_circuit_request(
            ctx.state.backend,
            request,
            materials=materials,
            job_id=job_id,
            device_id=device_id,
            salt=salt,
        )

    return _submit(
        ctx,
        device_id=device_id,
        tool="run_ion_circuit",
        submission_action="submit_ion_circuit",
        result_action="ion_circuit_evidence",
        request=request,
        raw_records=request.shots,
        runner_factory=runner_factory,
    )


def submit_randomized_measurement_batch(
    ctx: ActionContext, request: RandomizedMeasurementBatchRequestV2
) -> dict[str, Any]:
    device_id = _validate_state(ctx, action="submit_randomized_measurement_batch")
    materials = _materials(ctx)

    from qiqcbench.qsim.qtypes.ion_trap_gate_model.capabilities.randomized_measurement import (
        run_rm_batch_request,
        validate_rm_batch_request,
    )

    validate_rm_batch_request(request, ctx.state.public)
    if request.experiment_tag is not None and request.experiment_tag.task_id != ctx.state.task_id:
        raise ValueError("experiment_tag.task_id must match the active task")
    raw_records = request.n_unitaries * request.shots_per_unitary

    def runner_factory(job_id: str, salt: int) -> JobResult:
        return run_rm_batch_request(
            ctx.state.backend,
            request,
            materials=materials,
            job_id=job_id,
            device_id=device_id,
            salt=salt,
        )

    return _submit(
        ctx,
        device_id=device_id,
        tool="run_randomized_measurement_batch",
        submission_action="submit_randomized_measurement_batch",
        result_action="randomized_measurement_evidence",
        request=request,
        raw_records=raw_records,
        runner_factory=runner_factory,
    )
