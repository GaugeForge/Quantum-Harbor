"""Surface-neutral submit seam for digital VQE observable batches.

Used by future agent-facing surfaces (MCP, Qiskit) to enqueue a VQE-shaped
batch request through ``run_observable_batch_request`` without extending the
qtype backend protocols (controller decision).
"""

from __future__ import annotations

from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.actions.common import release_job_admission, reserve_job_admission
from qiqcbench.qsim.core.wire import (
    JobObservableBitstringData,
    JobResult,
    ObservableBatchPoint,
    ObservableBatchRequest,
)
from qiqcbench.qsim.jobs import new_job_id
from qiqcbench.qsim.qtypes.digital_gate_model.capabilities.vqe import (
    load_vqe_public_materials,
    run_observable_batch_request,
    validate_observable_batch_request,
)
from qiqcbench.qsim.qtypes.digital_gate_model.capabilities.vqe.observables import (
    OBSERVABLE_BATCH_EVIDENCE_CONTRACT,
    OBSERVABLE_BATCH_EVIDENCE_SCHEMA_VERSION,
    OBSERVABLE_BATCH_INTERNAL_FAILURE_ERROR,
    OBSERVABLE_BATCH_INTERNAL_FAILURE_KIND,
    OBSERVABLE_BATCH_JOB_ENQUEUE_FAILURE_STAGE,
    OBSERVABLE_BATCH_SUBMISSION_FAILURE_ACTION,
    OBSERVABLE_BATCH_SUBMISSION_FAILURE_ERROR,
    OBSERVABLE_BATCH_SUBMISSION_LOG_FAILURE_STAGE,
    OBSERVABLE_BATCH_TERMINAL_LOG_FAILURE_STAGE,
    build_point_value_digest_rows,
    compute_observable_batch_request_digest,
    compute_public_material_digests,
    summarize_observable_batch_result,
)
from qiqcbench.qsim.qtypes.digital_gate_model.capabilities.vqe.replay import (
    replay_observable_batch_request,
)

__all__ = ["submit_observable_batch"]


def _best_effort_log_submission_failure(
    ctx: ActionContext,
    *,
    task_id: str,
    device_id: str,
    job_id: str,
    failure_stage: str,
) -> None:
    """Publish a sanitized accepted-submission infrastructure failure."""

    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": OBSERVABLE_BATCH_SUBMISSION_FAILURE_ACTION,
                "tool": "run_observable_batch",
                "evidence_contract": OBSERVABLE_BATCH_EVIDENCE_CONTRACT,
                "evidence_schema_version": OBSERVABLE_BATCH_EVIDENCE_SCHEMA_VERSION,
                "device_id": device_id,
                "task_id": task_id,
                "job_id": job_id,
                "status": "failed",
                "error": OBSERVABLE_BATCH_SUBMISSION_FAILURE_ERROR,
                "failure_kind": OBSERVABLE_BATCH_INTERNAL_FAILURE_KIND,
                "failure_stage": failure_stage,
            }
        )
    except Exception:
        return


def submit_observable_batch(
    ctx: ActionContext,
    *,
    device_id: str,
    profile: str,
    points: list[dict[str, Any]],
    shots_per_setting: int,
    parameter_convention: str = "layer_major_ry_then_rz",
) -> dict[str, Any]:
    """Queue a VQE observable-batch run through the digital qtype runtime."""
    if "digital_vqe" not in ctx.state.active_capabilities:
        raise ValueError(
            "digital_vqe capability is not active for this run; cannot submit observable batch"
        )
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    task_id = ctx.state.task_id
    if not task_id:
        raise ValueError("submit_observable_batch requires ctx.state.task_id")

    # Deferred to avoid the actions/__init__ -> task_materials -> devices ->
    # qtypes.registry cycle on package import.
    from qiqcbench.qsim.task_materials import public_task_material_dir

    public_material_dir = public_task_material_dir(task_id)
    materials = load_vqe_public_materials(public_material_dir)

    # Bound point-object allocation itself. The full material-driven validator
    # below repeats this check after wire-model construction as defense in depth.
    max_points_per_call = materials.ansatz.evaluation_budget.max_parameter_points_per_call
    if len(points) > max_points_per_call:
        raise ValueError(
            f"observable-batch request contains {len(points)} parameter points; "
            f"per-call limit is {max_points_per_call}"
        )
    request = ObservableBatchRequest(
        shots_per_setting=shots_per_setting,
        profile=profile,
        parameter_convention=parameter_convention,
        points=[ObservableBatchPoint.model_validate(p) for p in points],
    )
    # Preflight every public semantic and per-call constraint synchronously.
    # Invalid work must not receive a job ID or enter the asynchronous queue.
    validate_observable_batch_request(request, materials)
    if request.shots_per_setting > ctx.state.public.max_shots:
        raise ValueError(
            f"shots_per_setting {request.shots_per_setting} exceeds public device "
            f"max_shots {ctx.state.public.max_shots}"
        )
    point_value_digest_rows = build_point_value_digest_rows(request.points)
    request_digest = compute_observable_batch_request_digest(request)
    public_material_digests = compute_public_material_digests(public_material_dir)

    evaluation_budget = materials.ansatz.evaluation_budget
    if request.profile == "ideal":
        point_budget = evaluation_budget.ideal_max_submitted_parameter_point_evaluations
    else:
        point_budget = evaluation_budget.public_noise_max_submitted_parameter_point_evaluations
    budget_key = f"digital_vqe:{task_id}:{request.profile}:submitted_parameter_point_evaluations"
    raw_records_key = f"digital_vqe:{task_id}:raw_shot_records"
    raw_shot_records = (
        len(request.points) * len(materials.hamiltonian.pauli_terms) * request.shots_per_setting
    )
    ctx.state.reserve_action_budgets(
        reservations={
            budget_key: (len(request.points), point_budget),
            raw_records_key: (
                raw_shot_records,
                evaluation_budget.max_raw_shot_records_total,
            ),
        }
    )
    # Exact rollback amounts if admission later refuses: slot and budgets
    # succeed or fail together.
    budget_rollback = {budget_key: len(request.points), raw_records_key: raw_shot_records}

    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        point_summaries: list[dict[str, Any]] | None = None
        unexpected_failure = False
        try:
            if ctx.state.backend_mode == "provider_replay":
                replay_root = getattr(ctx.state.backend, "replay_root", None)
                if replay_root is None:
                    result = JobResult(
                        job_id=job_id,
                        device_id=ctx.state.hidden.device_id,
                        status="failed",
                        shots=request.shots_per_setting,
                        error=(
                            "provider_replay VQE observable batches require a backend "
                            "with replay_root"
                        ),
                    )
                else:
                    result = replay_observable_batch_request(
                        request,
                        ctx.state.hidden,
                        materials,
                        task_id=task_id,
                        replay_root=replay_root,
                        job_id=job_id,
                    )
            else:
                result = run_observable_batch_request(
                    request,
                    ctx.state.hidden,
                    materials,
                    job_id,
                    salt,
                    # Per-attempt private shot entropy owned by the simulator
                    # backend; a backend without one gets a fresh per-call draw.
                    run_entropy=getattr(ctx.state.backend, "run_entropy", None),
                )

            if ctx.state.backend_mode == "simulator" and result.status == "failed":
                # Admission has already validated every model-controlled
                # request field. A simulator failure is therefore qsim-owned
                # infrastructure, whether the engine returned it explicitly
                # or raised. Replace any internal error detail before it can
                # reach the public JobResult or evidence log.
                result = JobResult(
                    job_id=job_id,
                    device_id=ctx.state.hidden.device_id,
                    status="failed",
                    shots=request.shots_per_setting,
                    error=OBSERVABLE_BATCH_INTERNAL_FAILURE_ERROR,
                )
                unexpected_failure = True
            elif result.status == "complete":
                if not isinstance(result.data, JobObservableBitstringData):
                    raise RuntimeError(
                        "completed observable-batch job returned invalid result data"
                    )
                point_summaries = summarize_observable_batch_result(
                    result.data,
                    point_value_digest_rows,
                    expected_shots_per_setting=request.shots_per_setting,
                )
            elif result.status != "failed":
                raise RuntimeError("observable-batch worker returned a nonterminal result")
        except Exception:
            # The JobManager cannot emit capability-specific evidence. Convert
            # unexpected worker or summarization failures here so every
            # accepted request still has one typed v2 terminal event. Do not
            # expose exception text: simulator internals may contain hidden
            # paths, configuration values, or provider details.
            result = JobResult(
                job_id=job_id,
                device_id=ctx.state.hidden.device_id,
                status="failed",
                shots=request.shots_per_setting,
                error=OBSERVABLE_BATCH_INTERNAL_FAILURE_ERROR,
            )
            unexpected_failure = True

        result_event: dict[str, Any] = {
            "surface": ctx.surface,
            "action": "observable_batch_result",
            "tool": "run_observable_batch",
            "evidence_contract": OBSERVABLE_BATCH_EVIDENCE_CONTRACT,
            "evidence_schema_version": OBSERVABLE_BATCH_EVIDENCE_SCHEMA_VERSION,
            "device_id": device_id,
            "task_id": task_id,
            "profile": request.profile,
            "parameter_convention": request.parameter_convention,
            "shots_per_setting": request.shots_per_setting,
            "n_points": len(request.points),
            "job_id": job_id,
            "request_digest": request_digest,
            "public_material_digests": public_material_digests,
            "status": result.status,
        }
        if result.status == "complete":
            if point_summaries is None:  # pragma: no cover - guarded above
                raise RuntimeError("completed observable-batch job has no evidence summary")
            result_event["points"] = point_summaries
        elif result.error is not None:
            result_event["error"] = result.error
        if unexpected_failure:
            result_event["failure_kind"] = OBSERVABLE_BATCH_INTERNAL_FAILURE_KIND
        try:
            ctx.state.log(result_event)
        except Exception:
            # Do not let JobManager turn a private logging exception into an
            # untyped public error. A one-shot publication fault can still be
            # represented by a second sanitized terminal attempt; persistent
            # log failure remains an execution infrastructure fault.
            result = JobResult(
                job_id=job_id,
                device_id=ctx.state.hidden.device_id,
                status="failed",
                shots=request.shots_per_setting,
                error=OBSERVABLE_BATCH_INTERNAL_FAILURE_ERROR,
            )
            result_event.pop("points", None)
            result_event["status"] = "failed"
            result_event["error"] = OBSERVABLE_BATCH_INTERNAL_FAILURE_ERROR
            result_event["failure_kind"] = OBSERVABLE_BATCH_INTERNAL_FAILURE_KIND
            result_event["failure_stage"] = OBSERVABLE_BATCH_TERMINAL_LOG_FAILURE_STAGE
            try:
                ctx.state.log(result_event)
            except Exception:
                pass
        return result

    # Preallocate and log before enqueue: a fast worker must never publish its
    # result evidence ahead of the corresponding submission evidence.
    # Reserve the queue slot first (one atomic admission lifecycle): a
    # queue-full refusal is the typed, model-actionable JobQueueFullError --
    # never an enqueue-infrastructure disposition -- and it fires before
    # anything is logged, rolling this call's budget reservations back.
    job_id = new_job_id()
    try:
        reserve_job_admission(ctx.state, job_id)
    except Exception:
        ctx.state.release_action_budgets(reservations=budget_rollback)
        raise
    try:
        ctx.state.register_job_result_poll_budget(
            job_id=job_id,
            budget_key=f"digital_vqe:{task_id}:job_result_polls",
            limit=evaluation_budget.max_job_result_polls,
        )
    except Exception:
        release_job_admission(ctx.state, job_id)
        ctx.state.release_action_budgets(reservations=budget_rollback)
        raise
    submit_event = {
        "surface": ctx.surface,
        "action": "submit_observable_batch",
        "tool": "run_observable_batch",
        "evidence_contract": OBSERVABLE_BATCH_EVIDENCE_CONTRACT,
        "evidence_schema_version": OBSERVABLE_BATCH_EVIDENCE_SCHEMA_VERSION,
        "device_id": device_id,
        "task_id": task_id,
        "profile": request.profile,
        "parameter_convention": request.parameter_convention,
        "shots_per_setting": request.shots_per_setting,
        "n_points": len(request.points),
        "job_id": job_id,
        "request_digest": request_digest,
        "public_material_digests": public_material_digests,
        "points": point_value_digest_rows,
        # Retain the schema-v1 field for development-log readers while the
        # v2 verifier migrates to the point-ID-bound rows above.
        "point_value_digests": [point["value_digest"] for point in point_value_digest_rows],
    }
    try:
        ctx.state.log(submit_event)
    except Exception:
        release_job_admission(ctx.state, job_id)
        _best_effort_log_submission_failure(
            ctx,
            task_id=task_id,
            device_id=device_id,
            job_id=job_id,
            failure_stage=OBSERVABLE_BATCH_SUBMISSION_LOG_FAILURE_STAGE,
        )
        raise RuntimeError(OBSERVABLE_BATCH_SUBMISSION_FAILURE_ERROR) from None
    try:
        ctx.state.jobs.submit(runner, job_id=job_id)
    except Exception:
        _best_effort_log_submission_failure(
            ctx,
            task_id=task_id,
            device_id=device_id,
            job_id=job_id,
            failure_stage=OBSERVABLE_BATCH_JOB_ENQUEUE_FAILURE_STAGE,
        )
        raise RuntimeError(OBSERVABLE_BATCH_SUBMISSION_FAILURE_ERROR) from None
    return {"job_id": job_id, "status": "queued"}
