"""Action seam for the bosonic_cavity_qec qtype.

Gates ``run_bosonic_program`` / ``run_bosonic_program_sweep`` on the
``bosonic_qec_control`` capability (fail-closed at the action layer, not only at
surface registration). Each completed job writes a task-neutral,
content-addressed execution artifact containing the exact validated request and
compact outcome histograms. Optional task-owned evidence tags add pre-execution
semantic commitments without constraining untagged exploratory runs.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.hidden_commitment import hidden_device_model_commitment
from qiqcbench.qsim.jobs import (
    JOB_EXECUTION_ERROR,
    JobQueueFullError,
    is_internal_job_failure,
    new_job_id,
)
from qiqcbench.qsim.qtypes.bosonic_cavity_qec import physics as bosonic_physics
from qiqcbench.qsim.qtypes.bosonic_cavity_qec.commitments import (
    BOSONIC_EVIDENCE_CONTRACT,
    BOSONIC_EVIDENCE_SCHEMA_VERSION,
    BOSONIC_EXECUTION_ARTIFACT_KIND,
    BOSONIC_EXECUTION_ARTIFACT_MEDIA_TYPE,
    canonical_json_bytes,
    public_device_model_commitment,
    sweep_coordinates_commitment,
)
from qiqcbench.qsim.qtypes.bosonic_cavity_qec.engine import program_digest
from qiqcbench.qsim.qtypes.bosonic_cavity_qec.wire import (
    MAX_ANCILLA_MEASUREMENTS_PER_CALL,
    MAX_FEEDBACK_SYNDROMES_PER_CALL,
    MAX_RECORDED_OUTCOMES_PER_CALL,
    BosonicProgramRequest,
    BosonicProgramSweepRequest,
    JobBosonicProgramData,
)

_CAPABILITY = "bosonic_qec_control"
_RESULT_ACTION = "bosonic_program_result"
_EVIDENCE_FAILURE_ACTION = "bosonic_execution_record_failure"
_SUBMISSION_FAILURE_ACTION = "bosonic_submission_failure"
_SEGMENT_EVIDENCE_CONTRACT = "bosonic_segment_evidence_v1"
_JOB_RESULT_POLLS_BUDGET_KEY = "bosonic_cavity_qec:job_result_polls"
_SUBMISSION_FAILURE_ERROR = "qsim bosonic submission infrastructure failure"


def _require_capability(ctx: ActionContext, action: str) -> None:
    if _CAPABILITY not in ctx.state.active_capabilities:
        raise ValueError(f"{_CAPABILITY} capability is not active for this run; cannot {action}")


def _program_summary(ops: list) -> dict[str, Any]:
    kinds = Counter(op.kind for op in ops)
    has_cond = kinds.get("conditional", 0) > 0
    n_ancilla = kinds.get("ancilla_measure", 0)
    n_unrecorded = sum(op.kind == "ancilla_measure" and not op.record for op in ops)
    n_number = kinds.get("photon_number_measure", 0)
    return {
        "program_digest": program_digest(ops),
        "op_kind_counts": dict(kinds),
        "n_ops": len(ops),
        "has_conditional_feedback": has_cond,
        "n_ancilla_measures": n_ancilla,
        "n_unrecorded_ancilla_measures": n_unrecorded,
        "n_number_measures": n_number,
    }


def _evidence_identity(ctx: ActionContext) -> dict[str, Any]:
    execution_context_id = getattr(ctx.state, "execution_context_id", None)
    commitment_secret = getattr(ctx.state, "hidden_commitment_secret", None)
    if execution_context_id is not None and commitment_secret is None:
        raise RuntimeError("execution-bound bosonic evidence requires a hidden commitment secret")
    if execution_context_id is None and commitment_secret is not None:
        raise RuntimeError("unbound bosonic evidence must not carry hidden commitment material")
    task_id = getattr(ctx.state, "task_id", None) or "unbound_task"
    identity = {
        "evidence_contract": BOSONIC_EVIDENCE_CONTRACT,
        "evidence_schema_version": BOSONIC_EVIDENCE_SCHEMA_VERSION,
        "public_device_model_commitment": public_device_model_commitment(ctx.state.public),
    }
    if execution_context_id is not None:
        identity["hidden_device_model_commitment"] = hidden_device_model_commitment(
            ctx.state.hidden,
            task_id=task_id,
            commitment_secret=commitment_secret,
        )
    return identity


def _canonical_ops(request: BosonicProgramRequest | BosonicProgramSweepRequest) -> list[dict]:
    return [op.model_dump(mode="json") for op in request.ops]


def _validate_execution_capacity(
    ctx: ActionContext,
    request: BosonicProgramRequest | BosonicProgramSweepRequest,
) -> None:
    public = ctx.state.public
    if request.shots > public.max_shots:
        raise ValueError(f"shots {request.shots} exceeds public max_shots {public.max_shots}")
    if len(request.ops) > public.max_ops_per_program:
        raise ValueError(
            f"program has {len(request.ops)} ops > public max_ops_per_program "
            f"{public.max_ops_per_program}"
        )
    measurement_columns = sum(
        (op.kind == "ancilla_measure" and op.record) or op.kind == "photon_number_measure"
        for op in request.ops
    )
    if measurement_columns == 0:
        raise ValueError(
            "bosonic program must contain at least one measurement that is recorded "
            "(an ancilla_measure with record=true or a photon_number_measure)"
        )
    points = len(request.sweep_values) if isinstance(request, BosonicProgramSweepRequest) else 1
    if points > public.max_sweep_points_per_call:
        raise ValueError(
            f"sweep has {points} points > public max_sweep_points_per_call "
            f"{public.max_sweep_points_per_call}"
        )
    recorded_outcomes = request.shots * measurement_columns * points
    recorded_outcome_limit = min(
        MAX_RECORDED_OUTCOMES_PER_CALL,
        public.max_recorded_outcomes_per_call,
    )
    if recorded_outcomes > recorded_outcome_limit:
        raise ValueError(
            "bosonic request exceeds the per-call recorded-outcome capacity "
            f"({recorded_outcomes} > {recorded_outcome_limit})"
        )
    ancilla_measurements = 0  # recorded measures only (unrecorded ones neither branch nor index)
    feedback_syndromes: set[int] = set()
    for op in request.ops:
        if op.kind == "ancilla_measure" and op.record:
            ancilla_measurements += 1
        elif op.kind == "conditional":
            if op.on_index >= ancilla_measurements:
                raise ValueError(
                    "conditional on_index must identify a previously recorded syndrome"
                )
            feedback_syndromes.add(op.on_index)
    if len(feedback_syndromes) > MAX_FEEDBACK_SYNDROMES_PER_CALL:
        raise ValueError(
            "bosonic request exceeds the exact feedback-branch capacity "
            f"({len(feedback_syndromes)} > {MAX_FEEDBACK_SYNDROMES_PER_CALL})"
        )
    ancilla_measurement_limit = min(
        MAX_ANCILLA_MEASUREMENTS_PER_CALL,
        public.max_ancilla_measurements_per_call,
    )
    if ancilla_measurements > ancilla_measurement_limit:
        raise ValueError(
            "bosonic request exceeds the ancilla-measurement branch capacity "
            f"({ancilla_measurements} > {ancilla_measurement_limit})"
        )


def _log_submission_failure(
    ctx: ActionContext,
    *,
    tool: str,
    failure_stage: str,
    job_id: str | None = None,
) -> None:
    """Best-effort bounded marker for an admitted submission-path failure."""

    if getattr(ctx.state.public, "budget", None) is None:
        return

    event = {
        "surface": ctx.surface,
        "action": _SUBMISSION_FAILURE_ACTION,
        "tool": tool,
        "failure_kind": "qsim_internal",
        "failure_stage": failure_stage,
    }
    if job_id is not None:
        event["job_id"] = job_id
    try:
        ctx.state.log(event)
    except Exception:
        return


def _reserve_public_job_admission(ctx: ActionContext, *, tool: str) -> str | None:
    """Reserve schema-v3 queue admission before costly or cumulative work."""

    budget = getattr(ctx.state.public, "budget", None)
    if budget is None:
        return None
    register = getattr(ctx.state, "register_job_result_poll_budget", None)
    unregister = getattr(ctx.state, "unregister_job_result_poll_budget", None)
    reserve_poll = getattr(ctx.state, "reserve_job_result_poll", None)
    reserve_admission = getattr(ctx.state.jobs, "reserve_admission", None)
    release_admission = getattr(ctx.state.jobs, "release_admission", None)
    peek = getattr(ctx.state.jobs, "peek", None)
    if not all(
        callable(value)
        for value in (
            register,
            unregister,
            reserve_poll,
            reserve_admission,
            release_admission,
            peek,
        )
    ):
        _log_submission_failure(
            ctx,
            tool=tool,
            failure_stage="poll_budget_binding",
        )
        raise RuntimeError(_SUBMISSION_FAILURE_ERROR)
    job_id = new_job_id()
    try:
        reserve_admission(job_id)
    except JobQueueFullError:
        raise
    except Exception:
        _log_submission_failure(
            ctx,
            tool=tool,
            job_id=job_id,
            failure_stage="job_admission",
        )
        raise RuntimeError(_SUBMISSION_FAILURE_ERROR) from None
    return job_id


def _release_public_job_admission(ctx: ActionContext, job_id: str | None) -> None:
    if job_id is None:
        return
    release = getattr(ctx.state.jobs, "release_admission", None)
    if callable(release):
        try:
            release(job_id)
        except Exception:
            pass


def _submit_with_public_poll_budget(
    ctx: ActionContext,
    runner,
    *,
    tool: str,
    reserved_job_id: str | None,
) -> str:
    """Enqueue one job and bind schema-v3 jobs to the shared poll meter."""

    budget = getattr(ctx.state.public, "budget", None)
    if budget is None:
        return ctx.state.jobs.submit(runner)
    if reserved_job_id is None:
        _log_submission_failure(ctx, tool=tool, failure_stage="job_admission")
        raise RuntimeError(_SUBMISSION_FAILURE_ERROR)
    job_id = reserved_job_id
    register = ctx.state.register_job_result_poll_budget
    unregister = ctx.state.unregister_job_result_poll_budget

    def rollback() -> None:
        try:
            unregister(job_id=job_id)
        except Exception:
            pass
        _release_public_job_admission(ctx, job_id)

    try:
        register(
            job_id=job_id,
            budget_key=_JOB_RESULT_POLLS_BUDGET_KEY,
            limit=budget.max_job_result_polls,
        )
    except Exception:
        rollback()
        _log_submission_failure(
            ctx,
            tool=tool,
            job_id=job_id,
            failure_stage="poll_budget_registration",
        )
        raise RuntimeError(_SUBMISSION_FAILURE_ERROR) from None
    try:
        submitted = ctx.state.jobs.submit(runner, job_id=job_id)
    except Exception:
        rollback()
        _log_submission_failure(
            ctx,
            tool=tool,
            job_id=job_id,
            failure_stage="job_enqueue",
        )
        raise RuntimeError(_SUBMISSION_FAILURE_ERROR) from None
    if submitted != job_id:
        rollback()
        _log_submission_failure(
            ctx,
            tool=tool,
            job_id=job_id,
            failure_stage="job_enqueue",
        )
        raise RuntimeError(_SUBMISSION_FAILURE_ERROR)
    return job_id


def _complex_vector_payload(vector: np.ndarray) -> list[list[float]]:
    return [[float(value.real), float(value.imag)] for value in vector]


def _complex_matrix_payload(matrix: np.ndarray) -> list[list[list[float]]]:
    return [_complex_vector_payload(row) for row in matrix]


def _ideal_cavity_unitary(ops: list, n_max: int) -> np.ndarray:
    unitary = np.eye(n_max, dtype=complex)
    for op in ops:
        if op.kind == "displace":
            primitive = bosonic_physics.displace_unitary(complex(op.alpha_re, op.alpha_im), n_max)
        elif op.kind == "snap":
            primitive = bosonic_physics.snap_unitary(op.thetas, n_max)
        else:
            raise ValueError(
                "bosonic_segment_evidence_v1 preparation/measurement segments "
                "may contain only displace and snap ops"
            )
        unitary = primitive @ unitary
    return unitary


def _segment_evidence_indices(
    request: BosonicProgramRequest | BosonicProgramSweepRequest,
) -> tuple[int, int] | None:
    """Validate the tag and segment shapes without doing matrix algebra."""

    tag = request.evidence_tag
    if tag is None or tag.get("contract") != _SEGMENT_EVIDENCE_CONTRACT:
        return None
    prep_end = tag.get("preparation_end_op_index")
    measurement_start = tag.get("measurement_start_op_index")
    if (
        not isinstance(prep_end, int)
        or isinstance(prep_end, bool)
        or not isinstance(measurement_start, int)
        or isinstance(measurement_start, bool)
        or not (1 <= prep_end <= measurement_start < len(request.ops))
    ):
        raise ValueError(
            "bosonic_segment_evidence_v1 requires valid exclusive preparation-end "
            "and inclusive measurement-start op indices"
        )
    if request.ops[-1].kind != "photon_number_measure":
        raise ValueError("bosonic segment evidence requires terminal photon_number_measure")
    if sum(op.kind == "photon_number_measure" for op in request.ops) != 1:
        raise ValueError("bosonic segment evidence requires exactly one photon_number_measure")
    if isinstance(request, BosonicProgramSweepRequest) and not (
        prep_end <= request.sweep_op_index < measurement_start
    ):
        raise ValueError("a scored sweep may vary only the storage/control middle segment")
    for op in (*request.ops[:prep_end], *request.ops[measurement_start:-1]):
        if op.kind not in {"displace", "snap"}:
            raise ValueError(
                "bosonic_segment_evidence_v1 preparation/measurement segments "
                "may contain only displace and snap ops"
            )
    return prep_end, measurement_start


def _segment_evidence(
    request: BosonicProgramRequest | BosonicProgramSweepRequest,
    *,
    n_max: int,
    indices: tuple[int, int] | None,
) -> dict[str, Any] | None:
    if indices is None:
        return None
    prep_end, measurement_start = indices

    preparation = _ideal_cavity_unitary(list(request.ops[:prep_end]), n_max)
    measurement = _ideal_cavity_unitary(list(request.ops[measurement_start:-1]), n_max)
    vacuum = np.zeros(n_max, dtype=complex)
    vacuum[0] = 1.0
    return {
        "contract": _SEGMENT_EVIDENCE_CONTRACT,
        "schema_version": 1,
        "preparation_end_op_index": prep_end,
        "measurement_start_op_index": measurement_start,
        "preparation_state": _complex_vector_payload(preparation @ vacuum),
        "measurement_unitary": _complex_matrix_payload(measurement),
    }


def _discard_backend_reservation(ctx: ActionContext, token: int | None) -> None:
    if token is None:
        return
    discard = getattr(ctx.state.backend, "discard_bosonic_reservation", None)
    if callable(discard):
        try:
            discard(token)
        except Exception:
            pass


def _prepare_submission(
    ctx: ActionContext,
    request: BosonicProgramRequest | BosonicProgramSweepRequest,
    *,
    tool: str,
) -> tuple[str | None, int | None, dict[str, Any] | None, int, dict[str, Any], dict[str, Any]]:
    """Admit, preflight, and derive evidence in side-effect-safe order."""

    indices = _segment_evidence_indices(request)
    budget = getattr(ctx.state.public, "budget", None)
    reserve_backend = getattr(ctx.state.backend, "reserve_bosonic_request", None)
    discard_backend = getattr(ctx.state.backend, "discard_bosonic_reservation", None)
    if budget is not None and not all(
        callable(value) for value in (reserve_backend, discard_backend)
    ):
        _log_submission_failure(ctx, tool=tool, failure_stage="backend_budget_binding")
        raise RuntimeError(_SUBMISSION_FAILURE_ERROR)

    reserved_job_id = _reserve_public_job_admission(ctx, tool=tool)
    reservation_token: int | None = None
    try:
        if callable(reserve_backend):
            reservation_token = reserve_backend(request)
    except ValueError:
        _release_public_job_admission(ctx, reserved_job_id)
        raise
    except Exception:
        _release_public_job_admission(ctx, reserved_job_id)
        if budget is None:
            raise
        _log_submission_failure(
            ctx,
            tool=tool,
            job_id=reserved_job_id,
            failure_stage="backend_budget_reservation",
        )
        raise RuntimeError(_SUBMISSION_FAILURE_ERROR) from None

    try:
        segment_evidence = _segment_evidence(
            request,
            n_max=int(ctx.state.public.n_max),
            indices=indices,
        )
        salt = ctx.state.next_salt()
        summary = _program_summary(list(request.ops))
        identity = _evidence_identity(ctx)
    except Exception:
        _discard_backend_reservation(ctx, reservation_token)
        _release_public_job_admission(ctx, reserved_job_id)
        if budget is None:
            raise
        _log_submission_failure(
            ctx,
            tool=tool,
            job_id=reserved_job_id,
            failure_stage="pre_enqueue_evidence_derivation",
        )
        raise RuntimeError(_SUBMISSION_FAILURE_ERROR) from None
    return reserved_job_id, reservation_token, segment_evidence, salt, summary, identity


def _sanitize_reserved_backend_failure(
    result: JobResult, *, reservation_token: int | None
) -> JobResult:
    if (
        reservation_token is not None
        and result.status == "failed"
        and not is_internal_job_failure(result.error)
    ):
        return result.model_copy(update={"error": JOB_EXECUTION_ERROR})
    return result


def _result_points(data: JobBosonicProgramData) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for point_index, point in enumerate(data.points):
        counts = Counter(tuple(int(value) for value in row) for row in point.outcomes)
        points.append(
            {
                "point_index": point_index,
                "sweep_value": point.sweep_value,
                "measurement_kinds": list(point.measurement_kinds),
                "measurement_op_indices": list(point.measurement_op_indices),
                "n_outcomes": len(point.outcomes),
                "outcome_counts": [
                    {"outcomes": list(outcome), "count": count}
                    for outcome, count in sorted(counts.items())
                ],
            }
        )
    return points


def _execution_artifact_payload(
    ctx: ActionContext,
    *,
    request: BosonicProgramRequest | BosonicProgramSweepRequest,
    result: JobResult,
    tool: str,
    segment_evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    request_payload: dict[str, Any] = {
        "schema_version": request.schema_version,
        "shots": request.shots,
        "ops": _canonical_ops(request),
        "program_digest": program_digest(list(request.ops)),
        "evidence_tag": request.evidence_tag,
    }
    if isinstance(request, BosonicProgramSweepRequest):
        request_payload["sweep_coordinates"] = sweep_coordinates_commitment(
            sweep_op_index=request.sweep_op_index,
            sweep_field=request.sweep_field,
            sweep_values=request.sweep_values,
        )

    result_payload: dict[str, Any] = {
        "schema_version": result.schema_version,
        "status": result.status,
        "shots": result.shots,
        "error": result.error,
        "metadata": result.metadata.model_dump(mode="json"),
    }
    if result.status == "complete":
        if not isinstance(result.data, JobBosonicProgramData):
            raise ValueError("completed bosonic result has the wrong data kind")
        result_payload.update(
            {
                "n_max": result.data.n_max,
                "program_digest": result.data.program_digest,
                "points": _result_points(result.data),
            }
        )

    return {
        "artifact_kind": BOSONIC_EXECUTION_ARTIFACT_KIND,
        "schema_version": 1,
        "job_id": result.job_id,
        "device_id": result.device_id,
        "tool": tool,
        **_evidence_identity(ctx),
        "request": request_payload,
        "result": result_payload,
        "segment_evidence": segment_evidence,
    }


def _persist_execution_artifact(
    ctx: ActionContext, payload: dict[str, Any]
) -> dict[str, Any] | None:
    log_dir_value = getattr(ctx.state, "log_dir", None)
    if log_dir_value is None:
        return None
    log_dir = Path(log_dir_value)
    artifact_dir = log_dir / "bosonic_evidence"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_bytes = canonical_json_bytes(payload)
    digest = hashlib.sha256(artifact_bytes).hexdigest()
    target = artifact_dir / f"{digest}.json"
    if target.exists():
        if target.read_bytes() != artifact_bytes:
            raise ValueError("bosonic evidence artifact digest collision")
    else:
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                "wb",
                dir=artifact_dir,
                prefix=f".{digest}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                handle.write(artifact_bytes)
                temporary = Path(handle.name)
            temporary.replace(target)
            target.chmod(0o644)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if target.stat().st_mode & 0o777 != 0o644:
        target.chmod(0o644)
    return {
        "artifact_kind": BOSONIC_EXECUTION_ARTIFACT_KIND,
        "schema_version": 1,
        "media_type": BOSONIC_EXECUTION_ARTIFACT_MEDIA_TYPE,
        "reference": target.relative_to(log_dir).as_posix(),
        "sha256": digest,
        "size_bytes": len(artifact_bytes),
    }


def _log_result(
    ctx: ActionContext,
    *,
    request: BosonicProgramRequest | BosonicProgramSweepRequest,
    result: JobResult,
    tool: str,
    segment_evidence: dict[str, Any] | None,
    sweep_op_index: int | None = None,
    sweep_field: str | None = None,
) -> None:
    event: dict[str, Any] = {
        "surface": ctx.surface,
        "action": _RESULT_ACTION,
        "tool": tool,
        "job_id": result.job_id,
        "status": result.status,
        "shots": result.shots,
        **_evidence_identity(ctx),
    }
    artifact = _persist_execution_artifact(
        ctx,
        _execution_artifact_payload(
            ctx,
            request=request,
            result=result,
            tool=tool,
            segment_evidence=segment_evidence,
        ),
    )
    if artifact is not None:
        event["result_artifact"] = artifact
    if result.status == "complete":
        event["program_digest"] = result.data.program_digest
        if sweep_op_index is not None and sweep_field is not None:
            coords = result.metadata.sweep_coords or {}
            values = coords.get("sweep_values")
            if not isinstance(values, list):
                raise ValueError("completed bosonic sweep result omitted sweep_values")
            event["sweep_coordinates"] = sweep_coordinates_commitment(
                sweep_op_index=sweep_op_index,
                sweep_field=sweep_field,
                sweep_values=values,
            )
    ctx.state.log(event)


def _log_execution_record_failure(
    ctx: ActionContext,
    *,
    job_id: str,
    tool: str,
    exc: Exception,
) -> None:
    """Best-effort bounded marker for qsim execution/evidence infrastructure failure."""
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": _EVIDENCE_FAILURE_ACTION,
            "tool": tool,
            "job_id": job_id,
            "status": "failed",
            "failure_class": "infrastructure",
            "reason_code": _EVIDENCE_FAILURE_ACTION,
            "error_type": type(exc).__name__[:128],
            **_evidence_identity(ctx),
        }
    )


def submit_bosonic_program(
    ctx: ActionContext,
    *,
    device_id: str,
    ops: list[dict],
    shots: int,
    evidence_tag: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Submit a single bosonic program; returns a job_id."""
    _require_capability(ctx, "run bosonic program")
    request = BosonicProgramRequest.model_validate(
        {"shots": shots, "ops": ops, "evidence_tag": evidence_tag}
    )
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    _validate_execution_capacity(ctx, request)
    (
        reserved_job_id,
        reservation_token,
        segment_evidence,
        salt,
        summary,
        identity,
    ) = _prepare_submission(ctx, request, tool="run_bosonic_program")
    submission_logged = threading.Event()

    def runner(job_id: str) -> JobResult:
        submission_logged.wait()
        try:
            if reservation_token is None:
                result = ctx.state.backend.run_bosonic_program(request, job_id, salt)
            else:
                result = ctx.state.backend.run_bosonic_program(
                    request,
                    job_id,
                    salt,
                    reservation_token=reservation_token,
                )
            result = _sanitize_reserved_backend_failure(result, reservation_token=reservation_token)
            _log_result(
                ctx,
                request=request,
                result=result,
                tool="run_bosonic_program",
                segment_evidence=segment_evidence,
            )
            return result
        except Exception as exc:
            try:
                _log_execution_record_failure(
                    ctx,
                    job_id=job_id,
                    tool="run_bosonic_program",
                    exc=exc,
                )
            except Exception:
                pass
            raise

    try:
        job_id = _submit_with_public_poll_budget(
            ctx,
            runner,
            tool="run_bosonic_program",
            reserved_job_id=reserved_job_id,
        )
    except Exception:
        _discard_backend_reservation(ctx, reservation_token)
        raise
    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": "submit_bosonic_program",
                "tool": "run_bosonic_program",
                "device_id": device_id,
                "shots": shots,
                # Full validated program payload: lets the hidden verifier REPLAY a
                # cited job under the true noise model (agent-authored data only).
                "ops_wire": _canonical_ops(request),
                "job_id": job_id,
                "evidence_tag": request.evidence_tag,
                **summary,
                **identity,
            }
        )
    except Exception:
        if getattr(ctx.state.public, "budget", None) is None:
            raise
        _log_submission_failure(
            ctx,
            tool="run_bosonic_program",
            job_id=job_id,
            failure_stage="submission_log_publication",
        )
        raise RuntimeError(_SUBMISSION_FAILURE_ERROR) from None
    finally:
        submission_logged.set()
    return {"job_id": job_id, "status": "queued"}


def submit_bosonic_program_sweep(
    ctx: ActionContext,
    *,
    device_id: str,
    ops: list[dict],
    sweep_op_index: int,
    sweep_field: str,
    sweep_values: list[float],
    shots: int,
    evidence_tag: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Submit a bosonic-program sweep over one scalar op field; returns a job_id."""
    _require_capability(ctx, "run bosonic program sweep")
    request = BosonicProgramSweepRequest.model_validate(
        {
            "shots": shots,
            "ops": ops,
            "sweep_op_index": sweep_op_index,
            "sweep_field": sweep_field,
            "sweep_values": sweep_values,
            "evidence_tag": evidence_tag,
        }
    )
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    _validate_execution_capacity(ctx, request)
    coordinates = sweep_coordinates_commitment(
        sweep_op_index=request.sweep_op_index,
        sweep_field=request.sweep_field,
        sweep_values=request.sweep_values,
    )
    (
        reserved_job_id,
        reservation_token,
        segment_evidence,
        salt,
        summary,
        identity,
    ) = _prepare_submission(ctx, request, tool="run_bosonic_program_sweep")
    submission_logged = threading.Event()

    def runner(job_id: str) -> JobResult:
        submission_logged.wait()
        try:
            if reservation_token is None:
                result = ctx.state.backend.run_bosonic_program_sweep(request, job_id, salt)
            else:
                result = ctx.state.backend.run_bosonic_program_sweep(
                    request,
                    job_id,
                    salt,
                    reservation_token=reservation_token,
                )
            result = _sanitize_reserved_backend_failure(result, reservation_token=reservation_token)
            _log_result(
                ctx,
                request=request,
                result=result,
                tool="run_bosonic_program_sweep",
                segment_evidence=segment_evidence,
                sweep_op_index=request.sweep_op_index,
                sweep_field=request.sweep_field,
            )
            return result
        except Exception as exc:
            try:
                _log_execution_record_failure(
                    ctx,
                    job_id=job_id,
                    tool="run_bosonic_program_sweep",
                    exc=exc,
                )
            except Exception:
                pass
            raise

    try:
        job_id = _submit_with_public_poll_budget(
            ctx,
            runner,
            tool="run_bosonic_program_sweep",
            reserved_job_id=reserved_job_id,
        )
    except Exception:
        _discard_backend_reservation(ctx, reservation_token)
        raise
    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": "submit_bosonic_program_sweep",
                "tool": "run_bosonic_program_sweep",
                "device_id": device_id,
                "shots": shots,
                # Full validated program payload: lets the hidden verifier replay
                # cited jobs and audit tomography phase-space coverage.
                "ops_wire": _canonical_ops(request),
                "sweep_op_index": sweep_op_index,
                "sweep_field": sweep_field,
                "sweep_values": coordinates["sweep_values"],
                "n_sweep_points": len(request.sweep_values),
                "sweep_coordinates": coordinates,
                "job_id": job_id,
                "evidence_tag": request.evidence_tag,
                **summary,
                **identity,
            }
        )
    except Exception:
        if getattr(ctx.state.public, "budget", None) is None:
            raise
        _log_submission_failure(
            ctx,
            tool="run_bosonic_program_sweep",
            job_id=job_id,
            failure_stage="submission_log_publication",
        )
        raise RuntimeError(_SUBMISSION_FAILURE_ERROR) from None
    finally:
        submission_logged.set()
    return {"job_id": job_id, "status": "queued"}


__all__ = [
    "submit_bosonic_program",
    "submit_bosonic_program_sweep",
]
