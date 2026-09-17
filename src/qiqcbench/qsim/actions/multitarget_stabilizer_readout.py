"""Capability-gated action seam for Rydberg stabilizer experiments."""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.rydberg_multitarget_surface_code.capabilities.multitarget_stabilizer_readout.runtime import (
    canonical_cycle,
    canonical_cycle_digest,
)
from qiqcbench.qsim.qtypes.rydberg_multitarget_surface_code.physics import cycle_duration_us
from qiqcbench.qsim.qtypes.rydberg_multitarget_surface_code.wire import (
    Cz2CharacterizationRequest,
    JobCz2CharacterizationData,
    JobMultitargetMemoryData,
    MultitargetMemoryRequest,
    MultitargetStabilizerCycle,
)

_CAP = "multitarget_stabilizer_readout"
EVIDENCE_CONTRACT = "rydberg_multitarget_raw_measurements_v2"
INFRASTRUCTURE_FAILURE = "QIQCBENCH_RYDBERG_MULTITARGET_INFRASTRUCTURE_FAILURE"
_JOB_BUDGET_KEY = f"{EVIDENCE_CONTRACT}:experiment_jobs"
_SHOT_BUDGET_KEY = f"{EVIDENCE_CONTRACT}:shots"


def _require(ctx: ActionContext, device_id: str) -> None:
    if _CAP not in ctx.state.active_capabilities:
        raise ValueError(f"{_CAP} capability is not active for this run")
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")


def _request_digest(request: Cz2CharacterizationRequest | MultitargetMemoryRequest) -> str:
    payload = json.dumps(
        request.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reserve(ctx: ActionContext, *, shots: int) -> tuple[int, int]:
    budgets = ctx.state.public.budgets
    evidence = ctx.state.reserve_action_budgets(
        reservations={
            _JOB_BUDGET_KEY: (1, budgets.experiment_job_budget),
            _SHOT_BUDGET_KEY: (shots, budgets.shot_budget),
        }
    )
    return evidence[_JOB_BUDGET_KEY][1], evidence[_SHOT_BUDGET_KEY][1]


def _infrastructure_failure(
    ctx: ActionContext,
    *,
    job_id: str,
    device_id: str,
    tool: str,
    submission_index: int,
    request_digest: str,
    stage: str,
) -> JobResult:
    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": "rydberg_multitarget_infrastructure_failure",
                "tool": tool,
                "device_id": device_id,
                "job_id": job_id,
                "evidence_contract": EVIDENCE_CONTRACT,
                "evidence_schema_version": 1,
                "submission_index": submission_index,
                "request_digest": request_digest,
                "failure_class": INFRASTRUCTURE_FAILURE,
                "failure_stage": stage,
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


def _submit(
    ctx: ActionContext,
    *,
    device_id: str,
    tool: str,
    submit_action: str,
    result_action: str,
    request: Cz2CharacterizationRequest | MultitargetMemoryRequest,
    result_type: type[JobCz2CharacterizationData] | type[JobMultitargetMemoryData],
    backend_method: str,
) -> dict[str, Any]:
    submission_index, accepted_shots_used = _reserve(ctx, shots=request.shots)
    digest = _request_digest(request)
    canonical_request = request.model_dump(mode="json")
    # Atomic admission already assigns a unique, logged one-based stream ID.
    # Reusing it preserves serial streams while removing scheduler-dependent binding.
    salt = submission_index
    submission_logged = threading.Event()

    def runner(job_id: str) -> JobResult:
        submission_logged.wait()
        try:
            result = getattr(ctx.state.backend, backend_method)(
                request,
                job_id,
                salt,
                accepted_shots_used=accepted_shots_used,
            )
        except Exception:
            return _infrastructure_failure(
                ctx,
                job_id=job_id,
                device_id=device_id,
                tool=tool,
                submission_index=submission_index,
                request_digest=digest,
                stage="backend_execution",
            )
        data = result.data
        if result.status != "complete" or not isinstance(data, result_type):
            return _infrastructure_failure(
                ctx,
                job_id=job_id,
                device_id=device_id,
                tool=tool,
                submission_index=submission_index,
                request_digest=digest,
                stage="backend_result",
            )
        try:
            if isinstance(data, JobCz2CharacterizationData):
                summary = {
                    **data.model_dump(exclude={"readout_bits_b64"}),
                    "raw_outcomes_sha256": hashlib.sha256(
                        data.readout_bits_b64.encode("ascii")
                    ).hexdigest(),
                }
            else:
                summary = {
                    **data.model_dump(
                        exclude={
                            "syndrome_measurement_bits_b64",
                            "final_data_measurement_bits_b64",
                        }
                    ),
                    "raw_records_sha256": hashlib.sha256(
                        (
                            data.syndrome_measurement_bits_b64
                            + data.final_data_measurement_bits_b64
                        ).encode("ascii")
                    ).hexdigest(),
                }
            ctx.state.log(
                {
                    "surface": ctx.surface,
                    "action": result_action,
                    "tool": tool,
                    "device_id": device_id,
                    "job_id": job_id,
                    "status": "complete",
                    "evidence_contract": EVIDENCE_CONTRACT,
                    "evidence_schema_version": 1,
                    "submission_index": submission_index,
                    "request_digest": digest,
                    "request": canonical_request,
                    "accepted_shots_used": accepted_shots_used,
                    "data": summary,
                }
            )
        except Exception:
            return _infrastructure_failure(
                ctx,
                job_id=job_id,
                device_id=device_id,
                tool=tool,
                submission_index=submission_index,
                request_digest=digest,
                stage="evidence_recording",
            )
        return result

    job_id = ctx.state.jobs.submit(runner)
    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": submit_action,
                "tool": tool,
                "device_id": device_id,
                "job_id": job_id,
                "evidence_contract": EVIDENCE_CONTRACT,
                "evidence_schema_version": 1,
                "submission_index": submission_index,
                "request_digest": digest,
                "request": canonical_request,
                "accepted_shots_used": accepted_shots_used,
                "experiment_jobs_used": submission_index,
            }
        )
    finally:
        submission_logged.set()
    return {
        "job_id": job_id,
        "status": "queued",
        "submission_index": submission_index,
        "accepted_shots_used": accepted_shots_used,
    }


def submit_cz2_echo_characterization(
    ctx: ActionContext,
    *,
    device_id: str,
    echo_protocol: str,
    repetitions: int,
    duration_scale: float,
    target_phase_compensation_rad: float,
    shots: int,
) -> dict[str, Any]:
    _require(ctx, device_id)
    request = Cz2CharacterizationRequest.model_validate(
        {
            "echo_protocol": echo_protocol,
            "repetitions": repetitions,
            "duration_scale": duration_scale,
            "target_phase_compensation_rad": target_phase_compensation_rad,
            "shots": shots,
        }
    )
    return _submit(
        ctx,
        device_id=device_id,
        tool="run_cz2_echo_characterization",
        submit_action="submit_cz2_echo_characterization",
        result_action="cz2_echo_characterization_result",
        request=request,
        result_type=JobCz2CharacterizationData,
        backend_method="run_cz2_echo_characterization",
    )


def validate_multitarget_stabilizer_cycle(
    ctx: ActionContext, *, device_id: str, cycle: dict[str, Any]
) -> dict[str, Any]:
    _require(ctx, device_id)
    parsed = MultitargetStabilizerCycle.model_validate(cycle)
    canonical = canonical_cycle(parsed, ctx.state.public)
    duration = cycle_duration_us(ctx.state.public, 1.0)
    return {
        "valid": True,
        "cycle_digest": canonical_cycle_digest(parsed, ctx.state.public),
        "canonical_cycle": canonical,
        "nominal_cycle_duration_us_at_duration_scale_1": duration,
    }


def submit_multitarget_stabilizer_memory(
    ctx: ActionContext,
    *,
    device_id: str,
    decoded_basis: str,
    cycle: dict[str, Any],
    duration_scale: float,
    target_phase_compensation_rad: float,
    rounds: int,
    shots: int,
) -> dict[str, Any]:
    _require(ctx, device_id)
    request = MultitargetMemoryRequest.model_validate(
        {
            "decoded_basis": decoded_basis,
            "cycle": cycle,
            "duration_scale": duration_scale,
            "target_phase_compensation_rad": target_phase_compensation_rad,
            "rounds": rounds,
            "shots": shots,
        }
    )
    # Semantic validation is part of admission. An invalid partition must not
    # consume a job/shot reservation and later masquerade as a backend failure.
    canonical_cycle(request.cycle, ctx.state.public)
    digest = canonical_cycle_digest(request.cycle, ctx.state.public)
    submitted = _submit(
        ctx,
        device_id=device_id,
        tool="run_multitarget_stabilizer_memory",
        submit_action="submit_multitarget_stabilizer_memory",
        result_action="multitarget_stabilizer_memory_result",
        request=request,
        result_type=JobMultitargetMemoryData,
        backend_method="run_multitarget_stabilizer_memory",
    )
    return {**submitted, "cycle_digest": digest}


__all__ = [
    "EVIDENCE_CONTRACT",
    "INFRASTRUCTURE_FAILURE",
    "submit_cz2_echo_characterization",
    "submit_multitarget_stabilizer_memory",
    "validate_multitarget_stabilizer_cycle",
]
