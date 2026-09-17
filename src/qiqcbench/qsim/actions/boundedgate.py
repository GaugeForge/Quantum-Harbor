"""Action seam for the blackbox bounded-gate qtype.

Routes ``run_basis_shots`` into the active backend and records the evidence
the verifier reads back: each block's exact requested input ``x``, the
per-setting basis/shots design, a digest of the returned raw bitstrings, and
the cumulative qsim-owned shot/job budget. The budget logged here is
authoritative, not the agent's self-report.

Admission is atomic and fail-closed at this seam: the backend reserves the
request's full shot cost before the job is enqueued, so an over-budget or
malformed request raises immediately and consumes nothing.
"""

from __future__ import annotations

import hashlib
from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.actions.common import release_job_admission, reserve_job_admission
from qiqcbench.qsim.actions.sealed_holdout import transaction_guard
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.jobs import new_job_id
from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.wire import (
    BasisShotRequest,
    JobBasisShotData,
)

CAPABILITY = "basis_shot_sampling"
SUBMISSION_FAILURE_ACTION = "submit_basis_shots_failure"
SUBMISSION_FAILURE_STAGES = frozenset(
    {"poll_budget_registration", "submission_log_publication", "job_enqueue"}
)
RESULT_FAILURE_ACTION = "basis_shots_result_failure"
RESULT_FAILURE_STAGE = "result_log_publication"


def _best_effort_log_submission_failure(
    ctx: ActionContext,
    *,
    device_id: str,
    job_id: str,
    admission_sequence: int,
    failure_stage: str,
) -> None:
    """Record a sanitized marker after admission-side infrastructure failure."""
    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": SUBMISSION_FAILURE_ACTION,
                "tool": "run_basis_shots",
                "device_id": device_id,
                "job_id": job_id,
                "admission_sequence": admission_sequence,
                "failure_kind": "qsim_internal",
                "failure_stage": failure_stage,
            }
        )
    except Exception:
        return


def _best_effort_log_result_failure(
    ctx: ActionContext,
    *,
    device_id: str,
    job_id: str,
    admission_sequence: int,
) -> None:
    """Record a sanitized marker when terminal result evidence cannot be published."""
    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": RESULT_FAILURE_ACTION,
                "tool": "run_basis_shots",
                "device_id": device_id,
                "job_id": job_id,
                "admission_sequence": admission_sequence,
                "failure_kind": "qsim_internal",
                "failure_stage": RESULT_FAILURE_STAGE,
            }
        )
    except Exception:
        return


def _setting_digest(bitstrings: list[str], random_bases: list[str] | None) -> str:
    hasher = hashlib.sha256()
    for row in bitstrings:
        hasher.update(row.encode("ascii"))
        hasher.update(b"\n")
    if random_bases is not None:
        for row in random_bases:
            hasher.update(row.encode("ascii"))
            hasher.update(b"\n")
    return hasher.hexdigest()


def submit_basis_shots(
    ctx: ActionContext, *, device_id: str, blocks: list[dict[str, Any]]
) -> dict[str, Any]:
    """Submit one batched basis-shot job through the active qsim backend."""
    if CAPABILITY not in ctx.state.active_capabilities:
        raise ValueError(f"{CAPABILITY} capability is not active for this run; cannot submit shots")
    request = BasisShotRequest.model_validate({"blocks": blocks})
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    # Hold the engine-owned RLock through admission, salt/job-id allocation,
    # submission logging, and enqueue. The synchronous seal action holds the
    # same guard through reveal logging, so evidence exposes one total order.
    with transaction_guard(ctx):
        # Reversible before irreversible: claim the queue slot
        # first, then the engine meter. The engine's shots/jobs usage and its
        # monotone admission_sequence are deliberately irreversible -- rolling
        # them back would corrupt the usage_before clock of concurrently
        # admitted jobs and the seal-order evidence -- so a queue-full refusal
        # must fire before the engine is charged, and an engine refusal
        # releases the (fully reversible) slot. The guard serializes the whole
        # section, so slot order equals admission_sequence order.
        job_id = new_job_id()
        reserve_job_admission(ctx.state, job_id)
        try:
            usage_before, admission_sequence = ctx.state.backend.reserve_basis_shots_with_sequence(
                request
            )
        except Exception:
            release_job_admission(ctx.state, job_id)
            raise
        salt = ctx.state.next_salt()

        def runner(job_id: str) -> JobResult:
            result = ctx.state.backend.run_basis_shots(request, job_id, salt, usage_before)
            data = result.data
            if isinstance(data, JobBasisShotData):
                result_event = {
                    "surface": ctx.surface,
                    "action": "basis_shots_result",
                    "tool": "run_basis_shots",
                    "device_id": device_id,
                    "job_id": job_id,
                    "admission_sequence": admission_sequence,
                    "blocks": [
                        {
                            "x": block.x,
                            "settings": [
                                {
                                    "basis": s.basis,
                                    "shots": s.shots,
                                    "digest": _setting_digest(s.bitstrings, s.random_bases),
                                }
                                for s in block.settings
                            ],
                        }
                        for block in data.blocks
                    ],
                    "shots_used": data.budget.shots_used,
                    "total_shot_budget": data.budget.total_shot_budget,
                    "jobs_used": data.budget.jobs_used,
                    "max_jobs": data.budget.max_jobs,
                    "elapsed_wall_clock_s": data.budget.elapsed_wall_clock_s,
                    "salt": salt,
                }
            elif result.status == "failed":
                result_event = {
                    "surface": ctx.surface,
                    "action": "basis_shots_result",
                    "tool": "run_basis_shots",
                    "device_id": device_id,
                    "job_id": job_id,
                    "admission_sequence": admission_sequence,
                    "status": "failed",
                    "error": result.error,
                }
            else:  # pragma: no cover - backend contract violation
                raise RuntimeError("qsim basis-shot backend returned an invalid terminal result")
            try:
                ctx.state.log(result_event)
            except Exception:
                _best_effort_log_result_failure(
                    ctx,
                    device_id=device_id,
                    job_id=job_id,
                    admission_sequence=admission_sequence,
                )
                raise RuntimeError("qsim basis-shot result publication failed") from None
            return result

        # Log the submission BEFORE the worker starts (pre-allocated job_id):
        # otherwise a fast worker's result event can precede the submission
        # event, breaking submit -> result ordering for verifiers. The queue
        # slot was reserved at the top of the guard, before the engine meter;
        # every later pre-submit failure rolls it back.
        try:
            ctx.state.register_job_result_poll_budget(
                job_id=job_id,
                budget_key="blackbox_boundedgate_circuit:job_result_polls",
                limit=ctx.state.public.budget.max_job_result_polls,
            )
        except Exception:
            release_job_admission(ctx.state, job_id)
            _best_effort_log_submission_failure(
                ctx,
                device_id=device_id,
                job_id=job_id,
                admission_sequence=admission_sequence,
                failure_stage="poll_budget_registration",
            )
            raise RuntimeError("qsim basis-shot poll-budget registration failed") from None
        submission_event = {
            "surface": ctx.surface,
            "action": "submit_basis_shots",
            "tool": "run_basis_shots",
            "device_id": device_id,
            "n_blocks": len(request.blocks),
            "requested_shots": sum(s.shots for block in request.blocks for s in block.settings),
            "job_id": job_id,
            "salt": salt,
            "admission_sequence": admission_sequence,
        }
        try:
            ctx.state.log(submission_event)
        except Exception:
            release_job_admission(ctx.state, job_id)
            _best_effort_log_submission_failure(
                ctx,
                device_id=device_id,
                job_id=job_id,
                admission_sequence=admission_sequence,
                failure_stage="submission_log_publication",
            )
            raise RuntimeError("qsim basis-shot submission publication failed") from None
        try:
            ctx.state.jobs.submit(runner, job_id=job_id)
        except Exception:
            _best_effort_log_submission_failure(
                ctx,
                device_id=device_id,
                job_id=job_id,
                admission_sequence=admission_sequence,
                failure_stage="job_enqueue",
            )
            raise RuntimeError("qsim basis-shot enqueue failed") from None
    return {"job_id": job_id, "status": "queued"}
