"""Lindblad-probe actions shared by every agent-facing surface.

Evidence contract (consumed by the task verifier):

* the submission event is logged with a pre-allocated ``job_id`` and the exact
  per-row request digests BEFORE the worker starts, so a fast worker's
  ``probe_batch_result`` event can never precede its submission event;
* the versioned result event binds each digest and complete public request
  preimage to its outcome status, so the hidden verifier can reconstruct the
  scientific design without changing the raw-result wire schema;
* probe admission closes once a final answer has been persisted (task seal) —
  post-final experiments are rejected at this action layer on every surface;
* transport/evidence budgets from the public spec are reserved before
  allocation: one cumulative probe-job meter here, a cumulative poll meter
  registered per job, and per-row field-size sanity bounds so a malformed
  request cannot grow logs or echoed evidence without limit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.actions.common import release_job_admission, reserve_job_admission
from qiqcbench.qsim.core.wire import HamiltonianProbeBatchRequest, JobProbeOutcomeData, JobResult
from qiqcbench.qsim.jobs import new_job_id

__all__ = ["probe_row_digest", "submit_lindblad_probe_batch"]

PROBE_BATCH_EVIDENCE_SCHEMA_VERSION = 1
SUBMISSION_FAILURE_ACTION = "submit_lindblad_probe_batch_failure"
RESULT_FAILURE_ACTION = "lindblad_probe_batch_result_failure"
_PROBE_JOBS_BUDGET_KEY = "blackbox_lindblad_dynamics:probe_jobs"
_JOB_RESULT_POLLS_BUDGET_KEY = "blackbox_lindblad_dynamics:job_result_polls"
# Transport sanity ceiling on echoed row fields, far above the length-6
# contract but low enough that an invalid row cannot bloat evidence.
_MAX_ROW_FIELD_CHARS = 64


def _best_effort_failure_marker(
    ctx: ActionContext,
    *,
    action: str,
    device_id: str,
    job_id: str,
    failure_stage: str,
) -> None:
    """Publish a sanitized qsim-owned failure marker when possible."""
    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": action,
                "tool": "run_lindblad_probe_batch",
                "device_id": device_id,
                "job_id": job_id,
                "failure_kind": "qsim_internal",
                "failure_stage": failure_stage,
            }
        )
    except Exception:
        return


def probe_row_digest(row: dict[str, Any]) -> str:
    """Stable digest of one validated probe row (binds request to outcome)."""
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def submit_lindblad_probe_batch(
    ctx: ActionContext, *, device_id: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Submit a Lindblad probe batch through the active qsim backend."""
    if "lindblad_probe" not in ctx.state.active_capabilities:
        raise ValueError(
            "lindblad_probe capability is not active for this run; cannot submit probe batch"
        )
    request = HamiltonianProbeBatchRequest.model_validate({"rows": rows})
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    for row in request.rows:
        if (
            len(row.observable_pauli) > _MAX_ROW_FIELD_CHARS
            or len(row.initial_state) > _MAX_ROW_FIELD_CHARS
            or any(len(label) > _MAX_ROW_FIELD_CHARS for label in row.initial_state)
        ):
            raise ValueError(
                "probe row field exceeds the transport sanity bound of "
                f"{_MAX_ROW_FIELD_CHARS} characters/entries"
            )
    # Task seal: a persisted final answer closes probe admission on every
    # surface. Jobs admitted before the final still complete and count.
    log_dir = ctx.state.log_dir
    if log_dir is not None and (Path(log_dir) / "final_answer.json").exists():
        raise ValueError("final answer already submitted; probe admission is closed")
    budget = ctx.state.public.budget
    jobs_before, jobs_after = ctx.state.reserve_action_budgets(
        reservations={_PROBE_JOBS_BUDGET_KEY: (1, budget.max_probe_jobs)}
    )[_PROBE_JOBS_BUDGET_KEY]
    salt = ctx.state.next_salt()
    requested = [
        {
            "initial_state": list(r.initial_state),
            "evolve_time_us": r.evolve_time_us,
            "observable_pauli": r.observable_pauli,
        }
        for r in request.rows
    ]
    row_digests = [probe_row_digest(req) for req in requested]

    def runner(job_id: str) -> JobResult:
        result = ctx.state.backend.run_lindblad_probe_batch(request, job_id, salt)
        data = result.data
        if isinstance(data, JobProbeOutcomeData):
            result_event = {
                "evidence_schema_version": PROBE_BATCH_EVIDENCE_SCHEMA_VERSION,
                "surface": ctx.surface,
                "action": "probe_batch_result",
                "tool": "run_lindblad_probe_batch",
                "device_id": device_id,
                "job_id": job_id,
                "rows": [
                    {
                        "digest": digest,
                        "initial_state": req["initial_state"],
                        "evolve_time_us": req["evolve_time_us"],
                        "observable_pauli": req["observable_pauli"],
                        "status": out.status,
                        "accepted_evolve_time_us": out.accepted_evolve_time_us,
                        "reject_reason": out.reject_reason,
                    }
                    for digest, req, out in zip(row_digests, requested, data.rows, strict=True)
                ],
                "budget_used_us": data.budget_used_us,
                "budget_remaining_us": data.budget_remaining_us,
                "accepted_row_count": data.accepted_row_count,
            }
            try:
                ctx.state.log(result_event)
            except Exception:
                _best_effort_failure_marker(
                    ctx,
                    action=RESULT_FAILURE_ACTION,
                    device_id=device_id,
                    job_id=job_id,
                    failure_stage="result_log_publication",
                )
                raise RuntimeError("qsim Lindblad result publication failed") from None
        return result

    # Pre-allocate the job ID and log the submission BEFORE the worker starts,
    # so evidence keeps submit -> result ordering even for a fast worker.
    # Reserve the queue slot first (one atomic admission lifecycle): a
    # queue-full refusal fires before anything is logged and rolls the
    # probe-jobs reservation back, so a rejected submit consumes nothing.
    job_id = new_job_id()
    try:
        reserve_job_admission(ctx.state, job_id)
    except Exception:
        ctx.state.release_action_budgets(reservations={_PROBE_JOBS_BUDGET_KEY: 1})
        raise
    try:
        ctx.state.register_job_result_poll_budget(
            job_id=job_id,
            budget_key=_JOB_RESULT_POLLS_BUDGET_KEY,
            limit=budget.max_job_result_polls,
        )
    except Exception:
        release_job_admission(ctx.state, job_id)
        _best_effort_failure_marker(
            ctx,
            action=SUBMISSION_FAILURE_ACTION,
            device_id=device_id,
            job_id=job_id,
            failure_stage="poll_budget_registration",
        )
        raise RuntimeError("qsim Lindblad poll-budget registration failed") from None
    submission_event = {
        "surface": ctx.surface,
        "action": "submit_lindblad_probe_batch",
        "tool": "run_lindblad_probe_batch",
        "device_id": device_id,
        "n_rows": len(requested),
        "row_digests": row_digests,
        "job_id": job_id,
        "probe_jobs_used": jobs_after,
        "max_probe_jobs": budget.max_probe_jobs,
    }
    try:
        ctx.state.log(submission_event)
    except Exception:
        release_job_admission(ctx.state, job_id)
        _best_effort_failure_marker(
            ctx,
            action=SUBMISSION_FAILURE_ACTION,
            device_id=device_id,
            job_id=job_id,
            failure_stage="submission_log_publication",
        )
        raise RuntimeError("qsim Lindblad submission publication failed") from None
    try:
        ctx.state.jobs.submit(runner, job_id=job_id)
    except Exception:
        _best_effort_failure_marker(
            ctx,
            action=SUBMISSION_FAILURE_ACTION,
            device_id=device_id,
            job_id=job_id,
            failure_stage="job_enqueue",
        )
        raise RuntimeError("qsim Lindblad enqueue failed") from None
    return {"job_id": job_id, "status": "queued"}
