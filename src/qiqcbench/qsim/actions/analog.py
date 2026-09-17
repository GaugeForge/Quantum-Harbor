"""Action seam for the blackbox analog-dynamics qtype.

Routes ``run_hamiltonian_probe_batch`` into the active backend and records the
per-batch evidence the verifier reads back: each probe's complete public request
metadata, digest, accepted/rejected status, and the cumulative evidence-owned
budget. The budget logged here is authoritative (qsim-owned), not the agent's
self-report.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.core.wire import (
    HamiltonianProbeBatchRequest,
    JobProbeOutcomeData,
    JobResult,
)

PROBE_BATCH_EVIDENCE_SCHEMA_VERSION = 1
_PROBE_JOBS_BUDGET_KEY = "blackbox_analog_dynamics:probe_jobs"


def probe_row_digest(row: dict[str, Any]) -> str:
    """Stable sha256 of a probe descriptor (state, time, observable).

    Uses the same compact canonicalization as the VQE point-value digest so the
    verifier can bind submitted probes to logged evidence byte-for-byte.
    """
    payload = {
        "initial_state": list(row["initial_state"]),
        "evolve_time_us": row["evolve_time_us"],
        "observable_pauli": row["observable_pauli"],
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _ensure_instance_commitment_logged(ctx: ActionContext) -> None:
    """Log the execution-binding instance commitment before the first probe.

    The verifier refuses to score probe evidence whose log carries no
    ``hidden_instance_commitment`` (or a mismatching one), so every surface
    that can produce probe evidence — MCP, in-process harnesses, future
    surfaces — must emit it. The MCP registrar additionally logs one at boot;
    duplicates are fine because the verifier requires every occurrence to
    match its recomputed digest.
    """
    if getattr(ctx.state, "_analog_instance_commitment_logged", False):
        return
    from qiqcbench.qsim.qtypes.blackbox_analog_dynamics.device import (
        hidden_instance_commitment,
    )

    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "hidden_instance_commitment",
            "qtype": "blackbox_analog_dynamics",
            "device_id": ctx.state.public.device_id,
            "task_id": ctx.state.task_id,
            "scheme": "sha256_omega_vector_v1",
            "commitment_sha256": hidden_instance_commitment(ctx.state.hidden, ctx.state.public),
        }
    )
    ctx.state._analog_instance_commitment_logged = True


def submit_hamiltonian_probe_batch(
    ctx: ActionContext, *, device_id: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Submit a Hamiltonian probe batch through the active qsim backend."""
    if "hamiltonian_probe" not in ctx.state.active_capabilities:
        raise ValueError(
            "hamiltonian_probe capability is not active for this run; cannot submit probe batch"
        )
    request = HamiltonianProbeBatchRequest.model_validate({"rows": rows})
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    _ensure_instance_commitment_logged(ctx)
    budget = ctx.state.public.budget
    _, probe_jobs_used = ctx.state.reserve_action_budgets(
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

    def runner(job_id: str) -> JobResult:
        result = ctx.state.backend.run_hamiltonian_probe_batch(request, job_id, salt)
        data = result.data
        if isinstance(data, JobProbeOutcomeData):
            ctx.state.log(
                {
                    "evidence_schema_version": PROBE_BATCH_EVIDENCE_SCHEMA_VERSION,
                    "surface": ctx.surface,
                    "action": "probe_batch_result",
                    "tool": "run_hamiltonian_probe_batch",
                    "device_id": device_id,
                    "job_id": job_id,
                    "rows": [
                        {
                            "digest": probe_row_digest(req),
                            "initial_state": req["initial_state"],
                            "evolve_time_us": req["evolve_time_us"],
                            "observable_pauli": req["observable_pauli"],
                            "status": out.status,
                            "accepted_evolve_time_us": out.accepted_evolve_time_us,
                            "reject_reason": out.reject_reason,
                        }
                        for req, out in zip(requested, data.rows, strict=True)
                    ],
                    "budget_used_us": data.budget_used_us,
                    "budget_remaining_us": data.budget_remaining_us,
                    "accepted_row_count": data.accepted_row_count,
                }
            )
        return result

    try:
        job_id = ctx.state.jobs.submit(runner)
    except Exception:
        ctx.state.release_action_budgets(reservations={_PROBE_JOBS_BUDGET_KEY: 1})
        raise
    # Cumulative result-poll cap (2026-09-02 audit): bind this job to the
    # qtype's public poll budget so get_job_result's generic enforcement
    # applies. The budget is a DoS-hardening cap far above legitimate polling.
    ctx.state.register_job_result_poll_budget(
        job_id=job_id,
        budget_key="blackbox_analog_dynamics:job_result_polls",
        limit=budget.max_job_result_polls,
    )
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "submit_hamiltonian_probe_batch",
            "tool": "run_hamiltonian_probe_batch",
            "device_id": device_id,
            "n_rows": len(requested),
            "job_id": job_id,
            "probe_jobs_used": probe_jobs_used,
            "max_probe_jobs": budget.max_probe_jobs,
        }
    )
    return {"job_id": job_id, "status": "queued"}
