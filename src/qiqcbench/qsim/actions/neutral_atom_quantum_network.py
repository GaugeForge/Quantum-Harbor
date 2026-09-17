"""Action seam for the ``neutral_atom_quantum_network`` qtype.

Routes ``run_distributed_estimation`` (async, budgeted, over the photonic link) into the active
backend, and records the per-call evidence the verifier reads back: the estimation shape
(estimator / channel / precision / register) and the cumulative qsim-owned **communication
budget**. Capability-gated and fail-closed at this layer (defense in depth alongside MCP
registration). The raw estimate goes to the agent through the job result; the log keeps the
communication meter so the verifier can confirm the agent actually ran the declared protocol
under the finite budget.
"""

from __future__ import annotations

from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.neutral_atom_quantum_network.wire import (
    DistributedEstimationRequest,
    JobEstimationData,
)

_CAP = "vfl_distributed_estimation"


def _require(ctx: ActionContext, device_id: str) -> None:
    if _CAP not in ctx.state.active_capabilities:
        raise ValueError(f"{_CAP} capability is not active for this run")
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")


def submit_distributed_estimation(
    ctx: ActionContext,
    *,
    device_id: str,
    estimator: str,
    precision: int,
    n_index_bits: int = 18,
) -> dict[str, Any]:
    _require(ctx, device_id)
    request = DistributedEstimationRequest.model_validate(
        {
            "estimator": estimator,
            "precision": precision,
            "n_index_bits": n_index_bits,
        }
    )
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        result = ctx.state.backend.run_distributed_estimation(request, job_id, salt)
        data = result.data
        if isinstance(data, JobEstimationData):
            ctx.state.log(
                {
                    "surface": ctx.surface,
                    "action": "distributed_estimation_result",
                    "tool": "run_distributed_estimation",
                    "device_id": device_id,
                    "job_id": job_id,
                    "estimator": data.estimator,
                    "channel": data.channel,
                    "precision": data.precision,
                    "n_index_bits": data.n_index_bits,
                    "communication_units_charged": data.communication_units_charged,
                    "budget": data.budget.model_dump(),
                }
            )
        return result

    job_id = ctx.state.jobs.submit(runner)
    ctx.state.log(
        {
            "action": "submit_distributed_estimation",
            "job_id": job_id,
            "surface": ctx.surface,
            "device_id": device_id,
            "estimator": estimator,
            "precision": precision,
            "n_index_bits": n_index_bits,
        }
    )
    return {"job_id": job_id, "status": "queued"}
