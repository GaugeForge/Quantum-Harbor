"""Action seam for the ``surface_code_memory`` qtype.

Routes ``run_memory_experiment`` into the active backend and records the per-call evidence
the verifier reads back: the experiment shape (rounds/shots) and the cumulative qsim-owned
shot budget. Capability-gated and fail-closed at this layer (defense in depth alongside MCP
registration). Raw detection events themselves go to the agent through the job result; the
log keeps the budget meter (so the verifier can bind the agent's self-reported shot total to
qsim-owned evidence) but not the bulky bit arrays.
"""

from __future__ import annotations

from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.surface_code_memory.wire import (
    JobDetectorData,
    MemoryExperimentRequest,
)

_CAP = "surface_code_memory_calibration"


def _require(ctx: ActionContext, device_id: str) -> None:
    if _CAP not in ctx.state.active_capabilities:
        raise ValueError(f"{_CAP} capability is not active for this run")
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")


def submit_memory_experiment(
    ctx: ActionContext, *, device_id: str, rounds: int, shots: int
) -> dict[str, Any]:
    _require(ctx, device_id)
    request = MemoryExperimentRequest.model_validate({"rounds": rounds, "shots": shots})
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        result = ctx.state.backend.run_memory_experiment(request, job_id, salt)
        data = result.data
        if isinstance(data, JobDetectorData):
            ctx.state.log(
                {
                    "surface": ctx.surface,
                    "action": "memory_experiment_result",
                    "tool": "run_memory_experiment",
                    "device_id": device_id,
                    "job_id": job_id,
                    "rounds": data.rounds,
                    "shots": data.shots,
                    "n_detectors": data.n_detectors,
                    "budget": data.budget.model_dump(),
                }
            )
        return result

    job_id = ctx.state.jobs.submit(runner)
    ctx.state.log(
        {
            "action": "submit_memory_experiment",
            "job_id": job_id,
            "surface": ctx.surface,
            "device_id": device_id,
            "rounds": rounds,
            "shots": shots,
        }
    )
    return {"job_id": job_id, "status": "queued"}
