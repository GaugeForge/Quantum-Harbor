"""Action seam for the ``surface_code_lattice_surgery`` qtype.

Routes the three experiment tools into the active backend and records the per-call
evidence the verifier reads back: the experiment shape and the cumulative qsim-owned
shot + shot-rounds budgets. Capability-gated and fail-closed at this layer (defense in
depth alongside MCP registration). Raw detection events go to the agent through the job
result; the log keeps the budget meters, not the bulky bit arrays.
"""

from __future__ import annotations

import threading
from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery.wire import (
    JobLatticeSurgeryData,
    LsCnotRequest,
    LsMemoryRequest,
    LsMergedRequest,
)

_CAP = "lattice_surgery_calibration"


def _require(ctx: ActionContext, device_id: str) -> None:
    if _CAP not in ctx.state.active_capabilities:
        raise ValueError(f"{_CAP} capability is not active for this run")
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")


def _submit(
    ctx: ActionContext, *, device_id: str, tool: str, request: Any, backend_method: str
) -> dict[str, Any]:
    salt = ctx.state.next_salt()
    del salt  # deterministic engine; salt reserved by the shared action contract
    submission_logged = threading.Event()

    def runner(job_id: str) -> JobResult:
        # A fast worker can otherwise append its result before the caller has appended
        # the matching submission.  Verifiers intentionally require the causal order
        # submit -> result -> completed poll -> final answer.
        submission_logged.wait()
        result = getattr(ctx.state.backend, backend_method)(request, job_id, 0)
        data = result.data
        if isinstance(data, JobLatticeSurgeryData):
            ctx.state.log(
                {
                    "surface": ctx.surface,
                    "action": f"{tool}_result",
                    "tool": tool,
                    "device_id": device_id,
                    "job_id": job_id,
                    "experiment": data.experiment,
                    "layout": data.layout,
                    "rounds": data.rounds,
                    "shots": data.shots,
                    "n_detectors": data.n_detectors,
                    "budget": data.budget.model_dump(),
                }
            )
        return result

    job_id = ctx.state.jobs.submit(runner)
    try:
        ctx.state.log(
            {
                "action": f"submit_{tool}",
                "job_id": job_id,
                "surface": ctx.surface,
                "device_id": device_id,
                "request": request.model_dump(),
            }
        )
    finally:
        submission_logged.set()
    return {"job_id": job_id, "status": "queued"}


def submit_memory_experiment(
    ctx: ActionContext, *, device_id: str, layout: str, rounds: int, shots: int
) -> dict[str, Any]:
    _require(ctx, device_id)
    request = LsMemoryRequest.model_validate({"layout": layout, "rounds": rounds, "shots": shots})
    return _submit(
        ctx,
        device_id=device_id,
        tool="run_memory_experiment",
        request=request,
        backend_method="run_memory_experiment",
    )


def submit_merged_memory(
    ctx: ActionContext,
    *,
    device_id: str,
    window: str,
    rounds: int,
    shots: int,
    include_transitions: bool,
) -> dict[str, Any]:
    _require(ctx, device_id)
    request = LsMergedRequest.model_validate(
        {
            "window": window,
            "rounds": rounds,
            "shots": shots,
            "include_transitions": include_transitions,
        }
    )
    return _submit(
        ctx,
        device_id=device_id,
        tool="run_merged_memory",
        request=request,
        backend_method="run_merged_memory",
    )


def submit_lattice_surgery_cnot(
    ctx: ActionContext, *, device_id: str, config: str, logical_input: str, shots: int
) -> dict[str, Any]:
    _require(ctx, device_id)
    request = LsCnotRequest.model_validate(
        {"config": config, "logical_input": logical_input, "shots": shots}
    )
    return _submit(
        ctx,
        device_id=device_id,
        tool="run_lattice_surgery_cnot",
        request=request,
        backend_method="run_lattice_surgery_cnot",
    )
