from __future__ import annotations

from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.core.wire import JobResult, PulseSequenceRequest, SweepRequest


def submit_pulse_sequence(
    ctx: ActionContext, *, device_id: str, sequence: list[dict[str, Any]], shots: int
) -> dict[str, Any]:
    """Submit a pulse sequence through the active qsim backend."""
    request = PulseSequenceRequest.model_validate({"shots": shots, "sequence": sequence})
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        return ctx.state.backend.run_pulse_sequence(request, job_id, salt)

    job_id = ctx.state.jobs.submit(runner)
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "submit_pulse_sequence",
            "tool": "run_pulse_sequence",
            "device_id": device_id,
            "shots": shots,
            "n_ops": len(sequence),
            "job_id": job_id,
        }
    )
    return {"job_id": job_id, "status": "queued"}


def submit_pulse_sweep(
    ctx: ActionContext,
    *,
    device_id: str,
    template_sequence: list[dict[str, Any]],
    sweep: dict[str, list[float]],
    shots: int,
    mode: str,
) -> dict[str, Any]:
    """Submit a pulse sweep through the active qsim backend."""
    request = SweepRequest.model_validate(
        {
            "shots": shots,
            "template_sequence": template_sequence,
            "sweep": sweep,
            "mode": mode,
        }
    )
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        return ctx.state.backend.run_sweep(request, job_id, salt)

    job_id = ctx.state.jobs.submit(runner)
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "submit_pulse_sweep",
            "tool": "run_sweep",
            "device_id": device_id,
            "shots": shots,
            "sweep_keys": list(sweep.keys()),
            "n_points": sum(len(v) for v in sweep.values()),
            "job_id": job_id,
        }
    )
    return {"job_id": job_id, "status": "queued"}
