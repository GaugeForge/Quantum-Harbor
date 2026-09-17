from __future__ import annotations

from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.core.wire import (
    JobResult,
    MultilevelPulseRequest,
    MultilevelPulseSweepRequest,
)

_CAPABILITY = "multilevel_pulse_control"


def _require_capability(ctx: ActionContext, action: str) -> None:
    if _CAPABILITY not in ctx.state.active_capabilities:
        raise ValueError(f"{_CAPABILITY} capability is not active for this run; cannot {action}")


def submit_pulse_sequence(
    ctx: ActionContext,
    *,
    device_id: str,
    segments: list[dict[str, Any]],
    shots: int,
) -> dict[str, Any]:
    """Submit a concatenated drive sequence + level-resolved readout."""
    _require_capability(ctx, "submit pulse sequence")
    request = MultilevelPulseRequest.model_validate({"shots": shots, "segments": segments})
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
            "n_segments": len(segments),
            "segments": [dict(s) for s in segments],
            "job_id": job_id,
        }
    )
    return {"job_id": job_id, "status": "queued"}


def submit_pulse_sweep(
    ctx: ActionContext,
    *,
    device_id: str,
    template_segments: list[dict[str, Any]],
    sweep: dict[str, list[float]],
    shots: int,
    mode: str = "product",
) -> dict[str, Any]:
    """Submit a sweep over named scalar placeholders in analytic segments."""
    _require_capability(ctx, "submit pulse sweep")
    request = MultilevelPulseSweepRequest.model_validate(
        {
            "shots": shots,
            "template_segments": template_segments,
            "sweep": sweep,
            "mode": mode,
        }
    )
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        return ctx.state.backend.run_pulse_sweep(request, job_id, salt)

    job_id = ctx.state.jobs.submit(runner)
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "submit_pulse_sweep",
            "tool": "run_pulse_sweep",
            "device_id": device_id,
            "shots": shots,
            "sweep": {k: list(v) for k, v in sweep.items()},
            "mode": mode,
            "job_id": job_id,
        }
    )
    return {"job_id": job_id, "status": "queued"}


__all__ = ["submit_pulse_sequence", "submit_pulse_sweep"]
