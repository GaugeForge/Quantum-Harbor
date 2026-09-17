from __future__ import annotations

from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.core.wire import (
    JobResult,
    NvSensingProbeRequest,
    NvSensingSweepRequest,
)

_CAPABILITY = "network_field_sensing"


def _require_capability(ctx: ActionContext, action: str) -> None:
    if _CAPABILITY not in ctx.state.active_capabilities:
        raise ValueError(f"{_CAPABILITY} capability is not active for this run; cannot {action}")


def submit_sensing_probe(
    ctx: ActionContext,
    *,
    device_id: str,
    probe_type: str,
    support: list[int],
    interrogation_time_s: list[float],
    echo_sign: list[int],
    analysis_phase_rad: float = 0.0,
    shots: int = 1024,
) -> dict[str, Any]:
    """Submit one NV sensor-network probe (separable / ghz / link_probe)."""
    _require_capability(ctx, "submit sensing probe")
    request = NvSensingProbeRequest.model_validate(
        {
            "probe_type": probe_type,
            "support": support,
            "interrogation_time_s": interrogation_time_s,
            "echo_sign": echo_sign,
            "analysis_phase_rad": analysis_phase_rad,
            "shots": shots,
        }
    )
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        return ctx.state.backend.run_sensing_probe(request, job_id, salt)

    job_id = ctx.state.jobs.submit(runner)
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "submit_sensing_probe",
            "tool": "run_sensing_probe",
            "device_id": device_id,
            "job_id": job_id,
            "request": request.model_dump(mode="json"),
        }
    )
    return {"job_id": job_id, "status": "queued"}


def submit_sensing_sweep(
    ctx: ActionContext,
    *,
    device_id: str,
    probe_type: str,
    support: list[int],
    interrogation_time_grid_s: list[float],
    echo_sign: list[int],
    analysis_phase_rad: float = 0.0,
    shots: int = 512,
) -> dict[str, Any]:
    """Submit an NV sensing sweep over a grid of interrogation times."""
    _require_capability(ctx, "submit sensing sweep")
    request = NvSensingSweepRequest.model_validate(
        {
            "probe_type": probe_type,
            "support": support,
            "interrogation_time_grid_s": interrogation_time_grid_s,
            "echo_sign": echo_sign,
            "analysis_phase_rad": analysis_phase_rad,
            "shots": shots,
        }
    )
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        return ctx.state.backend.run_sensing_sweep(request, job_id, salt)

    job_id = ctx.state.jobs.submit(runner)
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "submit_sensing_sweep",
            "tool": "run_sensing_sweep",
            "device_id": device_id,
            "job_id": job_id,
            "n_points": len(interrogation_time_grid_s),
            "request": request.model_dump(mode="json"),
        }
    )
    return {"job_id": job_id, "status": "queued"}


__all__ = ["submit_sensing_probe", "submit_sensing_sweep"]
