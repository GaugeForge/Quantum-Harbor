"""Surface-neutral submit seam for tunable-coupler CZ flux pulses.

Queues a flux-pulse experiment through the backend, gated on the
``netzero_flux_control`` capability, logging structural evidence (whether a flux
pulse / coupler bias / prep+tomography were supplied) for lightweight binding.
"""

from __future__ import annotations

from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.transmon_pair_tunable_coupler.wire import FluxPulseRequest

__all__ = ["submit_flux_pulse"]

_CAPABILITY = "netzero_flux_control"


def submit_flux_pulse(
    ctx: ActionContext,
    *,
    device_id: str,
    coupler_flux: float,
    programmed_flux_q1: list[float] | None = None,
    sample_dt_ns: float = 0.4,
    prep_ops: list[dict[str, Any]] | None = None,
    post_ops: list[dict[str, Any]] | None = None,
    idle_ns: float = 0.0,
    measure_qubits: list[int] | None = None,
    shots: int = 4096,
) -> dict[str, Any]:
    if _CAPABILITY not in ctx.state.active_capabilities:
        raise ValueError(
            f"{_CAPABILITY} capability is not active for this run; cannot submit flux pulses"
        )
    request = FluxPulseRequest.model_validate(
        {
            "shots": shots,
            "coupler_flux": coupler_flux,
            "programmed_flux_q1": programmed_flux_q1 or [],
            "sample_dt_ns": sample_dt_ns,
            "prep_ops": prep_ops or [],
            "post_ops": post_ops or [],
            "idle_ns": idle_ns,
            "measure_qubits": measure_qubits or [1, 2],
        }
    )
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        return ctx.state.backend.run_flux_pulse(request, job_id, salt)

    job_id = ctx.state.jobs.submit(runner)
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "submit_flux_pulse",
            "tool": "run_flux_pulse",
            "device_id": device_id,
            "shots": shots,
            "job_id": job_id,
            "coupler_flux": coupler_flux,
            "n_flux_samples": len(request.programmed_flux_q1),
            "idle_ns": idle_ns,
            "n_prep": len(request.prep_ops),
            "n_post": len(request.post_ops),
        }
    )
    return {"job_id": job_id, "status": "queued"}
