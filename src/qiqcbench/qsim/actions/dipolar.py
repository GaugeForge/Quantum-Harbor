from __future__ import annotations

from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.dipolar_spin_ensemble.wire import DipolarSequenceRequest

_CAPABILITY = "toggling_frame_control"


def _normalize_op(op: dict[str, Any]) -> dict[str, Any]:
    """Accept either ``op`` or ``kind`` as the op-kind key (agent ergonomics)."""
    if not isinstance(op, dict):
        return op
    out = dict(op)
    if "kind" not in out and "op" in out:
        out["kind"] = out.pop("op")
    return out


def _require_capability(ctx: ActionContext, action: str) -> None:
    if _CAPABILITY not in ctx.state.active_capabilities:
        raise ValueError(f"{_CAPABILITY} capability is not active for this run; cannot {action}")


def submit_pulse_train(
    ctx: ActionContext,
    *,
    device_id: str,
    sequence: list[dict[str, Any]],
    init_axis: str = "+x",
    n_cycles: int = 1,
    measure_axes: list[str] | None = None,
    shots: int = 2048,
    inject_rotation_error: bool = False,
) -> dict[str, Any]:
    """Submit a periodic global pulse sequence and read out stroboscopically."""
    _require_capability(ctx, "submit pulse train")
    request = DipolarSequenceRequest.model_validate(
        {
            "shots": shots,
            "init_axis": init_axis,
            "sequence": [_normalize_op(op) for op in sequence],
            "n_cycles": n_cycles,
            "measure_axes": measure_axes if measure_axes is not None else ["x"],
            "inject_rotation_error": inject_rotation_error,
        }
    )
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        return ctx.state.backend.run_pulse_train(request, job_id, salt)

    job_id = ctx.state.jobs.submit(runner)
    n_pulses = sum(1 for op in request.sequence if op.kind == "pulse")
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "submit_pulse_train",
            "tool": "run_pulse_train",
            "device_id": device_id,
            "shots": shots,
            "init_axis": init_axis,
            "n_cycles": n_cycles,
            "measure_axes": list(request.measure_axes),
            "n_windows": sum(1 for op in request.sequence if op.kind == "free"),
            "n_pulses": n_pulses,
            "inject_rotation_error": inject_rotation_error,
            "job_id": job_id,
        }
    )
    return {"job_id": job_id, "status": "queued"}


__all__ = ["submit_pulse_train"]
