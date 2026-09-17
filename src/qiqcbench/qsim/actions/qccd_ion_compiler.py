"""Action seam for the ``qccd_ion_compiler`` qtype.

Synchronous calculator tools (no async job, no shots), gated on the
``qccd_compilation`` capability. ``_require`` fails closed at the action layer
(not only at MCP registration). Both tools log a ``tool`` event so the verifier
sees genuine tool evidence in ``experiment_log.jsonl``.
"""

from __future__ import annotations

from typing import Any

from qiqcbench.qsim.actions.base import ActionContext

_CAPABILITY = "qccd_compilation"


def _require(ctx: ActionContext, device_id: str) -> None:
    if _CAPABILITY not in ctx.state.active_capabilities:
        raise ValueError(
            f"{_CAPABILITY} capability is not active for this run; cannot compile schedules"
        )
    if device_id != ctx.state.public.device_id:
        raise ValueError(f"Unknown device {device_id!r}")


def get_compilation_instance(ctx: ActionContext, *, device_id: str) -> dict[str, Any]:
    """Return the target QAOA circuit instance + public cost model."""
    _require(ctx, device_id)
    payload = ctx.state.backend.compilation_instance()
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "get_compilation_instance",
            "tool": "get_compilation_instance",
            "device_id": device_id,
            "instance_seed": payload.get("instance_seed"),
        }
    )
    return payload


def evaluate_schedule(
    ctx: ActionContext, *, device_id: str, schedule: dict[str, Any]
) -> dict[str, Any]:
    """Deterministic cost oracle for a candidate compiled schedule."""
    _require(ctx, device_id)
    result = ctx.state.backend.evaluate(schedule)
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "evaluate_schedule",
            "tool": "evaluate_schedule",
            "device_id": device_id,
            "accepted": result.get("accepted"),
            "legal": result.get("legal"),
            "circuit_correct": result.get("circuit_correct"),
            "total_infidelity": result.get("total_infidelity"),
            "evaluator_calls_used": result.get("evaluator_calls_used"),
        }
    )
    return result
