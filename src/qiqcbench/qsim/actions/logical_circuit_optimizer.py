"""Action seam for the ``logical_circuit_optimizer`` qtype.

One synchronous calculator (no async job, no shots): ``get_optimization_instance``
serves the opaque Clifford+T input circuit + objective. It fails closed on the
``clifford_t_optimization`` capability and logs a ``tool`` event so the verifier sees
genuine tool evidence.
"""

from __future__ import annotations

from typing import Any

from qiqcbench.qsim.actions.base import ActionContext

_CAPABILITY = "clifford_t_optimization"


def _require(ctx: ActionContext, device_id: str) -> None:
    if _CAPABILITY not in ctx.state.active_capabilities:
        raise ValueError(
            f"{_CAPABILITY} capability is not active for this run; cannot serve the instance"
        )
    if device_id != ctx.state.public.device_id:
        raise ValueError(f"Unknown device {device_id!r}")


def get_optimization_instance(ctx: ActionContext, *, device_id: str) -> dict[str, Any]:
    """Return the opaque input circuit, gate set, limits, and objective."""
    _require(ctx, device_id)
    payload = ctx.state.backend.optimization_instance()
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "get_optimization_instance",
            "tool": "get_optimization_instance",
            "device_id": device_id,
            "n_qubits": payload.get("n_qubits"),
            "initial_t_count": payload.get("initial_t_count"),
        }
    )
    return payload
