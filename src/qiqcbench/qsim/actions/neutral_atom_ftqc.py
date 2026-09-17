"""Action seam for the ``neutral_atom_ftqc_compiler`` qtype.

A single synchronous calculator tool (no async job, no shots):
``get_target_circuit`` returns the fixed modular-multiplier Clifford+T netlist +
dependency DAG for this run's instance. It logs an event (with a ``tool`` field)
so the verifier sees genuine tool evidence. The cost model itself is public (via
``get_device_spec``); there is no cost oracle.
"""

from __future__ import annotations

from typing import Any

from qiqcbench.qsim.actions.base import ActionContext

_CAPABILITY = "neutral_atom_ftqc_compilation"


def _require(ctx: ActionContext, device_id: str) -> None:
    if _CAPABILITY not in ctx.state.active_capabilities:
        raise ValueError(
            f"{_CAPABILITY} capability is not active for this run; cannot serve target"
        )
    if device_id != ctx.state.public.device_id:
        raise ValueError(f"Unknown device {device_id!r}")


def get_target_circuit(ctx: ActionContext, *, device_id: str) -> dict[str, Any]:
    """Return the public compilation-target netlist (gates + dependency DAG)."""
    _require(ctx, device_id)
    payload = ctx.state.backend.target_circuit()
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "get_target_circuit",
            "tool": "get_target_circuit",
            "device_id": device_id,
            "instance_seed": payload.get("instance_seed"),
            "t_count": payload.get("t_count"),
            "n_data_qubits": payload.get("n_data_qubits"),
        }
    )
    return payload


__all__ = ["get_target_circuit"]
