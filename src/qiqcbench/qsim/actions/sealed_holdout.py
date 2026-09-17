"""Surface-neutral measure -> seal -> reveal action for sealed holdouts.

The action and the bounded-gate measurement submission action share the
backend's re-entrant transaction guard. This makes the submission receipt and
the irreversible reveal receipt totally ordered in qsim evidence while the
engine independently enforces the same ordering at admission time.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

from qiqcbench.qsim.actions.base import ActionContext

CAPABILITY = "sealed_holdout_reveal"
REVEAL_FAILURE_ACTION = "sealed_holdout_reveal_failure"


def _best_effort_log_reveal_failure(
    ctx: ActionContext, *, task_id: str, device_id: str, failure_stage: str
) -> None:
    """Record a sanitized marker when an accepted reveal fails inside qsim."""
    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": REVEAL_FAILURE_ACTION,
                "tool": "lock_measurements_and_reveal_challenge",
                "task_id": task_id,
                "device_id": device_id,
                "failure_kind": "qsim_internal",
                "failure_stage": failure_stage,
            }
        )
    except Exception:
        return


def transaction_guard(ctx: ActionContext) -> AbstractContextManager:
    """Return the qtype-owned guard shared by measurement admission and seal."""
    guard = getattr(ctx.state.backend, "transaction_guard", None)
    if guard is None:
        raise RuntimeError("active backend does not provide a sealed-holdout transaction guard")
    return guard


def lock_measurements_and_reveal_challenge(ctx: ActionContext, *, device_id: str) -> dict[str, Any]:
    """Irreversibly seal measurement admission and return the committed targets."""
    if CAPABILITY not in ctx.state.active_capabilities:
        raise ValueError(f"{CAPABILITY} capability is not active for this run")
    if device_id != ctx.state.public.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    task_id = ctx.state.task_id
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("sealed holdout reveal requires an active task ID")

    with transaction_guard(ctx):
        reveal_limit = ctx.state.public.budget.max_sealed_holdout_reveals
        _, reveals_used = ctx.state.reserve_action_budgets(
            reservations={"sealed_holdout_reveals": (1, reveal_limit)}
        )["sealed_holdout_reveals"]
        try:
            reveal = ctx.state.backend.lock_measurements_and_reveal_challenge()
        except Exception:
            _best_effort_log_reveal_failure(
                ctx,
                task_id=task_id,
                device_id=device_id,
                failure_stage="reveal_execution",
            )
            raise RuntimeError("qsim sealed-holdout reveal failed") from None
        receipt = {
            "task_id": task_id,
            "device_id": device_id,
            **reveal.model_dump(mode="json"),
        }
        try:
            ctx.state.log(
                {
                    "surface": ctx.surface,
                    "action": "sealed_holdout_reveal",
                    "tool": "lock_measurements_and_reveal_challenge",
                    **receipt,
                    "sealed_holdout_reveals_used": reveals_used,
                    "max_sealed_holdout_reveals": reveal_limit,
                }
            )
        except Exception:
            _best_effort_log_reveal_failure(
                ctx,
                task_id=task_id,
                device_id=device_id,
                failure_stage="reveal_log_publication",
            )
            raise RuntimeError("qsim sealed-holdout reveal publication failed") from None
    return receipt


__all__ = ["CAPABILITY", "lock_measurements_and_reveal_challenge", "transaction_guard"]
