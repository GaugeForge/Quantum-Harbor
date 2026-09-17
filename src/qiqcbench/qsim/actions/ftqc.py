"""Action seam for the ``ftqc_resource_estimation`` qtype.

Synchronous calculator tools (no async job, no shots) for two task families:

* Physical (``ftqc_physical_estimation`` capability): ``get_algorithm_instance``
  + ``evaluate_factory_design`` (spacetime volume + feasibility only).
* MPS-QPE (``mps_qpe_resource_planning``): ``get_mps_qpe_instance`` +
  ``evaluate_mps_qpe_plan`` (logical Toffolis + logical high-water only).

Both oracles are minimal-output (objective + feasibility, no closed-form
internals) and each fails closed on its capability. Every call logs an event with
a ``tool`` field so the verifier sees genuine tool evidence.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.actions.common import QSIM_INTERNAL_FAILURE_KIND

_PHYSICAL_CAPABILITY = "ftqc_physical_estimation"
_MPS_QPE_CAPABILITY = "mps_qpe_resource_planning"
SYNCHRONOUS_CALL_FAILURE_ACTION = "ftqc_synchronous_call_infrastructure_failure"
_BACKEND_EXECUTION_STAGE = "backend_execution"
_EVIDENCE_LOG_PUBLICATION_STAGE = "evidence_log_publication"


def _best_effort_log_synchronous_failure(
    ctx: ActionContext,
    *,
    tool: str,
    device_id: str,
    failure_stage: str,
) -> None:
    """Record a sanitized marker for a qsim-owned synchronous-call fault."""

    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": SYNCHRONOUS_CALL_FAILURE_ACTION,
                "tool": tool,
                "device_id": device_id,
                "failure_kind": QSIM_INTERNAL_FAILURE_KIND,
                "failure_stage": failure_stage,
            }
        )
    except Exception:
        return


def _run_backend_call(
    ctx: ActionContext,
    *,
    tool: str,
    device_id: str,
    call: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Run one already-validated calculator call behind a fault marker."""

    try:
        return call()
    except Exception:
        _best_effort_log_synchronous_failure(
            ctx,
            tool=tool,
            device_id=device_id,
            failure_stage=_BACKEND_EXECUTION_STAGE,
        )
        raise RuntimeError("qsim FTQC synchronous calculation failed") from None


def _log_success(ctx: ActionContext, event: dict[str, Any]) -> None:
    """Publish successful synchronous-call evidence or mark qsim's log fault."""

    try:
        ctx.state.log(event)
    except Exception:
        _best_effort_log_synchronous_failure(
            ctx,
            tool=str(event["tool"]),
            device_id=str(event["device_id"]),
            failure_stage=_EVIDENCE_LOG_PUBLICATION_STAGE,
        )
        raise RuntimeError("qsim FTQC synchronous-call evidence publication failed") from None


def _require_physical(ctx: ActionContext, device_id: str) -> None:
    if _PHYSICAL_CAPABILITY not in ctx.state.active_capabilities:
        raise ValueError(
            f"{_PHYSICAL_CAPABILITY} capability is not active for this run; "
            "cannot evaluate factory designs"
        )
    if device_id != ctx.state.public.device_id:
        raise ValueError(f"Unknown device {device_id!r}")


def _require_mps_qpe(ctx: ActionContext, device_id: str) -> None:
    if _MPS_QPE_CAPABILITY not in ctx.state.active_capabilities:
        raise ValueError(
            f"{_MPS_QPE_CAPABILITY} capability is not active for this run; "
            "cannot evaluate MPS-QPE plans"
        )
    if device_id != ctx.state.public.device_id:
        raise ValueError(f"Unknown device {device_id!r}")


def get_algorithm_instance(ctx: ActionContext, *, device_id: str) -> dict[str, Any]:
    """Return the public algorithm summary + public CCZ2T cost-model semantics."""
    _require_physical(ctx, device_id)
    payload = _run_backend_call(
        ctx,
        tool="get_algorithm_instance",
        device_id=device_id,
        call=ctx.state.backend.algorithm_instance,
    )
    _log_success(
        ctx,
        {
            "surface": ctx.surface,
            "action": "get_algorithm_instance",
            "tool": "get_algorithm_instance",
            "device_id": device_id,
        },
    )
    return payload


def evaluate_factory_design(
    ctx: ActionContext,
    *,
    device_id: str,
    distillation_l1_d: int,
    distillation_l2_d: int,
    n_factories: int,
    data_block_d: int,
) -> dict[str, Any]:
    """Cost oracle (spacetime volume + feasibility) for a CCZ2T factory design."""
    _require_physical(ctx, device_id)
    result = _run_backend_call(
        ctx,
        tool="evaluate_factory_design",
        device_id=device_id,
        call=lambda: ctx.state.backend.evaluate_design(
            distillation_l1_d=distillation_l1_d,
            distillation_l2_d=distillation_l2_d,
            n_factories=n_factories,
            data_block_d=data_block_d,
        ),
    )
    _log_success(
        ctx,
        {
            "surface": ctx.surface,
            "action": "evaluate_factory_design",
            "tool": "evaluate_factory_design",
            "device_id": device_id,
            "distillation_l1_d": distillation_l1_d,
            "distillation_l2_d": distillation_l2_d,
            "n_factories": n_factories,
            "data_block_d": data_block_d,
            "accepted": result.get("accepted"),
            "valid": result.get("valid"),
            "spacetime_volume_qubit_seconds": result.get("spacetime_volume_qubit_seconds"),
            "evaluator_calls_used": result.get("evaluator_calls_used"),
        },
    )
    return result


def get_mps_qpe_instance(ctx: ActionContext, *, device_id: str) -> dict[str, Any]:
    """Return public case metadata and authoritative material handles."""
    _require_mps_qpe(ctx, device_id)
    payload = _run_backend_call(
        ctx,
        tool="get_mps_qpe_instance",
        device_id=device_id,
        call=ctx.state.backend.mps_qpe_instance,
    )
    _log_success(
        ctx,
        {
            "surface": ctx.surface,
            "action": "get_mps_qpe_instance",
            "tool": "get_mps_qpe_instance",
            "device_id": device_id,
        },
    )
    return payload


def evaluate_mps_qpe_plan(
    ctx: ActionContext,
    *,
    device_id: str,
    selected_block_encoding_id: str,
    candidate_methods: list[str],
    mps_descriptor_ids: list[str],
) -> dict[str, Any]:
    """Return minimal resource output for one compact plan selection."""
    _require_mps_qpe(ctx, device_id)
    result = _run_backend_call(
        ctx,
        tool="evaluate_mps_qpe_plan",
        device_id=device_id,
        call=lambda: ctx.state.backend.evaluate_mps_qpe_plan(
            selected_block_encoding_id=selected_block_encoding_id,
            candidate_methods=candidate_methods,
            mps_descriptor_ids=mps_descriptor_ids,
        ),
    )
    _log_success(
        ctx,
        {
            "surface": ctx.surface,
            "action": "evaluate_mps_qpe_plan",
            "tool": "evaluate_mps_qpe_plan",
            "device_id": device_id,
            "accepted": result.get("accepted"),
            "valid": result.get("valid"),
            "request_sha256": result.get("request_sha256"),
            "total_logical_toffoli_count": result.get("total_logical_toffoli_count"),
            "logical_qubit_high_water": result.get("logical_qubit_high_water"),
            "evaluator_calls_used": result.get("evaluator_calls_used"),
        },
    )
    return result
