from __future__ import annotations

import hashlib
import json
from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.core.wire import CircuitRequest, CircuitSweepRequest, JobResult


def _circuit_digest(circuit: list[dict[str, Any]]) -> str:
    """Canonical digest of a submitted circuit (matches the task-material digest scheme:
    ``sha256(json.dumps(ops, sort_keys=True, separators=(",", ":")))``). Logged so a verifier
    can bind a completed job to a pinned/canonical circuit without trusting agent-reported
    scalars; harmless for tasks that ignore it."""
    payload = json.dumps(circuit, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _measured_qubits(circuit: list[dict[str, Any]]) -> list[int]:
    for op in reversed(circuit):
        if isinstance(op, dict) and op.get("kind") == "circuit_measure":
            return list(op.get("qubits", []))
    return []


def submit_circuit(
    ctx: ActionContext, *, device_id: str, circuit: list[dict[str, Any]], shots: int
) -> dict[str, Any]:
    """Submit a digital circuit through the active qsim backend."""
    if "digital_circuit_execution" not in ctx.state.active_capabilities:
        raise ValueError(
            "digital_circuit_execution capability is not active for this run; cannot submit circuit"
        )
    request = CircuitRequest.model_validate({"shots": shots, "circuit": circuit})
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        return ctx.state.backend.run_circuit(request, job_id, salt)

    job_id = ctx.state.jobs.submit(runner)
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "submit_circuit",
            "tool": "run_circuit",
            "device_id": device_id,
            "shots": shots,
            "n_ops": len(circuit),
            "circuit_digest": _circuit_digest(circuit),
            "measured_qubits": _measured_qubits(circuit),
            "job_id": job_id,
        }
    )
    return {"job_id": job_id, "status": "queued"}


def submit_circuit_sweep(
    ctx: ActionContext,
    *,
    device_id: str,
    template_circuit: list[dict[str, Any]],
    parameters: list[str],
    sweep: dict[str, list[float]],
    shots: int,
    mode: str,
) -> dict[str, Any]:
    """Submit a digital circuit sweep through the active qsim backend."""
    if "digital_circuit_execution" not in ctx.state.active_capabilities:
        raise ValueError(
            "digital_circuit_execution capability is not active for this run; "
            "cannot submit circuit sweep"
        )
    request = CircuitSweepRequest.model_validate(
        {
            "shots": shots,
            "template_circuit": template_circuit,
            "parameters": parameters,
            "sweep": sweep,
            "mode": mode,
        }
    )
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        return ctx.state.backend.run_circuit_sweep(request, job_id, salt)

    job_id = ctx.state.jobs.submit(runner)
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "submit_circuit_sweep",
            "tool": "run_circuit_sweep",
            "device_id": device_id,
            "shots": shots,
            "parameters": parameters,
            "n_points": sum(len(v) for v in sweep.values()),
            "job_id": job_id,
        }
    )
    return {"job_id": job_id, "status": "queued"}
