"""Surface-neutral submit seam for chain pulse-compiler circuits.

Queues a scheduled gate program through the chain backend
(``run_chain_circuit``) without extending the pulse/circuit backend protocols
(the compilation surface is distinct, mirroring gmon). Gated on the
``chain_pulse_compilation`` capability and logs structural evidence fields the
verifier uses for lightweight evidence binding (gate counts, whether a CPhase
recalibration was supplied).
"""

from __future__ import annotations

from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.transmon_chain_pulse_compiler.wire import (
    ChainCircuitRequest,
    ChainGateOp,
)

__all__ = ["submit_chain_circuit"]

_CAPABILITY = "chain_pulse_compilation"


def _require_capability(ctx: ActionContext) -> None:
    if _CAPABILITY not in ctx.state.active_capabilities:
        raise ValueError(
            f"{_CAPABILITY} capability is not active for this run; cannot submit circuits"
        )


def _evidence_fields(circuit: list[ChainGateOp], calibration: object | None) -> dict[str, Any]:
    """Structural summary the verifier uses for lightweight evidence binding."""
    n_cphase = sum(1 for g in circuit if g.type == "cphase")
    n_virtual_rz = sum(1 for g in circuit if g.type == "rz" and g.virtual)
    n_physical_rz = sum(1 for g in circuit if g.type == "rz" and not g.virtual)
    n_1q = sum(1 for g in circuit if g.type in ("rx", "ry"))
    return {
        "n_ops": len(circuit),
        "n_cphase": n_cphase,
        "n_virtual_rz": n_virtual_rz,
        "n_physical_rz": n_physical_rz,
        "n_single_qubit_rot": n_1q,
        "recalibrated": calibration is not None,
    }


def submit_chain_circuit(
    ctx: ActionContext,
    *,
    device_id: str,
    circuit: list[dict[str, Any]],
    shots: int,
    calibration: dict[str, Any] | None = None,
    measure_qubits: list[int] | None = None,
) -> dict[str, Any]:
    """Queue one scheduled gate program through the active chain backend."""
    _require_capability(ctx)
    request = ChainCircuitRequest.model_validate(
        {
            "shots": shots,
            "circuit": circuit,
            "calibration": calibration,
            "measure_qubits": measure_qubits,
        }
    )
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        return ctx.state.backend.run_chain_circuit(request, job_id, salt)

    job_id = ctx.state.jobs.submit(runner)
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "submit_chain_circuit",
            "tool": "run_compiled_circuit",
            "device_id": device_id,
            "shots": shots,
            "job_id": job_id,
            **_evidence_fields(request.circuit, request.calibration),
        }
    )
    return {"job_id": job_id, "status": "queued"}
