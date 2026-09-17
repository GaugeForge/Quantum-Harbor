"""Action seam for the ``trapped_ion_state_copy_randomized_measurement`` qtype.

Routes the three async experiment tools (``run_readout_calibration``, ``run_local_pauli_batch``,
``run_copy_block_batch``) into the active backend and records the per-call evidence the verifier
reads back: the measurement shape and the cumulative qsim-owned state-copy / randomized-basis
budget. Capability-gated and fail-closed at this layer (defense in depth alongside MCP
registration). Raw samples go to the agent through the job result; the log keeps the budget
meter + tool-call record so the verifier can confirm the agent actually measured under budget.
"""

from __future__ import annotations

from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.trapped_ion_state_copy_randomized_measurement.wire import (
    CopyBlockBatchRequest,
    JobCopyBlockData,
    JobJointCopyBlockData,
    JobLocalPauliData,
    JobOrmData,
    JobReadoutCalData,
    LocalPauliBatchRequest,
    ReadoutCalibrationRequest,
)

_CAP = "nonlinear_randomized_measurement"


def _require(ctx: ActionContext, device_id: str) -> None:
    if _CAP not in ctx.state.active_capabilities:
        raise ValueError(f"{_CAP} capability is not active for this run")
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")


def _submit(
    ctx: ActionContext, *, device_id: str, tool: str, request, run_method: str, submit_extra: dict
) -> dict[str, Any]:
    _require(ctx, device_id)
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        result = getattr(ctx.state.backend, run_method)(request, job_id, salt)
        data = result.data
        if isinstance(
            data,
            (
                JobReadoutCalData,
                JobOrmData,
                JobLocalPauliData,
                JobCopyBlockData,
                JobJointCopyBlockData,
            ),
        ):
            ctx.state.log(
                {
                    "surface": ctx.surface,
                    "action": f"{tool}_result",
                    "tool": tool,
                    "device_id": device_id,
                    "job_id": job_id,
                    "kind": data.kind,
                    **_shape(data),
                    "budget": data.budget.model_dump(),
                }
            )
        return result

    job_id = ctx.state.jobs.submit(runner)
    ctx.state.log(
        {
            "action": f"submit_{tool}",
            "job_id": job_id,
            "surface": ctx.surface,
            "device_id": device_id,
            **submit_extra,
        }
    )
    return {"job_id": job_id, "status": "queued"}


def _shape(data) -> dict[str, Any]:
    if isinstance(data, (JobCopyBlockData, JobJointCopyBlockData)):
        first = next(iter(data.records.values()), {})
        column = first.get("den") or first.get("num") or []
        return {
            "observable": data.observable,
            "measurement_family": data.measurement_family,
            "block_size": data.block_size,
            # qsim-owned block count: the verifier derives the uncertainty implied by the
            # agent's actual measurement design from this record, never from self-reports.
            "num_blocks": len(column),
        }
    if isinstance(data, (JobOrmData, JobLocalPauliData)):
        return {"observable": data.observable, "basis_family": data.basis_family}
    return {"state_label": data.state_label}


def submit_readout_calibration(
    ctx: ActionContext, *, device_id: str, state_label: str, num_shots: int
) -> dict[str, Any]:
    request = ReadoutCalibrationRequest.model_validate(
        {"state_label": state_label, "num_shots": num_shots}
    )
    return _submit(
        ctx,
        device_id=device_id,
        tool="run_readout_calibration",
        request=request,
        run_method="run_readout_calibration",
        submit_extra={"state_label": state_label, "num_shots": num_shots},
    )


def submit_local_pauli_batch(
    ctx: ActionContext,
    *,
    device_id: str,
    basis_family: str,
    observable: str,
    num_bases: int,
    shots_per_basis: int,
    random_seed_label: str | None = None,
) -> dict[str, Any]:
    request = LocalPauliBatchRequest.model_validate(
        {
            "basis_family": basis_family,
            "observable": observable,
            "num_bases": num_bases,
            "shots_per_basis": shots_per_basis,
            "random_seed_label": random_seed_label,
        }
    )
    return _submit(
        ctx,
        device_id=device_id,
        tool="run_local_pauli_batch",
        request=request,
        run_method="run_local_pauli_batch",
        submit_extra={
            "basis_family": basis_family,
            "observable": observable,
            "num_bases": num_bases,
            "shots_per_basis": shots_per_basis,
        },
    )


def submit_copy_block_batch(
    ctx: ActionContext,
    *,
    device_id: str,
    block_size: int,
    observable: str,
    num_blocks: int,
    measurement_family: str,
) -> dict[str, Any]:
    request = CopyBlockBatchRequest.model_validate(
        {
            "block_size": block_size,
            "observable": observable,
            "num_blocks": num_blocks,
            "measurement_family": measurement_family,
        }
    )
    return _submit(
        ctx,
        device_id=device_id,
        tool="run_copy_block_batch",
        request=request,
        run_method="run_copy_block_batch",
        submit_extra={
            "block_size": block_size,
            "observable": observable,
            "num_blocks": num_blocks,
            "measurement_family": measurement_family,
        },
    )
