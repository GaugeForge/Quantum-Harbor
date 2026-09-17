"""Action seam for the ``cycle_error_recon`` qtype.

Routes ``run_readout_calibration_batch`` / ``run_folded_cer_batch`` into the active
backend and records the per-batch evidence the verifier reads back: each row's digest +
accepted/rejected status and the cumulative qsim-owned budget. Capability-gated and
fail-closed at this layer (defense in depth alongside MCP registration).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.cycle_error_recon.wire import (
    FoldedCerBatchRequest,
    JobCerCountsData,
    JobReadoutCalibData,
    ReadoutCalibBatchRequest,
)

_CAP = "folded_cer"


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def cer_row_digest(payload: dict[str, Any]) -> str:
    return _digest(payload)


def readout_row_digest(payload: dict[str, Any]) -> str:
    return _digest(payload)


def _require(ctx: ActionContext, device_id: str) -> None:
    if _CAP not in ctx.state.active_capabilities:
        raise ValueError(f"{_CAP} capability is not active for this run")
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")


def submit_readout_calibration_batch(
    ctx: ActionContext, *, device_id: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    _require(ctx, device_id)
    request = ReadoutCalibBatchRequest.model_validate({"rows": rows})
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        result = ctx.state.backend.run_readout_calibration_batch(request, job_id, salt)
        data = result.data
        if isinstance(data, JobReadoutCalibData):
            ctx.state.log(
                {
                    "evidence_schema_version": 1,
                    "surface": ctx.surface,
                    "action": "readout_calibration_result",
                    "tool": "run_readout_calibration_batch",
                    "device_id": device_id,
                    "job_id": job_id,
                    "rows": [
                        {
                            "digest": readout_row_digest(request_row.model_dump()),
                            **request_row.model_dump(),
                            "status": result_row.status,
                            "accepted_shots": result_row.shots,
                            "reject_reason": result_row.reject_reason,
                        }
                        for request_row, result_row in zip(request.rows, data.rows, strict=True)
                    ],
                    "budget": data.budget.model_dump(),
                }
            )
        return result

    job_id = ctx.state.jobs.submit(runner)
    ctx.state.log(
        {
            "evidence_schema_version": 1,
            "action": "submit_readout_calibration_batch",
            "job_id": job_id,
            "surface": ctx.surface,
            "device_id": device_id,
            "n_rows": len(rows),
            "rows": [
                {"digest": readout_row_digest(row.model_dump()), **row.model_dump()}
                for row in request.rows
            ],
        }
    )
    return {"job_id": job_id, "status": "queued"}


def submit_folded_cer_batch(
    ctx: ActionContext, *, device_id: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    _require(ctx, device_id)
    request = FoldedCerBatchRequest.model_validate({"rows": rows})
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        result = ctx.state.backend.run_folded_cer_batch(request, job_id, salt)
        data = result.data
        if isinstance(data, JobCerCountsData):
            ctx.state.log(
                {
                    "evidence_schema_version": 1,
                    "surface": ctx.surface,
                    "action": "folded_cer_result",
                    "tool": "run_folded_cer_batch",
                    "device_id": device_id,
                    "job_id": job_id,
                    "rows": [
                        {
                            "digest": cer_row_digest(request_row.model_dump()),
                            **request_row.model_dump(),
                            "status": result_row.status,
                            "reject_reason": result_row.reject_reason,
                        }
                        for request_row, result_row in zip(request.rows, data.rows, strict=True)
                    ],
                    "budget": data.budget.model_dump(),
                }
            )
        return result

    job_id = ctx.state.jobs.submit(runner)
    ctx.state.log(
        {
            "evidence_schema_version": 1,
            "action": "submit_folded_cer_batch",
            "job_id": job_id,
            "surface": ctx.surface,
            "device_id": device_id,
            "n_rows": len(rows),
            "rows": [
                {"digest": cer_row_digest(row.model_dump()), **row.model_dump()}
                for row in request.rows
            ],
        }
    )
    return {"job_id": job_id, "status": "queued"}
