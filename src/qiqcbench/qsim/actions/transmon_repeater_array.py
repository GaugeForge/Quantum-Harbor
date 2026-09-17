"""Action seam for the ``transmon_repeater_array`` qtype.

Routes ``run_purification_experiment`` into the active backend and records the
per-batch evidence the verifier reads back: each row's (depth, the agent's per-end local
circuit, basis, status, raw counts, success probability, pairs charged) + the cumulative
qsim-owned pair budget. The circuit is logged so the verifier can group evidence by
(circuit, depth) and recompute the achieved final fidelity. Capability-gated and
fail-closed at this layer. A party's circuit can only touch that party's two qubits, so
nothing acts across the two ends by construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.transmon_repeater_array.wire import (
    JobPurificationData,
    PurificationBatchRequest,
)

_CAP = "entanglement_purification"


def _require(ctx: ActionContext, device_id: str) -> None:
    if _CAP not in ctx.state.active_capabilities:
        raise ValueError(f"{_CAP} capability is not active for this run")
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")


INSTANCE_DIRNAME = "purification_instance"
INSTANCE_FILENAME = "realized_instance.json"


def _persist_realized_instance(ctx: ActionContext) -> None:
    """Record this attempt's realized hidden instance (the transport-bias axis)
    for the separate-mode verifier. Written once, under
    ``<log_dir>/purification_instance/realized_instance.json`` -- a qsim-private subtree that
    Harbor ferries to the verifier as a declared artifact. It is never published into
    ``public_job_results`` and never logged into ``experiment_log.jsonl``, and in separate
    mode the agent's container cannot see it."""
    log_dir = getattr(ctx.state, "log_dir", None)
    realize = getattr(ctx.state.backend, "realized_instance", None)
    if log_dir is None or realize is None:
        return
    payload = realize()
    if payload is None:
        return
    target = Path(log_dir) / INSTANCE_DIRNAME / INSTANCE_FILENAME
    if target.exists():
        return
    target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)


def submit_purification_batch(
    ctx: ActionContext, *, device_id: str, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    _require(ctx, device_id)
    request = PurificationBatchRequest.model_validate({"rows": rows})
    _persist_realized_instance(ctx)
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        result = ctx.state.backend.run_purification_batch(request, job_id, salt)
        data = result.data
        if isinstance(data, JobPurificationData):
            ctx.state.log(
                {
                    "surface": ctx.surface,
                    "action": "purification_result",
                    "tool": "run_purification_experiment",
                    "device_id": device_id,
                    "job_id": job_id,
                    "rows": [
                        {
                            "depth": r.depth,
                            "alice_circuit": [
                                g.model_dump(exclude_none=True) for g in req.alice_circuit
                            ],
                            "bob_circuit": [
                                g.model_dump(exclude_none=True) for g in req.bob_circuit
                            ],
                            "measure_basis": r.measure_basis,
                            "status": r.status,
                            "counts": r.counts,
                            "shots": r.shots,
                            "success_prob": r.success_prob,
                            "pairs_charged": r.pairs_charged,
                            "reject_reason": r.reject_reason,
                        }
                        for req, r in zip(request.rows, data.rows, strict=False)
                    ],
                    "budget": data.budget.model_dump(),
                }
            )
        return result

    job_id = ctx.state.jobs.submit(runner)
    ctx.state.log(
        {
            "action": "submit_purification_batch",
            "job_id": job_id,
            "surface": ctx.surface,
            "device_id": device_id,
            "n_rows": len(rows),
        }
    )
    return {"job_id": job_id, "status": "queued"}
