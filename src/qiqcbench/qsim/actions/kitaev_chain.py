"""Action seam for the ``kitaev_chain`` qtype charge-readout capability.

Routes the three charge-readout primitives into the active backend and records
per-call evidence the verifier reads back (which probe was run, on which
bond/window, how many shots). Capability-gated and fail-closed here (defense in
depth alongside MCP registration). Raw parity bits / gaps go to the agent via
the job result; the log keeps the experiment shape, not the bulk arrays.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.actions.common import release_job_admission, reserve_job_admission
from qiqcbench.qsim.backends.provider_artifacts import canonical_request_hash
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.jobs import new_job_id
from qiqcbench.qsim.qtypes.kitaev_chain.wire import (
    ChargeStabilityRequest,
    KitaevCliffordRBData,
    MajoranaCliffordRBRequest,
    MajoranaPulseBatchRequest,
    ProtectionSweepRequest,
    SubchainSpectroscopyRequest,
)

_CAP = "kitaev_charge_readout"
_CAP_RB = "majorana_pulse_control"
_RB_EVIDENCE_CONTRACT = "majorana_clifford_rb_v1"


def _require(ctx: ActionContext, device_id: str, action: str, cap: str = _CAP) -> None:
    if cap not in ctx.state.active_capabilities:
        raise ValueError(f"{cap} capability is not active for this run; cannot {action}")
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")


def submit_charge_stability(
    ctx: ActionContext,
    *,
    device_id: str,
    bond: int,
    mu_ld_values: list[float],
    mu_rd_values: list[float],
    shots: int,
) -> dict[str, Any]:
    _require(ctx, device_id, "submit charge stability")
    request = ChargeStabilityRequest.model_validate(
        {
            "bond": bond,
            "mu_ld_values": mu_ld_values,
            "mu_rd_values": mu_rd_values,
            "shots": shots,
        }
    )
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        return ctx.state.backend.run_charge_stability(request, job_id, salt)

    job_id = ctx.state.jobs.submit(runner)
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "submit_charge_stability",
            "tool": "run_charge_stability",
            "device_id": device_id,
            "bond": bond,
            "n_ld": len(mu_ld_values),
            "n_rd": len(mu_rd_values),
            "shots": shots,
            "job_id": job_id,
        }
    )
    return {"job_id": job_id, "status": "queued"}


def submit_protection_sweep(
    ctx: ActionContext,
    *,
    device_id: str,
    mu_common_values: list[float],
    shots: int,
) -> dict[str, Any]:
    _require(ctx, device_id, "submit protection sweep")
    request = ProtectionSweepRequest.model_validate(
        {"mu_common_values": mu_common_values, "shots": shots}
    )
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        return ctx.state.backend.run_protection_sweep(request, job_id, salt)

    job_id = ctx.state.jobs.submit(runner)
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "submit_protection_sweep",
            "tool": "run_protection_sweep",
            "device_id": device_id,
            "n_mu": len(mu_common_values),
            # The detunings themselves (agent inputs, never hidden truth): a slope is
            # only defined over distinct points, so the verifier counts them rather
            # than trusting a call count.
            "mu_common_values": list(request.mu_common_values),
            "shots": shots,
            "job_id": job_id,
        }
    )
    return {"job_id": job_id, "status": "queued"}


def submit_subchain_spectroscopy(
    ctx: ActionContext,
    *,
    device_id: str,
    site_a: int,
    site_b: int,
    shots: int,
) -> dict[str, Any]:
    _require(ctx, device_id, "submit subchain spectroscopy")
    request = SubchainSpectroscopyRequest.model_validate(
        {"site_a": site_a, "site_b": site_b, "shots": shots}
    )
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        return ctx.state.backend.run_subchain_spectroscopy(request, job_id, salt)

    job_id = ctx.state.jobs.submit(runner)
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "submit_subchain_spectroscopy",
            "tool": "run_subchain_spectroscopy",
            "device_id": device_id,
            "site_a": site_a,
            "site_b": site_b,
            "shots": shots,
            "job_id": job_id,
        }
    )
    return {"job_id": job_id, "status": "queued"}


def submit_pulse_batch(
    ctx: ActionContext,
    *,
    device_id: str,
    sequences: list[list[dict[str, Any]]],
    shots: int,
) -> dict[str, Any]:
    _require(ctx, device_id, "submit pulse batch", cap=_CAP_RB)
    request = MajoranaPulseBatchRequest.model_validate({"sequences": sequences, "shots": shots})
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        return ctx.state.backend.run_pulse_batch(request, job_id, salt)

    job_id = ctx.state.jobs.submit(runner)
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "submit_pulse_batch",
            "tool": "run_majorana_pulse_batch",
            "device_id": device_id,
            "n_sequences": len(request.sequences),
            "total_segments": request.total_segments,
            "shots": shots,
            "job_id": job_id,
        }
    )
    return {"job_id": job_id, "status": "queued"}


def submit_clifford_rb(
    ctx: ActionContext,
    *,
    device_id: str,
    gate_x90_segments: list[dict[str, Any]],
    gate_z90_segments: list[dict[str, Any]],
    lengths: list[int],
    sequences_per_length: int,
    shots_per_sequence: int,
) -> dict[str, Any]:
    """Submit authenticated qsim-owned uniform Clifford randomization."""

    _require(ctx, device_id, "submit Clifford RB", cap=_CAP_RB)
    request = MajoranaCliffordRBRequest.model_validate(
        {
            "gate_x90_segments": gate_x90_segments,
            "gate_z90_segments": gate_z90_segments,
            "lengths": lengths,
            "sequences_per_length": sequences_per_length,
            "shots_per_sequence": shots_per_sequence,
        }
    )
    salt = ctx.state.next_salt()
    job_id = new_job_id()
    request_payload = request.model_dump(mode="json")
    request_digest = canonical_request_hash(request)
    submission = {
        "surface": ctx.surface,
        "action": "submit_majorana_clifford_rb",
        "tool": "run_majorana_clifford_rb",
        "evidence_contract": _RB_EVIDENCE_CONTRACT,
        "evidence_schema_version": 1,
        "task_id": ctx.state.task_id,
        "device_id": device_id,
        "job_id": job_id,
        "request": request_payload,
        "request_digest": request_digest,
    }
    reserve_job_admission(ctx.state, job_id)
    try:
        ctx.state.log(submission)
    except Exception:
        release_job_admission(ctx.state, job_id)
        raise RuntimeError("qsim Clifford-RB submission evidence failed") from None

    def runner(accepted_job_id: str) -> JobResult:
        result = ctx.state.backend.run_clifford_rb(request, accepted_job_id, salt)
        terminal: dict[str, Any] = {
            "surface": ctx.surface,
            "action": "majorana_clifford_rb_result",
            "tool": "run_majorana_clifford_rb",
            "evidence_contract": _RB_EVIDENCE_CONTRACT,
            "evidence_schema_version": 1,
            "task_id": ctx.state.task_id,
            "device_id": device_id,
            "job_id": accepted_job_id,
            "request_digest": request_digest,
            "status": result.status,
        }
        if result.status == "complete" and isinstance(result.data, KitaevCliffordRBData):
            raw = json.dumps(
                result.data.model_dump(mode="json"),
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
            terminal.update(
                {
                    "raw_data_sha256": hashlib.sha256(raw).hexdigest(),
                    "n_random_sequences": len(request.lengths) * request.sequences_per_length,
                    "shots_per_sequence": request.shots_per_sequence,
                }
            )
        ctx.state.log(terminal)
        return result

    try:
        ctx.state.jobs.submit(runner, job_id=job_id)
    except Exception:
        release_job_admission(ctx.state, job_id)
        ctx.state.log(
            {
                **submission,
                "action": "majorana_clifford_rb_submission_failure",
                "status": "failed",
                "failure_kind": "qsim_internal",
            }
        )
        raise RuntimeError("qsim Clifford-RB submission failed") from None
    return {"job_id": job_id, "status": "queued"}


__all__ = [
    "submit_charge_stability",
    "submit_protection_sweep",
    "submit_subchain_spectroscopy",
    "submit_pulse_batch",
    "submit_clifford_rb",
]
