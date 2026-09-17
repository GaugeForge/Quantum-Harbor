"""Task-owned action seam for the heralded-leakage decoding capability.

Evidence discipline (/qsim_logs is agent-readable): log lines carry digests, counts, and
budget meters only — never the challenge truth. The full canonical prediction bits DO go
into the log (they are the agent's own submission, ~5 KB packed, and the verifier scores
from them); challenge/calibration bit streams are logged as sha256 digests only. Every
evidence log line binds a versioned opaque instance commitment — never the raw
graded instance seed — so a verifier or an offline rescore can prove producer/scorer
run-input agreement and regenerate the identical challenge from its own runtime seed.
"""

from __future__ import annotations

import hashlib
from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.surface_code_memory.capabilities.heralded_leakage_decoding.runtime import (  # noqa: E501
    HERALDED_EVIDENCE_SCHEMA_VERSION,
    instance_evidence_commitment,
)
from qiqcbench.qsim.qtypes.surface_code_memory.wire import (
    ChallengeBatchRequest,
    JobLeakageChallengeData,
    JobLeakageMemoryData,
    LeakageMemoryRequest,
)

_CAPABILITY = "heralded_leakage_decoding"


def _record_infrastructure_failure(
    ctx: ActionContext, *, device_id: str, job_id: str | None, tool: str, failure_stage: str
) -> None:
    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": "heralded_leakage_infrastructure_failure",
                "tool": tool,
                "device_id": device_id,
                "job_id": job_id,
                "failure_stage": failure_stage,
            }
        )
    except Exception:
        return


def _require_capability(ctx: ActionContext, device_id: str) -> None:
    if _CAPABILITY not in ctx.state.active_capabilities:
        raise ValueError(f"{_CAPABILITY} capability is not active for this run")
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")


def _sha(b64: str) -> str:
    return hashlib.sha256(b64.encode("ascii")).hexdigest()


def _evidence_provenance(ctx: ActionContext) -> dict[str, Any]:
    """Versioned opaque binding of evidence to the graded run inputs."""
    return {
        "evidence_schema_version": HERALDED_EVIDENCE_SCHEMA_VERSION,
        "instance_commitment": instance_evidence_commitment(
            ctx.state.hidden,
            task_id=ctx.state.task_id,
            instance_seed=ctx.state.backend.leakage_instance_seed(),
        ),
    }


def submit_leakage_memory_experiment(
    ctx: ActionContext, *, device_id: str, rounds: int, shots: int
) -> dict[str, Any]:
    _require_capability(ctx, device_id)
    request = LeakageMemoryRequest.model_validate({"rounds": rounds, "shots": shots})
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        try:
            result = ctx.state.backend.run_leakage_memory_experiment(request, job_id, salt)
            data = result.data
            if isinstance(data, JobLeakageMemoryData):
                ctx.state.log(
                    {
                        "surface": ctx.surface,
                        "action": "leakage_memory_result",
                        "tool": "run_leakage_memory_experiment",
                        "device_id": device_id,
                        "job_id": job_id,
                        **_evidence_provenance(ctx),
                        "rounds": data.rounds,
                        "shots": data.shots,
                        "n_detectors": data.n_detectors,
                        "n_herald_slots": data.n_herald_slots,
                        "raw_detector_sha256": _sha(data.detection_events_b64),
                        "raw_herald_sha256": _sha(data.herald_events_b64),
                        "budget": data.budget.model_dump(),
                    }
                )
            return result
        except Exception:
            _record_infrastructure_failure(
                ctx,
                device_id=device_id,
                job_id=job_id,
                tool="run_leakage_memory_experiment",
                failure_stage="backend_or_evidence_publication",
            )
            raise

    job_id = ctx.state.jobs.submit(runner)
    ctx.state.log(
        {
            "action": "submit_leakage_memory_experiment",
            "job_id": job_id,
            "surface": ctx.surface,
            "device_id": device_id,
            "rounds": rounds,
            "shots": shots,
        }
    )
    return {"job_id": job_id, "status": "queued"}


def submit_challenge_batch(
    ctx: ActionContext, *, device_id: str, chunk_index: int
) -> dict[str, Any]:
    _require_capability(ctx, device_id)
    request = ChallengeBatchRequest.model_validate({"chunk_index": chunk_index})
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        try:
            result = ctx.state.backend.run_leakage_challenge_batch(request, job_id, salt)
            data = result.data
            if isinstance(data, JobLeakageChallengeData):
                ctx.state.log(
                    {
                        "surface": ctx.surface,
                        "action": "leakage_challenge_served",
                        "tool": "fetch_challenge_batch",
                        "device_id": device_id,
                        "job_id": job_id,
                        **_evidence_provenance(ctx),
                        "challenge_set_id": data.challenge_set_id,
                        "challenge_digest": data.challenge_digest,
                        "chunk_index": data.chunk_index,
                        "n_chunks": data.n_chunks,
                        "chunk_shots": data.chunk_shots,
                        "rounds": data.rounds,
                        "chunk_detector_sha256": _sha(data.detection_events_b64),
                        "chunk_herald_sha256": _sha(data.herald_events_b64),
                    }
                )
            return result
        except Exception:
            _record_infrastructure_failure(
                ctx,
                device_id=device_id,
                job_id=job_id,
                tool="fetch_challenge_batch",
                failure_stage="backend_or_evidence_publication",
            )
            raise

    job_id = ctx.state.jobs.submit(runner)
    ctx.state.log(
        {
            "action": "submit_challenge_batch",
            "job_id": job_id,
            "surface": ctx.surface,
            "device_id": device_id,
            "chunk_index": chunk_index,
        }
    )
    return {"job_id": job_id, "status": "queued"}


def submit_challenge_predictions(
    ctx: ActionContext, *, device_id: str, chunk_index: int, predictions_b64: str
) -> dict[str, Any]:
    """Synchronous, free of charge, no truth-derived feedback (invariant 4 governs
    ``run_*`` experiment tools; this records a claim and runs no dynamics)."""
    _require_capability(ctx, device_id)
    # Malformed agent input raises out of accept_challenge_predictions as an
    # ordinary agent-facing error; only the evidence-publication step below is
    # an infrastructure fault when it fails.
    payload = ctx.state.backend.accept_challenge_predictions(chunk_index, predictions_b64)
    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": "challenge_predictions_accepted",
                "tool": "submit_challenge_predictions",
                "device_id": device_id,
                **_evidence_provenance(ctx),
                **payload,
            }
        )
    except Exception:
        _record_infrastructure_failure(
            ctx,
            device_id=device_id,
            job_id=None,
            tool="submit_challenge_predictions",
            failure_stage="evidence_publication",
        )
        raise RuntimeError("qsim challenge-prediction evidence publication failed") from None
    return {
        "accepted": True,
        "challenge_set_id": payload["challenge_set_id"],
        "chunk_index": payload["chunk_index"],
        "chunk_sha256": payload["chunk_sha256"],
        "chunks_accepted": payload["chunks_accepted"],
        "coverage_shots": payload["coverage_shots"],
        "combined_sha256": payload["combined_sha256"],
        "complete": payload["complete"],
    }


__all__ = [
    "submit_challenge_batch",
    "submit_challenge_predictions",
    "submit_leakage_memory_experiment",
]
