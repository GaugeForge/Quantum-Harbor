"""Surface-neutral submit seam for gmon-ring sequences and sweeps.

Queues gmon control jobs through the gmon backend (``run_gmon_sequence`` /
``run_gmon_sweep``) without extending the pulse/circuit backend protocols
(controller decision). Gated on the ``ring_modulation`` capability and logs
gmon-specific evidence fields the verifier uses for lightweight evidence
presence (e.g. ``has_modulation`` distinguishes a real parametric drive from a
static/DC coupling or a zero-amplitude no-op).
"""

from __future__ import annotations

from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.gmon_ring_3q.wire import (
    CouplerSetting,
    GmonEvolveOp,
    GmonMeasureOp,
    GmonSequenceOp,
    GmonSequenceRequest,
    GmonSweepRequest,
)

__all__ = ["submit_gmon_sequence", "submit_gmon_sweep"]

_CAPABILITY = "ring_modulation"


def _require_capability(ctx: ActionContext) -> None:
    if _CAPABILITY not in ctx.state.active_capabilities:
        raise ValueError(
            f"{_CAPABILITY} capability is not active for this run; cannot submit gmon jobs"
        )


def _executed_values(
    value: float | str, sweep: dict[str, list[float]] | None
) -> list[float] | None:
    """Every value a coupler field can take at execution, or None if unresolvable.

    A sweep placeholder resolves to the values the sweep actually binds to it, so
    classification follows what ran rather than what the template looked like.
    """
    if isinstance(value, str):
        bound = (sweep or {}).get(value.lstrip("$"))
        if not isinstance(bound, list) or not bound:
            return None
        try:
            return [float(v) for v in bound]
        except (TypeError, ValueError):
            return None
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return None


def _is_driving(setting: CouplerSetting, sweep: dict[str, list[float]] | None) -> bool:
    """True only when this coupler genuinely drives at EVERY executed configuration.

    A parametric drive needs a nonzero modulation frequency **and** a nonzero
    amplitude: ``freq_hz=70e6`` at ``amp=0`` modulates nothing and yields a
    no-information job. Classifying such a job as modulated let an agent submit
    zero-amplitude jobs and bind fabricated claims to them, since a per-point
    evidence gate can only be as strong as its notion of "a drive".

    Placeholders resolve through the sweep, and a swept amplitude that can reach 0
    does not count -- without enumerating joint sweep points we cannot tell which
    coordinate a later claim refers to, so this fails closed.
    """
    amps = _executed_values(setting.amp, sweep)
    freqs = _executed_values(setting.freq_hz, sweep)
    if amps is None or freqs is None:
        return False
    return min(amps) > 0.0 and min(freqs) > 0.0


def _evidence_fields(
    sequence: list[GmonSequenceOp], sweep: dict[str, list[float]] | None = None
) -> dict[str, Any]:
    """Structural summary the verifier uses for lightweight evidence binding."""
    n_evolve = 0
    n_coupler_settings = 0
    n_measure = 0
    has_modulation = False
    for op in sequence:
        if isinstance(op, GmonEvolveOp):
            n_evolve += 1
            for c in op.couplers:
                n_coupler_settings += 1
                if _is_driving(c, sweep):
                    has_modulation = True
        elif isinstance(op, GmonMeasureOp):
            n_measure += 1
    return {
        "n_ops": len(sequence),
        "n_evolve": n_evolve,
        "n_coupler_settings": n_coupler_settings,
        "n_measure": n_measure,
        "has_modulation": has_modulation,
    }


def _log_request(request: GmonSequenceRequest | GmonSweepRequest) -> dict[str, Any]:
    """The exact validated request, so a verifier can bind a claim to what ran.

    Mirrors the ``ramsey_sensing`` action seam: the JSONL log carries the validated
    request itself rather than a task-specific restatement of it, and per-task
    verifiers derive whatever they need (here: which commanded coupler phases a job
    actually drove) from the executed ops and sweep coordinates.
    """
    return {"request": request.model_dump(mode="json")}


def submit_gmon_sequence(
    ctx: ActionContext, *, device_id: str, sequence: list[dict[str, Any]], shots: int
) -> dict[str, Any]:
    """Queue one gmon control sequence through the active gmon backend."""
    _require_capability(ctx)
    request = GmonSequenceRequest.model_validate({"shots": shots, "sequence": sequence})
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        return ctx.state.backend.run_gmon_sequence(request, job_id, salt)

    job_id = ctx.state.jobs.submit(runner)
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "submit_gmon_sequence",
            "tool": "run_gmon_sequence",
            "device_id": device_id,
            "shots": shots,
            "job_id": job_id,
            **_evidence_fields(request.sequence),
            **_log_request(request),
        }
    )
    return {"job_id": job_id, "status": "queued"}


def submit_gmon_sweep(
    ctx: ActionContext,
    *,
    device_id: str,
    template_sequence: list[dict[str, Any]],
    sweep: dict[str, list[float]],
    shots: int,
    mode: str = "product",
) -> dict[str, Any]:
    """Queue one gmon control sweep through the active gmon backend."""
    _require_capability(ctx)
    request = GmonSweepRequest.model_validate(
        {
            "shots": shots,
            "template_sequence": template_sequence,
            "sweep": sweep,
            "mode": mode,
        }
    )
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        return ctx.state.backend.run_gmon_sweep(request, job_id, salt)

    job_id = ctx.state.jobs.submit(runner)
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": "submit_gmon_sweep",
            "tool": "run_gmon_sweep",
            "device_id": device_id,
            "shots": shots,
            "sweep_keys": list(sweep.keys()),
            "n_points": sum(len(v) for v in sweep.values()),
            "job_id": job_id,
            **_evidence_fields(request.template_sequence, request.sweep),
            **_log_request(request),
        }
    )
    return {"job_id": job_id, "status": "queued"}
