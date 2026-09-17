"""Pulse-level sequence and sweep runners for the transmon qtype.

Bridges wire schemas through the transmon engine into a ``JobResult``. This
module is the canonical home for the transmon runtime glue; the top-level
``qiqcbench.qsim.runner`` dispatcher routes to these entry points through
``descriptor_for_qtype``.
"""

from __future__ import annotations

import time

import numpy as np

from qiqcbench.qsim.core.wire import (
    DelayOp,
    JobIQData,
    JobResult,
    JobResultMetadata,
    PulseOp,
    PulseSequenceRequest,
    SweepRequest,
)
from qiqcbench.qsim.qtypes.transmon_pulse.device import (
    HiddenTransmonConfig,
)
from qiqcbench.qsim.qtypes.transmon_pulse.engine import TransmonEngine
from qiqcbench.qsim.sweep import expand_sweep_points


def _make_rng(hidden: HiddenTransmonConfig, salt: int) -> np.random.Generator:
    """Combine the hidden seed with a per-call salt so repeated calls differ."""
    return np.random.default_rng((hidden.seed * 1_000_003) ^ salt)


def run_pulse_sequence(
    request: PulseSequenceRequest,
    hidden: HiddenTransmonConfig,
    job_id: str,
    salt: int,
) -> JobResult:
    rng = _make_rng(hidden, salt)
    engine = TransmonEngine(hidden, rng)
    return engine.run_sequence(
        sequence=request.sequence,
        shots=request.shots,
        device_id=hidden.device_id,
        job_id=job_id,
    )


def _resolve_template(template: list, bindings: dict[str, float]) -> list:
    """Substitute $name placeholders inside a template sequence."""
    resolved = []
    for op in template:
        if isinstance(op, DelayOp) and isinstance(op.duration_ns, str):
            name = op.duration_ns.lstrip("$")
            if name not in bindings:
                raise ValueError(f"Unbound sweep placeholder ${name}")
            resolved.append(DelayOp(duration_ns=float(bindings[name])))
        elif isinstance(op, PulseOp):
            # Pulses don't carry placeholders in the MVP, but copy through.
            resolved.append(op)
        else:
            resolved.append(op)
    return resolved


def run_sweep(
    request: SweepRequest,
    hidden: HiddenTransmonConfig,
    job_id: str,
    salt: int,
) -> JobResult:
    keys = list(request.sweep.keys())
    try:
        points = expand_sweep_points(keys, request.sweep, request.mode)
    except ValueError as exc:
        return JobResult(
            job_id=job_id,
            device_id=hidden.device_id,
            status="failed",
            shots=request.shots,
            error=str(exc),
        )

    t_start = time.perf_counter()
    per_qubit_iq: dict[str, list] = {}
    coords = {k: [] for k in keys}
    seq_dur_total = 0.0
    rng = _make_rng(hidden, salt)
    for i, binding in enumerate(points):
        # Each sweep point uses an isolated engine so the previous point's
        # state does not leak into the next.
        per_point_engine = TransmonEngine(hidden, rng)
        resolved = _resolve_template(request.template_sequence, binding)
        result = per_point_engine.run_sequence(
            sequence=resolved,
            shots=request.shots,
            device_id=hidden.device_id,
            job_id=f"{job_id}_pt{i}",
        )
        if result.status != "complete":
            return JobResult(
                job_id=job_id,
                device_id=hidden.device_id,
                status="failed",
                shots=request.shots,
                error=f"sweep point {i} failed: {result.error}",
            )
        for q, samples in result.data.iq.items():
            per_qubit_iq.setdefault(q, []).append(samples)
        for k in keys:
            coords[k].append(float(binding[k]))
        seq_dur_total += result.metadata.sequence_duration_ns or 0.0

    return JobResult(
        job_id=job_id,
        device_id=hidden.device_id,
        status="complete",
        shots=request.shots,
        data=JobIQData(iq=per_qubit_iq),
        metadata=JobResultMetadata(
            sequence_duration_ns=seq_dur_total,
            wallclock_ms=int((time.perf_counter() - t_start) * 1000),
            sweep_coords=coords,
        ),
    )


__all__ = ["run_pulse_sequence", "run_sweep"]
