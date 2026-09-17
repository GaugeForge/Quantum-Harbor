"""Sequence and sweep runners for the gmon-ring qtype.

Bridges the gmon wire schemas through the joint-qutrit engine into a
``JobResult``. Mirrors ``transmon_pulse/runner.py`` but uses gmon-named entry
points (``run_gmon_sequence``/``run_gmon_sweep``) since the gmon control surface
is distinct from the pulse/circuit runner slots.
"""

from __future__ import annotations

import time

import numpy as np

from qiqcbench.qsim.core.wire import JobIQData, JobResult, JobResultMetadata
from qiqcbench.qsim.qtypes.gmon_ring_3q.device import HiddenGmonRingConfig
from qiqcbench.qsim.qtypes.gmon_ring_3q.engine import GmonRingEngine
from qiqcbench.qsim.qtypes.gmon_ring_3q.wire import (
    GmonDelayOp,
    GmonEvolveOp,
    GmonSequenceOp,
    GmonSequenceRequest,
    GmonSweepRequest,
)
from qiqcbench.qsim.sweep import expand_sweep_points


def _make_rng(hidden: HiddenGmonRingConfig, salt: int) -> np.random.Generator:
    """Combine the hidden seed with a per-call salt so repeated calls differ."""
    return np.random.default_rng((hidden.seed * 1_000_003) ^ salt)


def run_gmon_sequence(
    request: GmonSequenceRequest,
    hidden: HiddenGmonRingConfig,
    job_id: str,
    salt: int,
) -> JobResult:
    rng = _make_rng(hidden, salt)
    engine = GmonRingEngine(hidden, rng)
    return engine.run_sequence(
        sequence=request.sequence,
        shots=request.shots,
        device_id=hidden.device_id,
        job_id=job_id,
    )


def _sub(value: float | str, bindings: dict[str, float]) -> float | str:
    if isinstance(value, str):
        name = value.lstrip("$")
        if name not in bindings:
            raise ValueError(f"Unbound sweep placeholder ${name}")
        return float(bindings[name])
    return value


def _resolve_op(op: GmonSequenceOp, bindings: dict[str, float]) -> GmonSequenceOp:
    if isinstance(op, GmonEvolveOp):
        return op.model_copy(
            update={
                "duration_ns": _sub(op.duration_ns, bindings),
                "couplers": [
                    c.model_copy(
                        update={
                            "amp": _sub(c.amp, bindings),
                            "freq_hz": _sub(c.freq_hz, bindings),
                            "phase_rad": _sub(c.phase_rad, bindings),
                        }
                    )
                    for c in op.couplers
                ],
            }
        )
    if isinstance(op, GmonDelayOp):
        return op.model_copy(update={"duration_ns": _sub(op.duration_ns, bindings)})
    return op


def run_gmon_sweep(
    request: GmonSweepRequest,
    hidden: HiddenGmonRingConfig,
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
    coords: dict[str, list] = {k: [] for k in keys}
    seq_dur_total = 0.0
    rng = _make_rng(hidden, salt)
    for i, binding in enumerate(points):
        engine = GmonRingEngine(hidden, rng)
        try:
            resolved = [_resolve_op(op, binding) for op in request.template_sequence]
        except ValueError as exc:
            return JobResult(
                job_id=job_id,
                device_id=hidden.device_id,
                status="failed",
                shots=request.shots,
                error=str(exc),
            )
        result = engine.run_sequence(
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


__all__ = ["run_gmon_sequence", "run_gmon_sweep"]
