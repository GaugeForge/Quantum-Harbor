"""Circuit-level sequence and sweep runners for the digital qtype.

Bridges wire schemas through the digital engine into a ``JobResult``. The
top-level ``qiqcbench.qsim.runner`` dispatcher routes here through
``descriptor_for_qtype`` so qtype-specific imports stay confined to this
module.
"""

from __future__ import annotations

import secrets
import time

import numpy as np

from qiqcbench.qsim.core.wire import (
    CircuitRequest,
    CircuitSweepRequest,
    JobBitstringData,
    JobResult,
    JobResultMetadata,
)
from qiqcbench.qsim.qtypes.digital_gate_model.device import HiddenDigitalConfig
from qiqcbench.qsim.qtypes.digital_gate_model.engine import DigitalEngine
from qiqcbench.qsim.qtypes.digital_gate_model.policy import (
    expand_circuit_sweep_points,
    sweep_coords_from_points,
    validate_circuit_non_sweep_parameters,
)


def fresh_run_entropy() -> int:
    """Draw private per-attempt shot entropy (64 bits from the OS CSPRNG)."""
    return secrets.randbits(64)


def _make_rng(
    hidden: HiddenDigitalConfig, salt: int, run_entropy: int | None = None
) -> np.random.Generator:
    """Seed the shot sampler from (hidden seed, per-attempt entropy, per-call salt).

    ``hidden.seed`` is a committed constant and ``salt`` is a per-process
    counter that restarts at zero in every container, so seeding from those
    two alone replays byte-identical shot noise across independent benchmark
    attempts. ``run_entropy`` is drawn once per simulator backend (once per
    attempt) and breaks that replay; pinning it is a test/regression seam
    only. A ``None`` entropy draws a fresh value for this call so a direct
    runner invocation can never silently replay.
    """
    if run_entropy is None:
        run_entropy = fresh_run_entropy()
    return np.random.default_rng([hidden.seed, run_entropy, salt])


def run_circuit_sequence(
    request: CircuitRequest,
    hidden: HiddenDigitalConfig,
    job_id: str,
    salt: int,
    *,
    run_entropy: int | None = None,
) -> JobResult:
    error = validate_circuit_non_sweep_parameters(request)
    if error is not None:
        return JobResult(
            job_id=job_id,
            device_id=hidden.device_id,
            status="failed",
            shots=request.shots,
            error=error,
        )
    rng = _make_rng(hidden, salt, run_entropy)
    engine = DigitalEngine(hidden, rng)
    return engine.run_circuit(
        circuit=request.circuit,
        shots=request.shots,
        device_id=hidden.device_id,
        job_id=job_id,
        bindings=None,
    )


def run_circuit_sweep_request(
    request: CircuitSweepRequest,
    hidden: HiddenDigitalConfig,
    job_id: str,
    salt: int,
    *,
    run_entropy: int | None = None,
) -> JobResult:
    try:
        keys, points = expand_circuit_sweep_points(request)
    except ValueError as exc:
        return JobResult(
            job_id=job_id,
            device_id=hidden.device_id,
            status="failed",
            shots=request.shots,
            error=str(exc),
        )

    t_start = time.perf_counter()
    rng = _make_rng(hidden, salt, run_entropy)
    bitstrings_per_point: list[list[str]] = []
    measured_qubits: list[int] | None = None
    coords = sweep_coords_from_points(keys, points)

    for i, binding in enumerate(points):
        per_point_engine = DigitalEngine(hidden, rng)
        result = per_point_engine.run_circuit(
            circuit=request.template_circuit,
            shots=request.shots,
            device_id=hidden.device_id,
            job_id=f"{job_id}_pt{i}",
            bindings=binding,
        )
        if result.status != "complete":
            return JobResult(
                job_id=job_id,
                device_id=hidden.device_id,
                status="failed",
                shots=request.shots,
                error=f"sweep point {i} failed: {result.error}",
            )
        assert isinstance(result.data, JobBitstringData)
        # result.data.bitstrings has outer length 1 for a single circuit run.
        bitstrings_per_point.append(result.data.bitstrings[0])
        if measured_qubits is None:
            measured_qubits = result.data.measured_qubits
        elif measured_qubits != result.data.measured_qubits:
            return JobResult(
                job_id=job_id,
                device_id=hidden.device_id,
                status="failed",
                shots=request.shots,
                error="measured_qubits varied across sweep points",
            )

    return JobResult(
        job_id=job_id,
        device_id=hidden.device_id,
        status="complete",
        shots=request.shots,
        data=JobBitstringData(
            bitstrings=bitstrings_per_point,
            measured_qubits=measured_qubits or [],
        ),
        metadata=JobResultMetadata(
            wallclock_ms=int((time.perf_counter() - t_start) * 1000),
            sweep_coords=coords,
        ),
    )


__all__ = ["fresh_run_entropy", "run_circuit_sequence", "run_circuit_sweep_request"]
