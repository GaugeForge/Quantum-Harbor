"""Top-level qtype-agnostic runner façade.

This module is a thin dispatcher: it resolves the qtype-specific runner from
``descriptor_for_qtype(hidden.qtype)`` and forwards. The concrete sequence
and sweep implementations live under the qtype packages
(``qsim.qtypes.transmon_pulse.runner`` and
``qsim.qtypes.digital_gate_model.runner``).

The top-level names (``run_pulse_sequence``, ``run_sweep``,
``run_circuit_sequence``, ``run_circuit_sweep_request``, ``_make_rng``) are
preserved for backward compatibility with existing test imports.
"""

from __future__ import annotations

import secrets
from typing import Any

import numpy as np

from qiqcbench.qsim.core.wire import (
    CircuitRequest,
    CircuitSweepRequest,
    JobResult,
    PulseSequenceRequest,
    SweepRequest,
)
from qiqcbench.qsim.qtypes.registry import descriptor_for_qtype


def _make_rng(hidden: Any, salt: int, run_entropy: int | None = None) -> np.random.Generator:
    """Compatibility shim mirroring the digital runner's sampler seeding.

    Seeds from (hidden seed, per-attempt entropy, per-call salt); ``None``
    entropy draws a fresh private value so a direct call never replays across
    processes. Runtime paths use the qtype-local runners, not this shim.
    """
    if run_entropy is None:
        run_entropy = secrets.randbits(64)
    return np.random.default_rng([hidden.seed, run_entropy, salt])


def run_pulse_sequence(
    request: PulseSequenceRequest,
    hidden: Any,
    job_id: str,
    salt: int,
) -> JobResult:
    desc = descriptor_for_qtype(hidden.qtype)
    return desc.run_pulse_sequence(request, hidden, job_id, salt)


def run_sweep(
    request: SweepRequest,
    hidden: Any,
    job_id: str,
    salt: int,
) -> JobResult:
    desc = descriptor_for_qtype(hidden.qtype)
    return desc.run_sweep(request, hidden, job_id, salt)


def run_circuit_sequence(
    request: CircuitRequest,
    hidden: Any,
    job_id: str,
    salt: int,
) -> JobResult:
    desc = descriptor_for_qtype(hidden.qtype)
    return desc.run_circuit_sequence(request, hidden, job_id, salt)


def run_circuit_sweep_request(
    request: CircuitSweepRequest,
    hidden: Any,
    job_id: str,
    salt: int,
) -> JobResult:
    desc = descriptor_for_qtype(hidden.qtype)
    return desc.run_circuit_sweep_request(request, hidden, job_id, salt)


__all__ = [
    "_make_rng",
    "run_circuit_sequence",
    "run_circuit_sweep_request",
    "run_pulse_sequence",
    "run_sweep",
]
