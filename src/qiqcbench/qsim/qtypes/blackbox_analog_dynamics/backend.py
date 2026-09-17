"""Simulator backend for the blackbox analog-dynamics qtype.

Unlike the digital/transmon backends (which build a fresh stateless engine per
job), this backend holds **one** :class:`AnalogEngine` for the whole run, so the
total-evolution-time budget accumulates across probe batches. The backend mixes
``hidden.seed`` with fresh per-run entropy and injects one RNG into the engine.
The hidden Hamiltonian stays fixed while shot noise changes between benchmark
attempts.
"""

from __future__ import annotations

import secrets

import numpy as np

from qiqcbench.qsim.core.wire import HamiltonianProbeBatchRequest, JobResult
from qiqcbench.qsim.qtypes.blackbox_analog_dynamics.device import (
    HiddenAnalogConfig,
    PublicAnalogSpec,
)
from qiqcbench.qsim.qtypes.blackbox_analog_dynamics.engine import AnalogEngine

__all__ = ["AnalogSimulatorBackend", "build_analog_simulator_backend"]


class AnalogSimulatorBackend:
    """Holds the single run-long analog engine and answers probe batches."""

    def __init__(self, hidden: HiddenAnalogConfig, public: PublicAnalogSpec):
        self._device_id = hidden.device_id
        self._max_rows_per_batch = public.max_rows_per_batch
        rng = np.random.default_rng([hidden.seed, secrets.randbits(64)])
        self._engine = AnalogEngine(hidden, public, rng)

    @property
    def engine(self) -> AnalogEngine:
        return self._engine

    def _failed(self, job_id: str, error: str) -> JobResult:
        return JobResult(job_id=job_id, device_id=self._device_id, status="failed", error=error)

    def run_hamiltonian_probe_batch(
        self, request: HamiltonianProbeBatchRequest, job_id: str, salt: int
    ) -> JobResult:
        """Execute one probe batch. ``salt`` is unused: the run-long engine owns
        a single entropy-mixed RNG stream, so rows within a run remain
        independent and separate attempts receive fresh shot noise."""
        if len(request.rows) > self._max_rows_per_batch:
            return self._failed(
                job_id,
                f"probe batch has {len(request.rows)} rows; max is {self._max_rows_per_batch}",
            )
        data = self._engine.run_probe_batch(request.rows)
        return JobResult(
            job_id=job_id,
            device_id=self._device_id,
            status="complete",
            shots=data.rows[0].num_internal_repetitions if data.rows else None,
            data=data,
        )


def build_analog_simulator_backend(
    hidden: HiddenAnalogConfig, public: PublicAnalogSpec
) -> AnalogSimulatorBackend:
    return AnalogSimulatorBackend(hidden, public)
