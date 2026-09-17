"""Simulator backend for the blackbox Lindblad-dynamics qtype.

Like the analog backend (and unlike the per-job-stateless digital/transmon
backends), this backend holds **one** :class:`LindbladEngine` for the whole run,
so the total-evolution-time budget accumulates across probe batches. The
engine's RNG is injected once here (engine-contract invariant: the factory
injects, the engine never reseeds itself). Its stream comes only from fresh
private per-run entropy: the hidden truth (h*/c*) is fixed by the committed
instance, but realized shot noise must not replay across attempts.

Replay / live-provider modes are out of scope for v1 (simulator-only qtype).
"""

from __future__ import annotations

import secrets

import numpy as np

from qiqcbench.qsim.core.wire import HamiltonianProbeBatchRequest, JobResult
from qiqcbench.qsim.qtypes.blackbox_lindblad_dynamics.device import (
    HiddenLindbladConfig,
    PublicLindbladSpec,
)
from qiqcbench.qsim.qtypes.blackbox_lindblad_dynamics.engine import LindbladEngine

__all__ = ["LindbladSimulatorBackend", "build_lindblad_simulator_backend"]


class LindbladSimulatorBackend:
    """Holds the single run-long Lindblad engine and answers probe batches."""

    def __init__(self, hidden: HiddenLindbladConfig, public: PublicLindbladSpec):
        self._device_id = hidden.device_id
        self._max_rows_per_batch = public.max_rows_per_batch
        rng = np.random.default_rng(secrets.randbits(128))
        self._engine = LindbladEngine(hidden, public, rng)

    @property
    def engine(self) -> LindbladEngine:
        return self._engine

    def _failed(self, job_id: str, error: str) -> JobResult:
        return JobResult(job_id=job_id, device_id=self._device_id, status="failed", error=error)

    def run_lindblad_probe_batch(
        self, request: HamiltonianProbeBatchRequest, job_id: str, salt: int
    ) -> JobResult:
        """Execute one probe batch. ``salt`` is unused: the run-long engine owns
        a single entropy-mixed RNG stream, so noise is independent across rows
        within the run and independent across attempts."""
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


def build_lindblad_simulator_backend(
    hidden: HiddenLindbladConfig, public: PublicLindbladSpec
) -> LindbladSimulatorBackend:
    return LindbladSimulatorBackend(hidden, public)
