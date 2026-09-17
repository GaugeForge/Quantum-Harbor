"""Simulator backend for the ``cycle_error_recon`` qtype.

Holds **one** :class:`CerEngine` for the whole run (like blackbox_analog_dynamics), so
the hard-cycle-exposure / raw-shot / row budgets accumulate across batch calls. The RNG
is initialized once from fresh private run entropy (engine-contract invariant: the
factory injects, the engine never reseeds itself). Tests may inject that entropy
explicitly; production callers do not persist or expose it.
"""

from __future__ import annotations

import secrets

import numpy as np

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.cycle_error_recon.device import HiddenCerConfig, PublicCerSpec
from qiqcbench.qsim.qtypes.cycle_error_recon.engine import CerEngine
from qiqcbench.qsim.qtypes.cycle_error_recon.wire import (
    FoldedCerBatchRequest,
    ReadoutCalibBatchRequest,
)

__all__ = ["CerSimulatorBackend", "build_cycle_error_recon_simulator_backend"]


class CerSimulatorBackend:
    def __init__(
        self,
        hidden: HiddenCerConfig,
        public: PublicCerSpec,
        *,
        run_entropy: int | None = None,
    ):
        self._device_id = hidden.device_id
        self._rows_per_batch = public.budgets.rows_per_batch
        if run_entropy is None:
            run_entropy = secrets.randbits(128)
        if (
            not isinstance(run_entropy, int)
            or isinstance(run_entropy, bool)
            or not 0 <= run_entropy < 2**128
        ):
            raise ValueError("run_entropy must be a 128-bit non-negative integer")
        rng = np.random.default_rng(run_entropy)
        self._engine = CerEngine(hidden, public, rng)

    @property
    def engine(self) -> CerEngine:
        return self._engine

    def _failed(self, job_id: str, error: str) -> JobResult:
        return JobResult(job_id=job_id, device_id=self._device_id, status="failed", error=error)

    def run_readout_calibration_batch(
        self, request: ReadoutCalibBatchRequest, job_id: str, salt: int
    ) -> JobResult:
        if len(request.rows) > self._rows_per_batch:
            return self._failed(
                job_id, f"batch has {len(request.rows)} rows; max is {self._rows_per_batch}"
            )
        data = self._engine.run_readout_calibration(request)
        return JobResult(job_id=job_id, device_id=self._device_id, status="complete", data=data)

    def run_folded_cer_batch(
        self, request: FoldedCerBatchRequest, job_id: str, salt: int
    ) -> JobResult:
        if len(request.rows) > self._rows_per_batch:
            return self._failed(
                job_id, f"batch has {len(request.rows)} rows; max is {self._rows_per_batch}"
            )
        data = self._engine.run_folded_cer(request)
        return JobResult(job_id=job_id, device_id=self._device_id, status="complete", data=data)


def build_cycle_error_recon_simulator_backend(
    hidden: HiddenCerConfig,
    public: PublicCerSpec,
    *,
    run_entropy: int | None = None,
) -> CerSimulatorBackend:
    return CerSimulatorBackend(hidden, public, run_entropy=run_entropy)
