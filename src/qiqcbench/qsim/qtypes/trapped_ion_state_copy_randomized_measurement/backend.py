"""Simulator backend for ``trapped_ion_state_copy_randomized_measurement``.

Holds **one** :class:`StateCopyEngine` for the whole run so the state-copy and randomized-basis
budgets accumulate across calls. The RNG is injected once here (engine contract: the factory
injects, the engine never reseeds itself), seeded from ``hidden.seed`` mixed with fresh
per-process entropy: the hidden physics is deterministic in the seed, but the *shot noise* must
differ between trials. A pure ``hidden.seed`` stream made every trial that opened with the same
request replay byte-identical data (one effective noise sample across an entire run), and
the per-call ``salt`` cannot repair that because the salt counter itself restarts identically in
every process.
"""

from __future__ import annotations

import secrets

import numpy as np

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.trapped_ion_state_copy_randomized_measurement.device import (
    HiddenStateCopyConfig,
    PublicStateCopySpec,
)
from qiqcbench.qsim.qtypes.trapped_ion_state_copy_randomized_measurement.engine import (
    StateCopyEngine,
)
from qiqcbench.qsim.qtypes.trapped_ion_state_copy_randomized_measurement.wire import (
    CopyBlockBatchRequest,
    JobJointCopyBlockData,
    JobLocalPauliData,
    JobReadoutCalData,
    LocalPauliBatchRequest,
    ReadoutCalibrationRequest,
)

__all__ = [
    "StateCopySimulatorBackend",
    "build_trapped_ion_state_copy_randomized_measurement_simulator_backend",
]


class StateCopySimulatorBackend:
    def __init__(self, hidden: HiddenStateCopyConfig, public: PublicStateCopySpec):
        self._device_id = hidden.device_id
        rng = np.random.default_rng([hidden.seed, secrets.randbits(63)])
        self._engine = StateCopyEngine(hidden, rng, max_jobs=public.budgets.max_jobs)

    @property
    def engine(self) -> StateCopyEngine:
        return self._engine

    def _wrap(self, result, job_id: str, ok_type) -> JobResult:
        if isinstance(result, ok_type):
            return JobResult(
                job_id=job_id, device_id=self._device_id, status="complete", data=result
            )
        return JobResult(job_id=job_id, device_id=self._device_id, status="failed", error=result)

    def run_readout_calibration(
        self, request: ReadoutCalibrationRequest, job_id: str, salt: int
    ) -> JobResult:
        return self._wrap(self._engine.run_readout_calibration(request), job_id, JobReadoutCalData)

    def run_local_pauli_batch(
        self, request: LocalPauliBatchRequest, job_id: str, salt: int
    ) -> JobResult:
        return self._wrap(self._engine.run_local_pauli_batch(request), job_id, JobLocalPauliData)

    def run_copy_block_batch(
        self, request: CopyBlockBatchRequest, job_id: str, salt: int
    ) -> JobResult:
        return self._wrap(self._engine.run_copy_block_batch(request), job_id, JobJointCopyBlockData)


def build_trapped_ion_state_copy_randomized_measurement_simulator_backend(
    hidden: HiddenStateCopyConfig, public: PublicStateCopySpec
) -> StateCopySimulatorBackend:
    return StateCopySimulatorBackend(hidden, public)
