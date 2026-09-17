"""Simulator backend for the ``surface_code_memory`` qtype.

Holds **one** :class:`SurfaceCodeEngine` for the whole run so the shot budget accumulates
across calls. The RNG is injected once here from ``hidden.seed`` (engine-contract invariant:
the factory injects, the engine never reseeds itself).
"""

from __future__ import annotations

import os

import numpy as np

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.surface_code_memory.capabilities.drift_recalibration_scheduling.runtime import (
    DriftEngine,
)
from qiqcbench.qsim.qtypes.surface_code_memory.capabilities.heralded_leakage_decoding.runtime import (
    LeakageEngine,
)
from qiqcbench.qsim.qtypes.surface_code_memory.capabilities.syndrome_feedback_control.runtime import (
    FeedbackEngine,
)
from qiqcbench.qsim.qtypes.surface_code_memory.device import (
    HiddenSurfaceCodeConfig,
    PublicSurfaceCodeSpec,
)
from qiqcbench.qsim.qtypes.surface_code_memory.engine import SurfaceCodeEngine
from qiqcbench.qsim.qtypes.surface_code_memory.wire import (
    ChallengeBatchRequest,
    DriftControlRequest,
    DriftStreamRequest,
    JobDetectorData,
    JobDriftControlData,
    JobDriftStreamData,
    JobLeakageChallengeData,
    JobLeakageMemoryData,
    JobSyndromeControllerProgramData,
    JobSyndromeControlProbeDataV3,
    LeakageMemoryRequest,
    MemoryExperimentRequest,
    SyndromeControllerProgramRequest,
    SyndromeControlProbeRequest,
    _ControllerBudgetView,
)

__all__ = ["SurfaceCodeSimulatorBackend", "build_surface_code_memory_simulator_backend"]


class SurfaceCodeSimulatorBackend:
    def __init__(self, hidden: HiddenSurfaceCodeConfig, public: PublicSurfaceCodeSpec):
        self._device_id = hidden.device_id
        instance_seed = int(os.environ.get("QIQCBENCH_INSTANCE_SEED", "0") or "0")
        self._instance_seed = instance_seed
        # Mix the per-instance seed into the agent-facing memory realization so successive
        # runs draw INDEPENDENT characterization data (same pattern as the drift
        # engine below). Offset 63_017 is unique in this file, and the verifier
        # replay stream stays default_rng(seed + 1): a collision would need
        # instance_seed = 1 - 63_017 < 0, impossible for non-negative instance seeds.
        rng = np.random.default_rng(hidden.seed + 63_017 + instance_seed)
        self._engine = SurfaceCodeEngine(hidden, public, rng)
        self._feedback_engine = (
            FeedbackEngine(
                hidden,
                public,
                np.random.default_rng(hidden.seed + 41_009 + instance_seed),
            )
            if public.syndrome_feedback is not None
            else None
        )
        # Mix the per-instance seed into the agent-facing drift realization so
        # successive runs draw INDEPENDENT characterization data.
        self._drift_engine = (
            DriftEngine(hidden, public, np.random.default_rng(hidden.seed + 52_021 + instance_seed))
            if public.drift_reloqation is not None
            else None
        )
        # Same per-instance independence for the heralded-leakage instrument: both the
        # calibration realization and the sealed challenge mix the instance seed.
        self._leakage_engine = (
            LeakageEngine(hidden, public, instance_seed)
            if public.heralded_leakage is not None
            else None
        )

    @property
    def engine(self) -> SurfaceCodeEngine:
        return self._engine

    def run_memory_experiment(
        self, request: MemoryExperimentRequest, job_id: str, salt: int
    ) -> JobResult:
        # `salt` is deliberately unused: the run-long engine consumes one accumulating
        # stream (the budget-meter design); per-call reseeding is not wanted.
        result = self._engine.run_memory_experiment(request)
        if isinstance(result, JobDetectorData):
            return JobResult(
                job_id=job_id, device_id=self._device_id, status="complete", data=result
            )
        return JobResult(job_id=job_id, device_id=self._device_id, status="failed", error=result)

    def run_syndrome_control_probe(
        self,
        request: SyndromeControlProbeRequest,
        job_id: str,
        salt: int,
        budget: _ControllerBudgetView,
    ) -> JobResult:
        if self._feedback_engine is None:
            return JobResult(
                job_id=job_id,
                device_id=self._device_id,
                status="failed",
                error="syndrome-feedback instrument is not configured for this device",
            )
        result = self._feedback_engine.run_probe(request, budget=budget)
        if isinstance(result, JobSyndromeControlProbeDataV3):
            return JobResult(
                job_id=job_id, device_id=self._device_id, status="complete", data=result
            )
        return JobResult(job_id=job_id, device_id=self._device_id, status="failed", error=result)

    def validate_syndrome_controller_program(self) -> dict:
        if self._feedback_engine is None:
            raise ValueError("syndrome-feedback instrument is not configured for this device")
        return self._feedback_engine.validate_program()

    def run_syndrome_controller_program(
        self,
        request: SyndromeControllerProgramRequest,
        job_id: str,
        salt: int,
        budget: _ControllerBudgetView,
    ) -> JobResult:
        if self._feedback_engine is None:
            return JobResult(
                job_id=job_id,
                device_id=self._device_id,
                status="failed",
                error="syndrome-feedback instrument is not configured for this device",
            )
        result = self._feedback_engine.run_program(request, budget=budget)
        if isinstance(result, JobSyndromeControllerProgramData):
            return JobResult(
                job_id=job_id, device_id=self._device_id, status="complete", data=result
            )
        return JobResult(job_id=job_id, device_id=self._device_id, status="failed", error=result)

    def run_drift_syndrome_stream(
        self, request: DriftStreamRequest, job_id: str, salt: int
    ) -> JobResult:
        if self._drift_engine is None:
            return JobResult(
                job_id=job_id,
                device_id=self._device_id,
                status="failed",
                error="drift-reloqation instrument is not configured for this device",
            )
        result = self._drift_engine.run_stream(request)
        if isinstance(result, JobDriftStreamData):
            return JobResult(
                job_id=job_id, device_id=self._device_id, status="complete", data=result
            )
        return JobResult(job_id=job_id, device_id=self._device_id, status="failed", error=result)

    def run_drift_control(self, request: DriftControlRequest, job_id: str, salt: int) -> JobResult:
        if self._drift_engine is None:
            return JobResult(
                job_id=job_id,
                device_id=self._device_id,
                status="failed",
                error="drift-reloqation instrument is not configured for this device",
            )
        result = self._drift_engine.run_control(request)
        if isinstance(result, JobDriftControlData):
            return JobResult(
                job_id=job_id, device_id=self._device_id, status="complete", data=result
            )
        return JobResult(job_id=job_id, device_id=self._device_id, status="failed", error=result)

    def run_leakage_memory_experiment(
        self, request: LeakageMemoryRequest, job_id: str, salt: int
    ) -> JobResult:
        if self._leakage_engine is None:
            return JobResult(
                job_id=job_id,
                device_id=self._device_id,
                status="failed",
                error="heralded-leakage instrument is not configured for this device",
            )
        result = self._leakage_engine.run_calibration(request)
        if isinstance(result, JobLeakageMemoryData):
            return JobResult(
                job_id=job_id, device_id=self._device_id, status="complete", data=result
            )
        return JobResult(job_id=job_id, device_id=self._device_id, status="failed", error=result)

    def run_leakage_challenge_batch(
        self, request: ChallengeBatchRequest, job_id: str, salt: int
    ) -> JobResult:
        if self._leakage_engine is None:
            return JobResult(
                job_id=job_id,
                device_id=self._device_id,
                status="failed",
                error="heralded-leakage instrument is not configured for this device",
            )
        result = self._leakage_engine.challenge_chunk(request)
        if isinstance(result, JobLeakageChallengeData):
            return JobResult(
                job_id=job_id, device_id=self._device_id, status="complete", data=result
            )
        return JobResult(job_id=job_id, device_id=self._device_id, status="failed", error=result)

    def accept_challenge_predictions(self, chunk_index: int, predictions_b64: str) -> dict:
        if self._leakage_engine is None:
            raise ValueError("heralded-leakage instrument is not configured for this device")
        return self._leakage_engine.accept_predictions(chunk_index, predictions_b64)

    def leakage_instance_seed(self) -> int:
        if self._leakage_engine is None:
            raise ValueError("heralded-leakage instrument is not configured for this device")
        return self._leakage_engine.instance_seed

    def drift_instance_seed(self) -> int:
        if self._drift_engine is None:
            raise ValueError("drift-reloqation instrument is not configured for this device")
        return self._instance_seed

    def feedback_instance_seed(self) -> int:
        if self._feedback_engine is None:
            raise ValueError("syndrome-feedback instrument is not configured for this device")
        return self._instance_seed


def build_surface_code_memory_simulator_backend(
    hidden: HiddenSurfaceCodeConfig, public: PublicSurfaceCodeSpec
) -> SurfaceCodeSimulatorBackend:
    return SurfaceCodeSimulatorBackend(hidden, public)
