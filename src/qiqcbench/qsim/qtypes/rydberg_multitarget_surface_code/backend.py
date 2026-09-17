"""Simulator backend with domain-separated, schedule-independent job streams."""

from __future__ import annotations

import os

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.rydberg_multitarget_surface_code.capabilities.multitarget_stabilizer_readout.runtime import (
    MultitargetReadoutEngine,
    domain_separated_rng,
)
from qiqcbench.qsim.qtypes.rydberg_multitarget_surface_code.device import (
    HiddenRydbergMultitargetSurfaceCodeConfig,
    PublicRydbergMultitargetSurfaceCodeSpec,
)
from qiqcbench.qsim.qtypes.rydberg_multitarget_surface_code.wire import (
    Cz2CharacterizationRequest,
    JobCz2CharacterizationData,
    JobMultitargetMemoryData,
    MultitargetMemoryRequest,
)


class RydbergMultitargetSimulatorBackend:
    def __init__(
        self,
        hidden: HiddenRydbergMultitargetSurfaceCodeConfig,
        public: PublicRydbergMultitargetSurfaceCodeSpec,
        *,
        instance_seed: int | None = None,
    ):
        if instance_seed is None:
            raw_seed = os.environ.get("QIQCBENCH_INSTANCE_SEED", "0")
            try:
                instance_seed = int(raw_seed)
            except ValueError as exc:
                raise ValueError("QIQCBENCH_INSTANCE_SEED must be an integer") from exc
        if type(instance_seed) is not int:
            raise ValueError("instance_seed must be an integer")
        self._device_id = hidden.device_id
        self._construction_id = hidden.construction_id
        self._instance_seed = instance_seed
        self._engine = MultitargetReadoutEngine(hidden, public)

    @property
    def engine(self) -> MultitargetReadoutEngine:
        return self._engine

    def run_cz2_echo_characterization(
        self,
        request: Cz2CharacterizationRequest,
        job_id: str,
        salt: int,
        *,
        accepted_shots_used: int | None = None,
    ) -> JobResult:
        result = self._engine.run_characterization(
            request,
            rng=domain_separated_rng(
                construction_id=self._construction_id,
                attempt_seed=self._instance_seed,
                job_kind="cz2_echo_characterization",
                salt=salt,
            ),
            accepted_shots_used=accepted_shots_used,
        )
        if isinstance(result, JobCz2CharacterizationData):
            return JobResult(
                job_id=job_id, device_id=self._device_id, status="complete", data=result
            )
        return JobResult(job_id=job_id, device_id=self._device_id, status="failed", error=result)

    def run_multitarget_stabilizer_memory(
        self,
        request: MultitargetMemoryRequest,
        job_id: str,
        salt: int,
        *,
        accepted_shots_used: int | None = None,
    ) -> JobResult:
        result = self._engine.run_memory(
            request,
            rng=domain_separated_rng(
                construction_id=self._construction_id,
                attempt_seed=self._instance_seed,
                job_kind="multitarget_stabilizer_memory",
                salt=salt,
            ),
            accepted_shots_used=accepted_shots_used,
        )
        if isinstance(result, JobMultitargetMemoryData):
            return JobResult(
                job_id=job_id, device_id=self._device_id, status="complete", data=result
            )
        return JobResult(job_id=job_id, device_id=self._device_id, status="failed", error=result)


def build_rydberg_multitarget_simulator_backend(
    hidden: HiddenRydbergMultitargetSurfaceCodeConfig,
    public: PublicRydbergMultitargetSurfaceCodeSpec,
) -> RydbergMultitargetSimulatorBackend:
    return RydbergMultitargetSimulatorBackend(hidden, public)


__all__ = ["RydbergMultitargetSimulatorBackend", "build_rydberg_multitarget_simulator_backend"]
