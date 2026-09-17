"""Simulator backend for the ``surface_code_lattice_surgery`` qtype.

Holds **one** :class:`LatticeSurgeryEngine` for the whole run so the shot + shot-rounds
budgets accumulate across calls. The RNG is injected once here from ``hidden.seed``
(engine-contract invariant: the factory injects, the engine never reseeds itself).
"""

from __future__ import annotations

import numpy as np

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery.device import (
    HiddenLatticeSurgeryConfig,
    PublicLatticeSurgerySpec,
)
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery.engine import LatticeSurgeryEngine
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery.wire import (
    JobLatticeSurgeryData,
    LsCnotRequest,
    LsMemoryRequest,
    LsMergedRequest,
)

__all__ = [
    "LatticeSurgerySimulatorBackend",
    "build_surface_code_lattice_surgery_simulator_backend",
]


class LatticeSurgerySimulatorBackend:
    def __init__(self, hidden: HiddenLatticeSurgeryConfig, public: PublicLatticeSurgerySpec):
        self._device_id = hidden.device_id
        rng = np.random.default_rng(hidden.seed)
        self._engine = LatticeSurgeryEngine(hidden, public, rng)

    @property
    def engine(self) -> LatticeSurgeryEngine:
        return self._engine

    def _wrap(self, result: JobLatticeSurgeryData | str, job_id: str) -> JobResult:
        if isinstance(result, JobLatticeSurgeryData):
            return JobResult(
                job_id=job_id, device_id=self._device_id, status="complete", data=result
            )
        return JobResult(job_id=job_id, device_id=self._device_id, status="failed", error=result)

    def run_memory_experiment(self, request: LsMemoryRequest, job_id: str, salt: int) -> JobResult:
        return self._wrap(self._engine.run_memory_experiment(request), job_id)

    def run_merged_memory(self, request: LsMergedRequest, job_id: str, salt: int) -> JobResult:
        return self._wrap(self._engine.run_merged_memory(request), job_id)

    def run_lattice_surgery_cnot(self, request: LsCnotRequest, job_id: str, salt: int) -> JobResult:
        return self._wrap(self._engine.run_lattice_surgery_cnot(request), job_id)


def build_surface_code_lattice_surgery_simulator_backend(
    hidden: HiddenLatticeSurgeryConfig, public: PublicLatticeSurgerySpec
) -> LatticeSurgerySimulatorBackend:
    return LatticeSurgerySimulatorBackend(hidden, public)
