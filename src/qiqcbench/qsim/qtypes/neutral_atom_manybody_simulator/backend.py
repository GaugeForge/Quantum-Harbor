"""Simulator backend for stroboscopic Rydberg evolution."""

from __future__ import annotations

import os

import numpy as np

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.neutral_atom_manybody_simulator.device import (
    HiddenNeutralAtomManyBodyConfig,
    PublicNeutralAtomManyBodySpec,
)
from qiqcbench.qsim.qtypes.neutral_atom_manybody_simulator.engine import NeutralAtomManyBodyEngine
from qiqcbench.qsim.qtypes.neutral_atom_manybody_simulator.wire import RydbergTrajectoryRequest


class NeutralAtomManyBodySimulatorBackend:
    def __init__(
        self,
        hidden: HiddenNeutralAtomManyBodyConfig,
        public: PublicNeutralAtomManyBodySpec,
        *,
        instance_seed: int = 0,
    ):
        self._device_id = hidden.device_id
        # Fixed hidden truth, fresh realized data: the loss and readout parameters are the
        # device, while ``instance_seed`` selects which realization of them an attempt sees.
        # The verdict does not depend on it -- a rule's bias and retained fraction are
        # closed-form expectations -- so varying it varies the agent's evidence, not the gate.
        self.engine = NeutralAtomManyBodyEngine(
            hidden, public, np.random.default_rng([hidden.seed, instance_seed])
        )

    def run_stroboscopic_rydberg_evolution(
        self, request: RydbergTrajectoryRequest, job_id: str, salt: int
    ) -> JobResult:
        result = self.engine.run(request)
        if isinstance(result, str):
            return JobResult(
                job_id=job_id, device_id=self._device_id, status="failed", error=result
            )
        return JobResult(job_id=job_id, device_id=self._device_id, status="complete", data=result)


def build_neutral_atom_manybody_simulator_backend(
    hidden: HiddenNeutralAtomManyBodyConfig, public: PublicNeutralAtomManyBodySpec
) -> NeutralAtomManyBodySimulatorBackend:
    return NeutralAtomManyBodySimulatorBackend(
        hidden, public, instance_seed=int(os.environ.get("QIQCBENCH_INSTANCE_SEED", "0"))
    )


__all__ = ["NeutralAtomManyBodySimulatorBackend", "build_neutral_atom_manybody_simulator_backend"]
