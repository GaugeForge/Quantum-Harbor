"""Simulator backend for the ``neutral_atom_quantum_network`` qtype.

Holds **one** :class:`NetworkEngine` for the whole run so the communication budget accumulates
across calls. The RNG is injected once here from ``hidden.seed`` (engine-contract invariant:
the factory injects, the engine never reseeds itself).
"""

from __future__ import annotations

import numpy as np

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.neutral_atom_quantum_network.device import (
    HiddenNetworkConfig,
    PublicNetworkSpec,
)
from qiqcbench.qsim.qtypes.neutral_atom_quantum_network.engine import NetworkEngine
from qiqcbench.qsim.qtypes.neutral_atom_quantum_network.wire import (
    DistributedEstimationRequest,
    JobEstimationData,
)

__all__ = ["NetworkSimulatorBackend", "build_neutral_atom_quantum_network_simulator_backend"]


class NetworkSimulatorBackend:
    def __init__(self, hidden: HiddenNetworkConfig, public: PublicNetworkSpec):
        self._device_id = hidden.device_id
        rng = np.random.default_rng(hidden.seed)
        self._engine = NetworkEngine(hidden, public, rng)

    def run_distributed_estimation(
        self, request: DistributedEstimationRequest, job_id: str, salt: int
    ) -> JobResult:
        result = self._engine.run_distributed_estimation(request)
        if isinstance(result, JobEstimationData):
            return JobResult(
                job_id=job_id, device_id=self._device_id, status="complete", data=result
            )
        return JobResult(job_id=job_id, device_id=self._device_id, status="failed", error=result)


def build_neutral_atom_quantum_network_simulator_backend(
    hidden: HiddenNetworkConfig, public: PublicNetworkSpec
) -> NetworkSimulatorBackend:
    return NetworkSimulatorBackend(hidden, public)
