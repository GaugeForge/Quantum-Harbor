"""In-process simulator backend for the NV sensor-network qtype.

Validates the agent-visible request shape (fails closed before execution) and
delegates to the qtype-local runner. The fixed total-interrogation-time budget
``T_tot`` is a *task-level* fairness convention checked by the verifier, not a
device-level limit, so it is not enforced here.
"""

from __future__ import annotations

from qiqcbench.qsim.core.wire import (
    JobResult,
    NvSensingProbeRequest,
    NvSensingSweepRequest,
)
from qiqcbench.qsim.qtypes.nv_sensor_network.device import (
    HiddenNvSensorNetworkConfig,
    PublicNvSensorNetworkSpec,
)
from qiqcbench.qsim.qtypes.nv_sensor_network.runner import (
    run_sensing_probe,
    run_sensing_sweep,
)


class NvSensorNetworkSimulatorBackend:
    """NV sensor-network backend backed by the in-process analytic simulator."""

    def __init__(
        self, hidden: HiddenNvSensorNetworkConfig, public: PublicNvSensorNetworkSpec
    ) -> None:
        self.hidden = hidden
        self.public = public
        self.node_ids = {nd.id for nd in public.nodes}
        self.max_shots = public.max_shots
        self.resolution_s = public.time_resolution_s

    def _failed(self, job_id: str, shots: int, error: str) -> JobResult:
        return JobResult(
            job_id=job_id,
            device_id=self.hidden.device_id,
            status="failed",
            shots=shots,
            error=error,
        )

    def _on_grid(self, tau_s: float) -> bool:
        n = round(tau_s / self.resolution_s)
        return n >= 1 and abs(tau_s - n * self.resolution_s) <= self.resolution_s * 1e-6

    def _validate(self, *, support: list[int], taus: list[float], shots: int) -> str | None:
        if shots > self.max_shots:
            return f"shots exceeds max_shots {self.max_shots}"
        if any(s not in self.node_ids for s in support):
            return f"support nodes must be among {sorted(self.node_ids)!r}"
        if any(not self._on_grid(t) for t in taus):
            return (
                f"interrogation times must be positive multiples of the "
                f"{self.resolution_s * 1e9:.0f} ns grid"
            )
        return None

    def run_sensing_probe(
        self, request: NvSensingProbeRequest, job_id: str, salt: int
    ) -> JobResult:
        error = self._validate(
            support=list(request.support),
            taus=list(request.interrogation_time_s),
            shots=request.shots,
        )
        if error:
            return self._failed(job_id, request.shots, error)
        return run_sensing_probe(request, self.hidden, job_id, salt)

    def run_sensing_sweep(
        self, request: NvSensingSweepRequest, job_id: str, salt: int
    ) -> JobResult:
        error = self._validate(
            support=list(request.support),
            taus=list(request.interrogation_time_grid_s),
            shots=request.shots,
        )
        if error:
            return self._failed(job_id, request.shots, error)
        return run_sensing_sweep(request, self.hidden, job_id, salt)


def build_nv_sensor_network_simulator_backend(
    hidden: HiddenNvSensorNetworkConfig,
    public: PublicNvSensorNetworkSpec,
) -> NvSensorNetworkSimulatorBackend:
    """Construct the in-process NV sensor-network simulator backend."""
    return NvSensorNetworkSimulatorBackend(hidden, public)


__all__ = [
    "NvSensorNetworkSimulatorBackend",
    "build_nv_sensor_network_simulator_backend",
]
