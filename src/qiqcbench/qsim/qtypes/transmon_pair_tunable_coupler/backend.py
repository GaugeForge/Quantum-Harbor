"""Tunable-coupler CZ simulator backend (fails closed on agent-visible limits)."""

from __future__ import annotations

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.transmon_pair_tunable_coupler.device import (
    HiddenTunableCouplerConfig,
    PublicTunableCouplerSpec,
)
from qiqcbench.qsim.qtypes.transmon_pair_tunable_coupler.runner import run_flux_pulse
from qiqcbench.qsim.qtypes.transmon_pair_tunable_coupler.wire import FluxPulseRequest


class TunableCouplerSimulatorBackend:
    def __init__(self, hidden: HiddenTunableCouplerConfig) -> None:
        self.hidden = hidden
        self.max_shots: int | None = None
        self.max_abs_flux: float | None = None

    @classmethod
    def from_public(
        cls, hidden: HiddenTunableCouplerConfig, public: PublicTunableCouplerSpec
    ) -> TunableCouplerSimulatorBackend:
        b = cls(hidden)
        b.max_shots = public.max_shots
        b.max_abs_flux = public.max_abs_flux_q1
        return b

    def _failed(self, job_id: str, shots: int, error: str) -> JobResult:
        return JobResult(
            job_id=job_id,
            device_id=self.hidden.device_id,
            status="failed",
            shots=shots,
            error=error,
        )

    def _validate(self, request: FluxPulseRequest) -> str | None:
        if self.max_shots is not None and request.shots > self.max_shots:
            return f"shots exceeds max_shots {self.max_shots}"
        if not (0.0 <= request.coupler_flux <= 0.5):
            return f"coupler_flux {request.coupler_flux} out of range [0, 0.5]"
        for f in request.programmed_flux_q1:
            if abs(f) > 0.8:  # DAC full-scale (predistortion overshoot allowed up to ~0.7)
                return f"programmed flux sample |{f}| exceeds DAC range 0.8"
        for op in (*request.prep_ops, *request.post_ops):
            if op.qubit not in (1, 2):
                return f"rotation on invalid qubit {op.qubit}"
        return None

    def run_flux_pulse(self, request: FluxPulseRequest, job_id: str, salt: int) -> JobResult:
        if error := self._validate(request):
            return self._failed(job_id, request.shots, error)
        return run_flux_pulse(request, self.hidden, job_id, salt)


def build_tunable_coupler_simulator_backend(
    hidden: HiddenTunableCouplerConfig, public: PublicTunableCouplerSpec
) -> TunableCouplerSimulatorBackend:
    return TunableCouplerSimulatorBackend.from_public(hidden, public)


__all__ = ["TunableCouplerSimulatorBackend", "build_tunable_coupler_simulator_backend"]
