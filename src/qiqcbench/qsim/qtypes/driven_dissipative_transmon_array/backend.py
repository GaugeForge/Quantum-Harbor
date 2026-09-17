"""In-process simulator backend for the driven-dissipative array qtype.

Validates the agent-visible request shape against the public control envelope
(fails closed BEFORE execution) and delegates to the qtype-local runner.
"""

from __future__ import annotations

from collections.abc import Sequence

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.driven_dissipative_transmon_array.device import (
    HiddenDdtaConfig,
    PublicDdtaSpec,
)
from qiqcbench.qsim.qtypes.driven_dissipative_transmon_array.runner import (
    run_stabilization,
    run_stabilization_sweep,
)
from qiqcbench.qsim.qtypes.driven_dissipative_transmon_array.wire import (
    MeasureSetting,
    StabilizeRequest,
    StabilizeSweepRequest,
)


class DdtaSimulatorBackend:
    """Driven-dissipative array backend backed by the in-process Lindblad simulator."""

    def __init__(self, hidden: HiddenDdtaConfig, public: PublicDdtaSpec) -> None:
        self.hidden = hidden
        self.public = public
        self.n = len(public.qubits)
        self.limits = public.control_limits

    def _failed(self, job_id: str, shots: int, error: str) -> JobResult:
        return JobResult(
            job_id=job_id,
            device_id=self.hidden.device_id,
            status="failed",
            shots=shots,
            error=error,
        )

    def _validate(
        self,
        *,
        pair: str,
        delta_s_mhz: float,
        delta_d_mhz: float,
        g_s_mhz: float,
        g_d_mhz: float,
        durations: Sequence[float],
        measure: Sequence[MeasureSetting],
        shots: int,
    ) -> str | None:
        if shots > self.public.max_shots:
            return f"shots exceeds max_shots {self.public.max_shots}"
        if pair not in self.public.selectable_pairs:
            return f"pair must be one of {self.public.selectable_pairs!r}"
        ds_lo, ds_hi = self.limits.pump_detuning_mhz_range
        dd_lo, dd_hi = self.limits.loss_detuning_mhz_range
        if not (ds_lo <= delta_s_mhz <= ds_hi):
            return f"delta_s_mhz out of range [{ds_lo}, {ds_hi}]"
        if not (dd_lo <= delta_d_mhz <= dd_hi):
            return f"delta_d_mhz out of range [{dd_lo}, {dd_hi}]"
        g_lo, g_hi = self.limits.coupling_mhz_range
        if not (g_lo <= g_s_mhz <= g_hi) or not (g_lo <= g_d_mhz <= g_hi):
            return f"coupling out of range [{g_lo}, {g_hi}]"
        if any(t < 0 or t > self.limits.max_duration_us for t in durations):
            return f"duration must be in [0, {self.limits.max_duration_us}] us"
        sites = [m.site for m in measure]
        if len(sites) != len(set(sites)):
            return "measure sites must be distinct"
        if any(s < 0 or s >= self.n for s in sites):
            return f"measure sites must be in [0, {self.n - 1}]"
        return None

    def run_stabilization(self, request: StabilizeRequest, job_id: str, salt: int) -> JobResult:
        error = self._validate(
            pair=request.pair,
            delta_s_mhz=request.delta_s_mhz,
            delta_d_mhz=request.delta_d_mhz,
            g_s_mhz=request.g_s_mhz,
            g_d_mhz=request.g_d_mhz,
            durations=[request.duration_us],
            measure=request.measure,
            shots=request.shots,
        )
        if error:
            return self._failed(job_id, request.shots, error)
        return run_stabilization(request, self.hidden, job_id, salt)

    def run_stabilization_sweep(
        self, request: StabilizeSweepRequest, job_id: str, salt: int
    ) -> JobResult:
        error = self._validate(
            pair=request.pair,
            delta_s_mhz=request.delta_s_mhz,
            delta_d_mhz=request.delta_d_mhz,
            g_s_mhz=request.g_s_mhz,
            g_d_mhz=request.g_d_mhz,
            durations=list(request.duration_grid_us),
            measure=request.measure,
            shots=request.shots,
        )
        if error:
            return self._failed(job_id, request.shots, error)
        return run_stabilization_sweep(request, self.hidden, job_id, salt)


def build_ddta_simulator_backend(
    hidden: HiddenDdtaConfig, public: PublicDdtaSpec
) -> DdtaSimulatorBackend:
    """Construct the in-process driven-dissipative array simulator backend."""
    return DdtaSimulatorBackend(hidden, public)


__all__ = ["DdtaSimulatorBackend", "build_ddta_simulator_backend"]
