"""In-process simulator backend for the multilevel-transmon qtype.

Validates the agent-visible request shape against the public control contract
(fails closed before execution) and delegates to the qtype-local runner.
"""

from __future__ import annotations

import math

from qiqcbench.qsim.core.wire import (
    AnalyticDriveSegment,
    DriveSegment,
    JobResult,
    MultilevelPulseRequest,
    MultilevelPulseSweepRequest,
    SampledDriveSegment,
)
from qiqcbench.qsim.qtypes.transmon_multilevel_pulse.device import (
    HiddenTransmonMultilevelConfig,
    PublicTransmonMultilevelSpec,
)
from qiqcbench.qsim.qtypes.transmon_multilevel_pulse.runner import (
    run_pulse_sequence,
    run_pulse_sweep,
)


class TransmonMultilevelSimulatorBackend:
    """Multilevel-transmon backend backed by the in-process numpy simulator."""

    def __init__(
        self,
        hidden: HiddenTransmonMultilevelConfig,
        public: PublicTransmonMultilevelSpec,
    ) -> None:
        self.hidden = hidden
        self.public = public

    def _failed(self, job_id: str, shots: int, error: str) -> JobResult:
        return JobResult(
            job_id=job_id,
            device_id=self.hidden.device_id,
            status="failed",
            shots=shots,
            error=error,
        )

    def _validate_scalar_amp(self, value: float | str, what: str) -> str | None:
        if isinstance(value, str):
            return None  # sweep placeholder; bound value checked at render time is bounded below
        if abs(value) > self.public.max_amp_dac:
            return f"{what}={value} exceeds max_amp_dac {self.public.max_amp_dac}"
        return None

    def _validate_segment(self, seg: DriveSegment) -> str | None:
        p = self.public
        if isinstance(seg, SampledDriveSegment):
            n = len(seg.omega_x_dac)
            if not (len(seg.omega_y_dac) == n and len(seg.detuning_hz) == n):
                return "sampled segment quadrature/detuning arrays must share one length"
            if n < p.min_sample_count:
                return f"sampled segment has {n} samples; min_sample_count is {p.min_sample_count}"
            if not math.isclose(seg.sample_dt_ns, p.sample_dt_ns, rel_tol=1e-6):
                return f"sample_dt_ns must equal the device grid {p.sample_dt_ns} ns"
            if n * seg.sample_dt_ns > p.max_pulse_duration_ns:
                return f"segment exceeds max_pulse_duration_ns {p.max_pulse_duration_ns}"
            for arr, lbl in ((seg.omega_x_dac, "omega_x_dac"), (seg.omega_y_dac, "omega_y_dac")):
                if any(abs(v) > p.max_amp_dac for v in arr):
                    return f"{lbl} exceeds max_amp_dac {p.max_amp_dac}"
            if any(abs(v) > p.max_abs_detuning_hz for v in seg.detuning_hz):
                return f"detuning_hz exceeds max_abs_detuning_hz {p.max_abs_detuning_hz}"
            return None
        assert isinstance(seg, AnalyticDriveSegment)
        err = self._validate_scalar_amp(seg.amp_dac, "amp_dac")
        if err:
            return err
        if (
            isinstance(seg.carrier_detuning_hz, (int, float))
            and abs(seg.carrier_detuning_hz) > p.max_abs_detuning_hz
        ):
            return f"carrier_detuning_hz exceeds max_abs_detuning_hz {p.max_abs_detuning_hz}"
        if isinstance(seg.duration_ns, (int, float)):
            if seg.duration_ns <= 0 or seg.duration_ns > p.max_pulse_duration_ns:
                return f"duration_ns must be in (0, {p.max_pulse_duration_ns}] ns"
        return None

    def _validate(self, segments: list[DriveSegment], shots: int) -> str | None:
        if shots > self.public.max_shots:
            return f"shots exceeds max_shots {self.public.max_shots}"
        if len(segments) > self.public.max_segments:
            return f"too many segments (> {self.public.max_segments})"
        for seg in segments:
            err = self._validate_segment(seg)
            if err:
                return err
        return None

    def run_pulse_sequence(
        self, request: MultilevelPulseRequest, job_id: str, salt: int
    ) -> JobResult:
        err = self._validate(list(request.segments), request.shots)
        if err:
            return self._failed(job_id, request.shots, err)
        return run_pulse_sequence(request, self.hidden, self.public, job_id, salt)

    def run_pulse_sweep(
        self, request: MultilevelPulseSweepRequest, job_id: str, salt: int
    ) -> JobResult:
        err = self._validate(list(request.template_segments), request.shots)
        if err:
            return self._failed(job_id, request.shots, err)
        if not request.sweep:
            return self._failed(job_id, request.shots, "sweep must define at least one axis")
        return run_pulse_sweep(request, self.hidden, self.public, job_id, salt)


def build_transmon_multilevel_simulator_backend(
    hidden: HiddenTransmonMultilevelConfig,
    public: PublicTransmonMultilevelSpec,
) -> TransmonMultilevelSimulatorBackend:
    """Construct the in-process multilevel-transmon simulator backend."""
    return TransmonMultilevelSimulatorBackend(hidden, public)


__all__ = [
    "TransmonMultilevelSimulatorBackend",
    "build_transmon_multilevel_simulator_backend",
]
