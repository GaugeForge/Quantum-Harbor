"""Pulse-level transmon simulator backend adapter.

Wraps the qtype-local runner so the top-level
``qiqcbench.qsim.backends.simulator`` and the qtype registry can construct
a transmon backend without importing the engine directly.
"""

from __future__ import annotations

from qiqcbench.qsim.core.wire import (
    JobResult,
    MeasureOp,
    PulseOp,
    PulseSequenceRequest,
    SequenceOp,
    SweepRequest,
)
from qiqcbench.qsim.qtypes.transmon_pulse.device import (
    HiddenTransmonConfig,
    PublicTransmonSpec,
)
from qiqcbench.qsim.qtypes.transmon_pulse.runner import (
    run_pulse_sequence,
    run_sweep,
)


class TransmonSimulatorBackend:
    """Pulse-level transmon backend backed by the in-process simulator."""

    def __init__(self, hidden: HiddenTransmonConfig) -> None:
        self.hidden = hidden
        self.max_shots: int | None = None
        self.drive_channels: set[str] | None = None
        self.measure_qubits: tuple[str, ...] | None = None
        self.supported_shapes: set[str] | None = None
        self.max_amp: float | None = None
        self.min_duration_ns: float | None = None
        self.max_duration_ns: float | None = None

    @classmethod
    def from_public(
        cls,
        hidden: HiddenTransmonConfig,
        public: PublicTransmonSpec,
    ) -> TransmonSimulatorBackend:
        backend = cls(hidden)
        backend.max_shots = public.max_shots
        backend.drive_channels = {
            channel.id for channel in public.channels if channel.kind == "drive"
        }
        backend.measure_qubits = tuple(qubit.id for qubit in public.qubits)
        backend.supported_shapes = set(public.pulse_limits.supported_shapes)
        backend.max_amp = public.pulse_limits.max_amp
        backend.min_duration_ns = public.pulse_limits.min_duration_ns
        backend.max_duration_ns = public.pulse_limits.max_duration_ns
        return backend

    def _failed(self, job_id: str, shots: int, error: str) -> JobResult:
        return JobResult(
            job_id=job_id,
            device_id=self.hidden.device_id,
            status="failed",
            shots=shots,
            error=error,
        )

    def _validate_sequence(self, sequence: list[SequenceOp], shots: int) -> str | None:
        if self.max_shots is not None and shots > self.max_shots:
            return f"transmon pulse shots exceeds max_shots {self.max_shots}"
        for op in sequence:
            if isinstance(op, PulseOp):
                if self.drive_channels is not None and op.channel not in self.drive_channels:
                    return f"transmon pulse channel must be one of {sorted(self.drive_channels)!r}"
                if self.supported_shapes is not None and op.shape not in self.supported_shapes:
                    return f"transmon pulse shape {op.shape!r} is not supported"
                if self.max_amp is not None and not 0.0 <= op.amp <= self.max_amp:
                    return f"transmon pulse amp must be between 0.0 and {self.max_amp}"
                if self.min_duration_ns is not None and self.max_duration_ns is not None:
                    if not self.min_duration_ns <= op.duration_ns <= self.max_duration_ns:
                        return (
                            "transmon pulse duration must be between "
                            f"{self.min_duration_ns} and {self.max_duration_ns} ns"
                        )
            elif isinstance(op, MeasureOp):
                if self.measure_qubits is not None and tuple(op.qubits) != self.measure_qubits:
                    return f"transmon measure qubits must be {list(self.measure_qubits)!r}"
        return None

    def run_pulse_sequence(
        self, request: PulseSequenceRequest, job_id: str, salt: int
    ) -> JobResult:
        if error := self._validate_sequence(request.sequence, request.shots):
            return self._failed(job_id, request.shots, error)
        return run_pulse_sequence(request, self.hidden, job_id, salt)

    def run_sweep(self, request: SweepRequest, job_id: str, salt: int) -> JobResult:
        if error := self._validate_sequence(request.template_sequence, request.shots):
            return self._failed(job_id, request.shots, error)
        return run_sweep(request, self.hidden, job_id, salt)


def build_transmon_simulator_backend(
    hidden: HiddenTransmonConfig,
    public: PublicTransmonSpec,
) -> TransmonSimulatorBackend:
    """Construct the in-process transmon simulator backend.

    The simulator uses hidden truth for dynamics and the public spec for the
    agent-visible pulse limits that should fail closed before execution.
    """
    return TransmonSimulatorBackend.from_public(hidden, public)


__all__ = ["TransmonSimulatorBackend", "build_transmon_simulator_backend"]
