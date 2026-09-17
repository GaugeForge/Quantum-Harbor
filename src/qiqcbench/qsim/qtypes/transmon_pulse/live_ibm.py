"""Transmon qtype adapter for the limited IBM live-provider T1 bridge."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from qiskit import QuantumCircuit

from qiqcbench.qsim.backends.live_ibm import (
    LiveIBMCircuitBackend,
    _CompiledLiveRequest,
)
from qiqcbench.qsim.core.wire import (
    DelayOp,
    JobResult,
    JobResultMetadata,
    MeasureOp,
    PulseOp,
    PulseSequenceRequest,
    SequenceOp,
    SweepRequest,
)
from qiqcbench.qsim.qtypes.transmon_pulse.device import PublicTransmonSpec
from qiqcbench.qsim.sweep import expand_sweep_points

_T1_SHAPE_ERROR = "T1 live supports exactly: X180 pulse, delay, measure"
_DELAY_SWEEP_KEY = "delay_ns"
_MIN_T1_LIVE_DISTINCT_DELAYS = 4
_MAX_T1_LIVE_SWEEP_POINTS = 20
_CANONICAL_X180_CHANNEL = "q0.xy"
_CANONICAL_X180_SHAPE = "gaussian"
_CANONICAL_X180_AMP = 0.21
_CANONICAL_X180_DURATION_NS = 40.0
_CANONICAL_X180_SIGMA_NS = 8.0
_FLOAT_TOLERANCE = 1e-12


class TransmonLiveIBMBackend(LiveIBMCircuitBackend):
    """IBM live backend for the transmon T1 protocol subset only.

    This is intentionally not a pulse compiler. It accepts the benchmark's
    T1-shaped wire pulse program and translates it into a provider circuit with
    one X gate, one delay, and one measurement. Anything outside that subset
    fails before provider submission.
    """

    def __init__(
        self,
        *,
        task_id: str,
        public: PublicTransmonSpec,
        backend_name: str,
        log_dir: str | Path | None = None,
        token: str | None = None,
        instance: str | None = None,
        channel: str,
        allow_live_provider: bool,
        max_shots: int | None = None,
        max_jobs: int = 1,
        backend_allowlist: Iterable[str] | None = None,
        wall_clock_s: float | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        super().__init__(
            task_id=task_id,
            device_id=public.device_id,
            backend_name=backend_name,
            log_dir=log_dir,
            token=token,
            instance=instance,
            channel=channel,
            allow_live_provider=allow_live_provider,
            max_shots=max_shots,
            max_jobs=max_jobs,
            backend_allowlist=backend_allowlist,
            wall_clock_s=wall_clock_s,
            monotonic=monotonic,
        )
        self.qubits = tuple(qubit.id for qubit in public.qubits)
        self.drive_channels = {channel.id for channel in public.channels if channel.kind == "drive"}
        self.supported_shapes = set(public.pulse_limits.supported_shapes)
        self.max_amp = public.pulse_limits.max_amp
        self.min_duration_ns = public.pulse_limits.min_duration_ns
        self.max_duration_ns = public.pulse_limits.max_duration_ns

    def run_pulse_sequence(
        self,
        request: PulseSequenceRequest,
        job_id: str,
        salt: int,
    ) -> JobResult:
        del salt
        error = self._validate_shots(request.shots) or self._validate_t1_sequence(
            request.sequence,
            allow_delay_placeholder=False,
        )
        if error is not None:
            return self._failed_request_with_artifact(
                job_id,
                request.shots,
                error,
                self._measured_qubits(request.sequence),
                request.model_dump(mode="json"),
                circuit_count=1,
            )

        compiled = _CompiledLiveRequest(
            circuits=[self._sequence_to_qiskit(request.sequence)],
            measured_qubits=self._measured_qubits(request.sequence),
            metadata=JobResultMetadata(
                sequence_duration_ns=_sequence_duration_ns(request.sequence)
            ),
            request_payload=request.model_dump(mode="json"),
        )
        return self._submit(compiled, request.shots, job_id)

    def run_sweep(
        self,
        request: SweepRequest,
        job_id: str,
        salt: int,
    ) -> JobResult:
        del salt
        error = self._validate_shots(request.shots) or self._validate_t1_sequence(
            request.template_sequence,
            allow_delay_placeholder=True,
        )
        if error is not None:
            return self._failed_request_with_artifact(
                job_id,
                request.shots,
                error,
                self._measured_qubits(request.template_sequence),
                request.model_dump(mode="json"),
            )
        if tuple(request.sweep) != (_DELAY_SWEEP_KEY,):
            return self._failed_request_with_artifact(
                job_id,
                request.shots,
                f"T1 live supports only {_DELAY_SWEEP_KEY!r} sweeps",
                self._measured_qubits(request.template_sequence),
                request.model_dump(mode="json"),
            )

        try:
            points = expand_sweep_points([_DELAY_SWEEP_KEY], request.sweep, request.mode)
        except ValueError as exc:
            return self._failed_request_with_artifact(
                job_id,
                request.shots,
                str(exc),
                self._measured_qubits(request.template_sequence),
                request.model_dump(mode="json"),
            )
        if not points:
            return self._failed_request_with_artifact(
                job_id,
                request.shots,
                "T1 live delay sweep requires at least one delay point",
                self._measured_qubits(request.template_sequence),
                request.model_dump(mode="json"),
            )
        if len(points) > _MAX_T1_LIVE_SWEEP_POINTS:
            return self._failed_request_with_artifact(
                job_id,
                request.shots,
                f"T1 live supports at most {_MAX_T1_LIVE_SWEEP_POINTS} sweep points",
                self._measured_qubits(request.template_sequence),
                request.model_dump(mode="json"),
            )

        delay_values = [float(point[_DELAY_SWEEP_KEY]) for point in points]
        if any(delay_ns <= 0 for delay_ns in delay_values):
            return self._failed_request_with_artifact(
                job_id,
                request.shots,
                "T1 live delay sweep values must be positive",
                self._measured_qubits(request.template_sequence),
                request.model_dump(mode="json"),
            )
        if len(set(delay_values)) < _MIN_T1_LIVE_DISTINCT_DELAYS:
            return self._failed_request_with_artifact(
                job_id,
                request.shots,
                "T1 live delay sweep requires at least "
                f"{_MIN_T1_LIVE_DISTINCT_DELAYS} distinct delay points",
                self._measured_qubits(request.template_sequence),
                request.model_dump(mode="json"),
            )

        coords = {_DELAY_SWEEP_KEY: delay_values}
        compiled = _CompiledLiveRequest(
            circuits=[
                self._sequence_to_qiskit(request.template_sequence, bindings=point)
                for point in points
            ],
            measured_qubits=self._measured_qubits(request.template_sequence),
            metadata=JobResultMetadata(sweep_coords=coords),
            request_payload=request.model_dump(mode="json"),
        )
        return self._submit(compiled, request.shots, job_id)

    def _validate_shots(self, shots: int) -> str | None:
        if self.max_shots is not None and shots > self.max_shots:
            return f"shots exceeds live-provider max_shots {self.max_shots}"
        return None

    def _validate_t1_sequence(
        self,
        sequence: Sequence[SequenceOp],
        *,
        allow_delay_placeholder: bool,
    ) -> str | None:
        if len(sequence) != 3:
            return _T1_SHAPE_ERROR

        pulse, delay, measure = sequence
        if not isinstance(pulse, PulseOp):
            return _T1_SHAPE_ERROR
        if not isinstance(delay, DelayOp):
            return _T1_SHAPE_ERROR
        if not isinstance(measure, MeasureOp):
            return _T1_SHAPE_ERROR

        if pulse.channel not in self.drive_channels:
            return f"T1 live pulse channel must be one of {sorted(self.drive_channels)!r}"
        if pulse.shape not in self.supported_shapes:
            return f"T1 live pulse shape {pulse.shape!r} is not supported"
        if not 0.0 <= pulse.amp <= self.max_amp:
            return f"T1 live pulse amp must be between 0.0 and {self.max_amp}"
        if not self.min_duration_ns <= pulse.duration_ns <= self.max_duration_ns:
            return (
                "T1 live pulse duration must be between "
                f"{self.min_duration_ns} and {self.max_duration_ns} ns"
            )
        if pulse.shape == "gaussian" and pulse.sigma_ns is None:
            return "T1 live gaussian X180 pulse requires sigma_ns"
        if pulse.phase_rad != 0.0:
            return "T1 live X180 pulse requires phase_rad=0.0"
        if pulse.freq_hz is not None:
            return "T1 live X180 pulse must not override freq_hz"
        if not _is_canonical_x180(pulse):
            return (
                "T1 live requires the canonical X180 pulse: "
                f"channel={_CANONICAL_X180_CHANNEL!r}, "
                f"shape={_CANONICAL_X180_SHAPE!r}, "
                f"amp={_CANONICAL_X180_AMP}, "
                f"duration_ns={_CANONICAL_X180_DURATION_NS}, "
                f"sigma_ns={_CANONICAL_X180_SIGMA_NS}"
            )

        if isinstance(delay.duration_ns, str):
            if not allow_delay_placeholder:
                return "T1 live pulse sequences require a concrete delay duration"
            if delay.duration_ns != f"${_DELAY_SWEEP_KEY}":
                return f"T1 live delay placeholder must be '${_DELAY_SWEEP_KEY}'"
        elif allow_delay_placeholder:
            return f"T1 live delay sweep requires '${_DELAY_SWEEP_KEY}' placeholder"
        elif delay.duration_ns <= 0:
            return "T1 live delay duration must be positive"

        if measure.qubits != list(self.qubits):
            return f"T1 live measure qubits must be {list(self.qubits)!r}"
        return None

    def _sequence_to_qiskit(
        self,
        sequence: Sequence[SequenceOp],
        bindings: dict[str, float] | None = None,
    ) -> QuantumCircuit:
        _, delay, _ = sequence
        delay_ns = _resolve_delay_ns(delay, bindings)
        if delay_ns <= 0:
            raise ValueError("T1 live delay duration must be positive")
        qc = QuantumCircuit(1, 1)
        qc.x(0)
        qc.delay(delay_ns, 0, unit="ns")
        qc.measure(0, 0)
        return qc

    def _measured_qubits(self, sequence: Sequence[SequenceOp]) -> list[int]:
        for op in sequence:
            if isinstance(op, MeasureOp) and op.qubits == list(self.qubits):
                return [0]
        return []


def _resolve_delay_ns(
    delay: SequenceOp,
    bindings: dict[str, float] | None,
) -> float:
    if not isinstance(delay, DelayOp):
        raise ValueError(_T1_SHAPE_ERROR)
    if isinstance(delay.duration_ns, str):
        key = delay.duration_ns.lstrip("$")
        if bindings is None or key not in bindings:
            raise ValueError(f"unbound delay placeholder {delay.duration_ns!r}")
        return float(bindings[key])
    return float(delay.duration_ns)


def _is_canonical_x180(pulse: PulseOp) -> bool:
    return (
        pulse.channel == _CANONICAL_X180_CHANNEL
        and pulse.shape == _CANONICAL_X180_SHAPE
        and _float_equal(pulse.amp, _CANONICAL_X180_AMP)
        and _float_equal(pulse.duration_ns, _CANONICAL_X180_DURATION_NS)
        and _float_equal(pulse.sigma_ns, _CANONICAL_X180_SIGMA_NS)
    )


def _float_equal(left: float | None, right: float) -> bool:
    return left is not None and abs(float(left) - right) <= _FLOAT_TOLERANCE


def _sequence_duration_ns(sequence: Sequence[SequenceOp]) -> float:
    duration = 0.0
    for op in sequence:
        if isinstance(op, PulseOp):
            duration += op.duration_ns
        elif isinstance(op, DelayOp) and not isinstance(op.duration_ns, str):
            duration += op.duration_ns
    return duration


def build_transmon_live_provider_backend(
    *,
    task_id: str,
    public: PublicTransmonSpec,
    backend_name: str,
    log_dir: str | Path | None = None,
    token: str | None = None,
    instance: str | None = None,
    channel: str,
    allow_live_provider: bool,
    max_shots: int | None = None,
    max_jobs: int = 1,
    backend_allowlist: Iterable[str] | None = None,
    wall_clock_s: float | None = None,
) -> TransmonLiveIBMBackend:
    """Construct the limited transmon IBM live-provider bridge."""

    return TransmonLiveIBMBackend(
        task_id=task_id,
        public=public,
        backend_name=backend_name,
        log_dir=log_dir,
        token=token,
        instance=instance,
        channel=channel,
        allow_live_provider=allow_live_provider,
        max_shots=max_shots,
        max_jobs=max_jobs,
        backend_allowlist=backend_allowlist,
        wall_clock_s=wall_clock_s,
    )


__all__ = ["TransmonLiveIBMBackend", "build_transmon_live_provider_backend"]
