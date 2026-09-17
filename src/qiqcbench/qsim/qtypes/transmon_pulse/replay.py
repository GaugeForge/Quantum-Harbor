"""Transmon qtype adapter for provider replay backends."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel

from qiqcbench.qsim.backends.provider_artifacts import ProviderArtifactV1
from qiqcbench.qsim.backends.replay import ReplayFixtureBackend
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

_T1_SHAPE_ERROR = "T1 replay supports exactly: X180 pulse, delay, measure"
_DELAY_SWEEP_KEY = "delay_ns"


class TransmonReplayBackend(ReplayFixtureBackend):
    """Provider replay backend for the transmon T1 protocol subset."""

    def __init__(
        self,
        *,
        task_id: str,
        public: PublicTransmonSpec,
        replay_root: str | Path,
        log_dir: str | Path | None = None,
    ) -> None:
        super().__init__(
            task_id=task_id,
            device_id=public.device_id,
            replay_root=replay_root,
            log_dir=log_dir,
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
        error = self._validate_t1_sequence(request.sequence, allow_delay_placeholder=False)
        if error is not None:
            return self._failed(job_id, request.shots, error)

        metadata = JobResultMetadata(sequence_duration_ns=_sequence_duration_ns(request.sequence))
        result = self._replay_request(request, job_id, metadata=metadata)
        return self._validate_replayed_shape(result, job_id, request.shots, expected_points=1)

    def run_sweep(
        self,
        request: SweepRequest,
        job_id: str,
        salt: int,
    ) -> JobResult:
        error = self._validate_t1_sequence(
            request.template_sequence,
            allow_delay_placeholder=True,
        )
        if error is not None:
            return self._failed(job_id, request.shots, error)
        if tuple(request.sweep) != (_DELAY_SWEEP_KEY,):
            return self._failed(
                job_id,
                request.shots,
                f"T1 replay supports only {_DELAY_SWEEP_KEY!r} sweeps",
            )

        try:
            points = expand_sweep_points([_DELAY_SWEEP_KEY], request.sweep, request.mode)
        except ValueError as exc:
            return self._failed(job_id, request.shots, str(exc))

        coords = {_DELAY_SWEEP_KEY: [float(point[_DELAY_SWEEP_KEY]) for point in points]}
        metadata = JobResultMetadata(sweep_coords=coords)
        result = self._replay_request(request, job_id, metadata=metadata)
        return self._validate_replayed_shape(
            result,
            job_id,
            request.shots,
            expected_points=len(points),
        )

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
            return f"T1 replay pulse channel must be one of {sorted(self.drive_channels)!r}"
        if pulse.shape not in self.supported_shapes:
            return f"T1 replay pulse shape {pulse.shape!r} is not supported"
        if not 0.0 <= pulse.amp <= self.max_amp:
            return f"T1 replay pulse amp must be between 0.0 and {self.max_amp}"
        if not self.min_duration_ns <= pulse.duration_ns <= self.max_duration_ns:
            return (
                "T1 replay pulse duration must be between "
                f"{self.min_duration_ns} and {self.max_duration_ns} ns"
            )
        if pulse.shape == "gaussian" and pulse.sigma_ns is None:
            return "T1 replay gaussian X180 pulse requires sigma_ns"

        if isinstance(delay.duration_ns, str):
            if not allow_delay_placeholder:
                return "T1 replay pulse sequences require a concrete delay duration"
            if delay.duration_ns != f"${_DELAY_SWEEP_KEY}":
                return f"T1 replay delay placeholder must be '${_DELAY_SWEEP_KEY}'"
        elif delay.duration_ns < 0:
            return "T1 replay delay duration must be non-negative"

        if measure.qubits != list(self.qubits):
            return f"T1 replay measure qubits must be {list(self.qubits)!r}"
        return None

    def _validate_artifact(
        self,
        artifact: ProviderArtifactV1,
        request: BaseModel,
        shots: int,
    ) -> str | None:
        error = super()._validate_artifact(artifact, request, shots)
        if error is not None or artifact.failure is not None:
            return error

        metadata = artifact.usage_metadata or {}
        if isinstance(request, SweepRequest):
            if metadata.get("protocol") != "t1_delay_sweep":
                return "T1 replay fixture usage_metadata.protocol must be 't1_delay_sweep'"
            delay_ns = metadata.get(_DELAY_SWEEP_KEY)
            expected_delay_ns = [float(value) for value in request.sweep[_DELAY_SWEEP_KEY]]
            if delay_ns != expected_delay_ns:
                return "T1 replay fixture delay_ns metadata does not match request sweep"
        elif isinstance(request, PulseSequenceRequest):
            if metadata.get("protocol") != "t1_delay_sequence":
                return "T1 replay fixture usage_metadata.protocol must be 't1_delay_sequence'"
        return None

    def _validate_replayed_shape(
        self,
        result: JobResult,
        job_id: str,
        shots: int,
        *,
        expected_points: int,
    ) -> JobResult:
        if result.status != "complete":
            return result
        if result.data is None:
            return self._failed(job_id, shots, "replay fixture did not return data")
        if result.data.measured_qubits != [0]:
            return self._failed(job_id, shots, "T1 replay fixture must measure qubit 0")
        if len(result.data.bitstrings) != expected_points:
            return self._failed(
                job_id,
                shots,
                f"T1 replay fixture returned {len(result.data.bitstrings)} point(s); "
                f"expected {expected_points}",
            )
        if any(len(point) != shots for point in result.data.bitstrings):
            return self._failed(job_id, shots, "T1 replay fixture shot count mismatch")
        if any(bit not in {"0", "1"} for point in result.data.bitstrings for bit in point):
            return self._failed(
                job_id,
                shots,
                "T1 replay fixture must contain single-qubit bitstrings '0' or '1'",
            )
        return result


def _sequence_duration_ns(sequence: Sequence[SequenceOp]) -> float:
    duration = 0.0
    for op in sequence:
        if isinstance(op, PulseOp):
            duration += op.duration_ns
        elif isinstance(op, DelayOp) and not isinstance(op.duration_ns, str):
            duration += op.duration_ns
    return duration


def build_transmon_replay_backend(
    *,
    task_id: str,
    public: PublicTransmonSpec,
    replay_root: str | Path,
    log_dir: str | Path | None = None,
) -> TransmonReplayBackend:
    """Construct a transmon provider-replay backend from qtype-local data."""

    return TransmonReplayBackend(
        task_id=task_id,
        public=public,
        replay_root=replay_root,
        log_dir=log_dir,
    )


__all__ = ["TransmonReplayBackend", "build_transmon_replay_backend"]
