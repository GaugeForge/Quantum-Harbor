"""Gmon-ring simulator backend adapter.

Wraps the qtype-local runner and fails closed on agent-visible control limits
before any (expensive) simulation runs. Exposes gmon-named methods
(``run_gmon_sequence``/``run_gmon_sweep``); the gmon MCP actions call these
directly on ``state.backend`` rather than through the pulse/circuit registry
slots (controller decision — the gmon control surface is distinct).
"""

from __future__ import annotations

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.gmon_ring_3q.device import (
    HiddenGmonRingConfig,
    PublicGmonRingSpec,
)
from qiqcbench.qsim.qtypes.gmon_ring_3q.runner import run_gmon_sequence, run_gmon_sweep
from qiqcbench.qsim.qtypes.gmon_ring_3q.wire import (
    GmonDelayOp,
    GmonEvolveOp,
    GmonMeasureOp,
    GmonRotationOp,
    GmonSequenceOp,
    GmonSequenceRequest,
    GmonSweepRequest,
)


class GmonSimulatorBackend:
    """Gmon-ring backend backed by the in-process joint-qutrit simulator."""

    def __init__(self, hidden: HiddenGmonRingConfig) -> None:
        self.hidden = hidden
        self.max_shots: int | None = None
        self.qubit_ids: set[str] | None = None
        self.coupler_ids: set[str] | None = None
        self.supported_axes: set[str] | None = None
        self.max_amp: float | None = None
        self.max_mod_freq_hz: float | None = None
        self.max_sequence_ns: float | None = None
        self.supported_envelopes: set[str] | None = None

    @classmethod
    def from_public(
        cls,
        hidden: HiddenGmonRingConfig,
        public: PublicGmonRingSpec,
    ) -> GmonSimulatorBackend:
        backend = cls(hidden)
        limits = public.control_limits
        backend.max_shots = public.max_shots
        backend.qubit_ids = {q.id for q in public.qubits}
        backend.coupler_ids = {c.id for c in public.couplers}
        backend.supported_axes = set(limits.supported_rotation_axes)
        backend.max_amp = limits.max_amp
        backend.max_mod_freq_hz = limits.max_modulation_freq_hz
        backend.max_sequence_ns = limits.max_sequence_ns
        backend.supported_envelopes = set(limits.supported_envelopes)
        return backend

    def _failed(self, job_id: str, shots: int, error: str) -> JobResult:
        return JobResult(
            job_id=job_id,
            device_id=self.hidden.device_id,
            status="failed",
            shots=shots,
            error=error,
        )

    def _validate(
        self, sequence: list[GmonSequenceOp], shots: int, *, allow_placeholders: bool
    ) -> str | None:
        if self.max_shots is not None and shots > self.max_shots:
            return f"gmon shots exceeds max_shots {self.max_shots}"

        def num(value: float | str, what: str) -> tuple[float | None, str | None]:
            if isinstance(value, str):
                if allow_placeholders:
                    return None, None
                return None, f"unresolved sweep placeholder in {what}: {value!r}"
            return float(value), None

        has_measure = False
        for op in sequence:
            if isinstance(op, GmonRotationOp):
                if self.qubit_ids is not None and op.qubit not in self.qubit_ids:
                    return f"gmon rotation qubit {op.qubit!r} is not on this device"
                if self.supported_axes is not None and op.axis not in self.supported_axes:
                    return f"gmon rotation axis {op.axis!r} is not supported"
            elif isinstance(op, GmonEvolveOp):
                if (
                    self.supported_envelopes is not None
                    and op.envelope not in self.supported_envelopes
                ):
                    return f"gmon evolve envelope {op.envelope!r} is not supported"
                dur, err = num(op.duration_ns, "evolve duration")
                if err:
                    return err
                if (
                    dur is not None
                    and self.max_sequence_ns is not None
                    and not 0 < dur <= self.max_sequence_ns
                ):
                    return f"gmon evolve duration must be in (0, {self.max_sequence_ns}] ns"
                for c in op.couplers:
                    if self.coupler_ids is not None and c.coupler not in self.coupler_ids:
                        return f"gmon coupler {c.coupler!r} is not on this device"
                    amp, err = num(c.amp, "coupler amp")
                    if err:
                        return err
                    if (
                        amp is not None
                        and self.max_amp is not None
                        and not 0.0 <= amp <= self.max_amp
                    ):
                        return f"gmon coupler amp must be in [0, {self.max_amp}]"
                    freq, err = num(c.freq_hz, "coupler freq_hz")
                    if err:
                        return err
                    if (
                        freq is not None
                        and self.max_mod_freq_hz is not None
                        and not 0.0 <= freq <= self.max_mod_freq_hz
                    ):
                        return f"gmon coupler freq_hz must be in [0, {self.max_mod_freq_hz}]"
            elif isinstance(op, GmonDelayOp):
                dur, err = num(op.duration_ns, "delay duration")
                if err:
                    return err
                if dur is not None and dur <= 0:
                    return "gmon delay duration must be positive"
            elif isinstance(op, GmonMeasureOp):
                has_measure = True
                if self.qubit_ids is not None:
                    unknown = [q for q in op.qubits if q not in self.qubit_ids]
                    if unknown:
                        return f"gmon measure qubits {unknown!r} are not on this device"
        if not has_measure:
            return "gmon sequence must contain a gmon_measure op"
        return None

    def run_gmon_sequence(self, request: GmonSequenceRequest, job_id: str, salt: int) -> JobResult:
        if error := self._validate(request.sequence, request.shots, allow_placeholders=False):
            return self._failed(job_id, request.shots, error)
        return run_gmon_sequence(request, self.hidden, job_id, salt)

    def run_gmon_sweep(self, request: GmonSweepRequest, job_id: str, salt: int) -> JobResult:
        if error := self._validate(
            request.template_sequence, request.shots, allow_placeholders=True
        ):
            return self._failed(job_id, request.shots, error)
        return run_gmon_sweep(request, self.hidden, job_id, salt)


def build_gmon_simulator_backend(
    hidden: HiddenGmonRingConfig,
    public: PublicGmonRingSpec,
) -> GmonSimulatorBackend:
    """Construct the in-process gmon-ring simulator backend."""
    return GmonSimulatorBackend.from_public(hidden, public)


__all__ = ["GmonSimulatorBackend", "build_gmon_simulator_backend"]
