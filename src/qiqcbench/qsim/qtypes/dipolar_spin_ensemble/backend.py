"""In-process simulator backend for the dipolar-spin-ensemble qtype.

Validates the agent-visible request shape (fails closed before execution) and
delegates to the qtype-local runner.
"""

from __future__ import annotations

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.dipolar_spin_ensemble.device import (
    HiddenDipolarConfig,
    PublicDipolarSpec,
)
from qiqcbench.qsim.qtypes.dipolar_spin_ensemble.runner import run_pulse_train
from qiqcbench.qsim.qtypes.dipolar_spin_ensemble.wire import DipolarSequenceRequest


class DipolarSimulatorBackend:
    """Dipolar-spin-ensemble backend backed by the in-process simulator."""

    def __init__(self, hidden: HiddenDipolarConfig, public: PublicDipolarSpec) -> None:
        self.hidden = hidden
        self.public = public
        self.max_shots = public.max_shots
        self.max_sequence_length_ns = public.max_sequence_length_ns
        self.control_axes = set(public.control_axes)

    def _failed(self, job_id: str, shots: int, error: str) -> JobResult:
        return JobResult(
            job_id=job_id,
            device_id=self.hidden.device_id,
            status="failed",
            shots=shots,
            error=error,
        )

    # Cap the raw per-shot payload so a fine stroboscopic sweep cannot return a
    # multi-megabyte result that overflows the agent's context / the model API.
    MAX_RETURNED_OUTCOMES = 120_000

    # ``MAX_RETURNED_OUTCOMES`` bounds the returned payload, not the work needed
    # to produce it. Engine cost is one propagator build per sequence op plus one
    # stroboscopic propagation per cycle, repeated over every hidden disorder
    # realization, so ``len(sequence) + n_cycles`` tracks simulator time (an op
    # costs roughly twice a cycle). Neither driver is bounded by the payload cap:
    # ``shots=1, n_cycles=4096`` returns 4097 outcomes, and ``sequence`` has no
    # length bound in the wire schema at all, so a two-outcome request can hold a
    # worker for minutes. With only two worker slots that starves every job
    # behind it, so bound the work at admission rather than the bytes alone.
    #
    # ``max_sequence_length_ns`` already binds tighter than this for any sequence
    # with physically meaningful free windows; this only rejects the degenerate
    # near-zero-window and absurd-sequence-length corners. Calibrated against
    # every recorded run of this task: 1,909 submitted pulse trains, of which
    # 99.8% are under 500 work units and the largest ever is 1,208.
    MAX_SIMULATION_WORK_UNITS = 2_000

    def _validate(self, request: DipolarSequenceRequest) -> str | None:
        if request.shots > self.max_shots:
            return f"shots exceeds max_shots {self.max_shots}"
        n_outcomes = request.shots * (request.n_cycles + 1) * len(request.measure_axes)
        if n_outcomes > self.MAX_RETURNED_OUTCOMES:
            return (
                f"requested result is too large: shots*(n_cycles+1)*len(measure_axes) = "
                f"{n_outcomes} > {self.MAX_RETURNED_OUTCOMES}. Reduce shots, n_cycles, or "
                f"measure_axes (e.g. average a coarser stroboscopic grid)."
            )
        work_units = len(request.sequence) + request.n_cycles
        if work_units > self.MAX_SIMULATION_WORK_UNITS:
            return (
                f"requested simulation is too expensive: len(sequence)+n_cycles = "
                f"{work_units} > {self.MAX_SIMULATION_WORK_UNITS}. Simulator cost scales "
                f"with the number of sequence ops and cycles (not with the size of the "
                f"returned data), so shrink the base block or run fewer cycles."
            )
        t_cycle = sum(op.duration_ns for op in request.sequence if op.kind == "free")
        if t_cycle * request.n_cycles > self.max_sequence_length_ns:
            return (
                f"total sequence length {t_cycle * request.n_cycles} ns exceeds "
                f"max_sequence_length_ns {self.max_sequence_length_ns}"
            )
        for op in request.sequence:
            if op.kind == "pulse" and op.axis not in self.control_axes:
                return f"pulse axis {op.axis!r} not in device control axes {sorted(self.control_axes)!r}"
        return None

    def run_pulse_train(self, request: DipolarSequenceRequest, job_id: str, salt: int) -> JobResult:
        error = self._validate(request)
        if error:
            return self._failed(job_id, request.shots, error)
        return run_pulse_train(request, self.hidden, job_id, salt)


def build_dipolar_simulator_backend(
    hidden: HiddenDipolarConfig,
    public: PublicDipolarSpec,
) -> DipolarSimulatorBackend:
    """Construct the in-process dipolar-spin-ensemble simulator backend."""
    return DipolarSimulatorBackend(hidden, public)


__all__ = ["DipolarSimulatorBackend", "build_dipolar_simulator_backend"]
