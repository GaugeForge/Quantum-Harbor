"""Internal backend protocols for qsim task runners."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from qiqcbench.qsim.core.wire import (
    BoseHubbardEvolveRequest,
    BoseHubbardEvolveSweepRequest,
    CircuitRequest,
    CircuitSweepRequest,
    JobResult,
    PulseSequenceRequest,
    SweepRequest,
)


class PulseBackend(Protocol):
    """Backend capable of executing pulse-level transmon requests."""

    def run_pulse_sequence(
        self,
        request: PulseSequenceRequest,
        job_id: str,
        salt: int,
    ) -> JobResult:
        """Execute one pulse sequence request."""
        ...

    def run_sweep(
        self,
        request: SweepRequest,
        job_id: str,
        salt: int,
    ) -> JobResult:
        """Execute one pulse sweep request."""
        ...


class CircuitBackend(Protocol):
    """Backend capable of executing digital circuit requests."""

    def run_circuit(
        self,
        request: CircuitRequest,
        job_id: str,
        salt: int,
    ) -> JobResult:
        """Execute one concrete circuit request."""
        ...

    def run_circuit_sweep(
        self,
        request: CircuitSweepRequest,
        job_id: str,
        salt: int,
    ) -> JobResult:
        """Execute one circuit sweep request."""
        ...


class AnalogBackend(Protocol):
    """Backend capable of executing analog evolve-and-measure requests."""

    def run_evolution(
        self,
        request: BoseHubbardEvolveRequest,
        job_id: str,
        salt: int,
    ) -> JobResult:
        """Execute one prepare-evolve-measure request."""
        ...

    def run_evolution_sweep(
        self,
        request: BoseHubbardEvolveSweepRequest,
        job_id: str,
        salt: int,
    ) -> JobResult:
        """Execute one evolve-and-measure sweep over a time grid."""
        ...


@runtime_checkable
class PollingCircuitBackend(CircuitBackend, Protocol):
    """Circuit backend whose external provider jobs can be polled."""

    def poll_job_result(self, job_id: str, current: JobResult) -> JobResult | None:
        """Return an updated result, or None if the current result is unchanged."""
        ...
