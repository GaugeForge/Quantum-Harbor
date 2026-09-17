"""In-process simulator backend for the g-f erasure-qutrit qtype.

Validates the agent-visible request shape against the public control envelope
(fails closed before execution) and delegates to the qtype-local runner.
"""

from __future__ import annotations

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.transmon_erasure_qutrit.device import (
    HiddenErasureQutritConfig,
    PublicErasureQutritSpec,
)
from qiqcbench.qsim.qtypes.transmon_erasure_qutrit.model import cadence_grid_us
from qiqcbench.qsim.qtypes.transmon_erasure_qutrit.runner import (
    fresh_run_entropy,
    run_logical_memory,
    run_logical_memory_sweep,
)
from qiqcbench.qsim.qtypes.transmon_erasure_qutrit.wire import (
    LogicalMemoryRequest,
    LogicalMemorySweepRequest,
)

# Allowed matched-basis preparation states.
_Z_STATES = {"0L", "1L"}
_X_STATES = {"+X", "-X"}


class ErasureQutritSimulatorBackend:
    """g-f erasure-qutrit backend backed by the in-process Monte Carlo simulator."""

    def __init__(
        self,
        hidden: HiddenErasureQutritConfig,
        public: PublicErasureQutritSpec,
        *,
        run_entropy: int | None = None,
    ) -> None:
        self.hidden = hidden
        self.public = public
        if run_entropy is None:
            run_entropy = fresh_run_entropy()
        if isinstance(run_entropy, bool) or not isinstance(run_entropy, int) or run_entropy < 0:
            raise ValueError("run_entropy must be a non-negative integer")
        self._run_entropy = run_entropy

    @property
    def run_entropy(self) -> int:
        """Private per-attempt entropy; never serialize it into public evidence."""

        return self._run_entropy

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
        prep_basis: str,
        prep_state: str,
        measure_basis: str,
        n_rounds_grid: list[int],
        cycle_time_us: float,
        dd: str,
        shots: int,
    ) -> str | None:
        p = self.public
        if shots > p.max_shots:
            return f"shots exceeds max_shots {p.max_shots}"
        if prep_basis == "z" and prep_state not in _Z_STATES:
            return "prep_state must be '0L' or '1L' for prep_basis 'z'"
        if prep_basis == "x" and prep_state not in _X_STATES:
            return "prep_state must be '+X' or '-X' for prep_basis 'x'"
        if prep_basis not in ("z", "x"):
            return "prep_basis must be 'z' or 'x'"
        if measure_basis not in ("z", "x"):
            return "measure_basis must be 'z' or 'x'"
        if measure_basis != prep_basis:
            return "measure_basis must match prep_basis"
        if dd not in p.dd_options:
            return f"dd must be one of {p.dd_options!r}"
        if cycle_time_us < p.min_cycle_time_us - 1e-9:
            return f"cycle_time_us below usable minimum {p.min_cycle_time_us} us"
        if cycle_time_us > p.max_cycle_time_us + 1e-9:
            return f"cycle_time_us exceeds max {p.max_cycle_time_us} us"
        cadence_grid = cadence_grid_us(
            minimum_us=p.min_cycle_time_us,
            maximum_us=p.max_cycle_time_us,
            resolution_us=p.cycle_time_resolution_us,
        )
        nearest = min(cadence_grid, key=lambda value: abs(value - cycle_time_us))
        if abs(nearest - cycle_time_us) > 1e-9:
            return (
                "cycle_time_us must lie on the public cadence grid with resolution "
                f"{p.cycle_time_resolution_us} us"
            )
        if not n_rounds_grid:
            return "n_rounds grid must be non-empty"
        if any(n < 0 for n in n_rounds_grid):
            return "n_rounds must be non-negative"
        if any(n > p.max_rounds for n in n_rounds_grid):
            return f"n_rounds exceeds max_rounds {p.max_rounds}"
        if len(n_rounds_grid) > p.budget.max_sweep_points_per_call:
            return (
                f"sweep has {len(n_rounds_grid)} points > max_sweep_points_per_call "
                f"{p.budget.max_sweep_points_per_call}"
            )
        final_assignments = shots * len(n_rounds_grid)
        if final_assignments > p.budget.max_final_assignments_per_call:
            return (
                f"request records {final_assignments} final assignments > "
                "max_final_assignments_per_call "
                f"{p.budget.max_final_assignments_per_call}"
            )
        syndrome_bits = shots * sum(n_rounds_grid)
        if syndrome_bits > p.budget.max_syndrome_bits_per_call:
            return (
                f"request records {syndrome_bits} syndrome bits > "
                f"max_syndrome_bits_per_call {p.budget.max_syndrome_bits_per_call}"
            )
        max_total = max(n_rounds_grid) * cycle_time_us
        if max_total > p.max_total_evolution_us + 1e-6:
            return f"total evolution {max_total:.2f} us exceeds max {p.max_total_evolution_us} us"
        return None

    def run_logical_memory(
        self, request: LogicalMemoryRequest, job_id: str, salt: int
    ) -> JobResult:
        error = self._validate(
            prep_basis=request.prep_basis,
            prep_state=request.prep_state,
            measure_basis=request.measure_basis,
            n_rounds_grid=[request.n_rounds],
            cycle_time_us=request.cycle_time_us,
            dd=request.dd,
            shots=request.shots,
        )
        if error:
            return self._failed(job_id, request.shots, error)
        return run_logical_memory(
            request,
            self.hidden,
            job_id,
            salt,
            data_qubit=self.public.data_qubit,
            run_entropy=self._run_entropy,
        )

    def run_logical_memory_sweep(
        self, request: LogicalMemorySweepRequest, job_id: str, salt: int
    ) -> JobResult:
        error = self._validate(
            prep_basis=request.prep_basis,
            prep_state=request.prep_state,
            measure_basis=request.measure_basis,
            n_rounds_grid=[int(n) for n in request.n_rounds_grid],
            cycle_time_us=request.cycle_time_us,
            dd=request.dd,
            shots=request.shots,
        )
        if error:
            return self._failed(job_id, request.shots, error)
        return run_logical_memory_sweep(
            request,
            self.hidden,
            job_id,
            salt,
            data_qubit=self.public.data_qubit,
            run_entropy=self._run_entropy,
        )


def build_erasure_qutrit_simulator_backend(
    hidden: HiddenErasureQutritConfig,
    public: PublicErasureQutritSpec,
    *,
    run_entropy: int | None = None,
) -> ErasureQutritSimulatorBackend:
    """Construct the in-process g-f erasure-qutrit simulator backend."""
    return ErasureQutritSimulatorBackend(hidden, public, run_entropy=run_entropy)


__all__ = [
    "ErasureQutritSimulatorBackend",
    "build_erasure_qutrit_simulator_backend",
]
