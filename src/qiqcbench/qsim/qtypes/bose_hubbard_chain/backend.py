"""In-process simulator backend for the Bose-Hubbard chain qtype.

Validates the agent-visible request shape (fails closed before execution) and
delegates to the qtype-local runner.
"""

from __future__ import annotations

from qiqcbench.qsim.core.wire import (
    BoseHubbardEvolveRequest,
    BoseHubbardEvolveSweepRequest,
    JobResult,
)
from qiqcbench.qsim.qtypes.bose_hubbard_chain.device import (
    HiddenBoseHubbardConfig,
    PublicBoseHubbardSpec,
)
from qiqcbench.qsim.qtypes.bose_hubbard_chain.runner import (
    fresh_run_entropy,
    run_evolution,
    run_evolution_sweep,
)


class BoseHubbardSimulatorBackend:
    """Bose-Hubbard chain backend backed by the in-process analog simulator."""

    def __init__(
        self,
        hidden: HiddenBoseHubbardConfig,
        public: PublicBoseHubbardSpec,
        *,
        run_entropy: int | None = None,
    ) -> None:
        if run_entropy is None:
            run_entropy = fresh_run_entropy()
        if (
            isinstance(run_entropy, bool)
            or not isinstance(run_entropy, int)
            or not 0 <= run_entropy < 2**128
        ):
            raise ValueError("run_entropy must be a 128-bit non-negative integer")
        self.hidden = hidden
        self.public = public
        self._run_entropy = run_entropy
        self.site_ids = {s.id for s in public.sites}
        self.max_shots = public.max_shots
        self.max_excitations = public.max_excitations
        self.max_evolution_time_ns = public.max_evolution_time_ns

    @property
    def run_entropy(self) -> int:
        """Private per-attempt entropy; qsim-internal and never serialized."""

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
        init_sites: list[int],
        measure_sites: list[int],
        times: list[float],
        shots: int,
    ) -> str | None:
        if shots > self.max_shots:
            return f"shots exceeds max_shots {self.max_shots}"
        if len(init_sites) != len(set(init_sites)):
            return "init_excited_sites must be distinct"
        if len(init_sites) > self.max_excitations:
            return f"at most {self.max_excitations} excited sites may be prepared"
        if any(s not in self.site_ids for s in init_sites):
            return f"init_excited_sites must be among {sorted(self.site_ids)!r}"
        if len(measure_sites) != len(set(measure_sites)):
            return "measure sites must be distinct"
        if any(s not in self.site_ids for s in measure_sites):
            return f"measure sites must be among {sorted(self.site_ids)!r}"
        if any(t < 0 or t > self.max_evolution_time_ns for t in times):
            return f"evolution time must be in [0, {self.max_evolution_time_ns}] ns"
        return None

    def run_evolution(self, request: BoseHubbardEvolveRequest, job_id: str, salt: int) -> JobResult:
        error = self._validate(
            init_sites=list(request.init_excited_sites),
            measure_sites=[m.site for m in request.measure],
            times=[request.evolution_time_ns],
            shots=request.shots,
        )
        if error:
            return self._failed(job_id, request.shots, error)
        return run_evolution(
            request,
            self.hidden,
            self.public,
            job_id,
            salt,
            run_entropy=self._run_entropy,
        )

    def run_evolution_sweep(
        self, request: BoseHubbardEvolveSweepRequest, job_id: str, salt: int
    ) -> JobResult:
        error = self._validate(
            init_sites=list(request.init_excited_sites),
            measure_sites=[m.site for m in request.measure],
            times=list(request.time_grid_ns),
            shots=request.shots,
        )
        if error:
            return self._failed(job_id, request.shots, error)
        return run_evolution_sweep(
            request,
            self.hidden,
            self.public,
            job_id,
            salt,
            run_entropy=self._run_entropy,
        )


def build_bose_hubbard_simulator_backend(
    hidden: HiddenBoseHubbardConfig,
    public: PublicBoseHubbardSpec,
    *,
    run_entropy: int | None = None,
) -> BoseHubbardSimulatorBackend:
    """Construct the in-process Bose-Hubbard simulator backend."""
    return BoseHubbardSimulatorBackend(hidden, public, run_entropy=run_entropy)


__all__ = ["BoseHubbardSimulatorBackend", "build_bose_hubbard_simulator_backend"]
