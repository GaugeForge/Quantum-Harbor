"""In-process simulator backend for the Kitaev-chain qtype.

Validates the agent-visible request shape (fails closed before execution) and
delegates to the qtype-local runner. One backend per run; a fresh engine per
job is created inside the runner.
"""

from __future__ import annotations

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.kitaev_chain.device import (
    HiddenKitaevChainConfig,
    PublicKitaevChainSpec,
)
from qiqcbench.qsim.qtypes.kitaev_chain.runner import (
    fresh_run_entropy,
    run_charge_stability,
    run_clifford_rb,
    run_protection_sweep,
    run_pulse_batch,
    run_subchain_spectroscopy,
)
from qiqcbench.qsim.qtypes.kitaev_chain.wire import (
    ChargeStabilityRequest,
    MajoranaCliffordRBRequest,
    MajoranaPulseBatchRequest,
    ProtectionSweepRequest,
    SubchainSpectroscopyRequest,
)


class KitaevChainSimulatorBackend:
    def __init__(
        self,
        hidden: HiddenKitaevChainConfig,
        public: PublicKitaevChainSpec,
        *,
        run_entropy: int | None = None,
    ) -> None:
        if run_entropy is None:
            run_entropy = fresh_run_entropy()
        if isinstance(run_entropy, bool) or not isinstance(run_entropy, int) or run_entropy < 0:
            raise ValueError("run_entropy must be a non-negative integer")
        self.hidden = hidden
        self.public = public
        self._run_entropy = run_entropy
        self.n_sites = public.n_sites
        self.n_bonds = public.n_sites - 1
        self.mu_window = public.mu_window_ueV
        self.max_shots = public.max_shots

    @property
    def run_entropy(self) -> int:
        """Private per-attempt entropy for task-enabled stochastic actions."""

        return self._run_entropy

    def _failed(self, job_id: str, shots: int, error: str) -> JobResult:
        return JobResult(
            job_id=job_id,
            device_id=self.hidden.device_id,
            status="failed",
            shots=shots,
            error=error,
        )

    def run_charge_stability(
        self, request: ChargeStabilityRequest, job_id: str, salt: int
    ) -> JobResult:
        if request.shots > self.max_shots:
            return self._failed(job_id, request.shots, f"shots exceeds max_shots {self.max_shots}")
        if not (0 <= request.bond < self.n_bonds):
            return self._failed(job_id, request.shots, f"bond must be in [0, {self.n_bonds - 1}]")
        allv = list(request.mu_ld_values) + list(request.mu_rd_values)
        if any(abs(v) > self.mu_window for v in allv):
            return self._failed(
                job_id, request.shots, f"|mu| exceeds mu_window {self.mu_window} ueV"
            )
        return run_charge_stability(
            request,
            self.hidden,
            job_id,
            salt,
            run_entropy=self._run_entropy,
        )

    def run_protection_sweep(
        self, request: ProtectionSweepRequest, job_id: str, salt: int
    ) -> JobResult:
        if request.shots > self.max_shots:
            return self._failed(job_id, request.shots, f"shots exceeds max_shots {self.max_shots}")
        if any(abs(v) > self.mu_window for v in request.mu_common_values):
            return self._failed(
                job_id, request.shots, f"|mu| exceeds mu_window {self.mu_window} ueV"
            )
        return run_protection_sweep(
            request,
            self.hidden,
            job_id,
            salt,
            run_entropy=self._run_entropy,
        )

    def run_subchain_spectroscopy(
        self, request: SubchainSpectroscopyRequest, job_id: str, salt: int
    ) -> JobResult:
        if request.shots > self.max_shots:
            return self._failed(job_id, request.shots, f"shots exceeds max_shots {self.max_shots}")
        # site_a == site_b is a ONE-SITE window: it contains no bond, so its BdG
        # block has a single positive eigenvalue and there is no bulk excitation to
        # gap to (physics.subchain_bulk_gap returns +inf for it, which no job result
        # can publish). Refuse it here, before the job runs, so the model owns a
        # request it got wrong instead of collecting a qsim-stamped poll failure.
        if not (0 <= request.site_a < request.site_b < self.n_sites):
            return self._failed(
                job_id,
                request.shots,
                f"require 0 <= site_a < site_b < {self.n_sites}: the window must span at "
                "least one bond -- a one-site window has no bulk excitation, so it has "
                "no bulk gap to measure",
            )
        return run_subchain_spectroscopy(
            request,
            self.hidden,
            job_id,
            salt,
            run_entropy=self._run_entropy,
        )

    def run_pulse_batch(
        self, request: MajoranaPulseBatchRequest, job_id: str, salt: int
    ) -> JobResult:
        if request.shots > self.max_shots:
            return self._failed(job_id, request.shots, f"shots exceeds max_shots {self.max_shots}")
        if self.hidden.pulse_drive_rate_ueV is None:
            return self._failed(job_id, request.shots, "device has no pulse-gate parameters")
        if request.total_segments > 30_000:
            return self._failed(job_id, request.shots, "total pulse segments exceeds 30000")
        return run_pulse_batch(
            request,
            self.hidden,
            job_id,
            salt,
            run_entropy=self._run_entropy,
        )

    def run_clifford_rb(
        self, request: MajoranaCliffordRBRequest, job_id: str, salt: int
    ) -> JobResult:
        if self.hidden.pulse_drive_rate_ueV is None:
            return self._failed(
                job_id, request.shots_per_sequence, "device has no pulse-gate parameters"
            )
        # Bound the worst-case number of 3x3 segment propagations before qsim
        # samples the private Cliffords.  The canonical compiler has a small,
        # fixed maximum word length; using the upper bound keeps admission
        # independent of the random draw.
        from qiqcbench.qsim.qtypes.kitaev_chain.clifford_rb import clifford_compilations

        max_word = max(len(word) for word in clifford_compilations())
        gate_segments = max(len(request.gate_x90_segments), len(request.gate_z90_segments))
        segment_shot_upper_bound = (
            sum(length + 1 for length in request.lengths)
            * request.sequences_per_length
            * request.shots_per_sequence
            * max_word
            * gate_segments
        )
        if segment_shot_upper_bound > 25_000_000:
            return self._failed(
                job_id,
                request.shots_per_sequence,
                "Clifford RB request exceeds the 25000000 segment-shot evolution budget",
            )
        return run_clifford_rb(
            request,
            self.hidden,
            job_id,
            salt,
            run_entropy=self._run_entropy,
        )


def build_kitaev_chain_simulator_backend(
    hidden: HiddenKitaevChainConfig, public: PublicKitaevChainSpec
) -> KitaevChainSimulatorBackend:
    return KitaevChainSimulatorBackend(hidden, public)


__all__ = ["KitaevChainSimulatorBackend", "build_kitaev_chain_simulator_backend"]
