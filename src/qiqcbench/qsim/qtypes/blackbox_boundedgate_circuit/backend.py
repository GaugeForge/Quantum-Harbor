"""Simulator backend for the blackbox bounded-gate qtype.

Like the analog/lindblad backends (and unlike the per-job-stateless
digital/transmon backends), this backend holds **one** run-long
:class:`BoundedgateEngine`, so the cumulative shot/job budget accumulates
across jobs. Shot randomness is derived PER JOB by hashing
``(run_entropy, salt)`` through a domain-separated SHA-256 (a real KDF step:
one job's stream reveals nothing about another's derivation, unlike
SeedSequence's statistical mixing), so the mapping from an admitted job to its
stream is independent of the scheduling order of concurrent jobs (the engine
never holds a shared mutable RNG). PCG64 output itself is not
cryptographically hiding, but the agent only ever observes measurement
bitstrings drawn through it.

``run_entropy`` is the attempt's private sampling root. A production backend
draws it once from the OS CSPRNG at construction and never logs or returns
it, so two benchmark attempts on the same edition draw independent
measurement realizations. Certification, regression tests, and
certificate replay inject it explicitly (conventionally the edition's derived
shot seed) to reproduce exact streams.

Replay / live-provider modes are out of scope (simulator-only qtype).
"""

from __future__ import annotations

import hashlib
import secrets

import numpy as np

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.device import (
    HiddenBoundedgateConfig,
    PublicBoundedgateSpec,
)
from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.engine import BoundedgateEngine
from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.wire import (
    BasisShotRequest,
    SealedTargetChallengeReveal,
)

__all__ = ["BoundedgateSimulatorBackend", "build_boundedgate_simulator_backend"]


def _fresh_run_entropy() -> int:
    """Draw the attempt's private 256-bit sampling root from the OS CSPRNG."""
    return secrets.randbits(256)


class BoundedgateSimulatorBackend:
    """Holds the single run-long bounded-gate engine and answers shot jobs."""

    def __init__(
        self,
        hidden: HiddenBoundedgateConfig,
        public: PublicBoundedgateSpec,
        *,
        run_entropy: int | None = None,
    ):
        if run_entropy is None:
            run_entropy = _fresh_run_entropy()
        if (
            isinstance(run_entropy, bool)
            or not isinstance(run_entropy, int)
            or not 0 <= run_entropy < 2**256
        ):
            raise ValueError("run_entropy must be a non-negative integer below 2**256")
        self._device_id = hidden.device_id
        self._run_entropy = run_entropy
        self._engine = BoundedgateEngine(hidden, public)

    @property
    def engine(self) -> BoundedgateEngine:
        return self._engine

    @property
    def transaction_guard(self):
        """Admission/seal guard held through the corresponding evidence append."""
        return self._engine.transaction_guard

    def reserve_basis_shots(self, request: BasisShotRequest) -> int:
        """Fail-closed admission: raises (consuming nothing) on any violation.

        Returns this job's usage clock (shots consumed before it), which the
        caller must hand back to ``run_basis_shots`` so a drifting device stays
        deterministic under concurrent jobs.
        """
        return self._engine.reserve(request)

    def reserve_basis_shots_with_sequence(self, request: BasisShotRequest) -> tuple[int, int]:
        """Reserve a job and return its usage clock plus monotonic admission number."""
        return self._engine.reserve_with_sequence(request)

    def lock_measurements_and_reveal_challenge(self) -> SealedTargetChallengeReveal:
        """Seal further measurement admission and reveal the committed targets."""
        return self._engine.lock_measurements_and_reveal_challenge()

    def run_basis_shots(
        self,
        request: BasisShotRequest,
        job_id: str,
        salt: int,
        usage_before: int = 0,
    ) -> JobResult:
        """Execute one admitted request with a job-scoped RNG from (run_entropy, salt)."""
        digest = hashlib.sha256(
            f"qiqcbench/blackbox_boundedgate_circuit/v2/shot-job/{self._run_entropy}/{salt}".encode()
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "big"))
        try:
            data = self._engine.run_basis_shots(request, rng, usage_before)
        except Exception:  # qsim-owned execution failure: typed, but no private detail
            return JobResult(
                job_id=job_id,
                device_id=self._device_id,
                status="failed",
                error="qsim basis-shot execution failed",
            )
        return JobResult(
            job_id=job_id,
            device_id=self._device_id,
            status="complete",
            shots=sum(s.shots for block in data.blocks for s in block.settings),
            data=data,
        )


def build_boundedgate_simulator_backend(
    hidden: HiddenBoundedgateConfig,
    public: PublicBoundedgateSpec,
    *,
    run_entropy: int | None = None,
) -> BoundedgateSimulatorBackend:
    return BoundedgateSimulatorBackend(hidden, public, run_entropy=run_entropy)
