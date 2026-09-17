"""Run-long budgeted engine for the blackbox bounded-gate qtype.

Holds the single hidden circuit instance and the cumulative shot/job budget
for the whole run. Requests are admitted atomically at submission time
(``reserve``): a malformed or over-budget request raises and consumes
nothing; an admitted request has its full shot cost reserved before it is
enqueued, so concurrent jobs can never oversubscribe the budget.

Execution is exact: each block's circuit output is built as an exact MPS at
the requested input (bond-dimension guard fails loudly rather than
truncating), per-shot bitstrings are drawn by perfect sampling in the
requested bases, and the per-qubit depolarizing noise is applied exactly as
basis-independent outcome flips with the hidden per-qubit probabilities.

Those probabilities may DRIFT with device usage: a job's effective flip
probability is e_i + d_i * u, where u is the fraction of the total shot
budget already consumed when the job was admitted. The clock is fixed at
``reserve`` time and handed back to ``run_basis_shots``, so it is a
deterministic property of the agent's submission order rather than of the
worker scheduling — exact replay survives concurrency, like the per-job RNG.

Shot randomness is PER JOB: the caller passes a job-scoped generator derived
from the attempt-private sampling entropy and the job's salt. The job-to-stream
mapping is therefore independent of concurrent worker scheduling, while fresh
production attempts still receive different measurement realizations.
"""

from __future__ import annotations

import copy
import threading
import time

import numpy as np

from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.device import (
    HiddenBoundedgateConfig,
    PublicBoundedgateSpec,
)
from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.sim import (
    build_mps,
    sample_bitstrings,
)
from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.wire import (
    RANDOM_PAULI,
    BasisShotBlockResult,
    BasisShotBudgetView,
    BasisShotRequest,
    BasisShotSettingResult,
    JobBasisShotData,
    SealedTargetChallengeReveal,
)

_BASIS_CHARS = np.array(list("xyz"))


class BoundedgateEngine:
    """One hidden circuit + one cumulative budget per run."""

    def __init__(
        self,
        hidden: HiddenBoundedgateConfig,
        public: PublicBoundedgateSpec,
    ):
        # The device clock: a time-budgeted run reads elapsed time the way it
        # reads shots_used, from the instrument, not from its own bookkeeping.
        self._started_monotonic = time.monotonic()
        if public.n_qubits != hidden.n_qubits:
            raise ValueError(
                f"public n_qubits {public.n_qubits} != hidden n_qubits {hidden.n_qubits}"
            )
        if public.input_dimension != hidden.input_dimension:
            raise ValueError("public/hidden input_dimension mismatch")
        # Fail closed on the public identifiability promises: a future edition
        # whose hidden circuit violates the disclosed rotation-count or
        # per-angle harmonic-degree bound must refuse to serve, not silently
        # break the public contract. (Each occurrence of an angle adds at most
        # one harmonic degree, so occurrence count bounds realized degree.)
        rot_angles = [op.angle for op in hidden.circuit if op.gate == "rot"]
        if len(rot_angles) > public.max_rotation_gates:
            raise ValueError(
                f"hidden circuit has {len(rot_angles)} rotation gates > disclosed "
                f"bound {public.max_rotation_gates}"
            )
        per_angle_counts = [rot_angles.count(k) for k in range(hidden.input_dimension)]
        if max(per_angle_counts, default=0) > public.max_harmonic_degree_per_angle:
            raise ValueError(
                f"an angle drives {max(per_angle_counts)} rotations > disclosed "
                f"per-angle harmonic degree bound {public.max_harmonic_degree_per_angle}"
            )
        self._validate_target_challenge_contract(hidden, public)
        self._hidden = hidden
        self._public = public
        self._n = hidden.n_qubits
        self._d = hidden.input_dimension
        self._flip = np.asarray(hidden.depolarizing_flip_prob, dtype=float)
        self._drift = (
            np.zeros(self._n)
            if hidden.readout_drift_per_qubit is None
            else np.asarray(hidden.readout_drift_per_qubit, dtype=float)
        )
        # Admission and sealing are one state machine. Actions also hold this
        # same re-entrant lock across the associated evidence-log append, so a
        # submit receipt and a reveal receipt have a total order.
        self._lock = threading.RLock()
        self._shots_used = 0
        self._jobs_used = 0
        self._admission_sequence = 0
        self._measurements_sealed = False
        self._sealed_reveal_payload: dict[str, object] | None = None

    @staticmethod
    def _validate_target_challenge_contract(
        hidden: HiddenBoundedgateConfig,
        public: PublicBoundedgateSpec,
    ) -> None:
        public_challenge = public.target_challenge
        hidden_challenge = hidden.target_challenge
        if (public_challenge is None) != (hidden_challenge is None):
            raise ValueError(
                "public and hidden target_challenge must either both be set or both be absent"
            )
        if public_challenge is None or hidden_challenge is None:
            return
        if public_challenge.challenge_id != hidden_challenge.challenge_id:
            raise ValueError("public/hidden target_challenge challenge_id mismatch")
        if public_challenge.target_count != len(hidden_challenge.target_inputs):
            raise ValueError("public target_count does not match hidden target inputs")
        if public_challenge.input_dimension != hidden.input_dimension:
            raise ValueError("public target_challenge input_dimension mismatch")
        if public_challenge.target_commitment_sha256 != hidden_challenge.target_commitment_sha256:
            raise ValueError("public/hidden target challenge commitment mismatch")

    @property
    def transaction_guard(self):
        """Shared admission/seal guard used by the qtype action transaction."""
        return self._lock

    # ---------- budget ----------

    def budget_view(self) -> BasisShotBudgetView:
        with self._lock:
            return self._budget_view_locked()

    def elapsed_wall_clock_s(self) -> float:
        """Seconds since this run-long engine started (the device clock)."""
        return max(time.monotonic() - self._started_monotonic, 0.0)

    def _budget_view_locked(self) -> BasisShotBudgetView:
        return BasisShotBudgetView(
            shots_used=self._shots_used,
            total_shot_budget=self._public.budget.total_shot_budget,
            jobs_used=self._jobs_used,
            max_jobs=self._public.budget.max_jobs,
            elapsed_wall_clock_s=self.elapsed_wall_clock_s(),
        )

    def reserve(self, request: BasisShotRequest) -> int:
        """Validate + atomically reserve the request's full cost; raise on any violation.

        Returns the run's usage clock for this job: the number of shots already
        consumed BEFORE it. Fixing the clock at admission (not at execution)
        keeps the drifted device deterministic under concurrent jobs.
        """
        usage_before, _ = self.reserve_with_sequence(request)
        return usage_before

    def reserve_with_sequence(self, request: BasisShotRequest) -> tuple[int, int]:
        """Reserve a request and return ``(usage_before, admission_sequence)``."""
        budget = self._public.budget
        if len(request.blocks) > budget.max_blocks_per_job:
            raise ValueError(
                f"at most {budget.max_blocks_per_job} blocks per job (got {len(request.blocks)})"
            )
        n_settings = sum(len(block.settings) for block in request.blocks)
        if n_settings > budget.max_settings_per_job:
            raise ValueError(
                f"at most {budget.max_settings_per_job} settings per job (got {n_settings})"
            )
        for block in request.blocks:
            if len(block.x) != self._d:
                raise ValueError(f"x must have length {self._d}, got {len(block.x)}")
            for value in block.x:
                if not -np.pi <= value <= np.pi:
                    raise ValueError("every x component must lie in [-pi, pi]")
            for setting in block.settings:
                if setting.basis != RANDOM_PAULI and len(setting.basis) != self._n:
                    raise ValueError(
                        f"basis string must have length {self._n} (or be '{RANDOM_PAULI}')"
                    )
                if setting.shots > budget.max_shots_per_setting:
                    raise ValueError(
                        f"at most {budget.max_shots_per_setting} shots per setting "
                        f"(got {setting.shots})"
                    )
        cost = sum(s.shots for block in request.blocks for s in block.settings)
        with self._lock:
            if self._measurements_sealed:
                raise ValueError("measurements are sealed; no further basis-shot jobs are allowed")
            if self._jobs_used >= budget.max_jobs:
                raise ValueError(f"job budget exhausted ({budget.max_jobs} jobs)")
            remaining = budget.total_shot_budget - self._shots_used
            if cost > remaining:
                raise ValueError(
                    f"shot budget exceeded: requested {cost}, remaining {remaining} "
                    f"of {budget.total_shot_budget}"
                )
            usage_before = self._shots_used
            self._shots_used += cost
            self._jobs_used += 1
            self._admission_sequence += 1
            admission_sequence = self._admission_sequence
        return usage_before, admission_sequence

    # ---------- irreversible measurement seal ----------

    def lock_measurements_and_reveal_challenge(self) -> SealedTargetChallengeReveal:
        """Irreversibly close admission and reveal the precommitted target inputs.

        Jobs admitted before this transaction may still execute. Repeated calls
        return the same opening and frozen admission cutoff.
        """
        with self._lock:
            challenge = self._hidden.target_challenge
            if challenge is None or self._public.target_challenge is None:
                raise ValueError("this device has no sealed target challenge")
            is_repeat = self._measurements_sealed
            if not self._measurements_sealed:
                self._measurements_sealed = True
                self._sealed_reveal_payload = {
                    "challenge_id": challenge.challenge_id,
                    "commitment_scheme": challenge.commitment_scheme,
                    "target_inputs": copy.deepcopy(challenge.target_inputs),
                    "target_inputs_sha256": challenge.target_inputs_sha256,
                    "target_commitment_sha256": challenge.target_commitment_sha256,
                    "commitment_nonce": challenge.commitment_nonce,
                    "shots_used": self._shots_used,
                    "jobs_used": self._jobs_used,
                    "last_admission_sequence": self._admission_sequence,
                    "elapsed_wall_clock_s": self.elapsed_wall_clock_s(),
                }
            assert self._sealed_reveal_payload is not None
            return SealedTargetChallengeReveal.model_validate(
                {**copy.deepcopy(self._sealed_reveal_payload), "is_repeat": is_repeat}
            )

    # ---------- execution ----------

    def run_basis_shots(
        self,
        request: BasisShotRequest,
        rng: np.random.Generator,
        usage_before: int = 0,
    ) -> JobBasisShotData:
        """Execute one admitted request with the caller's job-scoped RNG.

        ``usage_before`` is the clock returned by :meth:`reserve` for this job.
        """
        flip = self._flip_at(usage_before)
        block_results: list[BasisShotBlockResult] = []
        for block in request.blocks:
            x = np.asarray(block.x, dtype=float)
            mps = build_mps(self._hidden.circuit, x, self._n, max_bond=self._hidden.max_bond)
            settings: list[BasisShotSettingResult] = []
            for setting in block.settings:
                if setting.basis == RANDOM_PAULI:
                    basis_idx = rng.integers(0, 3, size=(setting.shots, self._n))
                    bits = sample_bitstrings(mps, basis_idx, setting.shots, rng)
                    bits = self._apply_flips(bits, rng, flip)
                    random_bases = ["".join(row) for row in _BASIS_CHARS[basis_idx]]
                else:
                    bits = sample_bitstrings(mps, setting.basis, setting.shots, rng)
                    bits = self._apply_flips(bits, rng, flip)
                    random_bases = None
                settings.append(
                    BasisShotSettingResult(
                        basis=setting.basis,
                        shots=setting.shots,
                        bitstrings=["".join(row) for row in bits.astype(str)],
                        random_bases=random_bases,
                    )
                )
            block_results.append(
                BasisShotBlockResult(x=[float(v) for v in block.x], settings=settings)
            )
        return JobBasisShotData(blocks=block_results, budget=self.budget_view())

    def _flip_at(self, usage_before: int) -> np.ndarray:
        """Effective per-qubit flip probabilities at this job's usage clock."""
        if not self._drift.any():
            return self._flip
        u = min(max(usage_before / self._public.budget.total_shot_budget, 0.0), 1.0)
        return np.clip(self._flip + self._drift * u, 0.0, 0.499)

    def _apply_flips(
        self, bits: np.ndarray, rng: np.random.Generator, flip: np.ndarray | None = None
    ) -> np.ndarray:
        probs = self._flip if flip is None else flip
        flips = rng.random(bits.shape) < probs[None, :]
        return bits ^ flips.astype(np.uint8)
