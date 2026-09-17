"""Black-box analog-dynamics engine.

Generic, device-config-driven version of the feasibility-gate oracle: it builds
the hidden Hamiltonian ``M = sum_P omega_P P`` from the hidden coefficient list,
caches a dense ``eigh(M)`` once, and answers probe batches by evolving
``U(t)=exp(-i (t/2) M)`` and returning fixed-precision raw ±1 Pauli outcomes.

Two properties distinguish it from the per-job-stateless digital/transmon
engines and are load-bearing for the task semantics:

* **Budget persists across batches.** The scarce resource is total accepted
  evolution time, so a single engine instance (and its accumulating evidence)
  lives for the whole run. The backend builds exactly one engine.
* **RNG is injected, never reseeded internally** (engine-contract invariant):
  the backend factory mixes ``hidden.seed`` with fresh per-run entropy and
  injects the resulting generator once.

Strictness (per-row): off-grid / out-of-range times, malformed product states,
and invalid observables are **rejected** (consume zero budget, return no
outcomes). The accepted-row cap and total budget are enforced. Oversized
batches are a *batch-level* failure handled by the backend, not here.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from qiqcbench.qsim.core.wire import JobProbeOutcomeData, ProbeOutcomeRow, ProbeRowOp
from qiqcbench.qsim.qtypes.blackbox_analog_dynamics.device import (
    HiddenAnalogConfig,
    PublicAnalogSpec,
    hidden_coefficient_vector,
)

__all__ = ["AnalogEngine", "ProbeEvidence"]

_TOL = 1e-9

_PAULI_2x2 = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
_KET = {
    "z+": np.array([1, 0], dtype=complex),
    "z-": np.array([0, 1], dtype=complex),
    "x+": np.array([1, 1], dtype=complex) / np.sqrt(2),
    "x-": np.array([1, -1], dtype=complex) / np.sqrt(2),
    "y+": np.array([1, 1j], dtype=complex) / np.sqrt(2),
    "y-": np.array([1, -1j], dtype=complex) / np.sqrt(2),
}


@dataclass
class ProbeEvidence:
    """Authoritative record of what the engine actually ran (cumulative).

    Budget counts accepted rows only; rejected rows are retained for verifier
    readback but never charged.
    """

    accepted_rows: list[dict] = field(default_factory=list)
    rejected_rows: list[dict] = field(default_factory=list)
    budget_used_us: float = 0.0

    @property
    def accepted_row_count(self) -> int:
        return len(self.accepted_rows)

    @property
    def ran_nonzero_time_probe(self) -> bool:
        return any(r["evolve_time_us"] > 0.0 for r in self.accepted_rows)


def _full_pauli_matrix(pauli: str) -> np.ndarray:
    """Dense operator for a Pauli string (qubit 0 is the leftmost factor)."""
    op = np.array([[1.0 + 0j]])
    for char in pauli:
        op = np.kron(op, _PAULI_2x2[char])
    return op


class AnalogEngine:
    """Stateful hidden-dynamics engine for one run. RNG is injected."""

    def __init__(
        self,
        hidden: HiddenAnalogConfig,
        public: PublicAnalogSpec,
        rng: np.random.Generator,
    ):
        self._rng = rng
        self._n = public.n_qubits
        self._dim = 1 << self._n
        self._shots = public.num_internal_repetitions
        self._budget_us = public.total_evolution_time_budget_us
        self._max_rows = public.max_probe_rows
        self._t_min, self._t_max = public.evolve_time_range_us
        self._t_res = public.evolve_time_resolution_us
        self._w_min, self._w_max = public.allowed_observable_weight
        self._state_labels = set(public.allowed_initial_states)
        # Validate the independently parsed public/hidden halves before any
        # matrix is built. This is the same cross-contract used by the
        # execution commitment, so the digest cannot omit or reinterpret a
        # term that this engine would execute.
        hidden_coefficient_vector(hidden, public)
        # M = sum_P omega_P P ; U(t) = exp(-i (t/2) M). Dense build + dense eigh
        # is adequate for this fixed 10-qubit device; a sparse expm_multiply path
        # is the future option for larger devices.
        matrix = np.zeros((self._dim, self._dim), dtype=complex)
        for term in hidden.hamiltonian_terms:
            if term.coefficient != 0.0:
                matrix += term.coefficient * _full_pauli_matrix(term.pauli)
        self._evals, self._evecs = np.linalg.eigh(matrix)
        self._evidence = ProbeEvidence()
        self._lock = threading.Lock()

    @property
    def evidence(self) -> ProbeEvidence:
        return self._evidence

    # -- validation -----------------------------------------------------------

    def _on_grid(self, t: float) -> bool:
        if not (self._t_min - _TOL <= t <= self._t_max + _TOL):
            return False
        steps = t / self._t_res
        return abs(steps - round(steps)) <= 1e-6

    def _normalize_state(self, state: list[str]) -> list[str] | None:
        labels = list(state)
        if len(labels) != self._n:
            return None
        if any(lab not in self._state_labels for lab in labels):
            return None
        return labels

    def _observable_weight_ok(self, obs: str) -> bool:
        if len(obs) != self._n or any(c not in "IXYZ" for c in obs):
            return False
        weight = sum(1 for c in obs if c != "I")
        return self._w_min <= weight <= self._w_max

    # -- physics --------------------------------------------------------------

    def _apply_pauli(self, vec: np.ndarray, pauli: str) -> np.ndarray:
        tensor = vec.reshape((2,) * self._n)
        for q, ch in enumerate(pauli):
            if ch == "I":
                continue
            tensor = np.moveaxis(np.tensordot(_PAULI_2x2[ch], tensor, axes=([1], [q])), 0, q)
        return tensor.reshape(-1)

    def _exact_expectation(self, labels: list[str], obs: str, t: float) -> float:
        psi0 = np.array([1.0 + 0j])
        for lab in labels:
            psi0 = np.kron(psi0, _KET[lab])
        coeffs = self._evecs.conj().T @ psi0
        psit = self._evecs @ (np.exp(-1j * self._evals * t / 2.0) * coeffs)
        return float(np.real(np.vdot(psit, self._apply_pauli(psit, obs))))

    # -- batch execution ------------------------------------------------------

    def run_probe_batch(self, rows: list[ProbeRowOp]) -> JobProbeOutcomeData:
        """Process one valid-size batch; mutate cumulative evidence atomically."""
        with self._lock:
            out_rows = [self._run_row(row) for row in rows]
            return JobProbeOutcomeData(
                rows=out_rows,
                budget_used_us=self._evidence.budget_used_us,
                budget_remaining_us=max(0.0, self._budget_us - self._evidence.budget_used_us),
                accepted_row_count=self._evidence.accepted_row_count,
                max_probe_rows=self._max_rows,
            )

    def _reject(self, obs: str, reason: str, row: ProbeRowOp) -> ProbeOutcomeRow:
        self._evidence.rejected_rows.append(
            {
                "initial_state": list(row.initial_state),
                "evolve_time_us": row.evolve_time_us,
                "observable_pauli": obs,
                "reason": reason,
            }
        )
        return ProbeOutcomeRow(observable_pauli=obs, status="rejected", reject_reason=reason)

    def _run_row(self, row: ProbeRowOp) -> ProbeOutcomeRow:
        obs = row.observable_pauli
        labels = self._normalize_state(row.initial_state)
        if labels is None:
            return self._reject(obs, "malformed_initial_state", row)
        if not self._observable_weight_ok(obs):
            return self._reject(obs, "invalid_observable", row)
        if not self._on_grid(row.evolve_time_us):
            return self._reject(obs, "time_off_grid_or_out_of_range", row)
        if self._evidence.accepted_row_count >= self._max_rows:
            return self._reject(obs, "row_cap_exhausted", row)
        if self._evidence.budget_used_us + row.evolve_time_us > self._budget_us + _TOL:
            return self._reject(obs, "budget_exhausted", row)

        e = self._exact_expectation(labels, obs, row.evolve_time_us)
        p = min(max((1.0 + e) / 2.0, 0.0), 1.0)
        bits = self._rng.binomial(1, p, size=self._shots)
        outcomes = (2 * bits - 1).astype(int).tolist()

        self._evidence.budget_used_us += row.evolve_time_us
        self._evidence.accepted_rows.append(
            {
                "initial_state": labels,
                "evolve_time_us": row.evolve_time_us,
                "observable_pauli": obs,
                "raw_pauli_outcomes": outcomes,
                "num_internal_repetitions": self._shots,
            }
        )
        return ProbeOutcomeRow(
            observable_pauli=obs,
            status="accepted",
            accepted_evolve_time_us=row.evolve_time_us,
            raw_pauli_outcomes=outcomes,
            num_internal_repetitions=self._shots,
        )
