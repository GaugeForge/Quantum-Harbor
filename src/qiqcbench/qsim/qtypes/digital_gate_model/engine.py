"""Digital (gate-model) quantum-computer engine.

Mirrors the duck-typed contract of TransmonEngine: __init__(hidden, rng) and
a per-job entry point that takes a list of ops and returns a JobResult.

n-qubit flexible: state size and gate-embedding logic key off
`len(hidden.readout)`. The 3-qubit assumption lives only in the device YAML,
not in this engine. Density-matrix simulation: O(4^n) memory and O(4^n) per
gate. For n <= 6 this runs in milliseconds. State-vector + stochastic-noise
mode is a future extension when a >10-qubit device is added.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from qiqcbench.qsim.core.wire import (
    CircuitMeasureOp,
    CircuitOp,
    GateOp,
    JobBitstringData,
    JobResult,
    JobResultMetadata,
)
from qiqcbench.qsim.qtypes.digital_gate_model.device import HiddenDigitalConfig
from qiqcbench.qsim.qtypes.digital_gate_model.gates import (
    GATE_REGISTRY,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    embed_unitary,
)


@dataclass(frozen=True)
class PreparedDigitalState:
    """Opaque immutable snapshot of a density matrix prepared by DigitalEngine.

    The stored matrix is private; callers must use :meth:`density_matrix` to
    extract a fresh copy. The qtype-local runtime treats this snapshot as the
    public seam for reusing a prepared state across measurement settings.
    """

    n_qubits: int
    _rho: np.ndarray

    def density_matrix(self) -> np.ndarray:
        """Return an independent copy of the stored density matrix."""
        return self._rho.copy()


class DigitalEngine:
    """Stateful per-job density-matrix engine."""

    def __init__(self, hidden: HiddenDigitalConfig, rng: np.random.Generator):
        self.hidden = hidden
        self.rng = rng
        self._n = len(hidden.readout)
        self._dim = 1 << self._n
        self._reset_state()
        # Cache embedded 1-qubit Pauli operators for fast depolarizing channels.
        self._paulis_1q: list[list[np.ndarray]] = [
            [embed_unitary(P, [q], self._n) for P in (PAULI_X, PAULI_Y, PAULI_Z)]
            for q in range(self._n)
        ]
        # Per-qubit readout (id == index, validated below).
        self._readout: list[tuple[float, float]] = [(0.0, 0.0)] * self._n
        seen_ids: set[int] = set()
        for r in hidden.readout:
            if r.id in seen_ids:
                raise ValueError(f"Duplicate readout entry for qubit id {r.id}")
            if r.id >= self._n:
                raise ValueError(
                    f"Readout id {r.id} out of range for n={self._n} (ids must be 0..n-1)"
                )
            seen_ids.add(r.id)
            self._readout[r.id] = (r.p_0_to_1, r.p_1_to_0)
        if len(seen_ids) != self._n:
            missing = sorted(set(range(self._n)) - seen_ids)
            raise ValueError(f"Missing readout entries for qubit ids {missing}")
        # Pre-build the n-qubit confusion matrix once.
        self._conf_full = self._build_confusion_matrix()

    def _reset_state(self) -> None:
        self._rho = np.zeros((self._dim, self._dim), dtype=complex)
        self._rho[0, 0] = 1.0

    def _build_confusion_matrix(self) -> np.ndarray:
        """Per-qubit confusion as a Kronecker product. p_obs = M @ p_true."""
        per_q = []
        for q in range(self._n):
            p01, p10 = self._readout[q]
            per_q.append(
                np.array(
                    [[1.0 - p01, p10], [p01, 1.0 - p10]],
                    dtype=float,
                )
            )
        # Convention: qubit 0 is LSB of the index. np.kron(A, B) places B at LSB,
        # so we kron from MSB (qubit n-1) to LSB (qubit 0).
        m = per_q[self._n - 1]
        for q in range(self._n - 2, -1, -1):
            m = np.kron(m, per_q[q])
        return m

    @staticmethod
    def _resolve_param(p: float | str, bindings: dict[str, float] | None) -> float:
        if isinstance(p, str):
            if bindings is None or p not in bindings:
                raise ValueError(f"Unbound circuit parameter: {p!r}")
            return float(bindings[p])
        return float(p)

    def _apply_unitary(self, U: np.ndarray) -> None:
        self._rho = U @ self._rho @ U.conj().T

    def _apply_1q_depolarizing(self, q: int, p: float) -> None:
        if p <= 0:
            return
        new_rho = (1.0 - p) * self._rho
        b = p / 3.0
        for P in self._paulis_1q[q]:
            new_rho += b * (P @ self._rho @ P.conj().T)
        self._rho = new_rho

    def _apply_2q_depolarizing(self, qa: int, qb: int, p: float) -> None:
        if p <= 0:
            return
        new_rho = (1.0 - p) * self._rho
        b = p / 15.0
        single = [PAULI_X, PAULI_Y, PAULI_Z]
        # I⊗{X,Y,Z} and {X,Y,Z}⊗I — six single-qubit Pauli terms restricted to qa or qb.
        for P in single:
            E_a = embed_unitary(P, [qa], self._n)
            E_b = embed_unitary(P, [qb], self._n)
            new_rho += b * (E_a @ self._rho @ E_a.conj().T)
            new_rho += b * (E_b @ self._rho @ E_b.conj().T)
        # Nine {X,Y,Z}⊗{X,Y,Z} terms.
        for Pa in single:
            for Pb in single:
                # In our 2-qubit local basis qa is bit 0 (LSB), qb is bit 1, so
                # the local 4x4 operator is np.kron(Pb, Pa).
                P2q = np.kron(Pb, Pa)
                E = embed_unitary(P2q, [qa, qb], self._n)
                new_rho += b * (E @ self._rho @ E.conj().T)
        self._rho = new_rho

    def _apply_gate(
        self,
        name: str,
        qubits: list[int],
        params: list[float],
    ) -> None:
        gate_def = GATE_REGISTRY[name]
        if gate_def["n_qubits"] != len(qubits):
            raise ValueError(
                f"Gate {name!r} expects {gate_def['n_qubits']} qubits, got {len(qubits)}"
            )
        if gate_def["param_count"] > 0:
            U_local = gate_def["matrix"](*params[: gate_def["param_count"]])
        else:
            U_local = gate_def["matrix"]
        U_full = embed_unitary(U_local, qubits, self._n)
        self._apply_unitary(U_full)
        if gate_def["n_qubits"] == 1:
            rate = self.hidden.gate_errors.one_qubit_depolarizing
            self._apply_1q_depolarizing(qubits[0], rate)
        elif gate_def["n_qubits"] == 2:
            rate = self.hidden.gate_errors.two_qubit_cx_depolarizing
            self._apply_2q_depolarizing(qubits[0], qubits[1], rate)

    def _sample_bitstrings(
        self,
        observed_probs: np.ndarray,
        shots: int,
        measured_qubits: list[int],
    ) -> list[str]:
        probs = np.clip(observed_probs.real, 0.0, None)
        s = probs.sum()
        if s <= 0:
            raise ValueError("Probability distribution sums to zero (numerical error).")
        probs = probs / s
        outcomes = self.rng.choice(self._dim, size=shots, p=probs)
        result: list[str] = []
        for outcome in outcomes:
            bits = "".join(str((int(outcome) >> q) & 1) for q in measured_qubits)
            result.append(bits)
        return result

    def prepare_state(
        self,
        circuit: list[CircuitOp],
        bindings: dict[str, float] | None = None,
    ) -> PreparedDigitalState:
        """Reset, apply ``circuit`` (gate ops only), return an immutable snapshot.

        Measure ops are rejected; the prepared-state seam is for pre-measurement
        state preparation only.
        """
        self._reset_state()
        for op in circuit:
            if isinstance(op, GateOp):
                params = [self._resolve_param(p, bindings) for p in op.params]
                self._apply_gate(op.name, op.qubits, params)
            elif isinstance(op, CircuitMeasureOp):
                raise ValueError(
                    "prepare_state does not accept measure ops; use "
                    "sample_prepared_state to sample bitstrings."
                )
            else:  # pragma: no cover - defensive
                raise ValueError(f"Unknown op kind: {op}")
        return PreparedDigitalState(n_qubits=self._n, _rho=self._rho.copy())

    def sample_prepared_state(
        self,
        prepared: PreparedDigitalState,
        *,
        basis_ops: list[CircuitOp],
        measured_qubits: list[int],
        shots: int,
        bindings: dict[str, float] | None = None,
    ) -> list[str]:
        """Sample raw per-shot bitstrings from a prepared snapshot.

        Restores ``prepared`` into engine state via a defensive copy, applies
        only the supplied basis gate ops, applies the per-qubit readout
        confusion matrix, and returns per-shot bitstrings. The prepared
        snapshot is not mutated.
        """
        if prepared.n_qubits != self._n:
            raise ValueError(
                f"PreparedDigitalState n_qubits {prepared.n_qubits} does not "
                f"match engine n {self._n}"
            )
        if any(q < 0 or q >= self._n for q in measured_qubits):
            raise ValueError(f"Measure qubits out of range for n={self._n}: {measured_qubits}")
        # Restore the prepared snapshot via a defensive copy so subsequent
        # basis-op application cannot mutate the stored matrix.
        self._rho = prepared.density_matrix()
        for op in basis_ops:
            if isinstance(op, GateOp):
                params = [self._resolve_param(p, bindings) for p in op.params]
                self._apply_gate(op.name, op.qubits, params)
            elif isinstance(op, CircuitMeasureOp):
                raise ValueError("sample_prepared_state basis_ops must be gate ops only.")
            else:  # pragma: no cover - defensive
                raise ValueError(f"Unknown op kind: {op}")

        ideal_probs = np.real(np.diag(self._rho))
        ideal_probs = np.clip(ideal_probs, 0.0, None)
        s = ideal_probs.sum()
        if s <= 0:
            raise ValueError("Final density matrix has zero trace.")
        observed_probs = self._conf_full @ (ideal_probs / s)
        return self._sample_bitstrings(observed_probs, shots, list(measured_qubits))

    def run_circuit(
        self,
        circuit: list[CircuitOp],
        shots: int,
        device_id: str,
        job_id: str,
        bindings: dict[str, float] | None = None,
    ) -> JobResult:
        self._reset_state()
        t_start = time.perf_counter()
        measured_qubits: list[int] = []
        # No blanket except here. ``validate_digital_circuit`` runs first for both
        # entry points and refuses every agent-input fault this loop could hit --
        # unknown gate name (also a wire Literal), qubits outside the public
        # device, a measure qubit outside it, an unbound circuit parameter, a
        # two-qubit gate off the published connectivity, and a circuit with no
        # measurement -- so anything raised below is a qsim-owned execution fault.
        # Letting it propagate reaches ``JobManager``, which types it as the
        # sanctioned ``JOB_EXECUTION_ERROR`` that ``is_internal_job_failure``
        # recognizes and ``get_job_result`` stamps
        # ``failure_kind="qsim_internal"``. Returning ``error=str(exc)`` instead
        # replaced that marker with free-form text -- empty for ``MemoryError`` --
        # so a qsim fault was attributed to the model. The two explicit
        # raises below are retained as defence in depth for a direct engine
        # caller; neither is reachable from a validated request.
        for op in circuit:
            if isinstance(op, GateOp):
                params = [self._resolve_param(p, bindings) for p in op.params]
                self._apply_gate(op.name, op.qubits, params)
            elif isinstance(op, CircuitMeasureOp):
                if any(q < 0 or q >= self._n for q in op.qubits):
                    raise ValueError(f"Measure qubits out of range for n={self._n}: {op.qubits}")
                measured_qubits = list(op.qubits)
            else:  # pragma: no cover
                raise ValueError(f"Unknown op kind: {op}")

        if not measured_qubits:
            return JobResult(
                job_id=job_id,
                device_id=device_id,
                status="failed",
                shots=shots,
                error="Circuit did not contain a measure op.",
                metadata=JobResultMetadata(
                    wallclock_ms=int((time.perf_counter() - t_start) * 1000),
                ),
            )

        ideal_probs = np.real(np.diag(self._rho))
        ideal_probs = np.clip(ideal_probs, 0.0, None)
        s = ideal_probs.sum()
        if s <= 0:
            return JobResult(
                job_id=job_id,
                device_id=device_id,
                status="failed",
                shots=shots,
                error="Final density matrix has zero trace.",
                metadata=JobResultMetadata(
                    wallclock_ms=int((time.perf_counter() - t_start) * 1000),
                ),
            )
        ideal_probs = ideal_probs / s
        observed_probs = self._conf_full @ ideal_probs
        bitstrings = self._sample_bitstrings(observed_probs, shots, measured_qubits)
        return JobResult(
            job_id=job_id,
            device_id=device_id,
            status="complete",
            shots=shots,
            data=JobBitstringData(
                bitstrings=[bitstrings],
                measured_qubits=measured_qubits,
            ),
            metadata=JobResultMetadata(
                wallclock_ms=int((time.perf_counter() - t_start) * 1000),
            ),
        )
