"""Chain pulse-compiler device engine.

Duck-typed engine contract (``__init__(hidden, rng)`` + ``run_circuit(...) ->
JobResult``) shared with the other qtypes. The engine executes the agent's
scheduled gate program on the hidden noisy device and returns raw per-shot,
per-qubit level-resolved outcomes (``'0'``/``'1'``/``'2'``).

It walks the circuit on a ``2^n`` computational density matrix using the shared
``channels`` algebra so the device physics matches the verifier exactly. Leakage
to ``|2>`` removes population from the computational subspace; the engine
attributes each CPhase's leaked population to the higher-index edge qubit (the one
that leaks to ``|2>``) so a Stage-A leakage scan reads a level-2 signal.

The conditional-phase Ramsey (the Stage-A over-rotation probe) works because the
engine propagates the full density matrix: the agent appends its own basis-rotation
gates before measurement, and the joint computational distribution carries the
fringe.
"""

from __future__ import annotations

import time

import numpy as np

from qiqcbench.qsim.core.wire import JobResult, JobResultMetadata
from qiqcbench.qsim.qtypes.transmon_chain_pulse_compiler import channels as ch
from qiqcbench.qsim.qtypes.transmon_chain_pulse_compiler.device import (
    HiddenChainCompilerConfig,
)
from qiqcbench.qsim.qtypes.transmon_chain_pulse_compiler.wire import (
    ChainCircuitRequest,
    ChainEntanglingCalibration,
    ChainGateOp,
    JobChainLevelOutcomeData,
)

_PAULIS = (ch._PAULI_X, ch._PAULI_Y, ch._PAULI_Z)


class ChainCompilerEngine:
    """Stateful-per-call density-matrix engine for the chain pulse compiler."""

    def __init__(self, hidden: HiddenChainCompilerConfig, rng: np.random.Generator):
        self.hidden = hidden
        self.rng = rng
        self.n = hidden.n_qubits
        self.dim = 1 << self.n
        # Per-qubit readout rates indexed by qubit id.
        self._readout: list[tuple[float, float]] = [(0.0, 0.0)] * self.n
        self._p2_misread: list[float] = [0.0] * self.n
        for r in hidden.readout:
            if r.id >= self.n:
                raise ValueError(f"readout id {r.id} out of range for n={self.n}")
            self._readout[r.id] = (r.p_0_to_1, r.p_1_to_0)
            self._p2_misread[r.id] = r.p_2_misread

    # ---- physics with the agent's (optional) recalibration applied ----

    def _realized_physics(self, calib: ChainEntanglingCalibration | None) -> ch.ChainPhysics:
        over_corr = calib.over_rotation_correction if calib else 1.0
        leak_corr = calib.leakage_correction if calib else 1.0
        over_rotation = self.hidden.shipped_over_rotation_factor * over_corr
        # ACCEPTED RESIDUAL (audit 2026-08-26, owner ruling): the max(1.0, ·) clamp makes
        # over-correction free, so the leakage half of Stage A is obtainable without
        # measurement (any sufficiently small/zero/negative correction lands the well-tuned
        # floor). Race-neutral; kept for stored-row comparability. Revisit only with the
        # spectator-idling edition. Do NOT advertise this on any agent-facing surface.
        leak_factor = max(1.0, self.hidden.shipped_leakage_factor * leak_corr)
        cphase_leakage = self.hidden.cphase_leakage_well_tuned * leak_factor
        return ch.ChainPhysics(
            n_qubits=self.n,
            t1_ns=tuple(t * 1e3 for t in self.hidden.t1_us),  # us -> ns
            t2_ns=tuple(t * 1e3 for t in self.hidden.t2_us),
            rx_ry_duration_ns=self.hidden.rx_ry_duration_ns,
            rx_ry_depol=self.hidden.rx_ry_depolarizing,
            cphase_duration_ns=self.hidden.cphase_duration_ns,
            cphase_depol=self.hidden.cphase_depolarizing,
            cphase_leakage=cphase_leakage,
            over_rotation=over_rotation,
            prep_error=self.hidden.prep_error,
            readout_rates=tuple(self._readout),
        )

    # ---- density-matrix walk (tracks per-qubit leaked population) ----

    def _apply_prep(self, rho: np.ndarray, phys: ch.ChainPhysics) -> np.ndarray:
        e = phys.prep_error
        if e <= 0:
            return rho
        for q in range(self.n):
            xq = ch.embed_operator(ch._PAULI_X, [q], self.n)
            rho = (1.0 - e) * rho + e * (xq @ rho @ xq.conj().T)
        return rho

    def _apply_1q_depol(self, rho: np.ndarray, q: int, p: float) -> np.ndarray:
        if p <= 0:
            return rho
        out = (1.0 - p) * rho
        for pauli in _PAULIS:
            e = ch.embed_operator(pauli, [q], self.n)
            out = out + (p / 3.0) * (e @ rho @ e.conj().T)
        return out

    def _apply_2q_depol(self, rho: np.ndarray, a: int, b: int, p: float) -> np.ndarray:
        if p <= 0:
            return rho
        out = (1.0 - p) * rho
        paulis = (np.eye(2, dtype=complex), *_PAULIS)
        for pa in paulis:
            for pb in paulis:
                if pa is paulis[0] and pb is paulis[0]:
                    continue
                e = ch.embed_operator(np.kron(pb, pa), [a, b], self.n)
                out = out + (p / 15.0) * (e @ rho @ e.conj().T)
        return out

    def _qubit_one_population(self, rho: np.ndarray, q: int) -> float:
        proj = ch.embed_operator(np.array([[0, 0], [0, 1]], dtype=complex), [q], self.n)
        return float(np.real(np.trace(proj @ rho)))

    def _edge_11_population(self, rho: np.ndarray, a: int, b: int) -> float:
        n11 = ch.embed_operator(np.diag([0.0, 0.0, 0.0, 1.0]).astype(complex), [a, b], self.n)
        return float(np.real(np.trace(n11 @ rho)))

    def _run_one(
        self, circuit: list[ChainGateOp], phys: ch.ChainPhysics
    ) -> tuple[np.ndarray, list[float]]:
        """Walk the circuit on rho; return (final rho, per-qubit leaked population)."""
        rho = np.zeros((self.dim, self.dim), dtype=complex)
        rho[0, 0] = 1.0
        rho = self._apply_prep(rho, phys)
        leaked = [0.0] * self.n
        for g in circuit:
            if g.type == "rz":
                u = ch.embed_operator(ch.rz_unitary(g.angle), [g.qubits[0]], self.n)
                rho = u @ rho @ u.conj().T
                if not g.virtual:
                    rho = self._apply_1q_depol(rho, g.qubits[0], phys.rx_ry_depol)
            elif g.type in ("rx", "ry"):
                local = ch.rx_unitary(g.angle) if g.type == "rx" else ch.ry_unitary(g.angle)
                u = ch.embed_operator(local, [g.qubits[0]], self.n)
                rho = u @ rho @ u.conj().T
                rho = self._apply_1q_depol(rho, g.qubits[0], phys.rx_ry_depol)
            elif g.type == "cphase":
                a, b = g.qubits[0], g.qubits[1]
                theta = phys.over_rotation * g.angle
                u = ch.embed_operator(ch.cphase_local(theta), [a, b], self.n)
                rho = u @ rho @ u.conj().T
                # Attribute leaked |11> population to qubit b (the |2> leg).
                leaked[b] += phys.cphase_leakage * self._edge_11_population(rho, a, b)
                k0 = ch.embed_operator(
                    np.diag([1, 1, 1, np.sqrt(max(0.0, 1 - phys.cphase_leakage))]).astype(complex),
                    [a, b],
                    self.n,
                )
                rho = k0 @ rho @ k0.conj().T
                rho = self._apply_2q_depol(rho, a, b, phys.cphase_depol)
            else:  # pragma: no cover - validated upstream
                raise ValueError(f"unknown gate type {g.type!r}")
        # Decoherence applied once at the end keyed to each qubit's active time.
        circ_dicts = [
            {"type": g.type, "qubits": list(g.qubits), "virtual": g.virtual, "layer": g.layer}
            for g in circuit
        ]
        active = ch.per_qubit_active_ns(circ_dicts, phys)
        for q in range(self.n):
            kr = [
                ch.embed_operator(k, [q], self.n)
                for k in ch.amp_phase_damping_kraus(active[q], phys.t1_ns[q], phys.t2_ns[q])
            ]
            rho = sum(k @ rho @ k.conj().T for k in kr)
        return rho, leaked

    # ---- sampling ----

    def _sample(
        self,
        rho: np.ndarray,
        leaked: list[float],
        measure_qubits: list[int],
        shots: int,
    ) -> list[str]:
        probs = np.clip(np.real(np.diag(rho)), 0.0, None)
        t = probs.sum()
        if t <= 0:
            raise ValueError("final density matrix has zero computational trace")
        probs = probs / t  # condition on no-leakage; joint over {0,1}^n
        idx = self.rng.choice(self.dim, size=shots, p=probs)
        # Per-qubit leakage override probability (clamped).
        leak_p = [min(1.0, max(0.0, leaked[q])) for q in range(self.n)]
        out: list[str] = []
        for outcome in idx:
            chars: list[str] = []
            for q in measure_qubits:
                bit = (int(outcome) >> q) & 1
                # leakage event -> level '2'
                if leak_p[q] > 0 and self.rng.random() < leak_p[q]:
                    chars.append("2")
                    continue
                # asymmetric 0/1 readout confusion
                p01, p10 = self._readout[q]
                if bit == 0 and self.rng.random() < p01:
                    bit = 1
                elif bit == 1 and self.rng.random() < p10:
                    bit = 0
                chars.append(str(bit))
            out.append("".join(chars))
        return out

    def run_circuit(
        self,
        request: ChainCircuitRequest,
        device_id: str,
        job_id: str,
    ) -> JobResult:
        t0 = time.perf_counter()
        # No blanket except here. ``calibration`` is a strict wire model that
        # rejects unknown keys and out-of-range values, and
        # ``ChainCompilerSimulatorBackend`` checks shots, gate qubit ranges,
        # cphase distinctness, arity, the virtual-rz restriction, and measure
        # qubits before this call, so anything raised below is a qsim-owned
        # execution fault. Letting it propagate reaches ``JobManager``, which
        # types it as the sanctioned ``JOB_EXECUTION_ERROR`` that
        # ``is_internal_job_failure`` recognizes and ``get_job_result`` stamps
        # ``failure_kind="qsim_internal"``. Returning ``error=str(exc)`` instead
        # replaced that marker with free-form text -- empty for ``MemoryError``
        # -- so a qsim fault was attributed to the model.
        phys = self._realized_physics(request.calibration)
        measure_qubits = (
            list(request.measure_qubits)
            if request.measure_qubits is not None
            else list(range(self.n))
        )
        rho, leaked = self._run_one(list(request.circuit), phys)
        outcomes = self._sample(rho, leaked, measure_qubits, request.shots)
        circ_dicts = [
            {"type": g.type, "qubits": list(g.qubits), "virtual": g.virtual, "layer": g.layer}
            for g in request.circuit
        ]
        duration = float(sum(ch.per_qubit_active_ns(circ_dicts, phys))) / max(1, self.n)
        return JobResult(
            job_id=job_id,
            device_id=device_id,
            status="complete",
            shots=request.shots,
            data=JobChainLevelOutcomeData(
                outcomes=[outcomes],
                n_levels=3,
                measured_qubits=measure_qubits,
            ),
            metadata=JobResultMetadata(
                sequence_duration_ns=duration,
                wallclock_ms=int((time.perf_counter() - t0) * 1000),
            ),
        )


__all__ = ["ChainCompilerEngine"]
