"""IonChainEngine: pure-numpy state-vector simulator for trapped-ion chain.

Numerical strategy (spec § Engine API + § Physical-semantic chain § Engine modeling):

- State vector of size 2^N; N <= 16 fits comfortably (65536 complex amplitudes).
- prepare_state returns a noiseless |psi>; per-shot noise is sampled inside
  sample_prepared_state to keep the prepared snapshot reusable across
  measurement settings (parity-fringe sweep pattern).
- Gate noise is sampled per shot as stochastic Pauli kicks: prep-gate noise is
  applied at the end of preparation using a compact gate inventory carried by
  PreparedIonChainState, and analysis-gate noise is applied after each analysis
  gate. This is the O(p) end-of-prep approximation from the Engine API.
- Dephasing during free evolution:
    p = 2 (Gaussian): per-shot, per-ion sample xi ~ N(0, sigma_xi^2) with
        sigma_xi = sqrt(2) / T2. Phase per ion = (omega_total + xi) * t.
    p = 1 (Markovian): per-shot, per-ion apply Z-kick with probability
        (1 - exp(-t/T2)) / 2, then standard phase evolution omega_total * t.
- Per-ion asymmetric readout confusion applied to the sampled bitstring.

Unit ladder:
    Hidden config detuning_hz    | Hz (frequency)
    Wire reference_detuning_hz   | Hz (frequency)
    Engine internal omega_per_us | rad/us (after * 2*pi * 1e-6)
    Answer schema delta_omega_*  | rad/s

A factor-of-2pi conversion bug silently corrupts every Track B/C estimate;
the load-bearing phase-accumulation test catches it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from qiqcbench.qsim.core.wire import (
    CircuitMeasureOp,
    CircuitOp,
    GateOp,
)
from qiqcbench.qsim.qtypes.trapped_ion_chain.device import (
    HiddenTrappedIonChainConfig,
)
from qiqcbench.qsim.qtypes.trapped_ion_chain.gates import (
    ONE_QUBIT_GATE_MATRICES,
    embed_unitary,
    ms_unitary,
)

_TWO_PI = 2.0 * math.pi
_PARAMETERIZED_GATES = {"rx", "ry", "rz", "ms"}
_FIXED_GATES = {"i", "x", "y", "z", "h", "s", "sdg"}


@dataclass(frozen=True)
class PreparedIonChainState:
    """Noiseless prepared state snapshot + prep-circuit gate counts.

    Stores |psi> after applying the prepare circuit unitarily, WITHOUT gate
    noise. The prep-circuit gate inventory lets sample_prepared_state apply
    end-of-prep stochastic Pauli kicks per shot without carrying mutable GateOp
    objects or replaying the full prep circuit.
    """

    n_ions: int
    _psi: np.ndarray
    prep_one_q_gates: tuple[int, ...]
    prep_ms_gates: tuple[tuple[int, ...], ...]

    def state_vector(self) -> np.ndarray:
        """Return a copy; callers must not mutate ``_psi`` directly."""
        return self._psi.copy()


class IonChainEngine:
    def __init__(
        self,
        hidden: HiddenTrappedIonChainConfig,
        rng: np.random.Generator,
    ) -> None:
        self.hidden = hidden
        self.rng = rng
        self._n = len(hidden.ions)
        self._dim = 1 << self._n
        ids = {ion.id for ion in hidden.ions}
        if len(ids) != self._n or ids != set(range(self._n)):
            raise ValueError(f"Hidden ion ids must be 0..{self._n - 1}; got {sorted(ids)}")
        if len(hidden.readout) != self._n:
            raise ValueError(
                f"Hidden readout must have exactly {self._n} entries; got {len(hidden.readout)}"
            )
        ro_ids = {r.id for r in hidden.readout}
        if ro_ids != set(range(self._n)):
            raise ValueError(f"Hidden readout ids must match ions 0..{self._n - 1}")

        self._readout = [(0.0, 0.0)] * self._n
        for r in hidden.readout:
            self._readout[r.id] = (r.p_0_to_1, r.p_1_to_0)
        self._t2: list[float] = [0.0] * self._n
        self._p_dephase: list[float] = [0.0] * self._n
        self._detuning_hz: list[float] = [0.0] * self._n
        for ion in hidden.ions:
            self._t2[ion.id] = ion.t2_s
            self._p_dephase[ion.id] = ion.dephasing_exponent
            self._detuning_hz[ion.id] = ion.detuning_hz

    @staticmethod
    def _resolve_param(p: float | str, bindings: dict[str, float] | None) -> float:
        if isinstance(p, str):
            if bindings is None or p not in bindings:
                raise ValueError(f"Unbound circuit parameter: {p!r}")
            return float(bindings[p])
        return float(p)

    def _apply_gate(
        self,
        psi: np.ndarray,
        op: GateOp,
        bindings: dict[str, float] | None,
    ) -> np.ndarray:
        if op.name in _PARAMETERIZED_GATES:
            if len(op.params) != 1:
                raise ValueError(f"Gate {op.name!r} expects exactly 1 param, got {op.params}")
        elif op.name in _FIXED_GATES:
            if op.params:
                raise ValueError(f"Gate {op.name!r} expects 0 params, got {op.params}")
        else:
            raise ValueError(f"Unknown gate name: {op.name!r}")

        if op.name == "ms":
            theta = self._resolve_param(op.params[0], bindings)
            if self._n <= 14:
                return ms_unitary(self._n, theta, active=list(op.qubits)) @ psi
            return self._apply_ms_direct(psi, theta, list(op.qubits))

        if len(op.qubits) != 1:
            raise ValueError(f"Gate {op.name} expects 1 qubit, got {op.qubits}")
        q = op.qubits[0]
        mat_or_fn = ONE_QUBIT_GATE_MATRICES[op.name]
        if callable(mat_or_fn):
            theta = self._resolve_param(op.params[0], bindings)
            u_local = mat_or_fn(theta)
        else:
            u_local = mat_or_fn
        if self._n <= 14:
            return embed_unitary(u_local, [q], self._n) @ psi
        return self._apply_single_qubit_direct(psi, u_local, q)

    def _apply_single_qubit_direct(
        self, psi: np.ndarray, u_local: np.ndarray, q: int
    ) -> np.ndarray:
        """Apply a 1q unitary without materializing the full 2^N x 2^N matrix."""

        if not (0 <= q < self._n):
            raise ValueError(f"Qubit index {q} out of range for {self._n}-ion engine")
        shape = (2,) * self._n
        reshaped = psi.reshape(shape[::-1])
        axis = self._n - 1 - q
        moved = np.moveaxis(reshaped, axis, 0)
        flat = moved.reshape(2, -1)
        flat = u_local @ flat
        moved = flat.reshape((2,) + moved.shape[1:])
        return np.moveaxis(moved, 0, axis).reshape(-1)

    def _apply_ms_direct(self, psi: np.ndarray, theta: float, active: list[int]) -> np.ndarray:
        """Apply the MS gate in the X basis without dense-unitary materialization."""

        if any(q < 0 or q >= self._n for q in active):
            raise ValueError(f"MS active qubit out of range for n={self._n}: {active}")
        if len(set(active)) != len(active):
            raise ValueError(f"MS active qubits contain duplicates: {active}")
        hadamard = ONE_QUBIT_GATE_MATRICES["h"]
        out = psi
        for q in active:
            out = self._apply_single_qubit_direct(out, hadamard, q)
        idx = np.arange(self._dim)
        z_sum = np.zeros(self._dim, dtype=float)
        for q in active:
            z_sum += np.where(((idx >> q) & 1) == 0, 1.0, -1.0)
        eigvals = (z_sum * z_sum - len(active)) / 2.0
        out = out * np.exp(-1j * (theta / 2.0) * eigvals)
        for q in active:
            out = self._apply_single_qubit_direct(out, hadamard, q)
        return out

    def prepare_state(
        self,
        circuit: list[CircuitOp],
        bindings: dict[str, float] | None = None,
    ) -> PreparedIonChainState:
        psi = np.zeros(self._dim, dtype=complex)
        psi[0] = 1.0
        prep_one_q_gates: list[int] = []
        prep_ms_gates: list[tuple[int, ...]] = []
        for op in circuit:
            if isinstance(op, CircuitMeasureOp):
                raise ValueError(
                    "prepare_state does not accept measure ops; use sample_prepared_state."
                )
            if not isinstance(op, GateOp):
                raise ValueError(f"Unknown op kind: {op}")
            if op.name == "ms":
                prep_ms_gates.append(tuple(op.qubits))
            else:
                prep_one_q_gates.append(op.qubits[0])
            psi = self._apply_gate(psi, op, bindings)
        return PreparedIonChainState(
            n_ions=self._n,
            _psi=psi,
            prep_one_q_gates=tuple(prep_one_q_gates),
            prep_ms_gates=tuple(prep_ms_gates),
        )

    def _apply_free_evolution_phase(self, psi: np.ndarray, q: int, phase: float) -> np.ndarray:
        """Apply phase evolution equivalent to |1> accumulating exp(-i phase)."""
        idx = np.arange(self._dim)
        bit = (idx >> q) & 1
        factor = np.where(bit == 1, np.exp(-0.5j * phase), np.exp(0.5j * phase))
        return psi * factor

    def _apply_pauli_on_ion(
        self, psi: np.ndarray, q: int, kind: Literal["x", "y", "z"]
    ) -> np.ndarray:
        idx = np.arange(self._dim)
        bit = (idx >> q) & 1
        if kind == "z":
            return psi * np.where(bit == 1, -1.0, 1.0)
        flipped_idx = idx ^ (1 << q)
        if kind == "x":
            return psi[flipped_idx]
        if kind == "y":
            phase = np.where(bit == 1, 1j, -1j)
            return psi[flipped_idx] * phase
        raise ValueError(f"Unknown pauli kind: {kind}")

    def _sample_pauli_kick(self) -> Literal["x", "y", "z"]:
        return ("x", "y", "z")[int(self.rng.integers(0, 3))]

    def sample_prepared_state(
        self,
        prepared: PreparedIonChainState,
        *,
        free_evolution_us: float,
        phase_source: Literal["reference", "target"],
        reference_detuning_hz: float | None,
        analysis_circuit: list[CircuitOp],
        measured_ions: list[int],
        shots: int,
        bindings: dict[str, float] | None = None,
    ) -> list[str]:
        if prepared.n_ions != self._n:
            raise ValueError("PreparedIonChainState n_ions mismatch")
        if any(q < 0 or q >= self._n for q in measured_ions):
            raise ValueError(f"measured_ions out of range for n={self._n}: {measured_ions}")

        if phase_source == "reference":
            if reference_detuning_hz is None:
                raise ValueError("phase_source='reference' requires reference_detuning_hz")
            omega_per_ion_rad_per_us = [
                _TWO_PI * float(reference_detuning_hz) * 1e-6 for _ in range(self._n)
            ]
        elif phase_source == "target":
            omega_per_ion_rad_per_us = [_TWO_PI * d * 1e-6 for d in self._detuning_hz]
        else:
            raise ValueError(f"Unknown phase_source: {phase_source}")

        analysis_ops: list[GateOp] = []
        for op in analysis_circuit:
            if isinstance(op, CircuitMeasureOp):
                raise ValueError("analysis_circuit must not contain measure ops")
            if not isinstance(op, GateOp):
                raise ValueError(f"Unknown op kind: {op}")
            analysis_ops.append(op)

        gate_p = self.hidden.one_qubit_gate_errors.depolarizing
        ms_p = self.hidden.ms.depolarizing_error
        out: list[str] = []
        t_s = free_evolution_us * 1e-6
        prep_ms_active = [np.array(active, dtype=int) for active in prepared.prep_ms_gates]
        for _shot in range(shots):
            psi = prepared.state_vector()
            if gate_p > 0:
                for q in prepared.prep_one_q_gates:
                    if self.rng.random() < gate_p:
                        psi = self._apply_pauli_on_ion(psi, q, self._sample_pauli_kick())
            if ms_p > 0:
                for active in prep_ms_active:
                    if self.rng.random() < ms_p:
                        q = int(self.rng.choice(active))
                        psi = self._apply_pauli_on_ion(psi, q, self._sample_pauli_kick())
            for q in range(self._n):
                if free_evolution_us <= 0.0:
                    continue
                if self._p_dephase[q] == 2.0:
                    sigma_xi_rad_per_s = math.sqrt(2.0) / self._t2[q]
                    xi_rad_per_s = self.rng.normal(0.0, sigma_xi_rad_per_s)
                    phase_rad = omega_per_ion_rad_per_us[q] * free_evolution_us + xi_rad_per_s * t_s
                elif self._p_dephase[q] == 1.0:
                    p_kick = 0.5 * (1.0 - math.exp(-t_s / self._t2[q]))
                    if self.rng.random() < p_kick:
                        psi = self._apply_pauli_on_ion(psi, q, "z")
                    phase_rad = omega_per_ion_rad_per_us[q] * free_evolution_us
                else:
                    raise NotImplementedError(
                        f"dephasing_exponent={self._p_dephase[q]} not implemented; "
                        "engine supports only 1.0 and 2.0 (schema enforces this)."
                    )
                psi = self._apply_free_evolution_phase(psi, q, phase_rad)

            for op in analysis_ops:
                psi = self._apply_gate(psi, op, bindings)
                if op.name != "ms" and gate_p > 0:
                    q = op.qubits[0]
                    if self.rng.random() < gate_p:
                        psi = self._apply_pauli_on_ion(psi, q, self._sample_pauli_kick())
                elif op.name == "ms" and ms_p > 0 and self.rng.random() < ms_p:
                    q = int(self.rng.choice(np.array(op.qubits, dtype=int)))
                    psi = self._apply_pauli_on_ion(psi, q, self._sample_pauli_kick())

            probs = np.abs(psi) ** 2
            probs /= probs.sum()
            outcome = int(self.rng.choice(self._dim, p=probs))

            bits = ["0"] * len(measured_ions)
            for i, q in enumerate(measured_ions):
                true_bit = (outcome >> q) & 1
                p01, p10 = self._readout[q]
                if true_bit == 0:
                    obs = 1 if self.rng.random() < p01 else 0
                else:
                    obs = 0 if self.rng.random() < p10 else 1
                bits[i] = str(obs)
            out.append("".join(bits))
        return out
