"""IonTrapEngine: pure-numpy state-vector trajectory simulator (16 qubits).

Modeling:

- Exact per-shot Pauli-channel trajectories: every shot independently selects
  native-gate errors, and every nonzero schedule is evolved at its actual gate
  location. The zero-error mixture component reuses its one deterministic
  noiseless state; this is distribution-preserving, unlike moving a sampled
  mid-circuit error to the end of the preparation.
- Nonzero trajectories use a batch-shaped state array, but the 16-qubit
  deep-circuit path is deliberately cache-local at one trajectory per chunk.
  The zero/nonzero mixture draw, nonzero schedule, Born sample, and readout draw
  remain independent per shot and per setting.
- Fast matrix-free kernels: 1q gate = strided 2-index AXPY; ms(theta) =
  exp(-i theta/2 X_i X_j) = cos(theta/2) I - i sin(theta/2) X_iX_j applied as a
  single column gather (idx ^ mask_i ^ mask_j). No dense 2^n x 2^n matrices.

Convention: qubit 0 is the LEAST significant bit (matches digital/trapped-ion).
"""

from __future__ import annotations

import math

import numpy as np

from qiqcbench.qsim.core.wire import CircuitMeasureOp, CircuitOp, GateOp
from qiqcbench.qsim.qtypes.ion_trap_gate_model.device import (
    HiddenIonTrapGateModelConfig,
)

# State amplitudes are complex64 (8 bytes): precision is irrelevant since only
# bitstrings are sampled, and it halves the memory traffic that dominates a deep
# 16-qubit trajectory. Larger trajectory batches overflow the useful CPU cache and
# are slower per shot for the shipped ~2.4k-op circuits; one-row chunks also let the
# two async workers progress without competing over hundreds of MB of temporaries.
_B_MAX = 1
_CDTYPE = np.complex64

_FIXED_1Q = {
    "i": np.eye(2, dtype=complex),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
    "h": (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex),
    "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    "sdg": np.array([[1, 0], [0, -1j]], dtype=complex),
}
_PARAM_1Q = {"rx", "ry", "rz"}


def _rot(name: str, theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    if name == "rx":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if name == "ry":
        return np.array([[c, -s], [s, c]], dtype=complex)
    if name == "rz":
        return np.array(
            [
                [math.cos(theta / 2) - 1j * math.sin(theta / 2), 0],
                [0, math.cos(theta / 2) + 1j * math.sin(theta / 2)],
            ],
            dtype=complex,
        )
    raise ValueError(f"unknown rotation {name!r}")


def haar_su2(rng: np.random.Generator) -> np.ndarray:
    """A CUE(2) (Haar-random U(2)) 2x2 unitary."""
    z = (rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))) / math.sqrt(2)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    return q * ph


def _nonidentity_two_qubit_pauli(sample: int) -> tuple[int, int]:
    """Map a uniform integer in ``[0, 15)`` to one of the 15 non-identity pairs."""

    if not 0 <= sample < 15:
        raise ValueError("two-qubit Pauli sample must be in [0, 15)")
    return divmod(sample + 1, 4)


class IonTrapEngine:
    def __init__(
        self,
        hidden: HiddenIonTrapGateModelConfig,
        rng: np.random.Generator,
    ) -> None:
        self.hidden = hidden
        self.rng = rng
        self.n = hidden.n_qubits
        self.dim = 1 << self.n
        self.p_1q = float(hidden.one_qubit_depolarizing)
        self.p_ms = float(hidden.ms_depolarizing)
        self._p01 = np.zeros(self.n)
        self._p10 = np.zeros(self.n)
        for r in hidden.readout:
            self._p01[r.id] = r.p_0_to_1
            self._p10[r.id] = r.p_1_to_0
        self._idx = np.arange(self.dim)
        self._xperm: dict[int, np.ndarray] = {}
        self._zsign: dict[int, np.ndarray] = {}
        self._bit: dict[int, np.ndarray] = {}

    # ---------- cached index helpers (for per-row noise kicks) ----------

    def _xperm_for(self, q: int) -> np.ndarray:
        a = self._xperm.get(q)
        if a is None:
            a = self._idx ^ (1 << q)
            self._xperm[q] = a
        return a

    def _zsign_for(self, q: int) -> np.ndarray:
        a = self._zsign.get(q)
        if a is None:
            a = np.where(((self._idx >> q) & 1) == 1, -1.0, 1.0)
            self._zsign[q] = a
        return a

    def _bit_for(self, q: int) -> np.ndarray:
        a = self._bit.get(q)
        if a is None:
            a = (self._idx >> q) & 1
            self._bit[q] = a
        return a

    # ---------- coherent gates (batched, in-place via reshape-slicing) ----------
    #
    # state is (B, 2^n) C-contiguous; reshape to (B, 2, 2, ..., 2) is a VIEW whose
    # axis (n - q) is qubit q (qubit 0 = LSB = last axis). Operating on the size-2
    # slices is strided-but-contiguous arithmetic — far faster than fancy-index
    # gathers, with no whole-array permutation copy.

    def _apply_1q_batch(self, state: np.ndarray, u2: np.ndarray, q: int) -> None:
        # u2 is complex64; a0/a1 are strided views; compute both new blocks before
        # writing (no aliasing, no explicit copy).
        b = state.shape[0]
        t = state.reshape((b,) + (2,) * self.n)
        ax = self.n - q
        sl0 = (slice(None),) * ax + (0,) + (slice(None),) * q
        sl1 = (slice(None),) * ax + (1,) + (slice(None),) * q
        a0 = t[sl0]
        a1 = t[sl1]
        new0 = u2[0, 0] * a0 + u2[0, 1] * a1
        new1 = u2[1, 0] * a0 + u2[1, 1] * a1
        t[sl0] = new0
        t[sl1] = new1

    def _apply_xx_batch(self, state: np.ndarray, theta: float, i: int, j: int) -> None:
        b = state.shape[0]
        t = state.reshape((b,) + (2,) * self.n)
        c = np.complex64(math.cos(theta / 2.0))
        ms = np.complex64(-1j * math.sin(theta / 2.0))  # -i sin(theta/2)

        def sl(bi: int, bj: int) -> tuple:
            s = [slice(None)] * (self.n + 1)
            s[self.n - i] = bi
            s[self.n - j] = bj
            return tuple(s)

        a00 = t[sl(0, 0)]
        a01 = t[sl(0, 1)]
        a10 = t[sl(1, 0)]
        a11 = t[sl(1, 1)]
        # X_i X_j flips both bits: 00<->11, 01<->10. Compute all before writing.
        new00 = c * a00 + ms * a11
        new11 = c * a11 + ms * a00
        new01 = c * a01 + ms * a10
        new10 = c * a10 + ms * a01
        t[sl(0, 0)] = new00
        t[sl(1, 1)] = new11
        t[sl(0, 1)] = new01
        t[sl(1, 0)] = new10

    # ---------- per-row depolarizing noise ----------

    def _apply_pauli_row(self, state: np.ndarray, r: int, q: int, kind: int) -> None:
        # kind: 0=X, 1=Y, 2=Z
        if kind == 2:
            state[r] *= self._zsign_for(q)
            return
        xperm = self._xperm_for(q)
        if kind == 0:
            state[r] = state[r][xperm]
        else:  # Y = [[0,-i],[i,0]] : (Y psi)[idx] = psi[idx^mask] * (+i if bit==1 else -i)
            phase = np.where(self._bit_for(q) == 1, 1j, -1j)
            state[r] = state[r][xperm] * phase

    def _depolarize_1q(self, state: np.ndarray, q: int) -> None:
        if self.p_1q <= 0:
            return
        b = state.shape[0]
        hits = np.nonzero(self.rng.random(b) < self.p_1q)[0]
        for r in hits:
            self._apply_pauli_row(state, int(r), q, int(self.rng.integers(3)))

    def _depolarize_ms(self, state: np.ndarray, i: int, j: int) -> None:
        if self.p_ms <= 0:
            return
        b = state.shape[0]
        hits = np.nonzero(self.rng.random(b) < self.p_ms)[0]
        for r in hits:
            # 0=I, 1=X, 2=Y, 3=Z on each qubit; every non-II pair is equiprobable.
            pa, pb = _nonidentity_two_qubit_pauli(int(self.rng.integers(15)))
            if pa:
                self._apply_pauli_row(state, int(r), i, pa - 1)
            if pb:
                self._apply_pauli_row(state, int(r), j, pb - 1)

    # ---------- op dispatch + evolution ----------

    @staticmethod
    def _resolve(params: list) -> float:
        if len(params) != 1:
            raise ValueError(f"expected exactly 1 angle param, got {params}")
        p = params[0]
        if isinstance(p, str):
            raise ValueError(f"parameter placeholder {p!r} not supported; submit concrete angles")
        return float(p)

    def _apply_op_batch(self, state: np.ndarray, op: GateOp, *, noiseless: bool) -> None:
        name = op.name
        if name == "ms":
            if len(op.qubits) != 2:
                raise ValueError(f"ms requires exactly 2 qubits (a pair), got {op.qubits}")
            i, j = int(op.qubits[0]), int(op.qubits[1])
            if i == j or not (0 <= i < self.n) or not (0 <= j < self.n):
                raise ValueError(f"ms qubits out of range / equal: {op.qubits}")
            self._apply_xx_batch(state, self._resolve(op.params), i, j)
            if not noiseless:
                self._depolarize_ms(state, i, j)
            return
        if len(op.qubits) != 1:
            raise ValueError(f"gate {name!r} requires exactly 1 qubit, got {op.qubits}")
        q = int(op.qubits[0])
        if not (0 <= q < self.n):
            raise ValueError(f"qubit {q} out of range for n={self.n}")
        if name in _PARAM_1Q:
            u2 = _rot(name, self._resolve(op.params)).astype(_CDTYPE)
        elif name in _FIXED_1Q:
            if op.params:
                raise ValueError(f"gate {name!r} takes no params, got {op.params}")
            u2 = _FIXED_1Q[name].astype(_CDTYPE)
        else:
            raise ValueError(f"unsupported gate {name!r} for ion_trap_gate_model")
        self._apply_1q_batch(state, u2, q)
        if not noiseless:
            self._depolarize_1q(state, q)

    def _evolve_batch(
        self, circuit: list[CircuitOp], b: int, *, noiseless: bool = False
    ) -> np.ndarray:
        state = np.zeros((b, self.dim), dtype=_CDTYPE)
        state[:, 0] = 1.0
        for op in circuit:
            if isinstance(op, CircuitMeasureOp):
                raise ValueError("circuit must not contain measure ops")
            if not isinstance(op, GateOp):
                raise ValueError(f"unknown op kind: {op}")
            self._apply_op_batch(state, op, noiseless=noiseless)
        return state

    def _noise_probabilities(self, circuit: list[CircuitOp]) -> np.ndarray:
        """Return the independent native-gate error probabilities in execution order."""

        return np.asarray(
            [
                self.p_ms if isinstance(op, GateOp) and op.name == "ms" else self.p_1q
                for op in circuit
            ],
            dtype=np.float64,
        )

    def _sample_error_schedule_conditioned_nonempty(
        self,
        circuit: list[CircuitOp],
        probabilities: np.ndarray,
    ) -> dict[int, int]:
        """Sample the exact independent Pauli channel conditioned on an error."""

        if len(circuit) != len(probabilities) or not np.any(probabilities > 0.0):
            raise ValueError("a non-empty gate-error schedule cannot be sampled")
        while True:
            hit_indices = np.flatnonzero(self.rng.random(len(probabilities)) < probabilities)
            if len(hit_indices) > 0:
                break
        schedule: dict[int, int] = {}
        for raw_index in hit_indices:
            index = int(raw_index)
            op = circuit[index]
            schedule[index] = int(self.rng.integers(15 if op.name == "ms" else 3))
        return schedule

    def _evolve_with_error_schedule(
        self,
        circuit: list[CircuitOp],
        schedule: dict[int, int],
    ) -> np.ndarray:
        """Evolve one trajectory with a pre-sampled post-gate Pauli schedule."""

        state = np.zeros((1, self.dim), dtype=_CDTYPE)
        state[0, 0] = 1.0
        for index, op in enumerate(circuit):
            if isinstance(op, CircuitMeasureOp):
                raise ValueError("circuit must not contain measure ops")
            if not isinstance(op, GateOp):
                raise ValueError(f"unknown op kind: {op}")
            self._apply_op_batch(state, op, noiseless=True)
            if index not in schedule:
                continue
            sample = schedule[index]
            if op.name == "ms":
                left, right = _nonidentity_two_qubit_pauli(sample)
                if left:
                    self._apply_pauli_row(state, 0, int(op.qubits[0]), left - 1)
                if right:
                    self._apply_pauli_row(state, 0, int(op.qubits[1]), right - 1)
            else:
                self._apply_pauli_row(state, 0, int(op.qubits[0]), sample)
        return state

    # ---------- measurement ----------

    def _measure_repeated_state(
        self,
        state: np.ndarray,
        measured_qubits: list[int],
        shots: int,
    ) -> list[str]:
        """Draw independent Born/readout samples from one prepared state."""

        if state.shape != (1, self.dim):
            raise ValueError("repeated measurement requires one prepared state")
        if shots <= 0:
            return []
        probabilities = np.abs(state[0]).astype(np.float64) ** 2
        probabilities /= probabilities.sum()
        cdf = np.cumsum(probabilities)
        outcomes = np.searchsorted(cdf, self.rng.random(shots), side="left")
        outcomes = np.minimum(outcomes, self.dim - 1)
        columns = []
        for qubit in measured_qubits:
            true_bit = (outcomes >> qubit) & 1
            draw = self.rng.random(shots)
            flip_zero = (true_bit == 0) & (draw < self._p01[qubit])
            flip_one = (true_bit == 1) & (draw < self._p10[qubit])
            observed = np.where(flip_zero, 1, np.where(flip_one, 0, true_bit))
            columns.append(observed)
        values = np.stack(columns, axis=1).astype(np.int8)
        return ["".join(map(str, row)) for row in values.tolist()]

    def _split_trajectory_mixture(
        self,
        circuit: list[CircuitOp],
        shots: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split exact trajectories into reusable zero-error and noisy draws.

        Native Pauli errors are independent. Sampling the zero-error mixture
        component explicitly is distribution-preserving and avoids replaying a
        ~2.4k-gate state preparation for the majority of shots on the shipped
        low-error device. Nonzero schedules are sampled from the exact channel
        conditioned on at least one error and still evolve gate by gate.
        """

        clean_state = self._evolve_batch(circuit, 1, noiseless=True)
        probabilities = self._noise_probabilities(circuit)
        clean_probability = float(np.prod(1.0 - probabilities, dtype=np.float64))
        clean_mask = self.rng.random(shots) < clean_probability
        return clean_state, clean_mask, probabilities

    # ---------- public run methods ----------

    def run_circuit_shots(
        self, circuit: list[CircuitOp], measured_qubits: list[int], shots: int
    ) -> list[str]:
        if any(q < 0 or q >= self.n for q in measured_qubits):
            raise ValueError(f"measured_qubits out of range for n={self.n}: {measured_qubits}")
        clean_state, clean_mask, noise_probabilities = self._split_trajectory_mixture(
            circuit, shots
        )
        output = [""] * shots
        clean_indices = np.flatnonzero(clean_mask)
        clean_bits = self._measure_repeated_state(clean_state, measured_qubits, len(clean_indices))
        for index, bits in zip(clean_indices, clean_bits, strict=True):
            output[int(index)] = bits
        noisy_indices = np.flatnonzero(~clean_mask)
        for index in noisy_indices:
            schedule = self._sample_error_schedule_conditioned_nonempty(
                circuit,
                noise_probabilities,
            )
            state = self._evolve_with_error_schedule(circuit, schedule)
            output[int(index)] = self._measure_repeated_state(state, measured_qubits, 1)[0]
        return output

    def subsystem_purity(
        self,
        circuit: list[CircuitOp],
        subsystem_qubits: list[int],
        n_trajectories: int,
    ) -> float:
        """Developer diagnostic: unbiased trajectory estimate of ``Tr(rho_A^2)``.

        This method is not agent-facing and is not part of the Laughlin task
        verifier. That verifier replays the public randomized-measurement estimator
        from the realized raw bitstrings instead of substituting a sharp trajectory
        measurement.
        """
        if any(q < 0 or q >= self.n for q in subsystem_qubits):
            raise ValueError(f"subsystem_qubits out of range: {subsystem_qubits}")
        n_a = len(subsystem_qubits)
        dim_a = 1 << n_a
        rho_sum = np.zeros((dim_a, dim_a), dtype=np.complex128)
        diag_sum = 0.0
        k_total = 0
        remaining = n_trajectories
        while remaining > 0:
            b = min(remaining, _B_MAX)
            state = self._evolve_batch(circuit, b)
            m = self._split_ab(state, subsystem_qubits)  # (b, dim_a, dim_b)
            rho_k = np.einsum("kab,kcb->kac", m, m.conj())  # (b, dim_a, dim_a)
            rho_sum += rho_k.sum(axis=0).astype(np.complex128)
            diag_sum += float(np.sum(np.abs(rho_k) ** 2))
            k_total += b
            remaining -= b
        if k_total < 2:
            raise ValueError("need >= 2 trajectories for the unbiased purity")
        tr_sum_sq = float(np.sum(np.abs(rho_sum) ** 2))  # Tr(rho_sum^2), rho Hermitian
        return (tr_sum_sq - diag_sum) / (k_total * (k_total - 1))

    def _split_ab(self, state: np.ndarray, subsystem_qubits: list[int]) -> np.ndarray:
        """Reshape (b, 2^n) -> (b, 2^|A|, 2^|B|) grouping the subsystem qubits as A."""
        b = state.shape[0]
        t = state.reshape((b,) + (2,) * self.n)
        sub_axes = [self.n - q for q in subsystem_qubits]
        other_axes = [ax for ax in range(1, self.n + 1) if ax not in sub_axes]
        perm = [0] + sub_axes + other_axes
        t2 = np.transpose(t, perm)
        return t2.reshape(b, 1 << len(subsystem_qubits), -1)

    def run_rm_batch(
        self,
        prepare_circuit: list[CircuitOp],
        *,
        n_unitaries: int,
        shots_per_unitary: int,
        subsystem_qubits: list[int],
        basis_rng: np.random.Generator,
    ) -> list[tuple[int, list[str]]]:
        if any(q < 0 or q >= self.n for q in subsystem_qubits):
            raise ValueError(f"subsystem_qubits out of range for n={self.n}: {subsystem_qubits}")
        if len(set(subsystem_qubits)) != len(subsystem_qubits):
            raise ValueError(f"subsystem_qubits must be distinct: {subsystem_qubits}")
        clean_prepared = self._evolve_batch(prepare_circuit, 1, noiseless=True)
        noise_probabilities = self._noise_probabilities(prepare_circuit)
        clean_probability = float(np.prod(1.0 - noise_probabilities, dtype=np.float64))
        settings: list[tuple[int, list[str]]] = []
        for s in range(n_unitaries):
            haar = [haar_su2(basis_rng).astype(_CDTYPE) for _ in subsystem_qubits]
            clean_mask = self.rng.random(shots_per_unitary) < clean_probability
            bits = [""] * shots_per_unitary

            clean_state = clean_prepared.copy()
            for q, u2 in zip(subsystem_qubits, haar, strict=True):
                self._apply_1q_batch(clean_state, u2, q)
            clean_indices = np.flatnonzero(clean_mask)
            clean_bits = self._measure_repeated_state(
                clean_state, subsystem_qubits, len(clean_indices)
            )
            for index, value in zip(clean_indices, clean_bits, strict=True):
                bits[int(index)] = value

            for raw_index in np.flatnonzero(~clean_mask):
                schedule = self._sample_error_schedule_conditioned_nonempty(
                    prepare_circuit,
                    noise_probabilities,
                )
                state = self._evolve_with_error_schedule(prepare_circuit, schedule)
                for q, u2 in zip(subsystem_qubits, haar, strict=True):
                    self._apply_1q_batch(state, u2, q)
                bits[int(raw_index)] = self._measure_repeated_state(state, subsystem_qubits, 1)[0]
            settings.append((s, bits))
        return settings


__all__ = ["IonTrapEngine", "haar_su2"]
