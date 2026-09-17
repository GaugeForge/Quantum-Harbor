"""Run-long stateful, per-shot statevector trajectory engine for
``neutral_atom_logical_processor``.

Each shot re-executes the submitted op program from |0...0> on a single 2^n complex
statevector (n = atom count; 2^9 = 512 is tiny, so a per-shot Python loop is fine and
gives genuine mid-circuit measurement + per-shot atom loss + feedforward). Noise is
injected per-shot:
  - stochastic Pauli after each gate / CZ (depolarizing-style),
  - HONEST atom loss at CZ / move / readout: a lost atom is removed (subsequent gates on
    it no-op; a CZ with a lost partner is identity on the survivor) and reads dark "0",
  - idle T2 dephasing accrued over each op's public duration,
  - asymmetric fluorescence readout confusion.
The engine only SAMPLES: it returns the raw per-shot measurement bits (packbits). The
classical Hamming decode + post-selection + F(|H_L>) live verifier-side (steane.py).

Convention: qubit 0 = MSB (axis 0); flat index i has atom k at bit (n-1-k). This matches
``steane.ideal_h_logical`` so the engine and verifier share one physics.
"""

from __future__ import annotations

import base64
import math
import threading

import numpy as np

from qiqcbench.qsim.qtypes.neutral_atom_logical_processor.device import (
    HiddenNeutralAtomConfig,
    PublicNeutralAtomSpec,
)
from qiqcbench.qsim.qtypes.neutral_atom_logical_processor.wire import (
    AtomOp,
    AtomProgramRequest,
    JobAtomShotData,
)

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULIS = (_X, _Y, _Z)
_FIXED = {
    "h": (_X + _Z) / math.sqrt(2.0),
    "x": _X,
    "y": _Y,
    "z": _Z,
    "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    "sdg": np.array([[1, 0], [0, -1j]], dtype=complex),
}
# init state -> single-qubit prep gates applied to |0>
_INIT_GATES = {"0": [], "1": [_X], "+": [_FIXED["h"]], "-": [_FIXED["h"], _Z]}


def _rot(name: str, theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    if name == "rx":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if name == "ry":
        return np.array([[c, -s], [s, c]], dtype=complex)
    if name == "rz":
        return np.array([[c - 1j * s, 0], [0, c + 1j * s]], dtype=complex)
    raise ValueError(f"unknown rotation {name!r}")


def _pack_bits(arr: np.ndarray) -> str:
    packed = np.packbits(np.ascontiguousarray(arr.reshape(-1), dtype=np.uint8))
    return base64.b64encode(packed.tobytes()).decode("ascii")


class _Budget:
    __slots__ = ("used", "cap")

    def __init__(self, cap: int) -> None:
        self.used = 0
        self.cap = cap


class NeutralAtomEngine:
    def __init__(
        self,
        hidden: HiddenNeutralAtomConfig,
        public: PublicNeutralAtomSpec,
        rng: np.random.Generator,
    ) -> None:
        self.rng = rng
        self.hidden = hidden
        self.public = public
        self.n = public.n_atoms
        self.dim = 1 << self.n
        self._lock = threading.Lock()
        self._budget = _Budget(public.budgets.shot_budget)
        self._max_experiment_calls = public.budgets.max_experiment_calls
        self._experiment_calls = 0
        self._max_per_call = public.budgets.max_shots_per_call
        self._max_ops = public.budgets.max_ops
        self._max_recorded_bits_per_call = public.budgets.max_recorded_bits_per_call
        self._max_recorded_bits_per_run = public.budgets.max_recorded_bits_per_run
        self._recorded_bits = 0
        self.dur = public.durations_us
        nz = hidden.noise
        self.nz = nz
        # precompute per-atom bit masks (atom k -> bit (n-1-k), big-endian)
        idx = np.arange(self.dim)
        self._bit = [((idx >> (self.n - 1 - k)) & 1) for k in range(self.n)]

    # ---- single-state gate kernels (state is a (2,)*n complex tensor) ----

    def _apply_1q(self, st: np.ndarray, u2: np.ndarray, k: int) -> np.ndarray:
        return np.tensordot(u2, st, axes=([1], [k])).transpose(
            list(range(1, k + 1)) + [0] + list(range(k + 1, self.n))
        )

    def _apply_cz(self, st_flat: np.ndarray, a: int, b: int) -> None:
        mask = (self._bit[a] == 1) & (self._bit[b] == 1)
        st_flat[mask] *= -1.0

    def _random_pauli(self, st_flat: np.ndarray, k: int) -> np.ndarray:
        p = _PAULIS[int(self.rng.integers(3))]
        t = st_flat.reshape((2,) * self.n)
        return self._apply_1q(t, p, k).reshape(-1)

    # ---- per-op duration (drives idle dephasing) ----

    def _op_duration_us(self, op: AtomOp) -> float:
        if op.type == "gate":
            return self.dur.rotation
        if op.type == "cz":
            return self.dur.cz
        if op.type == "measure":
            return self.dur.readout
        if op.type == "move":
            return self.dur.move_per_um * 10.0  # nominal 10 um move if unspecified
        return 0.0

    def _idle_dephase(self, st_flat: np.ndarray, alive: list[bool], t_us: float) -> np.ndarray:
        if t_us <= 0 or self.nz.idle_t2_us <= 0:
            return st_flat
        p = 0.5 * (1.0 - math.exp(-t_us / self.nz.idle_t2_us))
        for k in range(self.n):
            if alive[k] and self.rng.random() < p:
                t = st_flat.reshape((2,) * self.n)
                st_flat = self._apply_1q(t, _Z, k).reshape(-1)
        return st_flat

    # ---- measurement of one atom in Z (collapse) ----

    def _measure_z(self, st_flat: np.ndarray, k: int) -> tuple[int, np.ndarray]:
        p1 = float(np.sum(np.abs(st_flat[self._bit[k] == 1]) ** 2))
        outcome = 1 if self.rng.random() < p1 else 0
        keep = self._bit[k] == outcome
        st_flat = st_flat * keep
        nrm = np.linalg.norm(st_flat)
        if nrm > 1e-12:
            st_flat = st_flat / nrm
        return outcome, st_flat

    # ---- run one shot ----

    def _run_shot(self, ops: list[AtomOp]) -> list[tuple[int, int]]:
        """Return the ordered list of (bit, lost_flag) recorded by measure ops this shot."""
        st = np.zeros(self.dim, dtype=complex)
        st[0] = 1.0
        alive = [True] * self.n
        recorded: list[tuple[int, int]] = []
        self._exec(ops, st, alive, recorded)
        return recorded

    def _exec(self, ops, st, alive, recorded) -> np.ndarray:
        for op in ops:
            st = self._apply_op(op, st, alive, recorded)
        return st

    def _apply_op(self, op: AtomOp, st: np.ndarray, alive: list[bool], recorded) -> np.ndarray:
        nz = self.nz
        if op.type == "init":
            # Prepare one atom (`atom`) or, as a convenience, a batch (`atoms`) to the same
            # `state`. run_atom_program validates the targets first, so they are in range here.
            targets = [int(op.atom)] if op.atom is not None else [int(a) for a in op.atoms or []]
            gates = [_rot("ry", math.pi / 4)] if op.state == "H" else _INIT_GATES[op.state or "0"]
            t = st.reshape((2,) * self.n)
            for k in targets:
                for g in gates:
                    t = self._apply_1q(t, g, k)
            return t.reshape(-1)

        if op.type == "gate":
            t = st.reshape((2,) * self.n)
            u2 = _rot(op.gate, op.angle) if op.gate in ("rx", "ry", "rz") else _FIXED[op.gate]
            for k in op.atoms or []:
                if not alive[k]:
                    continue
                t = self._apply_1q(t, u2, k)
            st = t.reshape(-1)
            if nz.rot_pauli_error > 0:
                for k in op.atoms or []:
                    if alive[k] and self.rng.random() < nz.rot_pauli_error:
                        st = self._random_pauli(st, k)
            return self._idle_dephase(st, alive, self._op_duration_us(op))

        if op.type == "cz":
            a, b = int(op.pair[0]), int(op.pair[1])
            if alive[a] and alive[b]:
                self._apply_cz(st, a, b)
                if nz.cz_pauli_error > 0 and self.rng.random() < nz.cz_pauli_error:
                    st = self._random_pauli(st, a)
                    st = self._random_pauli(st, b)
                if nz.cz_atom_loss > 0:
                    for k in (a, b):
                        if self.rng.random() < nz.cz_atom_loss:
                            alive[k] = False
            # if a partner is lost the CZ is identity on the survivor (honest loss)
            return self._idle_dephase(st, alive, self._op_duration_us(op))

        if op.type == "move":
            k = int(op.atom)
            dist = 10.0  # nominal; a real layout would compute |to_site - from_site|
            if alive[k] and self.rng.random() < nz.move_atom_loss_per_um * dist:
                alive[k] = False
            return self._idle_dephase(st, alive, self.dur.move_per_um * dist)

        if op.type == "measure":
            t_us = self._op_duration_us(op)
            for k in op.atoms or []:
                if not alive[k]:
                    recorded.append((0, 1))  # lost atom reads dark "0"
                    continue
                # rotate measured atom into Z: x-basis -> H ; y-basis -> Sdg then H
                tt = st.reshape((2,) * self.n)
                if op.basis == "x":
                    tt = self._apply_1q(tt, _FIXED["h"], k)
                elif op.basis == "y":
                    tt = self._apply_1q(tt, _FIXED["sdg"], k)
                    tt = self._apply_1q(tt, _FIXED["h"], k)
                st = tt.reshape(-1)
                outcome, st = self._measure_z(st, k)
                # asymmetric fluorescence confusion + readout loss
                if self.rng.random() < nz.readout_atom_loss:
                    recorded.append((0, 1))
                    alive[k] = False
                    continue
                if outcome == 1 and self.rng.random() < nz.readout_p1_to_dark:
                    outcome = 0
                elif outcome == 0 and self.rng.random() < nz.readout_p0_to_bright:
                    outcome = 1
                recorded.append((outcome, 0))
                if op.reset and outcome == 1:  # reset measured atom to |0>
                    st = self._apply_1q(st.reshape((2,) * self.n), _X, k).reshape(-1)
            return self._idle_dephase(st, alive, t_us)

        if op.type == "feedforward":
            idx = int(op.on_bit)
            if (
                0 <= idx < len(recorded)
                and recorded[idx][0] == int(op.value)
                and op.then is not None
            ):
                return self._apply_op(op.then, st, alive, recorded)
            return st

        raise ValueError(f"unknown op type {op.type!r}")

    def _op_target_error(self, op: AtomOp) -> str | None:
        """Legality message if `op` references a missing/out-of-range atom, else None.

        A malformed submission must be REJECTED (returned as an error string the caller
        surfaces to the agent / the scorer's replay), never crash the engine mid-run. `init`
        accepts either a single `atom` or a batch `atoms` list; both must be in range.
        """
        n = self.n

        def _bad(k) -> bool:
            return not (isinstance(k, int) and not isinstance(k, bool) and 0 <= k < n)

        if op.type == "init":
            targets = [op.atom] if op.atom is not None else list(op.atoms or [])
            if not targets:
                return "init op requires an 'atom' (int) or 'atoms' (list of ints)"
            if any(_bad(k) for k in targets):
                return f"init atom out of range [0,{n}): {targets}"
        elif op.type == "move":
            if _bad(op.atom):
                return f"move atom out of range [0,{n}): {op.atom!r}"
        elif op.type == "gate":
            if op.gate not in _FIXED and op.gate not in ("rx", "ry", "rz"):
                return f"unsupported gate {op.gate!r}"
            if any(_bad(k) for k in op.atoms or []):
                return f"gate atom out of range [0,{n}): {op.atoms}"
        elif op.type == "cz":
            if (
                not op.pair
                or len(op.pair) != 2
                or any(_bad(k) for k in op.pair)
                or op.pair[0] == op.pair[1]
            ):
                return f"cz requires a pair of two distinct in-range atoms; got {op.pair!r}"
        elif op.type == "measure":
            if any(_bad(k) for k in op.atoms or []):
                return f"measure atom out of range [0,{n}): {op.atoms}"
        elif op.type == "feedforward":
            if op.on_bit is None or op.value is None or op.then is None:
                return "feedforward requires 'on_bit', 'value', and 'then'"
            # The conditioned op executes through the same kernels, so it must pass
            # the same legality checks (recursively) — a nested op is not a bypass.
            return self._op_target_error(op.then)
        return None

    # ---- public run ----

    def _reserve_atom_program_locked(self, request: AtomProgramRequest) -> str | None:
        """Validate and charge one request while ``self._lock`` is held."""

        if len(request.ops) > self._max_ops:
            return f"program has {len(request.ops)} ops > max_ops {self._max_ops}"
        if request.shots > self._max_per_call:
            return f"shots {request.shots} exceeds max_shots_per_call {self._max_per_call}"
        if self._budget.used + request.shots > self._budget.cap:
            return (
                f"shot budget exhausted: {self._budget.used}+{request.shots} "
                f"> cap {self._budget.cap}"
            )
        if self._experiment_calls + 1 > self._max_experiment_calls:
            return (
                "experiment-call budget exhausted: "
                f"{self._experiment_calls}+1 > cap {self._max_experiment_calls}"
            )
        # Every schema-valid backend attempt consumes the public call quota,
        # including attempts rejected by semantic schedule validation below.
        # Otherwise rejected calls can grow durable submission/poll records
        # without bound while bypassing max_experiment_calls.
        self._experiment_calls += 1
        for op in request.ops:
            # Gate-name and target legality both live in _op_target_error so nested
            # feedforward `then` ops pass through the identical (recursive) checks.
            err = self._op_target_error(op)
            if err is not None:
                return err
        recorded_bits = request.shots * len(self._labels(request.ops))
        if recorded_bits > self._max_recorded_bits_per_call:
            return (
                "raw result exceeds max_recorded_bits_per_call: "
                f"{recorded_bits} > {self._max_recorded_bits_per_call}; "
                "split the experiment into smaller jobs"
            )
        if self._recorded_bits + recorded_bits > self._max_recorded_bits_per_run:
            return (
                "run-wide raw-result budget exhausted: "
                f"{self._recorded_bits}+{recorded_bits} "
                f"> cap {self._max_recorded_bits_per_run}"
            )
        # Reserve all run-wide resources before allocation/execution. A failed
        # execution still consumed qsim work and cannot be replayed to evade
        # the public run limits.
        self._recorded_bits += recorded_bits
        self._budget.used += request.shots
        return None

    def reserve_atom_program(self, request: AtomProgramRequest) -> str | None:
        """Atomically preflight and charge a request before async job enqueue."""

        with self._lock:
            return self._reserve_atom_program_locked(request)

    def run_atom_program(
        self,
        request: AtomProgramRequest,
        *,
        preadmitted: bool = False,
    ) -> JobAtomShotData | str:
        with self._lock:
            if not preadmitted:
                error = self._reserve_atom_program_locked(request)
                if error is not None:
                    return error
            labels = self._labels(request.ops)
            n_rec = len(labels)
            bits = np.zeros((request.shots, n_rec), dtype=np.uint8)
            lost = np.zeros((request.shots, n_rec), dtype=np.uint8)
            try:
                for s in range(request.shots):
                    rec = self._run_shot(request.ops)
                    if len(rec) != n_rec:
                        return f"recorded {len(rec)} bits but expected {n_rec} (data-dependent measures?)"
                    for j, (b, lo) in enumerate(rec):
                        bits[s, j] = b
                        lost[s, j] = lo
            except (TypeError, ValueError, IndexError, KeyError) as e:
                # A verifier must NEVER crash on agent input: surface any residual malformed-op
                # error as a rejection string (the caller/scorer turns it into a clean reject).
                return f"illegal schedule: {e}"
            return JobAtomShotData(
                shots=request.shots,
                n_recorded=n_rec,
                measure_labels=labels,
                measure_bits_b64=_pack_bits(bits),
                lost_b64=_pack_bits(lost),
            )

    def _labels(self, ops: list[AtomOp]) -> list[str]:
        labels: list[str] = []
        for i, op in enumerate(ops):
            if op.type == "measure":
                for k in op.atoms or []:
                    labels.append(f"{i}:{k}:{op.basis}")
            elif op.type == "feedforward" and op.then is not None and op.then.type == "measure":
                for k in op.then.atoms or []:
                    labels.append(f"{i}:{k}:{op.then.basis}")
        return labels


__all__ = ["NeutralAtomEngine"]
