"""Run-long scheduled Pauli-fault-frame engine.

The engine walks the actual frozen forward schedule and a mechanically compiled
native inverse. It evolves a vectorized Pauli fault frame through every ideal
native Clifford, maps accumulated overrotation coefficients to effective
Pauli-event probabilities, and samples the directed-ECR, simultaneous-layer,
idle, one-qubit, readout, and submission-index drift models. It is not a full
coherent state-vector evolution. Candidate IDs are only bank keys: there is no
per-candidate quality parameter or response table.
"""

from __future__ import annotations

import hashlib
import math
import secrets
from collections import Counter
from dataclasses import dataclass

import numpy as np

from qiqcbench.qsim.core.wire import JobBitstringData
from qiqcbench.qsim.qtypes.scheduled_transmon_gate_model.bank import (
    CandidateBank,
    CompiledCandidate,
    ScheduledNativeGate,
)
from qiqcbench.qsim.qtypes.scheduled_transmon_gate_model.device import (
    HiddenScheduledTransmonConfig,
    HiddenTemporalCoupling,
    PublicScheduledTransmonSpec,
)
from qiqcbench.qsim.qtypes.scheduled_transmon_gate_model.wire import (
    CompilationMirrorRequest,
    CompilationReadoutReferenceRequest,
)


@dataclass(frozen=True)
class _RuntimeGate:
    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()


@dataclass(frozen=True)
class _RuntimeLayer:
    duration_ns: int
    gates: tuple[_RuntimeGate, ...]


def _seed_from_parts(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def public_mirror_frame(mirror_seed: int, width: int) -> tuple[tuple[int, int], ...]:
    """Return legacy-named bits for the deterministic virtual coherent-axis twirl."""

    frames: list[tuple[int, int]] = []
    for qubit in range(width):
        digest = hashlib.sha256(
            f"qiqcbench_mirror_frame_v1|{mirror_seed}|{qubit}".encode()
        ).digest()
        frames.append((digest[0] & 1, digest[1] & 1))
    return tuple(frames)


def _gate_from_public(gate: ScheduledNativeGate) -> _RuntimeGate:
    return _RuntimeGate(gate.name, tuple(gate.qubits), tuple(gate.params))


def _forward_layers(candidate: CompiledCandidate) -> tuple[_RuntimeLayer, ...]:
    return tuple(
        _RuntimeLayer(
            duration_ns=layer.duration_ns,
            gates=tuple(_gate_from_public(candidate.gates[index]) for index in layer.gate_indices),
        )
        for layer in candidate.layers
    )


def _inverse_layers(
    candidate: CompiledCandidate,
    public: PublicScheduledTransmonSpec,
) -> tuple[_RuntimeLayer, ...]:
    """Compile a native inverse without an opaque candidate-specific artifact.

    ``ecr`` and ``x`` are self-inverse, ``rz(theta)`` becomes ``rz(-theta)``,
    and ``sx† = rz(pi) sx rz(-pi)`` up to global phase.  Gates sharing a
    forward layer are disjoint and remain parallel in each inverse sublayer.
    """

    layers: list[_RuntimeLayer] = []
    for public_layer in reversed(candidate.layers):
        gates = [candidate.gates[index] for index in public_layer.gate_indices]
        first: list[_RuntimeGate] = []
        middle: list[_RuntimeGate] = []
        last: list[_RuntimeGate] = []
        for gate in gates:
            qubits = tuple(gate.qubits)
            if gate.name == "rz":
                first.append(_RuntimeGate("rz", qubits, (-float(gate.params[0]),)))
            elif gate.name in {"x", "ecr"}:
                first.append(_RuntimeGate(gate.name, qubits))
            elif gate.name == "sx":
                first.append(_RuntimeGate("rz", qubits, (math.pi,)))
                middle.append(_RuntimeGate("sx", qubits))
                last.append(_RuntimeGate("rz", qubits, (-math.pi,)))
            else:  # pragma: no cover - bank schema is exhaustive
                raise ValueError(f"unsupported native gate {gate.name!r}")
        if first:
            first_duration = max(
                public.native_durations.ecr_ns
                if gate.name == "ecr"
                else public.native_durations.x_ns
                if gate.name == "x"
                else public.native_durations.sx_ns
                if gate.name == "sx"
                else public.native_durations.rz_ns
                for gate in first
            )
            layers.append(_RuntimeLayer(first_duration, tuple(first)))
        if middle:
            layers.append(_RuntimeLayer(public.native_durations.sx_ns, tuple(middle)))
        if last:
            layers.append(_RuntimeLayer(public.native_durations.rz_ns, tuple(last)))
    return tuple(layers)


class ScheduledTransmonEngine:
    def __init__(
        self,
        hidden: HiddenScheduledTransmonConfig,
        public: PublicScheduledTransmonSpec,
        bank: CandidateBank,
        *,
        run_entropy: int | None = None,
    ) -> None:
        self.hidden = hidden
        self.public = public
        self.bank = bank
        if run_entropy is None:
            run_entropy = secrets.randbits(64)
        if isinstance(run_entropy, bool) or not isinstance(run_entropy, int) or run_entropy < 0:
            raise ValueError("run_entropy must be a non-negative integer")
        self._run_entropy = run_entropy
        self._candidates = bank.by_id()
        self._width = len(public.measured_physical_qubits)
        if bank.measured_physical_qubits != public.measured_physical_qubits:
            raise ValueError("candidate bank measured-qubit order disagrees with public device")
        if set(self._candidates) != set(public.candidate_ids):
            raise ValueError("candidate bank IDs disagree with public device")
        if len(hidden.idle_t2_us) != public.n_physical_qubits:
            raise ValueError("hidden idle_t2_us length disagrees with physical qubit count")
        self._ecr = {entry.edge: entry for entry in hidden.ecr_noise}
        if set(self._ecr) != set(public.directed_ecr_edges):
            raise ValueError("hidden directed-ECR table does not cover the public edge set")
        self._ecr_pauli_cdf: dict[tuple[int, int], np.ndarray] = {}
        for edge, entry in self._ecr.items():
            weights = np.asarray(entry.stochastic_pauli_weights, dtype=float)
            cumulative = np.cumsum(weights / np.sum(weights))
            cumulative[-1] = 1.0
            self._ecr_pauli_cdf[edge] = cumulative
        self._readout = {
            entry.physical_qubit: (entry.p_0_to_1, entry.p_1_to_0) for entry in hidden.readout
        }
        if set(self._readout) != set(range(public.n_physical_qubits)):
            raise ValueError("hidden readout table must cover every physical qubit")
        self._conflicts = {
            frozenset((entry.edge_a, entry.edge_b)): entry for entry in hidden.conflict_classes
        }

    @property
    def run_entropy(self) -> int:
        """Per-run entropy; never serialize it into public evidence."""

        return self._run_entropy

    def _rng(
        self,
        kind: str,
        submission_index: int,
        shot_salt: int,
        *request_parts: object,
    ) -> np.random.Generator:
        return np.random.default_rng(
            _seed_from_parts(
                self.hidden.seed,
                self._run_entropy,
                kind,
                submission_index,
                shot_salt,
                *request_parts,
            )
        )

    def _drift_multiplier(self, submission_index: int) -> float:
        """Legacy global-v1 multiplier retained for schema-v1 compatibility."""

        drift = self.hidden.drift
        phase = 2.0 * math.pi * submission_index / drift.period_jobs + drift.phase_rad
        return 1.0 + drift.peak_fraction * math.sin(phase)

    def _channel_drift_multiplier(
        self,
        submission_index: int,
        coupling: HiddenTemporalCoupling,
    ) -> float:
        if self.hidden.schema_version == 1:
            return self._drift_multiplier(submission_index)
        drift = self.hidden.drift
        phase = (
            2.0 * math.pi * submission_index / drift.period_jobs
            + drift.phase_rad
            + coupling.phase_offset_rad
        )
        return 1.0 + drift.peak_fraction * coupling.sensitivity * math.sin(phase)

    def edge_drift_multiplier(
        self,
        edge: tuple[int, int],
        submission_index: int,
        *,
        component: str,
    ) -> float:
        """Return one physical directed-edge multiplier for tests/construction."""

        entry = self._ecr[edge]
        if component == "stochastic":
            coupling = entry.stochastic_drift
        elif component == "coherent":
            coupling = entry.coherent_drift
        else:
            raise ValueError("component must be 'stochastic' or 'coherent'")
        return self._channel_drift_multiplier(submission_index, coupling)

    @staticmethod
    def _conjugate_gate(x: np.ndarray, z: np.ndarray, gate: _RuntimeGate) -> None:
        if gate.name == "x":
            return  # Pauli signs are irrelevant to computational-basis outcomes.
        if gate.name == "sx":
            q = gate.qubits[0]
            x[:, q] ^= z[:, q]
            return
        if gate.name == "rz":
            quarter_turns = int(round(gate.params[0] / (math.pi / 2.0)))
            if quarter_turns & 1:
                q = gate.qubits[0]
                z[:, q] ^= x[:, q]
            return
        if gate.name == "ecr":
            a, b = gate.qubits
            xa = x[:, a].copy()
            xb = x[:, b].copy()
            za = z[:, a].copy()
            zb = z[:, b].copy()
            # Qiskit ECR Clifford symplectic action for qargs [a, b].
            x[:, a] = xa
            x[:, b] = xa ^ xb ^ zb
            z[:, a] = xa ^ za ^ zb
            z[:, b] = zb
            return
        raise ValueError(f"unsupported runtime gate {gate.name!r}")

    @staticmethod
    def _inject_one_qubit_pauli(
        x: np.ndarray,
        z: np.ndarray,
        q: int,
        event: np.ndarray,
        pauli_code: np.ndarray,
    ) -> None:
        x[:, q] ^= event & ((pauli_code == 1) | (pauli_code == 2))
        z[:, q] ^= event & ((pauli_code == 2) | (pauli_code == 3))

    @classmethod
    def _inject_two_qubit_pauli(
        cls,
        x: np.ndarray,
        z: np.ndarray,
        a: int,
        b: int,
        event: np.ndarray,
        code: np.ndarray,
    ) -> None:
        cls._inject_one_qubit_pauli(x, z, a, event, code & 3)
        cls._inject_one_qubit_pauli(x, z, b, event, (code >> 2) & 3)

    def _coherent_first_occurrences(
        self,
        layers: tuple[_RuntimeLayer, ...],
        mirror_seed: int,
        half: str,
    ) -> tuple[dict[tuple[int, int], tuple[int, int]], dict[frozenset, tuple[int, int]]]:
        edge_layers: dict[tuple[int, int], list[int]] = {}
        conflict_layers: dict[frozenset, list[int]] = {}
        for layer_index, layer in enumerate(layers):
            edges = [tuple(gate.qubits) for gate in layer.gates if gate.name == "ecr"]
            for edge in edges:
                edge_layers.setdefault(edge, []).append(layer_index)
            for left in range(len(edges)):
                for right in range(left + 1, len(edges)):
                    key = frozenset((edges[left], edges[right]))
                    if key in self._conflicts:
                        conflict_layers.setdefault(key, []).append(layer_index)
        frame = public_mirror_frame(mirror_seed, self.public.n_physical_qubits)
        half_offset = 0 if half == "forward" else 1
        edge_plan: dict[tuple[int, int], tuple[int, int]] = {}
        for edge, occurrences in edge_layers.items():
            frame_a = frame[edge[0]]
            frame_b = frame[edge[1]]
            frame_word = frame_a[0] | (frame_a[1] << 1) | (frame_b[0] << 2) | (frame_b[1] << 3)
            coherent_axis = 1 + ((frame_word + half_offset) % 3)
            edge_plan[edge] = (occurrences[0], coherent_axis)
        conflict_plan = {
            key: (occurrences[0], len(occurrences)) for key, occurrences in conflict_layers.items()
        }
        return edge_plan, conflict_plan

    def _run_layers(
        self,
        x: np.ndarray,
        z: np.ndarray,
        layers: tuple[_RuntimeLayer, ...],
        *,
        rng: np.random.Generator,
        mirror_seed: int,
        half: str,
        submission_index: int,
    ) -> None:
        shots = x.shape[0]
        edge_plan, conflict_plan = self._coherent_first_occurrences(layers, mirror_seed, half)
        for layer_index, layer in enumerate(layers):
            active: set[int] = set()
            ecr_edges: list[tuple[int, int]] = []
            for gate in layer.gates:
                active.update(gate.qubits)
                self._conjugate_gate(x, z, gate)
                if gate.name in {"sx", "x"}:
                    p = self.hidden.one_qubit_depolarizing
                    event = rng.random(shots) < p
                    code = rng.integers(1, 4, size=shots, dtype=np.uint8)
                    self._inject_one_qubit_pauli(x, z, gate.qubits[0], event, code)
                elif gate.name == "ecr":
                    edge = (gate.qubits[0], gate.qubits[1])
                    ecr_edges.append(edge)
                    stochastic_drift = self.edge_drift_multiplier(
                        edge,
                        submission_index,
                        component="stochastic",
                    )
                    p = min(1.0, self._ecr[edge].stochastic_rate * stochastic_drift)
                    event = rng.random(shots) < p
                    code = (
                        np.searchsorted(
                            self._ecr_pauli_cdf[edge], rng.random(shots), side="right"
                        ).astype(np.uint8)
                        + 1
                    )
                    self._inject_two_qubit_pauli(x, z, edge[0], edge[1], event, code)

            # Stable edge overrotation coefficients accumulate across repeated uses
            # in a half. The virtual public twirl selects one of three effective
            # Pauli-event axes, preventing systematic forward/inverse cancellation.
            for edge in ecr_edges:
                first_layer, axis = edge_plan[edge]
                if first_layer != layer_index:
                    continue
                count = sum(
                    tuple(gate.qubits) == edge
                    for item in layers
                    for gate in item.gates
                    if gate.name == "ecr"
                )
                coherent_drift = self.edge_drift_multiplier(
                    edge,
                    submission_index,
                    component="coherent",
                )
                angle = count * self._ecr[edge].coherent_overrotation_rad * coherent_drift
                probability = math.sin(angle / 2.0) ** 2
                event = rng.random(shots) < probability
                code_value = {1: 6, 2: 9, 3: 14}[axis]
                code = np.full(shots, code_value, dtype=np.uint8)
                self._inject_two_qubit_pauli(x, z, edge[0], edge[1], event, code)

            for left in range(len(ecr_edges)):
                for right in range(left + 1, len(ecr_edges)):
                    key = frozenset((ecr_edges[left], ecr_edges[right]))
                    plan = conflict_plan.get(key)
                    if plan is None or plan[0] != layer_index:
                        continue
                    conflict = self._conflicts[key]
                    conflict_drift = self._channel_drift_multiplier(
                        submission_index,
                        conflict.drift,
                    )
                    angle = plan[1] * conflict.correlated_zz_angle_rad * conflict_drift
                    # ``correlated_zz_angle_rad`` is the coefficient in
                    # exp(-i * angle * ZZ), so a coherently accumulated class
                    # has Pauli-event weight sin(angle)^2.
                    event = rng.random(shots) < math.sin(angle) ** 2
                    for qubit in sorted(set(conflict.edge_a + conflict.edge_b)):
                        z[:, qubit] ^= event

            if layer.duration_ns:
                for qubit in self.public.measured_physical_qubits:
                    if qubit in active:
                        continue
                    t2_ns = self.hidden.idle_t2_us[qubit] * 1_000.0
                    probability = 0.5 * (1.0 - math.exp(-layer.duration_ns / t2_ns))
                    z[:, qubit] ^= rng.random(shots) < probability

    def run_mirror(
        self,
        request: CompilationMirrorRequest,
        *,
        submission_index: int,
        shot_salt: int | None = None,
    ) -> JobBitstringData:
        candidate = self._candidates[request.candidate_id]
        # ``mirror_seed`` controls only the published virtual coherent-axis twirl. Hidden
        # drift follows accepted submission order, and shot noise is independently
        # keyed by the hidden device seed, candidate, and that physical ordering.
        if shot_salt is None:
            shot_salt = submission_index
        rng = self._rng("mirror", submission_index, shot_salt, request.candidate_id)
        x = np.zeros((request.shots, self.public.n_physical_qubits), dtype=np.bool_)
        z = np.zeros_like(x)
        forward = _forward_layers(candidate)
        inverse = _inverse_layers(candidate, self.public)
        self._run_layers(
            x,
            z,
            forward,
            rng=rng,
            mirror_seed=request.mirror_seed,
            half="forward",
            submission_index=submission_index,
        )
        self._run_layers(
            x,
            z,
            inverse,
            rng=rng,
            mirror_seed=request.mirror_seed,
            half="inverse",
            submission_index=submission_index,
        )

        measured = self.public.measured_physical_qubits
        observed = x[:, measured].copy()
        for column, qubit in enumerate(measured):
            p01, p10 = self._readout[qubit]
            draw = rng.random(request.shots)
            observed[:, column] ^= np.where(observed[:, column], draw < p10, draw < p01)
        bitstrings = ["".join("1" if bit else "0" for bit in row) for row in observed]
        return JobBitstringData(bitstrings=[bitstrings], measured_qubits=measured)

    def run_readout_reference(
        self,
        request: CompilationReadoutReferenceRequest,
        *,
        submission_index: int,
        shot_salt: int | None = None,
    ) -> JobBitstringData:
        if shot_salt is None:
            shot_salt = submission_index
        rng = self._rng(
            "readout_reference",
            submission_index,
            shot_salt,
            request.prepared_bitstring,
        )
        measured = self.public.measured_physical_qubits
        truth = np.array([bit == "1" for bit in request.prepared_bitstring], dtype=np.bool_)
        observed = np.repeat(truth[None, :], request.shots, axis=0)
        for column, qubit in enumerate(measured):
            p01, p10 = self._readout[qubit]
            draw = rng.random(request.shots)
            observed[:, column] ^= np.where(observed[:, column], draw < p10, draw < p01)
        bitstrings = ["".join("1" if bit else "0" for bit in row) for row in observed]
        return JobBitstringData(bitstrings=[bitstrings], measured_qubits=measured)


def bitstring_counts(data: JobBitstringData) -> dict[str, int]:
    if len(data.bitstrings) != 1:
        raise ValueError("compilation jobs must return exactly one result point")
    return dict(sorted(Counter(data.bitstrings[0]).items()))


__all__ = ["ScheduledTransmonEngine", "bitstring_counts", "public_mirror_frame"]
