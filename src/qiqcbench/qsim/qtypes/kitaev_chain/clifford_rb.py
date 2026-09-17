"""Canonical single-qubit Clifford compiler and qsim-owned RB sampler."""

from __future__ import annotations

from collections import deque
from functools import lru_cache

import numpy as np

from qiqcbench.qsim.qtypes.kitaev_chain import pulse as PU
from qiqcbench.qsim.qtypes.kitaev_chain.wire import (
    KitaevCliffordRBData,
    MajoranaCliffordRBRequest,
)

_I3 = np.eye(3, dtype=int)
_ROTATIONS = {
    "x90": np.array(((1, 0, 0), (0, 0, -1), (0, 1, 0)), dtype=int),
    "z90": np.array(((0, -1, 0), (1, 0, 0), (0, 0, 1)), dtype=int),
}
_TOKENS = ("x90", "z90")


def _key(rotation: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in rotation.reshape(-1))


def ideal_rotation(word: tuple[str, ...]) -> np.ndarray:
    """Return the exact signed-permutation Bloch rotation for ``word``."""

    rotation = _I3.copy()
    for token in word:
        try:
            generator = _ROTATIONS[token]
        except KeyError as exc:
            raise ValueError(f"unknown Clifford compiler token {token!r}") from exc
        rotation = generator @ rotation
    return rotation


@lru_cache(maxsize=1)
def clifford_compilations() -> tuple[tuple[str, ...], ...]:
    """Shortest canonical words for all 24 single-qubit Cliffords.

    Breadth-first order is the public, stable Clifford-index convention.  Only
    positive X90/Z90 primitives are needed; inverse Cliffords are compiled by
    the same table rather than synthesized through an unphysical inverse pulse.
    """

    words: list[tuple[str, ...]] = [()]
    rotations: list[np.ndarray] = [_I3.copy()]
    seen = {_key(_I3)}
    pending: deque[int] = deque((0,))
    while pending:
        index = pending.popleft()
        for token in _TOKENS:
            rotation = _ROTATIONS[token] @ rotations[index]
            key = _key(rotation)
            if key in seen:
                continue
            seen.add(key)
            words.append((*words[index], token))
            rotations.append(rotation)
            pending.append(len(words) - 1)
    if len(words) != 24:  # pragma: no cover - module invariant
        raise RuntimeError(f"X90/Z90 compiler generated {len(words)} Cliffords, expected 24")
    return tuple(words)


@lru_cache(maxsize=1)
def _index_by_rotation() -> dict[tuple[int, ...], int]:
    return {_key(ideal_rotation(word)): index for index, word in enumerate(clifford_compilations())}


def recovery_word(clifford_indices: tuple[int, ...] | list[int]) -> tuple[str, ...]:
    """Compile the exact inverse of an ideal Clifford sequence."""

    words = clifford_compilations()
    net = _I3.copy()
    for raw_index in clifford_indices:
        index = int(raw_index)
        if not 0 <= index < len(words):
            raise ValueError("Clifford index is out of range")
        net = ideal_rotation(words[index]) @ net
    inverse_index = _index_by_rotation()[_key(net.T)]
    return words[inverse_index]


def _wire_segments(raw) -> list[PU.PulseSegment]:
    return [
        PU.PulseSegment(
            amp=segment.amp,
            phase_rad=segment.phase_rad,
            detuning_ueV=segment.detuning_ueV,
            duration_ns=segment.duration_ns,
            drag=segment.drag,
        )
        for segment in raw
    ]


def compile_pulse_sequence(
    clifford_indices: tuple[int, ...] | list[int],
    *,
    x90: list[PU.PulseSegment],
    z90: list[PU.PulseSegment],
) -> tuple[list[PU.PulseSegment], int]:
    """Compile random Cliffords and their exact inverse through one table."""

    words = clifford_compilations()
    sequence: list[PU.PulseSegment] = []
    primitive_count = 0
    for index in clifford_indices:
        for token in words[int(index)]:
            sequence.extend(x90 if token == "x90" else z90)
            primitive_count += 1
    for token in recovery_word(clifford_indices):
        sequence.extend(x90 if token == "x90" else z90)
        primitive_count += 1
    return sequence, primitive_count


def run_clifford_rb(
    request: MajoranaCliffordRBRequest,
    params: PU.PulseParams,
    rng: np.random.Generator,
) -> KitaevCliffordRBData:
    """Draw uniform Cliffords privately and return raw level-resolved shots."""

    x90 = _wire_segments(request.gate_x90_segments)
    z90 = _wire_segments(request.gate_z90_segments)
    # Circuit selection and physical outcomes are scientifically distinct
    # random stages. Spawn separate streams so changing the shot precision does
    # not silently change the Clifford population selected by the same
    # regression entropy. Production still receives fresh parent entropy.
    seed_words = rng.integers(0, 2**32, size=8, dtype=np.uint32)
    clifford_seed, outcome_seed = np.random.SeedSequence(seed_words).spawn(2)
    clifford_rng = np.random.default_rng(clifford_seed)
    outcome_seeds = iter(outcome_seed.spawn(len(request.lengths) * request.sequences_per_length))
    shape = (len(request.lengths), request.sequences_per_length, request.shots_per_sequence)
    levels = np.empty(shape, dtype=np.uint8)
    all_indices: list[list[list[int]]] = []
    all_counts: list[list[int]] = []
    for length_index, length in enumerate(request.lengths):
        indices_at_length: list[list[int]] = []
        counts_at_length: list[int] = []
        for sequence_index in range(request.sequences_per_length):
            indices = [
                int(value)
                for value in clifford_rng.integers(
                    0,
                    len(clifford_compilations()),
                    size=length,
                )
            ]
            sequence, primitive_count = compile_pulse_sequence(indices, x90=x90, z90=z90)
            levels[length_index, sequence_index] = PU.sample_outcomes(
                sequence,
                params,
                np.random.default_rng(next(outcome_seeds)),
                request.shots_per_sequence,
            )
            indices_at_length.append(indices)
            counts_at_length.append(primitive_count)
        all_indices.append(indices_at_length)
        all_counts.append(counts_at_length)
    import base64

    return KitaevCliffordRBData(
        lengths=list(request.lengths),
        sequences_per_length=request.sequences_per_length,
        shots_per_sequence=request.shots_per_sequence,
        clifford_indices=all_indices,
        compiled_primitive_counts=all_counts,
        levels_b64=base64.b64encode(np.ascontiguousarray(levels).tobytes()).decode("ascii"),
    )


__all__ = [
    "clifford_compilations",
    "compile_pulse_sequence",
    "ideal_rotation",
    "recovery_word",
    "run_clifford_rb",
]
