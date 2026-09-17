"""Circuit-level Rydberg physics and the fixed unrotated-code decoder."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import stim

from qiqcbench.qsim.qtypes.rydberg_multitarget_surface_code.device import (
    HiddenRydbergMultitargetNoise,
    PublicRydbergMultitargetSurfaceCodeSpec,
    StabilizerCheckSpec,
)
from qiqcbench.qsim.qtypes.rydberg_multitarget_surface_code.wire import (
    MultitargetStabilizerCycle,
)


def pack_bits(bits: np.ndarray) -> str:
    """Pack a C-order binary array using the public big-bit-order contract."""
    value = np.ascontiguousarray(bits, dtype=np.uint8)
    if np.any((value != 0) & (value != 1)):
        raise ValueError("only binary arrays can be packed")
    return base64.b64encode(np.packbits(value.reshape(-1), bitorder="big").tobytes()).decode(
        "ascii"
    )


def unpack_bits(value: str, shape: tuple[int, ...]) -> np.ndarray:
    """Decode packed bits and reject non-canonical padding or encoded length."""
    if not shape or any(type(item) is not int or item < 0 for item in shape):
        raise ValueError("packed-bit shape must contain non-negative JSON integers")
    count = math.prod(shape)
    expected_bytes = (count + 7) // 8
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("packed bits are not canonical Base64") from exc
    if len(raw) != expected_bytes:
        raise ValueError("packed-bit byte length does not match shape")
    unpacked = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="big")
    if np.any(unpacked[count:]):
        raise ValueError("packed-bit trailing padding must be zero")
    return unpacked[:count].reshape(shape)


def _normalized_pairs(pairs: list[list[str]]) -> list[list[str]]:
    if len(pairs) != 2:
        raise ValueError("each bulk check must declare exactly two target_role_pairs")
    normalized: list[list[str]] = []
    for pair in pairs:
        if len(pair) != 2 or any(not isinstance(role, str) for role in pair):
            raise ValueError("every target role pair must contain exactly two role strings")
        if pair[0] == pair[1]:
            raise ValueError("a target role pair cannot repeat a data role")
        normalized.append(sorted(pair))
    normalized.sort()
    return normalized


def canonical_cycle(
    cycle: MultitargetStabilizerCycle,
    public: PublicRydbergMultitargetSurfaceCodeSpec,
) -> dict[str, Any]:
    """Canonicalize a complete legal partition without classifying its performance."""
    public_checks = {
        check.bulk_design_id: check
        for check in public.bulk_checks
        if check.bulk_design_id is not None
    }
    submitted: dict[str, list[list[str]]] = {}
    for partition in cycle.bulk_check_partitions:
        if partition.check_id not in public_checks or partition.check_id in submitted:
            raise ValueError("bulk_check_partitions must contain every public bulk ID once")
        pairs = _normalized_pairs(partition.target_role_pairs)
        flattened = sorted(role for pair in pairs for role in pair)
        if flattened != sorted(public_checks[partition.check_id].data_roles):
            raise ValueError(
                f"{partition.check_id} target_role_pairs must partition its four data roles"
            )
        submitted[partition.check_id] = pairs
    if set(submitted) != set(public_checks):
        raise ValueError("bulk_check_partitions must contain every public bulk ID once")
    return {
        "mode": "cz2_depth_reduced",
        "bulk_check_partitions": [
            {"check_id": bulk_id, "target_role_pairs": submitted[bulk_id]}
            for bulk_id in public_checks
        ],
    }


def canonical_cycle_digest(
    cycle: MultitargetStabilizerCycle,
    public: PublicRydbergMultitargetSurfaceCodeSpec,
) -> str:
    payload = json.dumps(
        canonical_cycle(cycle, public),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def cycle_duration_us(
    public: PublicRydbergMultitargetSurfaceCodeSpec, duration_scale: float
) -> float:
    gates = public.native_gates
    return (
        4.0 * gates.cz2_us * duration_scale
        + 4.0 * gates.local_layer_us
        + gates.measurement_reset_us
    )


def correlated_pair_probability(
    noise: HiddenRydbergMultitargetNoise,
    *,
    duration_scale: float,
    target_phase_compensation_rad: float,
) -> float:
    probability = (
        noise.correlated_pair_error_at_optimum
        + noise.correlated_pair_duration_curvature
        * (duration_scale - noise.duration_scale_optimum) ** 2
        + noise.correlated_pair_phase_curvature
        * (target_phase_compensation_rad - noise.target_phase_compensation_rad_optimum) ** 2
    )
    return float(np.clip(probability, 1.0e-9, 0.20))


def adjacent_arm_partitions(check: StabilizerCheckSpec) -> tuple[tuple[tuple[str, str], ...], ...]:
    """Return the two reflection-equivalent fault-preserving local partitions."""
    if len(check.data_roles) != 4:
        raise ValueError("adjacent-arm partitions apply only to weight-four checks")
    north, west, east, south = check.data_roles
    return (
        ((north, west), (east, south)),
        ((north, east), (west, south)),
    )


def axial_partition(check: StabilizerCheckSpec) -> tuple[tuple[str, str], ...]:
    """Return the topology-derived hook-unsafe axial partition for tests/construction."""
    if len(check.data_roles) != 4:
        raise ValueError("axial partitions apply only to weight-four checks")
    north, west, east, south = check.data_roles
    return ((north, south), (west, east))


def _role_to_simulator_index(public: PublicRydbergMultitargetSurfaceCodeSpec) -> dict[str, int]:
    return {item.role: item.simulator_index for item in public.data_qubits}


def _append_pair_channels(
    circuit: stim.Circuit,
    *,
    role_pairs: list[list[str]] | tuple[tuple[str, str], ...],
    role_indices: dict[str, int],
    probability: float,
) -> None:
    for first, second in role_pairs:
        circuit.append("DEPOLARIZE2", [role_indices[first], role_indices[second]], probability)


def _generated_memory_circuit(
    *,
    basis: Literal["Z", "X"],
    rounds: int,
    independent_error: float,
    measurement_flip_probability: float,
    reset_flip_probability: float,
) -> stim.Circuit:
    return stim.Circuit.generated(
        f"surface_code:unrotated_memory_{basis.lower()}",
        distance=3,
        rounds=rounds,
        after_clifford_depolarization=independent_error,
        before_measure_flip_probability=measurement_flip_probability,
        after_reset_flip_probability=reset_flip_probability,
    )


def _compile_pair_faults(
    base: stim.Circuit,
    *,
    pair_groups: list[list[list[str]] | tuple[tuple[str, str], ...]],
    role_indices: dict[str, int],
    per_group_probability: float,
) -> stim.Circuit:
    """Insert native pair channels immediately before every round's ancilla readout."""
    compiled = stim.Circuit()
    for instruction in base.flattened():
        if instruction.name == "MR":
            for role_pairs in pair_groups:
                _append_pair_channels(
                    compiled,
                    role_pairs=role_pairs,
                    role_indices=role_indices,
                    probability=per_group_probability,
                )
        compiled.append(instruction)
    return compiled


def build_memory_circuit(
    public: PublicRydbergMultitargetSurfaceCodeSpec,
    noise: HiddenRydbergMultitargetNoise,
    *,
    cycle: MultitargetStabilizerCycle,
    duration_scale: float,
    target_phase_compensation_rad: float,
    basis: Literal["Z", "X"],
    rounds: int,
) -> stim.Circuit:
    """Compile a submitted design into one noisy circuit-level memory experiment."""
    canonical = canonical_cycle(cycle, public)
    duration = cycle_duration_us(public, duration_scale)
    independent_error = min(
        0.05,
        noise.independent_clifford_error + noise.idle_depolarization_per_us * duration,
    )
    base = _generated_memory_circuit(
        basis=basis,
        rounds=rounds,
        independent_error=independent_error,
        measurement_flip_probability=noise.measurement_flip_probability,
        reset_flip_probability=noise.reset_flip_probability,
    )
    groups = [item["target_role_pairs"] for item in canonical["bulk_check_partitions"]]
    return _compile_pair_faults(
        base,
        pair_groups=groups,
        role_indices=_role_to_simulator_index(public),
        per_group_probability=correlated_pair_probability(
            noise,
            duration_scale=duration_scale,
            target_phase_compensation_rad=target_phase_compensation_rad,
        ),
    )


def build_fixed_decoder_circuit(
    public: PublicRydbergMultitargetSurfaceCodeSpec,
    *,
    basis: Literal["Z", "X"],
    rounds: int,
) -> stim.Circuit:
    """Build the frozen public nominal model containing both safe reflections."""
    spec = public.fixed_decoder
    base = _generated_memory_circuit(
        basis=basis,
        rounds=rounds,
        independent_error=spec.nominal_independent_clifford_error,
        measurement_flip_probability=spec.nominal_measurement_flip_probability,
        reset_flip_probability=spec.nominal_reset_flip_probability,
    )
    groups: list[tuple[tuple[str, str], ...]] = []
    for check in public.bulk_checks:
        groups.extend(adjacent_arm_partitions(check))
    return _compile_pair_faults(
        base,
        pair_groups=groups,
        role_indices=_role_to_simulator_index(public),
        per_group_probability=spec.nominal_correlated_pair_error / 2.0,
    )


def build_fixed_decoder(
    public: PublicRydbergMultitargetSurfaceCodeSpec,
    *,
    basis: Literal["Z", "X"],
    rounds: int,
) -> Any:
    """Construct MWPM from the frozen public nominal detector error model."""
    import pymatching

    decoder_circuit = build_fixed_decoder_circuit(public, basis=basis, rounds=rounds)
    detector_error_model = decoder_circuit.detector_error_model(decompose_errors=True)
    return pymatching.Matching.from_detector_error_model(detector_error_model)


@dataclass(frozen=True)
class RawMemorySample:
    circuit: stim.Circuit
    complete_measurements: np.ndarray
    syndrome_measurements: np.ndarray
    final_data_measurements: np.ndarray


def sample_memory_measurements(
    circuit: stim.Circuit,
    *,
    shots: int,
    rng: np.random.Generator,
    rounds: int,
) -> RawMemorySample:
    """Sample raw digital records and split their two public measurement regions."""
    seed = int(rng.integers(1, np.iinfo(np.uint64).max, dtype=np.uint64))
    measurements = circuit.compile_sampler(seed=seed).sample(shots).astype(np.uint8, copy=False)
    expected = rounds * 12 + 13
    if measurements.shape != (shots, expected):
        raise RuntimeError("unrotated memory circuit emitted an unexpected measurement record")
    syndrome = measurements[:, : rounds * 12].reshape(shots, rounds, 12)
    final_data = measurements[:, rounds * 12 :]
    return RawMemorySample(
        circuit=circuit,
        complete_measurements=measurements,
        syndrome_measurements=syndrome,
        final_data_measurements=final_data,
    )


def decode_memory_measurements(
    public: PublicRydbergMultitargetSurfaceCodeSpec,
    sample: RawMemorySample,
    *,
    basis: Literal["Z", "X"],
    rounds: int,
) -> np.ndarray:
    """Return per-shot decoder failures from raw measurements and the fixed decoder."""
    converter = sample.circuit.compile_m2d_converter()
    detectors, observables = converter.convert(
        measurements=sample.complete_measurements.astype(np.bool_, copy=False),
        separate_observables=True,
    )
    predictions = build_fixed_decoder(public, basis=basis, rounds=rounds).decode_batch(detectors)
    if observables.shape[1] != 1 or predictions.shape[1] != 1:
        raise RuntimeError("the fixed memory circuit must expose exactly one logical observable")
    return np.asarray(predictions[:, 0] ^ observables[:, 0], dtype=np.uint8)


__all__ = [
    "RawMemorySample",
    "adjacent_arm_partitions",
    "axial_partition",
    "build_fixed_decoder",
    "build_fixed_decoder_circuit",
    "build_memory_circuit",
    "canonical_cycle",
    "canonical_cycle_digest",
    "correlated_pair_probability",
    "cycle_duration_us",
    "decode_memory_measurements",
    "pack_bits",
    "sample_memory_measurements",
    "unpack_bits",
]
