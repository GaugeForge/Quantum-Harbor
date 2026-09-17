"""Efficient density-matrix runtime for local randomized measurements."""

from __future__ import annotations

import time

import numpy as np

from qiqcbench.qsim.core.wire import JobResult, JobResultMetadata
from qiqcbench.qsim.qtypes.digital_gate_model.capabilities.randomized_measurement.materials import (
    LocalRandomizedMeasurementProtocol,
)
from qiqcbench.qsim.qtypes.digital_gate_model.capabilities.randomized_measurement.schemas import (
    JobLocalRandomizedMeasurementData,
    LocalRandomCircuitBitstrings,
)
from qiqcbench.qsim.qtypes.digital_gate_model.device import HiddenDigitalConfig
from qiqcbench.qsim.qtypes.digital_gate_model.gates import CNOT_GATE, PAULI_X
from qiqcbench.qsim.qtypes.digital_gate_model.runner import _make_rng


def _haar_su2(rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    unitary, triangular = np.linalg.qr(raw)
    diagonal = np.diag(triangular)
    phases = diagonal / np.where(np.abs(diagonal) == 0.0, 1.0, np.abs(diagonal))
    unitary = unitary @ np.diag(np.conjugate(phases))
    determinant = np.linalg.det(unitary)
    return np.asarray(unitary / determinant**0.5, dtype=np.complex128)


def _initial_density(n_qubits: int) -> np.ndarray:
    density = np.zeros((1 << n_qubits, 1 << n_qubits), dtype=np.complex128)
    density[0, 0] = 1.0
    return density


def apply_local_noisy_unitary(
    density: np.ndarray,
    unitary: np.ndarray,
    qubits: tuple[int, ...],
    *,
    n_qubits: int,
    total_nonidentity_pauli_probability: float,
    noise_repetitions: int = 1,
) -> np.ndarray:
    """Apply one local unitary and repeated local Pauli-depolarizing channels."""

    local_dim = 1 << len(qubits)
    if unitary.shape != (local_dim, local_dim):
        raise ValueError("local unitary shape does not match its qubits")
    if len(set(qubits)) != len(qubits) or any(q < 0 or q >= n_qubits for q in qubits):
        raise ValueError("local unitary qubits are invalid")
    if not (0.0 <= total_nonidentity_pauli_probability < 1.0):
        raise ValueError("depolarizing probability must lie in [0,1)")
    if noise_repetitions < 1:
        raise ValueError("noise_repetitions must be positive")

    ket_axes = [n_qubits - 1 - qubit for qubit in reversed(qubits)]
    bra_axes = [2 * n_qubits - 1 - qubit for qubit in reversed(qubits)]
    front = ket_axes + bra_axes
    permutation = front + [axis for axis in range(2 * n_qubits) if axis not in front]
    inverse_permutation = np.argsort(permutation)
    block = density.reshape((2,) * (2 * n_qubits)).transpose(permutation)
    rest_shape = block.shape[2 * len(qubits) :]
    block = block.reshape(local_dim, local_dim, -1)
    block = np.einsum("ai,bj,ijr->abr", unitary, unitary.conj(), block, optimize=True)

    if total_nonidentity_pauli_probability > 0.0:
        one_pass_mix = (
            total_nonidentity_pauli_probability
            * local_dim
            * local_dim
            / (local_dim * local_dim - 1.0)
        )
        total_mix = 1.0 - (1.0 - one_pass_mix) ** noise_repetitions
        partial_trace = np.einsum("iir->r", block)
        block *= 1.0 - total_mix
        diagonal_addition = (total_mix / local_dim) * partial_trace
        for index in range(local_dim):
            block[index, index, :] += diagonal_addition

    restored = block.reshape((2,) * (2 * len(qubits)) + rest_shape).transpose(inverse_permutation)
    return np.asarray(restored.reshape(density.shape), dtype=np.complex128)


def _prepare_random_circuit(
    *,
    depth: int,
    hidden: HiddenDigitalConfig,
    protocol: LocalRandomizedMeasurementProtocol,
    rng: np.random.Generator,
) -> np.ndarray:
    density = _initial_density(protocol.n_qubits)
    p1 = hidden.gate_errors.one_qubit_depolarizing
    p2 = hidden.gate_errors.two_qubit_cx_depolarizing
    for _ in range(depth):
        for qubit in range(protocol.n_qubits):
            density = apply_local_noisy_unitary(
                density,
                _haar_su2(rng),
                (qubit,),
                n_qubits=protocol.n_qubits,
                total_nonidentity_pauli_probability=p1,
                noise_repetitions=3,
            )
        for edge in protocol.depth_cycle.even_cnot_edges:
            density = apply_local_noisy_unitary(
                density,
                CNOT_GATE,
                edge,
                n_qubits=protocol.n_qubits,
                total_nonidentity_pauli_probability=p2,
            )
        for qubit in range(protocol.n_qubits):
            density = apply_local_noisy_unitary(
                density,
                _haar_su2(rng),
                (qubit,),
                n_qubits=protocol.n_qubits,
                total_nonidentity_pauli_probability=p1,
                noise_repetitions=3,
            )
        for edge in protocol.depth_cycle.odd_cnot_edges:
            density = apply_local_noisy_unitary(
                density,
                CNOT_GATE,
                edge,
                n_qubits=protocol.n_qubits,
                total_nonidentity_pauli_probability=p2,
            )
    return density


def _apply_readout(
    probabilities: np.ndarray,
    hidden: HiddenDigitalConfig,
    *,
    n_qubits: int,
) -> np.ndarray:
    by_id = {entry.id: entry for entry in hidden.readout}
    if set(by_id) != set(range(n_qubits)):
        raise ValueError("hidden readout configuration does not cover the full register")
    tensor = probabilities.reshape((2,) * n_qubits)
    for qubit in range(n_qubits):
        entry = by_id[qubit]
        axis = n_qubits - 1 - qubit
        shape_without_axis = tuple(size for index, size in enumerate(tensor.shape) if index != axis)
        moved = np.moveaxis(tensor, axis, 0).reshape(2, -1)
        confusion = np.array(
            [
                [1.0 - entry.p_0_to_1, entry.p_1_to_0],
                [entry.p_0_to_1, 1.0 - entry.p_1_to_0],
            ],
            dtype=np.float64,
        )
        moved = confusion @ moved
        tensor = np.moveaxis(moved.reshape((2,) + shape_without_axis), 0, axis)
    observed = np.clip(tensor.reshape(-1).real, 0.0, None)
    total = float(observed.sum())
    if total <= 0.0:
        raise ValueError("measurement distribution has zero mass")
    return observed / total


def _sample_bitstrings(
    probabilities: np.ndarray,
    *,
    shots: int,
    measured_qubits: tuple[int, ...],
    rng: np.random.Generator,
) -> list[str]:
    outcomes = rng.choice(len(probabilities), size=shots, p=probabilities)
    return [
        "".join(str((int(outcome) >> qubit) & 1) for qubit in measured_qubits)
        for outcome in outcomes
    ]


def _measurement_probabilities(
    prepared: np.ndarray,
    *,
    hidden: HiddenDigitalConfig,
    protocol: LocalRandomizedMeasurementProtocol,
    rng: np.random.Generator,
) -> np.ndarray:
    density = prepared.copy()
    p1 = hidden.gate_errors.one_qubit_depolarizing
    for qubit in range(protocol.n_qubits):
        density = apply_local_noisy_unitary(
            density,
            _haar_su2(rng),
            (qubit,),
            n_qubits=protocol.n_qubits,
            total_nonidentity_pauli_probability=p1,
            noise_repetitions=3,
        )
    return _apply_readout(np.diag(density).real, hidden, n_qubits=protocol.n_qubits)


def _calibration_shots(
    hidden: HiddenDigitalConfig,
    protocol: LocalRandomizedMeasurementProtocol,
    rng: np.random.Generator,
) -> tuple[list[str], list[str]]:
    measured = protocol.measurement.measured_qubits
    shots = protocol.sampling.readout_calibration_shots_per_preparation
    zero = _initial_density(protocol.n_qubits)
    one = _initial_density(protocol.n_qubits)
    for qubit in range(protocol.n_qubits):
        one = apply_local_noisy_unitary(
            one,
            PAULI_X,
            (qubit,),
            n_qubits=protocol.n_qubits,
            total_nonidentity_pauli_probability=hidden.gate_errors.one_qubit_depolarizing,
        )
    zero_probabilities = _apply_readout(np.diag(zero).real, hidden, n_qubits=protocol.n_qubits)
    one_probabilities = _apply_readout(np.diag(one).real, hidden, n_qubits=protocol.n_qubits)
    return (
        _sample_bitstrings(zero_probabilities, shots=shots, measured_qubits=measured, rng=rng),
        _sample_bitstrings(one_probabilities, shots=shots, measured_qubits=measured, rng=rng),
    )


def run_local_randomized_measurement(
    *,
    depth: int,
    hidden: HiddenDigitalConfig,
    protocol: LocalRandomizedMeasurementProtocol,
    job_id: str,
    salt: int,
    run_entropy: int | None = None,
) -> JobResult:
    started = time.perf_counter()
    try:
        if len(hidden.readout) != protocol.n_qubits:
            raise ValueError("protocol qubit count does not match the active device")
        if not protocol.depth_limits.minimum <= depth <= protocol.depth_limits.maximum:
            raise ValueError(
                f"depth must lie in [{protocol.depth_limits.minimum}, "
                f"{protocol.depth_limits.maximum}]"
            )
        rng = _make_rng(hidden, salt, run_entropy)
        calibration_zero, calibration_one = _calibration_shots(hidden, protocol, rng)
        circuits: list[LocalRandomCircuitBitstrings] = []
        for circuit_index in range(protocol.sampling.n_random_circuits):
            prepared = _prepare_random_circuit(
                depth=depth,
                hidden=hidden,
                protocol=protocol,
                rng=rng,
            )
            basis_bitstrings: list[list[str]] = []
            for _ in range(protocol.sampling.n_measurement_bases):
                probabilities = _measurement_probabilities(
                    prepared,
                    hidden=hidden,
                    protocol=protocol,
                    rng=rng,
                )
                basis_bitstrings.append(
                    _sample_bitstrings(
                        probabilities,
                        shots=protocol.sampling.shots_per_basis,
                        measured_qubits=protocol.measurement.measured_qubits,
                        rng=rng,
                    )
                )
            circuits.append(
                LocalRandomCircuitBitstrings(
                    circuit_index=circuit_index,
                    basis_bitstrings=basis_bitstrings,
                )
            )
        data = JobLocalRandomizedMeasurementData(
            depth=depth,
            measured_qubits=list(protocol.measurement.measured_qubits),
            n_random_circuits=protocol.sampling.n_random_circuits,
            n_measurement_bases=protocol.sampling.n_measurement_bases,
            shots_per_basis=protocol.sampling.shots_per_basis,
            readout_calibration_shots_per_preparation=(
                protocol.sampling.readout_calibration_shots_per_preparation
            ),
            readout_calibration_zero_bitstrings=calibration_zero,
            readout_calibration_one_bitstrings=calibration_one,
            circuits=circuits,
        )
    except ValueError as exc:
        return JobResult(
            job_id=job_id,
            device_id=hidden.device_id,
            status="failed",
            error=str(exc),
            metadata=JobResultMetadata(wallclock_ms=int((time.perf_counter() - started) * 1000)),
        )
    return JobResult(
        job_id=job_id,
        device_id=hidden.device_id,
        status="complete",
        shots=protocol.sampling.shots_per_basis,
        data=data,
        metadata=JobResultMetadata(wallclock_ms=int((time.perf_counter() - started) * 1000)),
    )
