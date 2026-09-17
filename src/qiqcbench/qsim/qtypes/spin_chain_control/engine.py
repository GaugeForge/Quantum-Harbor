"""Full 8x8 coherent propagation and raw readout sampling."""

from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy.linalg import expm

from qiqcbench.qsim.core.wire import JobBitstringData
from qiqcbench.qsim.qtypes.spin_chain_control.device import (
    HiddenSpinChainControlConfig,
    PublicSpinChainControlSpec,
)
from qiqcbench.qsim.qtypes.spin_chain_control.wire import (
    ControlReadoutReferenceRequest,
    ControlTransferRequest,
)

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.diag([1, -1]).astype(complex)


def _seed_from_parts(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _operator(single: np.ndarray, qubit: int) -> np.ndarray:
    result = np.array([[1.0 + 0.0j]])
    for index in range(3):
        result = np.kron(result, single if index == qubit else _I)
    return result


_X0 = _operator(_X, 0) / 2.0
_Y0 = _operator(_Y, 0) / 2.0
_EXCHANGE = (
    sum(
        (
            _operator(pauli, 0) @ _operator(pauli, 1) + _operator(pauli, 1) @ _operator(pauli, 2)
            for pauli in (_X, _Y, _Z)
        ),
        np.zeros((8, 8), dtype=complex),
    )
    / 4.0
)


class SpinChainControlEngine:
    def __init__(
        self,
        hidden: HiddenSpinChainControlConfig,
        public: PublicSpinChainControlSpec,
        *,
        run_entropy: int,
    ) -> None:
        if (
            isinstance(run_entropy, bool)
            or not isinstance(run_entropy, int)
            or not 0 <= run_entropy < 2**128
        ):
            raise ValueError("run_entropy must be a 128-bit non-negative integer")
        self.hidden = hidden
        self.public = public
        self._run_entropy = run_entropy
        self._controllers = {item.controller_id: item for item in hidden.controllers}
        self._readout = {item.qubit: (item.p_0_to_1, item.p_1_to_0) for item in hidden.readout}
        offset_rng = np.random.default_rng(_seed_from_parts(run_entropy, "run_offset"))
        self.run_offset_fraction = float(
            np.clip(
                offset_rng.normal(0.0, hidden.run_offset_sigma_fraction),
                -hidden.run_offset_clip_fraction,
                hidden.run_offset_clip_fraction,
            )
        )

    def _rng(self, kind: str, submission_index: int, *request: object) -> np.random.Generator:
        return np.random.default_rng(
            _seed_from_parts(self._run_entropy, kind, submission_index, *request)
        )

    def drift_fraction(self, submission_index: int) -> float:
        hidden = self.hidden
        phase = 2.0 * math.pi * (submission_index - 1) / hidden.drift_period_jobs
        return hidden.drift_peak_fraction * math.sin(phase + hidden.drift_phase_rad)

    def exact_populations(
        self,
        controller_id: str,
        x_drive_scale_fraction: float,
        *,
        initial_bitstring: str = "100",
    ) -> np.ndarray:
        controller = self._controllers[controller_id]
        state = np.zeros(8, dtype=complex)
        state[int(initial_bitstring, 2)] = 1.0
        exchange = self.hidden.exchange_rad_per_us * _EXCHANGE
        for x_drive, y_drive in zip(
            controller.x_drive_rad_per_us,
            controller.y_drive_rad_per_us,
            strict=True,
        ):
            hamiltonian = exchange + (1.0 + x_drive_scale_fraction) * x_drive * _X0 + y_drive * _Y0
            state = expm(-1j * controller.interval_duration_us * hamiltonian) @ state
        probabilities = np.abs(state) ** 2
        return probabilities / probabilities.sum()

    def _sample_with_readout(
        self, probabilities: np.ndarray, shots: int, rng: np.random.Generator
    ) -> list[str]:
        basis = rng.choice(8, size=shots, p=probabilities)
        observed = np.array(
            [[bool((int(value) >> (2 - qubit)) & 1) for qubit in range(3)] for value in basis],
            dtype=np.bool_,
        )
        for qubit in range(3):
            p01, p10 = self._readout[qubit]
            draw = rng.random(shots)
            observed[:, qubit] ^= np.where(observed[:, qubit], draw < p10, draw < p01)
        return ["".join("1" if bit else "0" for bit in row) for row in observed]

    def run_transfer(
        self, request: ControlTransferRequest, *, submission_index: int
    ) -> JobBitstringData:
        realized = (
            request.x_drive_scale_fraction
            + self.run_offset_fraction
            + self.drift_fraction(submission_index)
        )
        success = self.exact_populations(request.controller_id, realized, initial_bitstring="100")
        failed = self.exact_populations(request.controller_id, realized, initial_bitstring="000")
        preparation_failure = self.hidden.controller_preparation_failure
        probabilities = (1.0 - preparation_failure) * success + preparation_failure * failed
        rng = self._rng(
            "transfer",
            submission_index,
            request.controller_id,
            f"{request.x_drive_scale_fraction:.12g}",
        )
        return JobBitstringData(
            bitstrings=[self._sample_with_readout(probabilities, request.shots, rng)],
            measured_qubits=[0, 1, 2],
        )

    def run_readout_reference(
        self, request: ControlReadoutReferenceRequest, *, submission_index: int
    ) -> JobBitstringData:
        probabilities = np.zeros(8, dtype=float)
        probabilities[int(request.prepared_bitstring, 2)] = 1.0
        rng = self._rng("readout_reference", submission_index, request.prepared_bitstring)
        return JobBitstringData(
            bitstrings=[self._sample_with_readout(probabilities, request.shots, rng)],
            measured_qubits=[0, 1, 2],
        )


__all__ = ["SpinChainControlEngine"]
