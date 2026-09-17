"""Descriptive evidence compatibility for time_budgeted_hamlearn_10q.

The diagnostic compares the submitted Hamiltonian with the raw outcomes from
completed qsim probes. It is intentionally not a binary gate: the same
adaptively chosen data are used to estimate the Hamiltonian, and a fixed
per-attempt goodness-of-fit cutoff would have an uncontrolled false-rejection
rate. Experiment-design identifiability belongs to task construction and
reference feasibility, not to the agent's binary verifier contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from qiqcbench.qsim.hidden_dynamics.time_budgeted_hamlearn_10q.dictionary import (
    N_QUBITS,
    N_TERMS,
    TERM_ORDER,
    pauli_string,
)

MAX_MEAN_BINOMIAL_DEVIANCE = 3.0
MAX_P90_BINOMIAL_DEVIANCE = 6.0

_VALID_STATES = frozenset({"x+", "x-", "y+", "y-", "z+", "z-"})
_PAULI_2X2 = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
_KET = {
    "z+": np.array([1, 0], dtype=complex),
    "z-": np.array([0, 1], dtype=complex),
    "x+": np.array([1, 1], dtype=complex) / np.sqrt(2),
    "x-": np.array([1, -1], dtype=complex) / np.sqrt(2),
    "y+": np.array([1, 1j], dtype=complex) / np.sqrt(2),
    "y-": np.array([1, -1j], dtype=complex) / np.sqrt(2),
}
_MULTIPLY = {
    ("I", "I"): (1.0 + 0.0j, "I"),
    ("I", "X"): (1.0 + 0.0j, "X"),
    ("I", "Y"): (1.0 + 0.0j, "Y"),
    ("I", "Z"): (1.0 + 0.0j, "Z"),
    ("X", "I"): (1.0 + 0.0j, "X"),
    ("Y", "I"): (1.0 + 0.0j, "Y"),
    ("Z", "I"): (1.0 + 0.0j, "Z"),
    ("X", "X"): (1.0 + 0.0j, "I"),
    ("Y", "Y"): (1.0 + 0.0j, "I"),
    ("Z", "Z"): (1.0 + 0.0j, "I"),
    ("X", "Y"): (1.0j, "Z"),
    ("Y", "X"): (-1.0j, "Z"),
    ("Y", "Z"): (1.0j, "X"),
    ("Z", "Y"): (-1.0j, "X"),
    ("Z", "X"): (1.0j, "Y"),
    ("X", "Z"): (-1.0j, "Y"),
}
_TERM_PAULIS = tuple(pauli_string(term) for term in TERM_ORDER)


@dataclass(frozen=True)
class EvidenceCertification:
    """Compatibility summary retained for diagnosis, never pass authority."""

    mean_binomial_deviance: float | None
    p90_binomial_deviance: float | None
    data_compatible: bool


def _product_expectation(labels: list[str], left: str, right: str) -> complex:
    phase = 1.0 + 0.0j
    value = 1.0
    for label, lhs, rhs in zip(labels, left, right, strict=True):
        local_phase, local_pauli = _MULTIPLY[(lhs, rhs)]
        phase *= local_phase
        if local_pauli == "I":
            continue
        if label[0].upper() != local_pauli:
            return 0.0 + 0.0j
        value *= 1.0 if label[1] == "+" else -1.0
    return phase * value


def linear_response_design_row(labels: list[str], observable: str) -> np.ndarray:
    """Return the construction-only t->0 design row for all public terms.

    Reference feasibility and identifiability tests use this helper. Runtime
    scoring deliberately does not call it.
    """
    row = np.zeros(N_TERMS, dtype=float)
    for index, term_pauli in enumerate(_TERM_PAULIS):
        po = _product_expectation(labels, term_pauli, observable)
        op = _product_expectation(labels, observable, term_pauli)
        row[index] = float(np.real(0.5j * (po - op)))
    return row


def _validated_rows(accepted_rows: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for row in accepted_rows:
        labels = row.get("initial_state")
        observable = row.get("observable_pauli")
        outcomes = row.get("raw_pauli_outcomes")
        repetitions = row.get("num_internal_repetitions")
        try:
            evolve_time = float(row.get("evolve_time_us"))
        except (TypeError, ValueError):
            continue
        if (
            not isinstance(labels, list)
            or len(labels) != N_QUBITS
            or any(label not in _VALID_STATES for label in labels)
            or not isinstance(observable, str)
            or len(observable) != N_QUBITS
            or any(character not in "IXYZ" for character in observable)
            or not math.isfinite(evolve_time)
            or evolve_time <= 0.0
            or not isinstance(outcomes, list)
            or type(repetitions) is not int
            or repetitions <= 0
            or len(outcomes) != repetitions
            or any(type(value) is not int or value not in (-1, 1) for value in outcomes)
        ):
            continue
        rows.append(
            {
                "initial_state": labels,
                "evolve_time_us": evolve_time,
                "observable_pauli": observable,
                "raw_pauli_outcomes": outcomes,
                "num_internal_repetitions": repetitions,
            }
        )
    return rows


def _full_pauli_matrix(pauli: str) -> np.ndarray:
    operator = np.array([[1.0 + 0.0j]])
    for character in pauli:
        operator = np.kron(operator, _PAULI_2X2[character])
    return operator


def _state_vector(labels: list[str]) -> np.ndarray:
    state = np.array([1.0 + 0.0j])
    for label in labels:
        state = np.kron(state, _KET[label])
    return state


def _apply_pauli(vector: np.ndarray, pauli: str) -> np.ndarray:
    tensor = vector.reshape((2,) * N_QUBITS)
    for qubit, character in enumerate(pauli):
        if character == "I":
            continue
        tensor = np.moveaxis(
            np.tensordot(_PAULI_2X2[character], tensor, axes=([1], [qubit])), 0, qubit
        )
    return tensor.reshape(-1)


def _exact_predictions(omega_hat: list[float], rows: list[dict]) -> list[float]:
    dimension = 1 << N_QUBITS
    hamiltonian = np.zeros((dimension, dimension), dtype=complex)
    for coefficient, pauli in zip(omega_hat, _TERM_PAULIS, strict=True):
        if coefficient != 0.0:
            hamiltonian += coefficient * _full_pauli_matrix(pauli)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

    predictions: list[float] = []
    for row in rows:
        state = _state_vector(row["initial_state"])
        coefficients = eigenvectors.conj().T @ state
        evolved = eigenvectors @ (
            np.exp(-0.5j * eigenvalues * row["evolve_time_us"]) * coefficients
        )
        expectation = float(
            np.real(np.vdot(evolved, _apply_pauli(evolved, row["observable_pauli"])))
        )
        predictions.append(min(max(expectation, -1.0), 1.0))
    return predictions


def _binomial_deviance(outcomes: list[int], expectation: float) -> float:
    repetitions = len(outcomes)
    positive = sum(value == 1 for value in outcomes)
    observed_probability = positive / repetitions
    predicted_probability = min(max((1.0 + expectation) / 2.0, 1e-12), 1.0 - 1e-12)
    deviance = 0.0
    if positive:
        deviance += positive * math.log(observed_probability / predicted_probability)
    negative = repetitions - positive
    if negative:
        deviance += negative * math.log(
            (1.0 - observed_probability) / (1.0 - predicted_probability)
        )
    return 2.0 * deviance


def certify_evidence(
    omega_hat: list[float],
    accepted_rows: list[dict],
) -> EvidenceCertification:
    """Summarize exact finite-time compatibility with verified outcomes."""
    rows = _validated_rows(accepted_rows)
    if not rows:
        return EvidenceCertification(
            mean_binomial_deviance=None,
            p90_binomial_deviance=None,
            data_compatible=False,
        )

    predictions = _exact_predictions(omega_hat, rows)
    deviances = np.asarray(
        [
            _binomial_deviance(row["raw_pauli_outcomes"], prediction)
            for row, prediction in zip(rows, predictions, strict=True)
        ],
        dtype=float,
    )
    mean_deviance = float(np.mean(deviances))
    p90_deviance = float(np.percentile(deviances, 90.0))
    compatible = bool(
        math.isfinite(mean_deviance)
        and math.isfinite(p90_deviance)
        and mean_deviance <= MAX_MEAN_BINOMIAL_DEVIANCE
        and p90_deviance <= MAX_P90_BINOMIAL_DEVIANCE
    )
    return EvidenceCertification(
        mean_binomial_deviance=mean_deviance,
        p90_binomial_deviance=p90_deviance,
        data_compatible=compatible,
    )


__all__ = [
    "EvidenceCertification",
    "MAX_MEAN_BINOMIAL_DEVIANCE",
    "MAX_P90_BINOMIAL_DEVIANCE",
    "certify_evidence",
    "linear_response_design_row",
]
