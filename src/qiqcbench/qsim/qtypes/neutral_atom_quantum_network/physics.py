"""Estimator statistics + communication-cost model for the two-node quantum network.

Pure-numpy. This module is the **single source of truth** shared by the engine (the agent's
experiment surface) and the hidden verifier (final-answer re-execution): both the deterministic
per-call communication cost and the estimator's sampled output come from here, so the agent's
measured behaviour and the scored re-execution cannot drift.

What it models (protocol/algorithm level — there is no atomic-physics pulse design):

* The decisive cross-node quantity is **one** distributed association ``c`` — the normalized
  co-occurrence of two binary cohorts (``x`` on Node A, ``y`` on Node B) over the same ``N``
  patients, a fixed value in ``[-1, 1]`` held jointly across the two nodes. Estimating it to
  additive error ``eps`` over the photonic link has a sharp communication hierarchy. We
  reproduce that hierarchy with faithful *statistical* models of each primitive (no 2^28
  statevector is ever built — the index register is ``n`` qubits and the estimators' output
  distributions are known closed-forms):

  - ``estimator_a`` (classical channel): additive std ``sqrt((1-c^2)/precision)``;
    cost ``precision`` bits. Error ~ ``1/sqrt(precision)`` -> ``1/eps^2``.
  - ``estimator_b`` (quantum channel): additive std ``sqrt((1-c^2)/precision)``; each unit of
    precision round-trips one ``(n+1)``-qubit register, so cost
    ``precision * (n+1)`` qubit-transmissions. Error ~ ``1/sqrt(precision)`` -> ``1/eps^2``.
  - ``estimator_c`` (quantum channel): additive std
    ``A_ae * sqrt(1-c^2) * 2^(-precision)``; cost
    ``2 * (n+1) * (2^precision - 1)`` qubit-transmissions. Error ~ ``2^(-precision)`` ->
    ``1/eps``.

* Every estimate is floored by a hidden **effective per-correlation noise floor** (residual
  error of the error-detected/heralded link + oracle/readout); the achievable accuracy is
  ``sqrt(estimator_var + floor^2)``. The floor is hidden truth (the public/stale notebook
  quotes a *different, pessimistic* value). Repeated device measurements can reveal the
  achievable accuracy at a chosen precision.

The instance values (the true association ``c`` and the floor) live in the hidden device
config; this module only holds the generic, instance-free math.
"""

from __future__ import annotations

import numpy as np

# Estimator method identifiers (the device's selectable primitives). Listing them is an API
# affordance, not a hint: the device does not rank them, and the accuracy each achieves is
# hidden behind the calibratable noise floor.
ESTIMATOR_A = "estimator_a"
ESTIMATOR_B = "estimator_b"
ESTIMATOR_C = "estimator_c"
ESTIMATORS = (ESTIMATOR_A, ESTIMATOR_B, ESTIMATOR_C)
QUANTUM_ESTIMATORS = (ESTIMATOR_B, ESTIMATOR_C)

# Pinned amplitude-estimation prefactor (the additive error on the association per precision
# register; folds the variance prefactor sqrt(1-c^2) and the canonical AE constant).
A_AE = 1.0

# Precision-range guards (the budget makes the real ceiling much tighter).
MAX_T = 16
MAX_SAMPLES = 5_000_000


class EstimatorError(ValueError):
    """Raised on a malformed estimator / precision request."""


def channel_of(estimator: str) -> str:
    if estimator == ESTIMATOR_A:
        return "classical"
    if estimator in QUANTUM_ESTIMATORS:
        return "quantum"
    raise EstimatorError(f"unknown estimator {estimator!r}; expected one of {ESTIMATORS}")


def validate_precision(estimator: str, precision: int) -> int:
    p = int(precision)
    if estimator == ESTIMATOR_C:
        if not (1 <= p <= MAX_T):
            raise EstimatorError(f"{estimator} precision must be in [1,{MAX_T}]")
    else:
        if not (1 <= p <= MAX_SAMPLES):
            raise EstimatorError(f"{estimator} precision must be in [1,{MAX_SAMPLES}]")
    return p


def cost(estimator: str, precision: int, n_index_bits: int) -> int:
    """Communication units to estimate the association ONCE (deterministic, public cost model).

    Quantum-channel costs are qubit-transmissions of an ``(n_index_bits + 1)``-qubit register;
    the classical-channel cost is in bits. Both count against the same communication budget.
    """
    precision = validate_precision(estimator, precision)
    reg = n_index_bits + 1
    if estimator == ESTIMATOR_C:
        return 2 * reg * (2**precision - 1)
    if estimator == ESTIMATOR_B:
        return precision * reg
    if estimator == ESTIMATOR_A:
        return precision  # bits
    raise EstimatorError(f"unknown estimator {estimator!r}")


def estimator_std(estimator: str, precision: int, c: float, floor: float) -> float:
    """Achievable additive std of the association estimate at the given precision + floor."""
    base = max(0.0, 1.0 - c * c)
    if estimator == ESTIMATOR_C:
        var = (A_AE * np.sqrt(base) * 2.0 ** (-precision)) ** 2
    elif estimator in (ESTIMATOR_A, ESTIMATOR_B):
        var = base / float(precision)
    else:
        raise EstimatorError(f"unknown estimator {estimator!r}")
    return float(np.sqrt(var + floor * floor))


def draw_estimate(
    true_c: float,
    estimator: str,
    precision: int,
    n_index_bits: int,
    floor: float,
    rng: np.random.Generator,
) -> float:
    """Sample one association estimate from the estimator's output distribution.

    Faithful statistical model: ``c_hat = clip(c + N(0, std^2), -1, 1)`` with ``std`` the
    estimator's achievable additive error at the given precision and the hidden floor. The
    cost is charged separately via :func:`cost`.
    """
    precision = validate_precision(estimator, precision)
    std = estimator_std(estimator, precision, float(true_c), floor)
    return float(np.clip(float(true_c) + rng.normal(0.0, std), -1.0, 1.0))
