"""Physics + estimator-sampling model for the trapped-ion state-copy device.

Pure-numpy. This module is the **single source of truth** shared by the engine (the agent's
measurement surface) and the hidden verifier (moment/ratio anchors), so the raw samples the
agent post-processes and the exact moments the verifier scores against cannot drift.

Scientific content (protocol/algorithm level):

* The hidden state is a six-ion long-range XX thermal state plus a small depolarizing admixture,
  ``rho = (1-lam) exp(-beta H_XX)/Z + lam I/64`` with
  ``H_XX = sum_{i<j} 2 J0/|i-j|^alpha (Xi Xj + Yi Yj) + sum_i B_i Zi`` (:func:`build_rho`). The
  64x64 matrix exponential is built once; the exact nonlinear moments ``Tr(rho^k)`` and
  ``Tr(Z0Z1 rho^k)`` come from :func:`exact_moments`.

* Virtual cooling (Cotler 2019) measures ``C_k = Tr(O rho^k)/Tr(rho^k)``. A degree-k functional
  needs k correlated copies. The device exposes the symmetrized weighted-cycle observables of
  Chen et al.: a block of ``m`` copies yields, for each power ``j <= m``, a bounded record whose
  mean is the moment. The observables are Hermitian and pairwise commuting, so
  :func:`sample_copy_block` samples their exact joint spectral measure without constructing the
  ``64**m`` copy state.

* The scored quantity is the RATIO ``C_k = num_k/den_k`` with a *shrinking* denominator
  ``Tr(rho^k)``. Estimating numerator and denominator from the SAME block (``observable="Z0Z1"``,
  ``measurement_family="simultaneous_weighted_cycle"``) gives the covariance fixed by those
  commuting observables; estimating them from separate blocks (``separate_cyclic_shift`` /
  ``observable="I"`` for the denominator) removes it. The joint law is not chosen from the
  marginal means: for ``k=2``, for example, ``E[num*den] = Tr(O rho)``.

* Single-copy randomized measurements (``run_local_pauli_batch``) rotate each copy by an
  independent uniformly random local Pauli basis per ion and read out (:func:`sample_local_pauli`) —
  a faithful local-shadow surface (with the well-known ~2^n purity variance; it cannot reach
  the task's precision under the basis budget and serves as an exploration surface).

* Asymmetric computational-basis readout applies to the single-copy local-Pauli and calibration
  surfaces. The collective weighted-cycle surface is the ideal joint-observable measurement;
  adding noise there would require a complete noisy POVM rather than rescaling one marginal.

Instance values (J0, alpha, fields, beta, lambda, readout rates) live in the hidden device
config; this module holds only the generic, instance-free math.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations, product
from math import factorial

import numpy as np
from scipy.linalg import expm

N_IONS = 6
DIM = 2**N_IONS
OBSERVABLE = "Z0Z1"
MAX_BLOCK_SIZE = 4
POWERS = (2, 3, 4)

_I2 = np.eye(2)
_X = np.array([[0.0, 1.0], [1.0, 0.0]])
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
_Z = np.array([[1.0, 0.0], [0.0, -1.0]])


class MeasurementError(ValueError):
    """Raised on a malformed measurement request."""


def _op_on(op: np.ndarray, i: int) -> np.ndarray:
    m = np.array([[1.0 + 0.0j]])
    for q in range(N_IONS):
        m = np.kron(m, op if q == i else _I2)
    return m


def _z0z1() -> np.ndarray:
    return (_op_on(_Z, 0) @ _op_on(_Z, 1)).real


def build_rho(
    J0: float,
    alpha: float,
    B_over_J0: float,
    delta_B_over_J0: list[float],
    beta_J0: float,
    lam: float,
) -> np.ndarray:
    """Build the hidden 6-ion density matrix ``rho`` (64x64, real, trace 1)."""
    xops = [_op_on(_X, i) for i in range(N_IONS)]
    yops = [_op_on(_Y, i) for i in range(N_IONS)]
    zops = [_op_on(_Z, i) for i in range(N_IONS)]
    H = np.zeros((DIM, DIM), dtype=complex)
    for i in range(N_IONS):
        for j in range(i + 1, N_IONS):
            H += (2.0 * J0 / abs(i - j) ** alpha) * (xops[i] @ xops[j] + yops[i] @ yops[j])
    for i in range(N_IONS):
        H += (B_over_J0 + delta_B_over_J0[i]) * J0 * zops[i]
    beta = beta_J0 / J0
    rt = expm(-beta * H.real)
    rt = rt / np.trace(rt)
    rho = (1.0 - lam) * rt + lam * np.eye(DIM) / DIM
    return rho.real


def exact_moments(rho: np.ndarray) -> dict[str, float]:
    """Exact nonlinear moments and cooled ratios for ``rho`` and ``O = Z0Z1``."""
    o = _z0z1()
    out: dict[str, float] = {}
    powk = {1: rho}
    for k in range(2, 5):
        powk[k] = powk[k - 1] @ rho
    for k in range(1, 5):
        out[f"p{k}"] = float(np.trace(powk[k]).real)
        out[f"num{k}"] = float(np.trace(o @ powk[k]).real)
    out["linear"] = out["num1"]
    for k in POWERS:
        out[f"C{k}"] = out[f"num{k}"] / out[f"p{k}"]
    return out


# --------------------------------------------------------------------------- #
# Readout model
# --------------------------------------------------------------------------- #


def readout_confusion(p01: list[float], p10: list[float]) -> list[np.ndarray]:
    """Per-ion 2x2 confusion matrices ``M[a,b] = P(read a | prepared b)``."""
    mats = []
    for i in range(N_IONS):
        mats.append(
            np.array(
                [[1.0 - p01[i], p10[i]], [p01[i], 1.0 - p10[i]]],
                dtype=np.float64,
            )
        )
    return mats


def readout_suppression(
    p01: list[float], p10: list[float], qubits: tuple[int, int] = (0, 1)
) -> float:
    """Two-qubit Z0Z1 multiplicative suppression ``d0*d1``, ``d_i = 1 - (p01_i + p10_i)``."""
    d = [1.0 - (p01[i] + p10[i]) for i in range(N_IONS)]
    return float(d[qubits[0]] * d[qubits[1]])


# --------------------------------------------------------------------------- #
# Copy-block (collective cyclic-shift) estimator sampling
# --------------------------------------------------------------------------- #

MEASUREMENT_FAMILIES = ("simultaneous_weighted_cycle", "separate_cyclic_shift")


def validate_copy_block(block_size: int, num_blocks: int, observable: str, family: str) -> None:
    if block_size not in (2, 3, 4):
        raise MeasurementError("block_size must be 2, 3, or 4")
    if num_blocks < 1:
        raise MeasurementError("num_blocks must be >= 1")
    if observable not in ("Z0Z1", "I"):
        raise MeasurementError("observable must be 'Z0Z1' or 'I'")
    if family not in MEASUREMENT_FAMILIES:
        raise MeasurementError(f"measurement_family must be one of {MEASUREMENT_FAMILIES}")


_Permutation = tuple[int, ...]
_Weight = tuple[int, ...]
_WeightedPermutation = tuple[_Permutation, _Weight]


def _compose(left: _Permutation, right: _Permutation) -> _Permutation:
    return tuple(left[right[i]] for i in range(len(left)))


def _inverse(permutation: _Permutation) -> _Permutation:
    result = [0] * len(permutation)
    for i, image in enumerate(permutation):
        result[image] = i
    return tuple(result)


def _multiply(left: _WeightedPermutation, right: _WeightedPermutation) -> _WeightedPermutation:
    """Multiply ``U_pi O^w`` elements for the involutory Pauli observable O."""
    pi, weight = left
    sigma, other_weight = right
    return (
        _compose(pi, sigma),
        tuple(weight[sigma[i]] ^ other_weight[i] for i in range(len(pi))),
    )


def _conjugate(permutation: _Permutation, item: _WeightedPermutation) -> _WeightedPermutation:
    zeros = (0,) * len(permutation)
    inverse = _inverse(permutation)
    return _multiply(
        _multiply((permutation, zeros), item),
        (inverse, zeros),
    )


def _cycle(block_size: int, power: int) -> _Permutation:
    return tuple((i + 1) % power if i < power else i for i in range(block_size))


def _symmetrized_observable(
    block_size: int, power: int, *, weighted: bool
) -> dict[_WeightedPermutation, float]:
    """Return Chen et al. Definition 2.14 in the signed-permutation group algebra."""
    weight = (1,) + (0,) * (block_size - 1) if weighted else (0,) * block_size
    base = (_cycle(block_size, power), weight)
    coefficient = 1.0 / factorial(block_size)
    result: dict[_WeightedPermutation, float] = {}
    for permutation in permutations(range(block_size)):
        item = _conjugate(permutation, base)
        result[item] = result.get(item, 0.0) + coefficient
    return result


@lru_cache(maxsize=3)
def _joint_spectral_measure(
    block_size: int,
) -> tuple[
    tuple[tuple[str, int], ...],
    tuple[_WeightedPermutation, ...],
    np.ndarray,
    np.ndarray,
]:
    """Joint spectrum and projector coefficients for every numerator/denominator power.

    The physical Hilbert space has dimension ``64**block_size``. Because all observables are
    elements of the signed-permutation group algebra and ``block_size <= 4``, their complete
    joint spectrum can instead be obtained in its at-most-384-dimensional regular
    representation. Each joint spectral projector remains a group-algebra element; the returned
    coefficients let :func:`weighted_cycle_joint_distribution` evaluate its probability on
    ``rho**tensor m``
    by cycle traces.
    """
    group = tuple(
        (permutation, weight)
        for permutation in permutations(range(block_size))
        for weight in product((0, 1), repeat=block_size)
    )
    index = {item: i for i, item in enumerate(group)}
    labels: list[tuple[str, int]] = []
    observables: list[np.ndarray] = []

    for weighted, kind in ((True, "num"), (False, "den")):
        for power in range(2, block_size + 1):
            coefficients = _symmetrized_observable(block_size, power, weighted=weighted)
            matrix = np.zeros((len(group), len(group)), dtype=np.float64)
            for column, right in enumerate(group):
                for left, coefficient in coefficients.items():
                    matrix[index[_multiply(left, right)], column] += coefficient
            labels.append((kind, power))
            observables.append(matrix)

    # Square roots of distinct primes make different rational joint eigenvalue tuples separate
    # in one diagonalization. The matrices commute and are real symmetric by construction.
    separators = np.sqrt(np.array([2, 3, 5, 7, 11, 13], dtype=np.float64))
    combined = sum(
        coefficient * observable
        for coefficient, observable in zip(separators[: len(observables)], observables, strict=True)
    )
    _, eigenvectors = np.linalg.eigh(combined)
    eigenvalues = np.column_stack(
        [
            np.einsum("ij,ij->j", eigenvectors, observable @ eigenvectors)
            for observable in observables
        ]
    )
    eigenvalues = np.round(eigenvalues, decimals=10)
    eigenvalues[np.abs(eigenvalues) < 1e-10] = 0.0
    outcomes, membership = np.unique(eigenvalues, axis=0, return_inverse=True)

    identity_index = index[(tuple(range(block_size)), (0,) * block_size)]
    projector_coefficients = np.empty((len(outcomes), len(group)), dtype=np.float64)
    for outcome_index in range(len(outcomes)):
        columns = np.flatnonzero(membership == outcome_index)
        vectors = eigenvectors[:, columns]
        projector_coefficients[outcome_index] = vectors @ vectors[identity_index]
    return tuple(labels), group, outcomes, projector_coefficients


def _weighted_permutation_expectation(
    item: _WeightedPermutation, rho: np.ndarray, observable: np.ndarray
) -> complex:
    """Evaluate ``Tr[U_pi O^w rho^tensor_m]`` as a product of cycle traces."""
    permutation, weight = item
    seen: set[int] = set()
    expectation = 1.0 + 0.0j
    identity = np.eye(rho.shape[0])
    local = [(observable if bit else identity) @ rho for bit in weight]
    for start in range(len(permutation)):
        if start in seen:
            continue
        cycle: list[int] = []
        i = start
        while i not in seen:
            seen.add(i)
            cycle.append(i)
            i = permutation[i]
        product_matrix = np.eye(rho.shape[0], dtype=np.complex128)
        for site in reversed(cycle):
            product_matrix = product_matrix @ local[site]
        expectation *= np.trace(product_matrix)
    return expectation


@lru_cache(maxsize=12)
def _weighted_cycle_joint_distribution_cached(
    block_size: int, shape: tuple[int, int], dtype: str, rho_bytes: bytes
) -> tuple[tuple[tuple[str, int], ...], np.ndarray, np.ndarray]:
    rho = np.frombuffer(rho_bytes, dtype=np.dtype(dtype)).reshape(shape)
    labels, group, outcomes, projector_coefficients = _joint_spectral_measure(block_size)
    observable = _z0z1()
    group_expectations = np.array(
        [_weighted_permutation_expectation(item, rho, observable) for item in group]
    )
    complex_probabilities = projector_coefficients @ group_expectations
    if float(np.max(np.abs(complex_probabilities.imag))) > 1e-8:
        raise RuntimeError("weighted-cycle spectral distribution is not real")
    probabilities = complex_probabilities.real
    if float(np.min(probabilities)) < -1e-8:
        raise RuntimeError("weighted-cycle spectral distribution has a negative probability")
    probabilities = np.clip(probabilities, 0.0, None)
    probabilities /= probabilities.sum()
    return labels, outcomes, probabilities


def weighted_cycle_joint_distribution(
    rho: np.ndarray, block_size: int
) -> tuple[tuple[tuple[str, int], ...], np.ndarray, np.ndarray]:
    """Return the exact joint spectral distribution of the Chen weighted-cycle observables."""
    if block_size not in (2, 3, 4):
        raise MeasurementError("block_size must be 2, 3, or 4")
    contiguous = np.ascontiguousarray(rho)
    return _weighted_cycle_joint_distribution_cached(
        block_size,
        contiguous.shape,
        contiguous.dtype.str,
        contiguous.tobytes(),
    )


def ratio_block_variance(rho: np.ndarray, block_size: int, power: int, family: str) -> float:
    """Delta-method variance of the cooled ratio ``Tr(O rho^k)/Tr(rho^k)`` per copy block.

    One block of ``block_size`` copies measured with ``observable="Z0Z1"`` contributes one
    numerator and one denominator record for ``power``. For ``n`` blocks the ratio of sample
    means has variance ``ratio_block_variance / n`` to leading order:

    - ``simultaneous_weighted_cycle`` reads both records off the same block, so the variance is
      ``E[(N - C D)^2] / E[D]^2`` under the exact joint spectral law;
    - ``separate_cyclic_shift`` reads them off independent block families (the engine charges
      ``2 * block_size`` copies per record pair), so the cross term vanishes and the variance is
      ``(Var N + C^2 Var D) / E[D]^2`` per record pair.

    Instance-free math: the verifier uses it to compute the uncertainty implied by an agent's
    actual measurement design and the best uncertainty available at a given copy expenditure.
    """
    if family not in MEASUREMENT_FAMILIES:
        raise MeasurementError(f"measurement_family must be one of {MEASUREMENT_FAMILIES}")
    if not 2 <= power <= block_size:
        raise MeasurementError("power must satisfy 2 <= power <= block_size")
    labels, outcomes, probabilities = weighted_cycle_joint_distribution(rho, block_size)
    column = {label: i for i, label in enumerate(labels)}
    num = outcomes[:, column[("num", power)]]
    den = outcomes[:, column[("den", power)]]
    mean_num = float(probabilities @ num)
    mean_den = float(probabilities @ den)
    ratio = mean_num / mean_den
    if family == "simultaneous_weighted_cycle":
        residual = float(probabilities @ (num - ratio * den) ** 2)
        return residual / mean_den**2
    var_num = float(probabilities @ num**2) - mean_num**2
    var_den = float(probabilities @ den**2) - mean_den**2
    return (var_num + ratio**2 * var_den) / mean_den**2


def ratio_block_covariance(
    rho: np.ndarray, block_size: int, power_a: int, power_b: int, family: str
) -> float:
    """Per-block covariance of the cooled-ratio estimators for two powers on one design.

    Uses the same influence values as :func:`ratio_block_variance` (``(N_k - C_k D_k)/E[D_k]``)
    under the exact joint law. For the aligned family every record of one block shares one
    joint outcome; for independent families the numerator records share one outcome and the
    denominator records another, so the cross term between numerators and denominators
    vanishes. ``power_a == power_b`` reproduces :func:`ratio_block_variance`.
    """
    if family not in MEASUREMENT_FAMILIES:
        raise MeasurementError(f"measurement_family must be one of {MEASUREMENT_FAMILIES}")
    for power in (power_a, power_b):
        if not 2 <= power <= block_size:
            raise MeasurementError("power must satisfy 2 <= power <= block_size")
    labels, outcomes, probabilities = weighted_cycle_joint_distribution(rho, block_size)
    column = {label: i for i, label in enumerate(labels)}
    parts = {}
    for power in (power_a, power_b):
        num = outcomes[:, column[("num", power)]]
        den = outcomes[:, column[("den", power)]]
        mean_num = float(probabilities @ num)
        mean_den = float(probabilities @ den)
        parts[power] = (num, den, mean_num / mean_den, mean_den)
    num_a, den_a, ratio_a, mean_a = parts[power_a]
    num_b, den_b, ratio_b, mean_b = parts[power_b]
    if family == "simultaneous_weighted_cycle":
        inf_a = (num_a - ratio_a * den_a) / mean_a
        inf_b = (num_b - ratio_b * den_b) / mean_b
        return float(probabilities @ (inf_a * inf_b)) - float(probabilities @ inf_a) * float(
            probabilities @ inf_b
        )
    cov_num = float(probabilities @ (num_a * num_b)) - float(probabilities @ num_a) * float(
        probabilities @ num_b
    )
    cov_den = float(probabilities @ (den_a * den_b)) - float(probabilities @ den_a) * float(
        probabilities @ den_b
    )
    return (cov_num + ratio_a * ratio_b * cov_den) / (mean_a * mean_b)


def sample_copy_block(
    rho: np.ndarray,
    block_size: int,
    num_blocks: int,
    observable: str,
    family: str,
    rng: np.random.Generator,
) -> dict[str, dict[str, list[float]]]:
    """Sample the raw weighted-cycle records for a copy-block batch.

    Returns ``{"j2": {"num": [...], "den": [...]}, ...}`` for every power ``2 <= j <= block_size``.
    Records are the bounded real eigenvalues of the commuting Chen weighted-cycle observables.
    The denominator records estimate ``Tr(rho^j)``; numerator records (present only for
    ``observable="Z0Z1"``) estimate ``Tr(Z0Z1 rho^j)``.

    - ``observable="Z0Z1"`` + ``simultaneous_weighted_cycle``: num and den are drawn from the
      exact joint spectral measure on the SAME block.
    - ``observable="Z0Z1"`` + ``separate_cyclic_shift``: num and den are drawn from independent
      physical blocks. The engine charges both block families.
    - ``observable="I"``: denominator-only diagnostic (no numerator records).
    """
    validate_copy_block(block_size, num_blocks, observable, family)
    labels, outcomes, probabilities = weighted_cycle_joint_distribution(rho, block_size)
    column = {label: i for i, label in enumerate(labels)}
    records: dict[str, dict[str, list[float]]] = {}
    correlated = observable == "Z0Z1" and family == "simultaneous_weighted_cycle"
    shared_indices = rng.choice(len(probabilities), size=num_blocks, p=probabilities)
    if observable == "Z0Z1" and not correlated:
        numerator_indices = rng.choice(len(probabilities), size=num_blocks, p=probabilities)
    else:
        numerator_indices = shared_indices
    for j in range(2, block_size + 1):
        den = outcomes[shared_indices, column[("den", j)]]
        if observable == "I":
            records[f"j{j}"] = {"den": [float(x) for x in den]}
            continue
        num = outcomes[numerator_indices, column[("num", j)]]
        records[f"j{j}"] = {
            "num": [float(x) for x in num],
            "den": [float(x) for x in den],
        }
    return records


# --------------------------------------------------------------------------- #
# Single-copy randomized (local Pauli) sampling
# --------------------------------------------------------------------------- #

# One family: every basis is an independent uniformly random local Pauli setting per ion. The
# previous three names (local/global Clifford ORM, Pauli shadow) all ran this same sampler.
LOCAL_PAULI_FAMILIES = ("random_local_pauli",)
_PAULI_BASES = ("X", "Y", "Z")
# Single-qubit rotations mapping the Pauli eigenbasis onto the computational basis (U s.t.
# U P U^dagger = Z): X -> H, Y -> H S^dagger, Z -> I.
_H = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
_SDAG = np.array([[1.0, 0.0], [0.0, -1.0j]])
_ROT = {"X": _H, "Y": _H @ _SDAG, "Z": _I2.astype(complex)}


def validate_local_pauli(
    basis_family: str, observable: str, num_bases: int, shots_per_basis: int
) -> None:
    if basis_family not in LOCAL_PAULI_FAMILIES:
        raise MeasurementError(f"basis_family must be one of {LOCAL_PAULI_FAMILIES}")
    if observable not in ("Z0Z1", "I"):
        raise MeasurementError("observable must be 'Z0Z1' or 'I'")
    if num_bases < 1:
        raise MeasurementError("num_bases must be >= 1")
    if shots_per_basis < 1:
        raise MeasurementError("shots_per_basis must be >= 1")


def _rotated_probs(rho: np.ndarray, basis: list[str]) -> np.ndarray:
    u = np.array([[1.0 + 0.0j]])
    for i in range(N_IONS):
        u = np.kron(u, _ROT[basis[i]])
    probs = np.real(np.diag(u @ rho @ u.conj().T))
    probs = np.clip(probs, 0.0, None)
    return probs / probs.sum()


def _apply_readout(
    bits: np.ndarray, p01: list[float], p10: list[float], rng: np.random.Generator
) -> np.ndarray:
    """Flip each ion's bit with its asymmetric readout error. ``bits`` shape (shots, N_IONS)."""
    out = bits.copy()
    for i in range(N_IONS):
        flip0 = (bits[:, i] == 0) & (rng.random(bits.shape[0]) < p01[i])
        flip1 = (bits[:, i] == 1) & (rng.random(bits.shape[0]) < p10[i])
        out[flip0, i] = 1
        out[flip1, i] = 0
    return out


def sample_local_pauli(
    rho: np.ndarray,
    basis_family: str,
    num_bases: int,
    shots_per_basis: int,
    p01: list[float],
    p10: list[float],
    rng: np.random.Generator,
) -> list[dict]:
    """Single-copy randomized measurement: per basis, a random local Pauli setting + bitstrings.

    Returns ``[{"basis": ["X","Z",...], "bitstrings": ["010110", ...]}, ...]``. Bit position ``i``
    of each string is ion ``i``, after asymmetric readout, in the same ion order as
    ``basis[i]``, the calibration surface, and the target observable ``Z0Z1``. Faithful shadow
    evidence; the agent forms the nonlinear estimator in post-processing.
    """
    validate_local_pauli(basis_family, "Z0Z1", num_bases, shots_per_basis)
    out: list[dict] = []
    idx = np.arange(DIM)
    # ``_op_on`` places ion 0 in the most-significant tensor factor, so computational index k
    # has ion i at bit (N_IONS - 1 - i). Decode in that order so string position i is ion i
    # (the little-index-first decode reversed the ion labels).
    bit_table = np.array(
        [[(k >> (N_IONS - 1 - i)) & 1 for i in range(N_IONS)] for k in idx], dtype=np.int8
    )
    for _ in range(num_bases):
        basis = [_PAULI_BASES[rng.integers(3)] for _ in range(N_IONS)]
        probs = _rotated_probs(rho, basis)
        drawn = rng.choice(DIM, size=shots_per_basis, p=probs)
        bits = bit_table[drawn]
        bits = _apply_readout(bits, p01, p10, rng)
        strings = ["".join(str(int(b)) for b in row) for row in bits]
        out.append({"basis": basis, "bitstrings": strings})
    return out


def sample_readout_calibration(
    state_label: str,
    num_shots: int,
    p01: list[float],
    p10: list[float],
    rng: np.random.Generator,
) -> list[str]:
    """Prepare a computational calibration state and return raw (readout-corrupted) bitstrings.

    ``state_label`` is a 6-bit string (e.g. ``"000000"``) or ``"all_zero"`` / ``"all_one"``.
    """
    if state_label in ("all_zero", "zeros"):
        prep = np.zeros(N_IONS, dtype=np.int8)
    elif state_label in ("all_one", "ones"):
        prep = np.ones(N_IONS, dtype=np.int8)
    elif len(state_label) == N_IONS and set(state_label) <= {"0", "1"}:
        prep = np.array([int(c) for c in state_label], dtype=np.int8)
    else:
        raise MeasurementError(
            f"state_label must be a {N_IONS}-bit string or 'all_zero'/'all_one', got {state_label!r}"
        )
    bits = np.tile(prep, (num_shots, 1))
    bits = _apply_readout(bits, p01, p10, rng)
    return ["".join(str(int(b)) for b in row) for row in bits]
