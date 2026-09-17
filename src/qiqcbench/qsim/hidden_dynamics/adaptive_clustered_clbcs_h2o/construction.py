"""Physics core for ``adaptive_clustered_clbcs_h2o`` (C-LBCS composite measurement).

This module is the single source of truth for:

* the hidden 14-qubit H2O/STO-3G state (Givens / double-excitation / Rz sequence),
* loading the public Jordan-Wigner Pauli Hamiltonian,
* the C-LBCS coverage probability ``h_j`` and the paper's state-independent
  ``V_Haar`` objective (Zhang et al. 2305.02439, Definition 5 / Eq. 11),
* the clustered finite-shot ``V_between`` / ``V_within`` / ``sigma_private``
  decomposition that is the primary Track-B score.

It is imported by the qtype engine (to sample raw counts) and by the hidden
scorer (to recompute hidden quantities). It is qsim-side only: the agent
container has no path to import it.

The functions are a faithful port of the maintainer reference-check script and
reproduce the validated reference numbers (E* = -74.83309335909544,
V_Haar(best-known) = 570.28886401966963, sigma_private(oracle) = 0.0367446746).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

NQ = 14
DIM = 1 << NQ
PAULI_TO_AXIS = {"X": 0, "Y": 1, "Z": 2}
SCHEME_NORMALIZATION_ATOL = 1e-8
CANONICAL_FLOAT_DECIMALS = 9

# Hidden state construction. q0-q3 are untouched spectators,
# so Rz(q2)/Rz(q3) would be global-phase no-ops and are dropped (every observable
# is invariant).
INITIAL_BITSTRING_Q0_TO_Q13 = "11111111110000"

# (kind, qubits, theta, phi)
GIVENS_DOUBLE_SEQUENCE: tuple[tuple[str, tuple[int, ...], float, float], ...] = (
    ("givens", (8, 10), 0.27, 0.19),
    ("givens", (9, 11), -0.23, -0.31),
    ("givens", (6, 12), 0.19, 0.27),
    ("givens", (7, 13), -0.16, 0.43),
    ("double_excitation", (8, 9, 10, 11), 0.15, 0.37),
    ("double_excitation", (6, 7, 12, 13), -0.13, -0.29),
    ("givens", (4, 10), 0.11, -0.21),
    ("givens", (5, 11), -0.09, 0.33),
)

# (qubit, angle) — q2/q3 no-ops dropped.
RZ_SEQUENCE: tuple[tuple[int, float], ...] = (
    (8, 0.23),
    (11, -0.19),
)

HAMILTONIAN_FILENAME = "h2o_sto3g_jw_pauli.csv"


# --------------------------------------------------------------------------- #
# State construction
# --------------------------------------------------------------------------- #
def bitpos(q: int) -> int:
    """Bit index for qubit ``q``; q0 is the most-significant bit (leftmost char)."""
    return NQ - 1 - q


def apply_givens(state: np.ndarray, p: int, q: int, theta: float, phi: float) -> np.ndarray:
    bp, bq = bitpos(p), bitpos(q)
    inds = np.arange(DIM)
    mask = (((inds >> bp) & 1) == 1) & (((inds >> bq) & 1) == 0)
    xs = inds[mask]
    ys = xs ^ (1 << bp) ^ (1 << bq)
    out = state.copy()
    c, s = np.cos(theta), np.sin(theta)
    phase = np.exp(1j * phi)
    a, b = state[xs], state[ys]
    out[xs] = c * a - np.conj(phase) * s * b
    out[ys] = phase * s * a + c * b
    return out


def apply_double(
    state: np.ndarray, occ1: int, occ2: int, vir1: int, vir2: int, theta: float, phi: float
) -> np.ndarray:
    bo1, bo2, bv1, bv2 = (bitpos(x) for x in (occ1, occ2, vir1, vir2))
    inds = np.arange(DIM)
    mask = (
        (((inds >> bo1) & 1) == 1)
        & (((inds >> bo2) & 1) == 1)
        & (((inds >> bv1) & 1) == 0)
        & (((inds >> bv2) & 1) == 0)
    )
    xs = inds[mask]
    ys = xs ^ (1 << bo1) ^ (1 << bo2) ^ (1 << bv1) ^ (1 << bv2)
    out = state.copy()
    c, s = np.cos(theta), np.sin(theta)
    phase = np.exp(1j * phi)
    a, b = state[xs], state[ys]
    out[xs] = c * a - np.conj(phase) * s * b
    out[ys] = phase * s * a + c * b
    return out


def apply_rz(state: np.ndarray, q: int, angle: float) -> np.ndarray:
    bp = bitpos(q)
    inds = np.arange(DIM)
    phases = np.where(((inds >> bp) & 1) == 0, np.exp(-0.5j * angle), np.exp(+0.5j * angle))
    return state * phases


@lru_cache(maxsize=1)
def build_hidden_state() -> np.ndarray:
    """Return the normalized 2^14 hidden statevector (q0 = most-significant bit)."""
    index = 0
    for q, char in enumerate(INITIAL_BITSTRING_Q0_TO_Q13):
        if char == "1":
            index |= 1 << bitpos(q)
    state = np.zeros(DIM, dtype=np.complex128)
    state[index] = 1.0
    for kind, qubits, theta, phi in GIVENS_DOUBLE_SEQUENCE:
        if kind == "givens":
            state = apply_givens(state, qubits[0], qubits[1], theta, phi)
        elif kind == "double_excitation":
            state = apply_double(state, qubits[0], qubits[1], qubits[2], qubits[3], theta, phi)
        else:  # pragma: no cover - guarded by the literal sequence above
            raise ValueError(f"Unknown hidden operation: {kind}")
    for q, angle in RZ_SEQUENCE:
        state = apply_rz(state, q, angle)
    norm = float(np.vdot(state, state).real)
    if not np.isclose(norm, 1.0, atol=1e-12):  # pragma: no cover
        raise AssertionError(f"Hidden state norm is {norm}")
    return state


# --------------------------------------------------------------------------- #
# Pauli expectations on the (sparse) hidden state
# --------------------------------------------------------------------------- #
def encode_pauli(pauli: str) -> tuple[int, int, int]:
    xmask = zmask = y_count = 0
    for q, char in enumerate(pauli):
        b = bitpos(q)
        if char in ("X", "Y"):
            xmask |= 1 << b
        if char in ("Z", "Y"):
            zmask |= 1 << b
        if char == "Y":
            y_count += 1
    return xmask, zmask, y_count


def make_sparse_expectation(state: np.ndarray | None = None):
    """Return ``(expectation(pauli)->float, support_size)`` for the hidden state."""
    if state is None:
        state = build_hidden_state()
    support = np.flatnonzero(np.abs(state) > 1e-14)
    amplitudes = {int(i): state[i] for i in support}

    def expectation(pauli: str) -> float:
        xmask, zmask, y_count = encode_pauli(pauli)
        prefactor = 1j**y_count
        value = 0.0j
        for basis, amplitude in amplitudes.items():
            target = basis ^ xmask
            target_amp = amplitudes.get(target)
            if target_amp is None:
                continue
            parity = (basis & zmask).bit_count() & 1
            value += np.conj(target_amp) * prefactor * (-1 if parity else 1) * amplitude
        if abs(value.imag) > 1e-9:  # pragma: no cover
            raise AssertionError(f"Non-real expectation {value} for Hermitian Pauli {pauli}")
        return float(value.real)

    return expectation, int(len(support))


# --------------------------------------------------------------------------- #
# Public Hamiltonian
# --------------------------------------------------------------------------- #
def _task_materials_root() -> Path:
    override = os.environ.get("QIQCBENCH_TASK_MATERIALS_DIR")
    if override:
        return Path(override)
    # Fall back to the configs root baked into qsim/verifier images
    # (QIQCBENCH_CONFIGS=/app/qiqcbench_configs), so the separate-mode verifier
    # resolves task materials without needing its own dedicated env var.
    configs_env = os.environ.get("QIQCBENCH_CONFIGS")
    if configs_env:
        candidate = Path(configs_env) / "task_materials"
        if candidate.is_dir():
            return candidate
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "configs" / "task_materials"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Could not locate configs/task_materials from construction.py")


def hamiltonian_csv_path(task_id: str = "adaptive_clustered_clbcs_h2o") -> Path:
    return _task_materials_root() / task_id / "public" / HAMILTONIAN_FILENAME


@lru_cache(maxsize=2)
def hamiltonian_sha256(task_id: str = "adaptive_clustered_clbcs_h2o") -> str:
    return hashlib.sha256(hamiltonian_csv_path(task_id).read_bytes()).hexdigest()


@lru_cache(maxsize=2)
def load_hamiltonian(
    task_id: str = "adaptive_clustered_clbcs_h2o",
) -> tuple[tuple[str, ...], np.ndarray, tuple[int, ...], float]:
    """Return ``(terms, coeffs, term_indices, identity_coeff)`` for non-identity Paulis.

    ``terms[j]`` is the length-14 Pauli string, ``coeffs[j]`` its coefficient,
    ``term_indices[j]`` the public CSV ``term_index``.
    """
    identity = "I" * NQ
    terms: list[str] = []
    coeffs: list[float] = []
    indices: list[int] = []
    identity_coeff = 0.0
    with hamiltonian_csv_path(task_id).open(newline="") as f:
        for row in csv.DictReader(f):
            pauli = row["pauli_q0_to_q13"]
            coeff = float(row["coefficient_hartree"])
            if pauli == identity:
                identity_coeff = coeff
                continue
            terms.append(pauli)
            coeffs.append(coeff)
            indices.append(int(row["term_index"]))
    return tuple(terms), np.asarray(coeffs, dtype=float), tuple(indices), identity_coeff


# --------------------------------------------------------------------------- #
# C-LBCS coverage, V_Haar, and clustered variance
# --------------------------------------------------------------------------- #
def coverage(pauli: str, weights: np.ndarray, beta: np.ndarray) -> float:
    component_probability = np.ones(len(weights), dtype=float)
    for i, char in enumerate(pauli):
        if char != "I":
            component_probability *= beta[:, i, PAULI_TO_AXIS[char]]
    return float(np.dot(weights, component_probability))


def validate_scheme(weights: np.ndarray, beta: np.ndarray, *, num_components: int = 16) -> None:
    if weights.shape != (num_components,):
        raise ValueError(f"Bad mixture shape {weights.shape}; expected ({num_components},)")
    if beta.shape != (num_components, NQ, 3):
        raise ValueError(f"Bad beta shape {beta.shape}; expected ({num_components}, {NQ}, 3)")
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(beta)):
        raise ValueError("Non-finite probability in scheme")
    if np.min(weights) < 0.0 or np.min(beta) < 0.0:
        raise ValueError("Negative probability in scheme")
    if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=SCHEME_NORMALIZATION_ATOL):
        raise ValueError(f"Mixture weights sum to {weights.sum()}, not 1")
    if not np.allclose(
        beta.sum(axis=-1),
        1.0,
        rtol=0.0,
        atol=SCHEME_NORMALIZATION_ATOL,
    ):
        raise ValueError("Local probabilities do not sum to one (axis tolerance 1e-8)")


def normalize_scheme(
    weights: np.ndarray, beta: np.ndarray, *, num_components: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return the unique normalized law represented by a scheme input.

    The public tolerance is an input/serialization allowance, not a second
    probability law.  Every consumer downstream of this boundary must use the
    returned arrays rather than the raw request values.
    """
    weights = np.asarray(weights, dtype=float)
    beta = np.asarray(beta, dtype=float)
    validate_scheme(weights, beta, num_components=num_components)
    normalized_weights = weights / weights.sum()
    normalized_beta = beta / beta.sum(axis=-1, keepdims=True)
    return normalized_weights, normalized_beta


def coverage_floor(terms, weights: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, float]:
    h = np.array([coverage(p, weights, beta) for p in terms])
    return h, float(np.min(h))


def track_a_variance(terms, coeffs: np.ndarray, weights: np.ndarray, beta: np.ndarray) -> float:
    """Paper Eq. 11 state-independent V_Haar = 2^n/(2^n+1) * sum_j c_j^2 / h_j."""
    h = np.array([coverage(p, weights, beta) for p in terms])
    if np.min(h) < 1e-9:
        raise ValueError(f"Coverage floor violated: {np.min(h)}")
    factor = (2**NQ) / (2**NQ + 1)
    return float(factor * np.sum(coeffs**2 / h))


def qwc_union_product(p: str, q: str) -> tuple[str, str] | None:
    union, product = [], []
    for a, b in zip(p, q, strict=True):
        if a != "I" and b != "I" and a != b:
            return None
        if a == "I":
            union.append(b)
            product.append(b)
        elif b == "I":
            union.append(a)
            product.append(a)
        else:
            union.append(a)
            product.append("I")
    return "".join(union), "".join(product)


def private_cluster_variance(
    terms,
    coeffs: np.ndarray,
    true_means: np.ndarray,
    control_means: np.ndarray,
    expectation,
    weights: np.ndarray,
    beta: np.ndarray,
    *,
    production_settings: int = 192,
    production_shots: int = 256,
) -> dict:
    """Exact clustered private variance for the locked Track-B design."""
    h = np.array([coverage(p, weights, beta) for p in terms])
    if np.min(h) < 1e-9:
        raise ValueError(f"Coverage floor violated: {np.min(h)}")
    residual = true_means - control_means
    design_second = float(np.sum(coeffs**2 * residual**2 / h))
    within = float(np.sum(coeffs**2 * (1.0 - true_means**2) / h))

    union_cache: dict[str, float] = {}
    product_cache: dict[str, float] = {}
    n = len(terms)
    for j in range(n):
        for ell in range(j + 1, n):
            up = qwc_union_product(terms[j], terms[ell])
            if up is None:
                continue
            union, product = up
            g = union_cache.get(union)
            if g is None:
                g = coverage(union, weights, beta)
                union_cache[union] = g
            ratio = g / (h[j] * h[ell])
            if residual[j] != 0.0 and residual[ell] != 0.0:
                design_second += 2.0 * coeffs[j] * coeffs[ell] * residual[j] * residual[ell] * ratio
            product_mean = product_cache.get(product)
            if product_mean is None:
                product_mean = expectation(product)
                product_cache[product] = product_mean
            covariance = product_mean - true_means[j] * true_means[ell]
            if covariance != 0.0:
                within += 2.0 * coeffs[j] * coeffs[ell] * covariance * ratio

    residual_energy = float(np.dot(coeffs, residual))
    between = design_second - residual_energy**2
    total_variance = (between + within / production_shots) / production_settings
    return {
        "min_coverage": float(np.min(h)),
        "between": float(between),
        "within": float(within),
        "variance": float(total_variance),
        "sigma": float(np.sqrt(max(total_variance, 0.0))),
    }


# --------------------------------------------------------------------------- #
# Canonical locked-request digest
# --------------------------------------------------------------------------- #
def _quantize_probability_vector(values: np.ndarray) -> np.ndarray:
    """Round a probability vector to the canonical decimal grid with exact mass 1.

    Independent element-wise rounding is not canonical: its entries need not sum
    to one, and normalizing the decoded values can then change their last digit.
    Largest-remainder allocation on an integer grid is deterministic, preserves
    non-negativity, and makes canonicalization idempotent.
    """
    values = np.asarray(values, dtype=float)
    total = float(values.sum())
    if not np.all(np.isfinite(values)) or np.min(values) < 0.0 or total <= 0.0:
        raise ValueError("Cannot canonicalize an invalid probability vector")
    scale = 10**CANONICAL_FLOAT_DECIMALS
    scaled = values / total * scale
    units = np.floor(scaled).astype(np.int64)
    remainder = int(scale - int(units.sum()))
    if remainder:
        fractional = scaled - units
        recipients = np.argsort(-fractional, kind="stable")[:remainder]
        units[recipients] += 1
    return units.astype(float) / scale


def canonical_request(
    weights, beta, control_entries: list[dict], task_id: str, hamiltonian_sha: str
) -> dict:
    """Section 4b/8e canonical form: drop zero-weight + merge duplicate components,
    renormalize weights, fixed float precision, sorted components + control variates,
    with the task_id and Hamiltonian SHA in the preimage."""
    w, b = normalize_scheme(np.asarray(weights, dtype=float), np.asarray(beta, dtype=float))
    merged: dict[str, float] = {}
    for k in range(len(w)):
        wk = float(w[k])
        if wk <= 0.0:
            continue
        canonical_beta = [_quantize_probability_vector(row).tolist() for row in b[k]]
        key = json.dumps(canonical_beta)
        merged[key] = merged.get(key, 0.0) + wk
    ordered_components = sorted(merged.items(), key=lambda item: item[0])
    canonical_weights = _quantize_probability_vector(
        np.asarray([weight for _, weight in ordered_components], dtype=float)
    )
    comps = [
        {"weight": float(weight), "beta": json.loads(key)}
        for (key, _), weight in zip(ordered_components, canonical_weights, strict=True)
        if weight > 0.0
    ]
    cvs = []
    for entry in control_entries:
        mean = round(float(entry["mean"]), CANONICAL_FLOAT_DECIMALS)
        cvs.append(
            {
                "term_index": int(entry["term_index"]),
                "mean": 0.0 if mean == 0.0 else mean,
            }
        )
    cvs.sort(key=lambda entry: entry["term_index"])
    return {
        "task_id": task_id,
        "hamiltonian_sha256": hamiltonian_sha,
        "components": comps,
        "control_variates": cvs,
    }


def locked_request_from_canonical(
    canonical: dict, *, num_components: int = 16
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Decode the digest-bearing request into its authoritative probability law.

    Canonicalization may merge duplicate components or drop zero-weight slots.
    The internal locked representation pads back to the public slot count with
    zero-weight uniform components, then normalizes the rounded canonical
    numbers once.  Thus the canonical JSON fully determines sampling and score.
    """
    components = canonical.get("components")
    if not isinstance(components, list) or not 1 <= len(components) <= num_components:
        raise ValueError("Canonical request has an invalid component list")

    weights = np.zeros(num_components, dtype=float)
    beta = np.full((num_components, NQ, 3), 1.0 / 3.0, dtype=float)
    for k, component in enumerate(components):
        if not isinstance(component, dict):
            raise ValueError("Canonical component must be an object")
        weights[k] = float(component["weight"])
        component_beta = np.asarray(component["beta"], dtype=float)
        if component_beta.shape != (NQ, 3):
            raise ValueError(f"Bad canonical beta shape {component_beta.shape}")
        beta[k] = component_beta

    weights, beta = normalize_scheme(weights, beta, num_components=num_components)
    controls = [
        {"term_index": int(entry["term_index"]), "mean": float(entry["mean"])}
        for entry in canonical.get("control_variates", [])
    ]
    return weights, beta, controls


def request_digest(canonical: dict) -> str:
    return hashlib.sha256(
        json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
