"""Black-box open-system (Lindblad/GKSL) engine.

Generic, device-config-driven open-system engine. It **never assembles the
4^n x 4^n Liouvillian** (dense or sparse). Instead it implements the Lindbladian
*directly*: the generator is applied to the density matrix ``rho`` (a
``2^n x 2^n`` matrix) as

    L(rho) = -i [H, rho] + sum_k c_k ( P_a rho P_b - 1/2 { P_b P_a, rho } ),
    H = (1/2) sum_P h_P P,

i.e. ordinary ``2^n``-dimensional matrix products ``H @ rho``, ``P_a @ rho @ P_b``.
The exact state at time ``t``, ``rho(t) = e^{tL}(rho0)``, is obtained as the
*action* of the matrix exponential on ``rho0`` via :func:`scipy...expm_multiply`,
driven by a matrix-free :class:`LinearOperator` whose matvec is the superoperator
above (and whose ``rmatvec`` is the Hilbert-Schmidt adjoint ``L†``, used only by
the norm estimator). The only ``4^n``-sized object is the (vectorized) state
itself — which a density matrix unavoidably is — so this scales to larger ``n``
far better than diagonalizing a ``4^n x 4^n`` operator.

Conventions (pinned — see the substrate scipy.expm test):

* **Row-stacking (row-major) vec**: ``vec(rho) = rho.reshape(-1)`` (numpy
  C-order). The dense reference Liouvillian (kept only for the cross-check test /
  the oracle compat helper) is
  ``L = -i(H (x) I - I (x) H^T) + sum_k c_k (P_a (x) P_b^T - 1/2 (P_b P_a) (x) I
  - 1/2 I (x) (P_b P_a)^T)``.
* Expectation: ``<O>_t = u . vec(rho_t)`` with ``u = vec(O^T) = O.conj().reshape(-1)``
  (``O`` Hermitian).

The dissipator is supplied as a flat list of GKSL terms ``(P_a, P_b, c)``: a
Hermitian single-site Kossakowski block expands to its nine ``(a,b)`` entries
(off-diagonal pairs carry conjugate ``c``); a correlated diagonal rate is one
``(P, P)`` term. This keeps the engine free of any task dictionary — the
task-specific expansion lives in the hidden-dynamics ``device_configs``.

Two properties match the analog engine and are load-bearing:

* **Budget persists across batches.** The scarce resource is total accepted
  evolution time, so a single engine instance (and its accumulating evidence)
  lives for the whole run. The backend builds exactly one engine.
* **RNG is injected, never reseeded internally** (engine-contract invariant).

Caches are module-level and keyed on the generator digest so they are shared
across engines built from the same hidden config (the multi-seed gate builds one
engine per noise seed but the noiseless trajectories are identical): the per-
generator operators are built once, and each ``(state, time)`` trajectory is
computed once and reused across seeds.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import LinearOperator, expm_multiply

from qiqcbench.qsim.core.wire import JobProbeOutcomeData, ProbeOutcomeRow, ProbeRowOp
from qiqcbench.qsim.qtypes.blackbox_lindblad_dynamics.device import (
    HiddenLindbladConfig,
    PublicLindbladSpec,
)

__all__ = ["LindbladEngine", "ProbeEvidence", "assemble_liouvillian"]

_TOL = 1e-9

_PAULI_2x2 = {
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


# Accepted rows are bounded by ``max_probe_rows``; rejected rows have no
# scientific cap, so retention is bounded to keep run-long evidence memory
# finite under a pathological all-rejected workload (overflow keeps a count).
MAX_RETAINED_REJECTED_ROWS = 256


@dataclass
class ProbeEvidence:
    """Authoritative cumulative record (budget counts accepted rows only)."""

    accepted_rows: list[dict] = field(default_factory=list)
    rejected_rows: list[dict] = field(default_factory=list)
    rejected_row_overflow: int = 0
    budget_used_us: float = 0.0

    @property
    def accepted_row_count(self) -> int:
        return len(self.accepted_rows)

    @property
    def ran_nonzero_time_probe(self) -> bool:
        return any(r["evolve_time_us"] > 0.0 for r in self.accepted_rows)


def _full_pauli_matrix(pauli: str) -> np.ndarray:
    """Dense ``2^n`` operator for a Pauli string (qubit 0 is the leftmost factor)."""
    op = np.array([[1.0 + 0j]])
    for char in pauli:
        op = np.kron(op, _PAULI_2x2[char])
    return op


def _super_d(pa: np.ndarray, pb: np.ndarray, eye: np.ndarray) -> np.ndarray:
    """Row-stacking dissipator super-operator for one (P_a, P_b) pair (dense ref)."""
    pba = pb @ pa
    return np.kron(pa, pb.T) - 0.5 * np.kron(pba, eye) - 0.5 * np.kron(eye, pba.T)


def assemble_liouvillian(
    n_qubits: int,
    coherent_terms: list[tuple[str, float]],
    dissipator_terms: list[tuple[str, str, complex]],
) -> np.ndarray:
    """Dense ``4^n x 4^n`` Liouvillian — reference only.

    The engine does **not** use this for propagation (it applies the Lindbladian
    matrix-free; see :class:`_Generator`). It is kept as a readable spec and as
    the reference the substrate scipy.expm test binds the matrix-free action to,
    and for the oracle's compat ``build_liouvillian(h_star, c_star)`` helper.
    """
    dim = 1 << n_qubits
    eye = np.eye(dim, dtype=complex)

    h_mat = np.zeros((dim, dim), dtype=complex)
    for pauli, coeff in coherent_terms:
        if coeff != 0.0:
            h_mat += 0.5 * coeff * _full_pauli_matrix(pauli)
    liou = -1j * (np.kron(h_mat, eye) - np.kron(eye, h_mat.T))

    for pa, pb, c in dissipator_terms:
        if c != 0:
            liou += c * _super_d(_full_pauli_matrix(pa), _full_pauli_matrix(pb), eye)
    return liou


# --- direct (matrix-free) Lindbladian generator ----------------------------


@dataclass
class _Generator:
    """Precomputed pieces of one hidden generator + its matrix-free operator.

    ``H`` and the dissipator operators ``(P_a, P_b, P_b P_a, P_a P_b)`` are
    ordinary ``2^n x 2^n`` matrices — the physical Hamiltonian and jump
    operators, never a ``4^n`` superoperator. ``lop`` applies ``L`` (and ``L†``)
    to a vectorized density matrix without assembling any Liouvillian.
    """

    n: int
    dim: int
    h_mat: np.ndarray
    terms: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, complex]]
    trace_l: float
    lop: LinearOperator


_GEN_CACHE: dict[str, _Generator] = {}
# vectorized initial states / observable covectors / trajectories, shared across
# engines built from the same generator (keyed including the generator digest).
_STATE_CACHE: dict[tuple[str, ...], np.ndarray] = {}
_OBS_CACHE: dict[str, np.ndarray] = {}
_TRAJ_CACHE: dict[tuple[str, tuple[str, ...], float], np.ndarray] = {}


def _generator_key(
    n_qubits: int,
    coherent_terms: list[tuple[str, float]],
    dissipator_terms: list[tuple[str, str, complex]],
) -> str:
    payload = {
        "n": n_qubits,
        "h": sorted((p, float(c)) for p, c in coherent_terms),
        "d": sorted((pa, pb, c.real, c.imag) for pa, pb, c in dissipator_terms),
    }
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _build_generator(
    n_qubits: int,
    coherent_terms: list[tuple[str, float]],
    dissipator_terms: list[tuple[str, str, complex]],
) -> _Generator:
    dim = 1 << n_qubits
    h_mat = np.zeros((dim, dim), dtype=complex)
    for pauli, coeff in coherent_terms:
        if coeff != 0.0:
            h_mat += 0.5 * coeff * _full_pauli_matrix(pauli)

    terms: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, complex]] = []
    trace_diag = 0.0
    for pa, pb, c in dissipator_terms:
        if c == 0:
            continue
        a = _full_pauli_matrix(pa)
        b = _full_pauli_matrix(pb)
        terms.append((a, b, b @ a, a @ b, c))
        if pa == pb:  # P_b P_a = I -> contributes to tr(L)
            trace_diag += c.real
    # tr(L) of the superoperator: only the anticommutator parts of diagonal terms
    # survive (commutator + off-diagonal jump parts are traceless).
    trace_l = -(dim * dim) * trace_diag

    def matvec(v: np.ndarray) -> np.ndarray:
        rho = v.reshape(dim, dim)
        out = -1j * (h_mat @ rho - rho @ h_mat)
        for a, b, ba, _ab, c in terms:
            out = out + c * (a @ rho @ b - 0.5 * (ba @ rho + rho @ ba))
        return out.reshape(-1)

    def rmatvec(w: np.ndarray) -> np.ndarray:
        sig = w.reshape(dim, dim)
        out = 1j * (h_mat @ sig - sig @ h_mat)
        for a, b, _ba, ab, c in terms:
            out = out + np.conj(c) * (a @ sig @ b - 0.5 * (ab @ sig + sig @ ab))
        return out.reshape(-1)

    d2 = dim * dim
    lop = LinearOperator((d2, d2), matvec=matvec, rmatvec=rmatvec, dtype=complex)
    return _Generator(n=n_qubits, dim=dim, h_mat=h_mat, terms=terms, trace_l=trace_l, lop=lop)


def _generator(
    n_qubits: int,
    coherent_terms: list[tuple[str, float]],
    dissipator_terms: list[tuple[str, str, complex]],
) -> tuple[str, _Generator]:
    key = _generator_key(n_qubits, coherent_terms, dissipator_terms)
    gen = _GEN_CACHE.get(key)
    if gen is None:
        gen = _build_generator(n_qubits, coherent_terms, dissipator_terms)
        _GEN_CACHE[key] = gen
    return key, gen


def _state_vector(labels: tuple[str, ...]) -> np.ndarray:
    cached = _STATE_CACHE.get(labels)
    if cached is not None:
        return cached
    psi = np.array([1.0 + 0j])
    for lab in labels:
        psi = np.kron(psi, _KET[lab])
    v0 = np.outer(psi, psi.conj()).reshape(-1)
    _STATE_CACHE[labels] = v0
    return v0


def _obs_covector(obs: str) -> np.ndarray:
    cached = _OBS_CACHE.get(obs)
    if cached is not None:
        return cached
    u = _full_pauli_matrix(obs).conj().reshape(-1)  # vec(O^T) for Hermitian O
    _OBS_CACHE[obs] = u
    return u


class LindbladEngine:
    """Stateful open-system probe engine for one run. RNG is injected."""

    def __init__(
        self,
        hidden: HiddenLindbladConfig,
        public: PublicLindbladSpec,
        rng: np.random.Generator,
    ):
        self._rng = rng
        self._n = public.n_qubits
        self._shots = public.num_internal_repetitions
        self._budget_us = public.total_evolution_time_budget_us
        self._max_rows = public.max_probe_rows
        self._t_min, self._t_max = public.evolve_time_range_us
        self._t_res = public.evolve_time_resolution_us
        self._w_min, self._w_max = public.allowed_observable_weight
        self._state_labels = set(public.allowed_initial_states)
        coherent_terms = [(t.pauli, t.coefficient) for t in hidden.coherent_terms]
        dissipator_terms = [(t.pa, t.pb, complex(t.c_re, t.c_im)) for t in hidden.dissipator_terms]
        self._gen_key, self._gen = _generator(self._n, coherent_terms, dissipator_terms)
        self._evidence = ProbeEvidence()
        self._lock = threading.Lock()

    @property
    def evidence(self) -> ProbeEvidence:
        return self._evidence

    # -- validation -----------------------------------------------------------

    def _on_grid(self, t: float) -> bool:
        if not (self._t_min - _TOL <= t <= self._t_max + _TOL):
            return False
        steps = t / self._t_res
        return abs(steps - round(steps)) <= 1e-6

    def _normalize_state(self, state: list[str]) -> list[str] | None:
        labels = list(state)
        if len(labels) != self._n or any(lab not in self._state_labels for lab in labels):
            return None
        return labels

    def _observable_ok(self, obs: str) -> bool:
        if len(obs) != self._n or any(c not in "IXYZ" for c in obs):
            return False
        weight = sum(1 for c in obs if c != "I")
        return self._w_min <= weight <= self._w_max

    # -- physics (direct, matrix-free Lindbladian) ----------------------------

    def _trajectory(self, labels: tuple[str, ...], t: float) -> np.ndarray:
        """``vec(rho_t) = e^{tL}(rho0)`` via the matrix-free exponential action.

        Cached per (generator, state, time): the noiseless value is deterministic,
        so it is computed once and reused across noise seeds / repeated probes.
        """
        key = (self._gen_key, labels, round(t, 9))
        cached = _TRAJ_CACHE.get(key)
        if cached is not None:
            return cached
        v0 = _state_vector(labels)
        vt = expm_multiply(t * self._gen.lop, v0, traceA=t * self._gen.trace_l)
        _TRAJ_CACHE[key] = vt
        return vt

    def _exact_expectation(self, labels: list[str], obs: str, t: float) -> float:
        vt = self._trajectory(tuple(labels), t)
        return float(np.real(_obs_covector(obs) @ vt))

    # -- batch execution ------------------------------------------------------

    def run_probe_batch(self, rows: list[ProbeRowOp]) -> JobProbeOutcomeData:
        """Process one valid-size batch; mutate cumulative evidence atomically."""
        with self._lock:
            out_rows = [self._run_row(row) for row in rows]
            return JobProbeOutcomeData(
                rows=out_rows,
                budget_used_us=self._evidence.budget_used_us,
                budget_remaining_us=max(0.0, self._budget_us - self._evidence.budget_used_us),
                accepted_row_count=self._evidence.accepted_row_count,
                max_probe_rows=self._max_rows,
            )

    def _reject(self, obs: str, reason: str, row: ProbeRowOp) -> ProbeOutcomeRow:
        if len(self._evidence.rejected_rows) < MAX_RETAINED_REJECTED_ROWS:
            self._evidence.rejected_rows.append(
                {
                    "initial_state": list(row.initial_state),
                    "evolve_time_us": row.evolve_time_us,
                    "observable_pauli": obs,
                    "reason": reason,
                }
            )
        else:
            self._evidence.rejected_row_overflow += 1
        return ProbeOutcomeRow(observable_pauli=obs, status="rejected", reject_reason=reason)

    def _run_row(self, row: ProbeRowOp) -> ProbeOutcomeRow:
        obs = row.observable_pauli
        labels = self._normalize_state(row.initial_state)
        if labels is None:
            return self._reject(obs, "malformed_initial_state", row)
        if not self._observable_ok(obs):
            return self._reject(obs, "invalid_observable", row)
        if not self._on_grid(row.evolve_time_us):
            return self._reject(obs, "time_off_grid_or_out_of_range", row)
        if self._evidence.accepted_row_count >= self._max_rows:
            return self._reject(obs, "row_cap_exhausted", row)
        if self._evidence.budget_used_us + row.evolve_time_us > self._budget_us + _TOL:
            return self._reject(obs, "budget_exhausted", row)

        e = self._exact_expectation(labels, obs, row.evolve_time_us)
        p = min(max((1.0 + e) / 2.0, 0.0), 1.0)
        bits = self._rng.binomial(1, p, size=self._shots)
        outcomes = (2 * bits - 1).astype(int).tolist()

        self._evidence.budget_used_us += row.evolve_time_us
        self._evidence.accepted_rows.append(
            {
                "initial_state": labels,
                "evolve_time_us": row.evolve_time_us,
                "observable_pauli": obs,
                "num_internal_repetitions": self._shots,
            }
        )
        return ProbeOutcomeRow(
            observable_pauli=obs,
            status="accepted",
            accepted_evolve_time_us=row.evolve_time_us,
            raw_pauli_outcomes=outcomes,
            num_internal_repetitions=self._shots,
        )
