"""Exact MPS simulator for the bounded-gate circuit qtype (pure numpy).

The hidden circuit is built from nearest-neighbour gates only, so an exact
matrix-product-state representation stays cheap: every two-qubit gate is a
local contraction + SVD split with singular values kept down to a relative
tolerance (no physical truncation — an exceeded ``max_bond`` raises rather
than silently approximating).

Per-shot bitstrings are drawn by exact sequential ("perfect") sampling from
the right-canonical form, vectorised across a whole shot batch. Measurement
bases are per-qubit Pauli settings; ``sample_bitstrings`` accepts either one
fixed setting string per call or an integer array of per-shot per-qubit basis
indices (shadow-style randomized measurement).
"""

from __future__ import annotations

import numpy as np
import scipy.linalg

from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.circuit import (
    BASIS_ORDER,
    BASIS_ROTATIONS,
    CircuitOp,
    one_qubit_unitary,
    two_qubit_unitary,
)

DEFAULT_SVD_TOL = 1e-12

_BASIS_MATS = np.stack([BASIS_ROTATIONS[name] for name in BASIS_ORDER])


class BondDimensionExceeded(RuntimeError):
    """The circuit needs a larger exact bond dimension than the configured cap."""


def _finite_factors(uu: np.ndarray, sv: np.ndarray, vh: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(uu)) and np.all(np.isfinite(sv)) and np.all(np.isfinite(vh)))


def _robust_svd(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin SVD that survives LAPACK ``gesdd`` misbehaviour on well-formed input.

    numpy's default divide-and-conquer driver misbehaves on well-formed
    unitary-like blocks at bond dimension 16-48 in two ways, both observed on
    routed parity circuits (singular values 1, 1, ..., 1e-15): it may raise
    ``LinAlgError`` (non-convergence), or -- OpenBLAS 0.3.31 on aarch64 Linux,
    48x32 complex isometry block -- it may return finite singular values with
    NaN in ``U``/``Vh`` *without raising*.  Either way the slower ``gesvd``
    driver reconstructs the same matrix to 1e-14, so validate the factors and
    fall back to it.  A non-finite input is a caller bug and raises.
    """
    if not np.all(np.isfinite(mat)):
        raise ValueError("SVD input contains non-finite entries")
    try:
        uu, sv, vh = np.linalg.svd(mat, full_matrices=False)
    except np.linalg.LinAlgError:
        uu, sv, vh = scipy.linalg.svd(mat, full_matrices=False, lapack_driver="gesvd")
    else:
        if not _finite_factors(uu, sv, vh):
            uu, sv, vh = scipy.linalg.svd(mat, full_matrices=False, lapack_driver="gesvd")
    if not _finite_factors(uu, sv, vh):
        raise np.linalg.LinAlgError("SVD returned non-finite factors under both LAPACK drivers")
    return uu, sv, vh


class MPS:
    """Open-boundary MPS; tensors[i] has shape (chi_left, 2, chi_right)."""

    def __init__(self, tensors: list[np.ndarray]):
        self.tensors = tensors

    @classmethod
    def computational_zero(cls, n: int) -> MPS:
        return cls([np.array([1.0, 0.0], dtype=complex).reshape(1, 2, 1) for _ in range(n)])

    @property
    def n(self) -> int:
        return len(self.tensors)

    def bond_dims(self) -> list[int]:
        return [t.shape[2] for t in self.tensors[:-1]]

    def apply_1q(self, u: np.ndarray, q: int) -> None:
        self.tensors[q] = np.einsum("ab,lbr->lar", u, self.tensors[q])

    def apply_2q(
        self,
        u4: np.ndarray,
        q_lo: int,
        *,
        svd_tol: float = DEFAULT_SVD_TOL,
        max_bond: int | None = None,
    ) -> None:
        """Apply a 4x4 unitary indexed by (hi, lo) pairs to sites (q_lo, q_lo+1)."""
        a, b = self.tensors[q_lo], self.tensors[q_lo + 1]
        # theta[l, lo, hi, m]
        theta = np.einsum("lar,rbm->labm", a, b)
        u = u4.reshape(2, 2, 2, 2)  # (hi', lo', hi, lo)
        theta = np.einsum("abcd,ldcm->lbam", u, theta)  # -> (l, lo', hi', m)
        left_dim, _, _, right_dim = theta.shape
        mat = theta.reshape(left_dim * 2, 2 * right_dim)
        uu, sv, vh = _robust_svd(mat)
        if sv[0] > 0:
            keep = int(np.sum(sv > svd_tol * sv[0]))
        else:
            keep = 1
        keep = max(keep, 1)
        if max_bond is not None and keep > max_bond:
            raise BondDimensionExceeded(f"exact bond dimension {keep} exceeds max_bond={max_bond}")
        self.tensors[q_lo] = uu[:, :keep].reshape(left_dim, 2, keep)
        self.tensors[q_lo + 1] = (sv[:keep, None] * vh[:keep, :]).reshape(keep, 2, right_dim)

    def right_canonicalize(self) -> None:
        """Bring every tensor to right-canonical form; normalises the state."""
        for i in range(self.n - 1, 0, -1):
            t = self.tensors[i]
            left_dim, _, right_dim = t.shape
            mat = t.reshape(left_dim, 2 * right_dim)
            uu, sv, vh = _robust_svd(mat)
            keep = vh.shape[0]
            self.tensors[i] = vh.reshape(keep, 2, right_dim)
            carry = uu * sv[None, :]
            self.tensors[i - 1] = np.einsum("lbr,rk->lbk", self.tensors[i - 1], carry)
        norm = np.linalg.norm(self.tensors[0])
        if norm <= 0:
            raise RuntimeError("MPS collapsed to zero norm")
        self.tensors[0] = self.tensors[0] / norm

    def to_dense(self) -> np.ndarray:
        """Dense statevector with qubit 0 as least-significant bit (small n only)."""
        if self.n > 20:
            raise ValueError("to_dense limited to n <= 20")
        # Contract left-to-right; accumulator indexed (basis_prefix, chi_right)
        acc = self.tensors[0].reshape(2, -1)  # (b0, r)
        for i in range(1, self.n):
            t = self.tensors[i]
            acc = np.einsum("pr,rbm->bpm", acc, t).reshape(acc.shape[0] * 2, -1)
        return acc.reshape(-1)

    def expectation_product_op(self, ops_by_site: dict[int, np.ndarray]) -> float:
        """<psi| prod_site O_site |psi> for single-site operators (identity elsewhere)."""
        env = np.ones((1, 1), dtype=complex)
        for i, t in enumerate(self.tensors):
            op = ops_by_site.get(i)
            if op is None:
                env = np.einsum("lk,lbr,kbs->rs", env, t.conj(), t)
            else:
                env = np.einsum("lk,lbr,bc,kcs->rs", env, t.conj(), op, t)
        return float(np.real(env[0, 0]))


def build_mps(
    ops: list[CircuitOp],
    x: np.ndarray,
    n: int,
    *,
    svd_tol: float = DEFAULT_SVD_TOL,
    max_bond: int | None = None,
) -> MPS:
    """Run the circuit at input ``x`` and return the right-canonical output MPS."""
    mps = MPS.computational_zero(n)
    for op in ops:
        if op.gate in ("cx", "cz"):
            assert op.q2 is not None
            mps.apply_2q(
                two_qubit_unitary(op),
                min(op.q, op.q2),
                svd_tol=svd_tol,
                max_bond=max_bond,
            )
        else:
            mps.apply_1q(one_qubit_unitary(op, x), op.q)
    mps.right_canonicalize()
    return mps


def sample_bitstrings(
    mps: MPS,
    bases: str | np.ndarray,
    shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``shots`` exact samples; returns uint8 array (shots, n), column q = qubit q.

    ``bases`` is either a length-n string over {x, y, z} (one fixed setting) or an
    integer array of shape (shots, n) with values indexing ``BASIS_ORDER``.
    """
    n = mps.n
    fixed_setting = isinstance(bases, str)
    if fixed_setting:
        if len(bases) != n:
            raise ValueError(f"basis string length {len(bases)} != n={n}")
        tensors = [
            np.einsum("ab,lbr->lar", BASIS_ROTATIONS[bases[i]], mps.tensors[i]) for i in range(n)
        ]
        basis_idx = None
    else:
        basis_idx = np.asarray(bases)
        if basis_idx.shape != (shots, n):
            raise ValueError(f"basis index array must have shape ({shots}, {n})")
        tensors = mps.tensors

    bits = np.empty((shots, n), dtype=np.uint8)
    left = np.ones((shots, 1), dtype=complex)
    for i in range(n):
        pre = np.einsum("sl,lbr->sbr", left, tensors[i])
        if basis_idx is not None:
            u_shots = _BASIS_MATS[basis_idx[:, i]]  # (shots, 2, 2)
            pre = np.einsum("sab,sbr->sar", u_shots, pre)
        prob = np.sum(np.abs(pre) ** 2, axis=2)  # (shots, 2)
        total = prob.sum(axis=1)
        p1 = prob[:, 1] / total
        outcome = (rng.random(shots) < p1).astype(np.uint8)
        bits[:, i] = outcome
        chosen = pre[np.arange(shots), outcome, :]
        amp = np.sqrt(prob[np.arange(shots), outcome])
        left = chosen / amp[:, None]
    return bits
