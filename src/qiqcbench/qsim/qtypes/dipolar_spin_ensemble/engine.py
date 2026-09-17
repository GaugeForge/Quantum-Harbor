"""Dipolar-spin-ensemble analog engine (disorder-dominated NV ensemble).

Physics model (numpy only — exact small-cluster propagation, disorder-averaged):

1. Represent the ensemble by a small cluster of ``N`` spin-1/2 (default 6 ->
   2^6 = 64-dim exact statevector). The collective magnetization of the real
   (~10^9-spin) ensemble is the disorder average over single-spin environments,
   which we estimate by averaging ``<sigma_a^i>`` over the ``N`` cluster spins
   and over ``R`` independent on-site-disorder realizations.

2. Static (rotating-frame) Hamiltonian, energies in MHz:

       H_s = sum_i h_i Sz_i  +  sum_{i<j} J_ij (c_x Sx_i Sx_j + c_y Sy_i Sy_j + c_z Sz_i Sz_j)

   with S = sigma/2 and the **traceless secular dipolar** tensor
   ``(c_x, c_y, c_z) = (-1/2, -1/2, 1)``. The on-site fields ``h_i ~ N(0, W^2)``
   are redrawn per realization (W = disorder std in MHz). The couplings
   ``J_ij = J0 (1 - 3 cos^2 theta_ij) / r_ij^3`` come from FIXED seeded random
   3D positions (the interaction structure is one fixed disorder of the medium),
   normalized so the median ``|J_ij|`` equals the hidden ``J``.

3. Global pulses: a rotation about ``+-x/+-y`` is propagated under
   ``H_s + H_drive`` for its finite duration ``t_p`` (so finite-pulse and
   disorder-during-pulse effects are captured). ``H_drive = Omega sum_i
   (cos phi Sx_i + sin phi Sy_i)``. On ``inject_error`` every pulse over-rotates
   by the hidden fractional ``rotation_error`` (Omega -> Omega (1+eps)).

Units: energies MHz, times ns. Evolution phase is ``exp(-i 2 pi E[MHz] t[ns]/1000)``,
so a Gaussian disorder of ``W/2pi`` MHz gives a free-induction 1/e time
``sqrt(2)*1000/(2 pi W)`` ns, and ``Omega/2pi`` MHz gives a ``pi/2`` pulse of
``250/Omega`` ns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_TWO_PI_OVER_NS = 2.0 * math.pi / 1000.0  # exp(-i * 2pi * E[MHz] * t[ns] / 1000)

# Single-spin Pauli matrices.
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)
_ID = np.eye(2, dtype=complex)

# Single-spin states (|0> is the +1 eigenstate of sigma_z).
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_AXIS_STATE = {
    "+z": np.array([1, 0], dtype=complex),
    "-z": np.array([0, 1], dtype=complex),
    "+x": np.array([_INV_SQRT2, _INV_SQRT2], dtype=complex),
    "-x": np.array([_INV_SQRT2, -_INV_SQRT2], dtype=complex),
    "+y": np.array([_INV_SQRT2, 1j * _INV_SQRT2], dtype=complex),
    "-y": np.array([_INV_SQRT2, -1j * _INV_SQRT2], dtype=complex),
}
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _embed(op: np.ndarray, site: int, n: int) -> np.ndarray:
    """Embed a single-spin operator on ``site`` into the ``n``-spin Hilbert space."""
    mats = [op if s == site else _ID for s in range(n)]
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


@dataclass(frozen=True)
class DipolarPhysics:
    """Realized (hidden) physical parameters in engine-native units."""

    n_spins: int
    n_realizations: int
    w_mhz: float  # on-site disorder std (W/2pi)
    j_mhz: float  # median dipolar coupling |J0|/2pi
    coeffs: tuple[float, float, float]  # (c_x, c_y, c_z), traceless
    rabi_mhz: float  # Omega/2pi
    rotation_error: float  # fractional over-rotation on injection
    readout_p_pm: float  # collective readout flip +1 -> -1
    readout_p_mp: float  # collective readout flip -1 -> +1


@dataclass(frozen=True)
class _PulseOp:
    axis: str  # "x" | "y"
    sign: int  # +1 | -1
    angle_deg: float


@dataclass(frozen=True)
class _FreeOp:
    duration_ns: float


class DipolarEnsembleEngine:
    """Exact small-cluster, disorder-averaged dipolar-ensemble engine."""

    def __init__(self, physics: DipolarPhysics, rng: np.random.Generator) -> None:
        self.phys = physics
        self.rng = rng
        n = physics.n_spins
        self.n = n
        self.dim = 2**n

        # Single-site Pauli operators embedded in the full space.
        self.sx = [_embed(_SX, i, n) for i in range(n)]
        self.sy = [_embed(_SY, i, n) for i in range(n)]
        self.sz = [_embed(_SZ, i, n) for i in range(n)]

        # Collective drive operators sum_i sigma_a^i (Sx = sigma/2 -> factor 1/2).
        self.drive_x = 0.5 * sum(self.sx)
        self.drive_y = 0.5 * sum(self.sy)

        # Fixed dipolar interaction (seeded positions -> one fixed J_ij structure).
        self._h_dip = self._build_dipolar()

        # Per-realization on-site fields (redrawn from the Gaussian).
        self._h_fields = self.rng.normal(0.0, physics.w_mhz, size=(physics.n_realizations, n))

    # ---- Hamiltonian construction ----

    def _build_dipolar(self) -> np.ndarray:
        n = self.n
        # Seeded positions on a small 3D lattice-ish cloud; deterministic given rng
        # is NOT used here so the interaction structure is fixed across calls.
        pos_rng = np.random.default_rng(0xD1B01A2B ^ n)
        pos = pos_rng.uniform(-1.0, 1.0, size=(n, 3))
        # Ensure no coincident spins.
        cx, cy, cz = self.phys.coeffs
        raw = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                r = float(np.linalg.norm(d))
                r = max(r, 0.25)
                cos_t = d[2] / r
                raw[i, j] = (1.0 - 3.0 * cos_t**2) / r**3
        # Normalize so the median |coupling| equals the realized J.
        offdiag = np.abs(raw[np.triu_indices(n, 1)])
        med = float(np.median(offdiag)) if offdiag.size else 1.0
        scale = (self.phys.j_mhz / med) if med > 0 else 0.0
        j = raw * scale

        h = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(n):
            for jj in range(i + 1, n):
                jij = j[i, jj]
                if jij == 0.0:
                    continue
                # S_a S_a = (sigma_a/2)(sigma_a/2) = sigma_a sigma_a / 4
                h += (
                    jij
                    * (
                        cx * (self.sx[i] @ self.sx[jj])
                        + cy * (self.sy[i] @ self.sy[jj])
                        + cz * (self.sz[i] @ self.sz[jj])
                    )
                    / 4.0
                )
        return h

    def _static_h(self, fields: np.ndarray) -> np.ndarray:
        # H_s = H_dip + sum_i h_i Sz_i  (Sz = sigma_z/2)
        h = self._h_dip.copy()
        for i in range(self.n):
            h += fields[i] * (self.sz[i] / 2.0)
        return h

    # ---- core: exact collective magnetization stroboscopically ----

    def collective_magnetization(
        self,
        *,
        init_axis: str,
        sequence: list,
        n_cycles: int,
        measure_axes: list[str],
        inject_error: bool,
    ) -> dict[str, np.ndarray]:
        """Exact (noiseless) collective magnetization M_a at cycles 0..n_cycles.

        Returns ``{axis: array of length n_cycles+1}`` with M_a in [-1, 1],
        averaged over cluster spins and disorder realizations.
        """
        ops = _normalize_ops(sequence)
        psi0 = _product_state(init_axis, self.n)
        eps = self.phys.rotation_error if inject_error else 0.0

        accum = {a: np.zeros(n_cycles + 1) for a in measure_axes}
        op_a = {"x": self.sx, "y": self.sy, "z": self.sz}

        for r in range(self.phys.n_realizations):
            h_s = self._static_h(self._h_fields[r])
            es, vs = np.linalg.eigh(h_s)
            block = self._block_unitary(ops, h_s, es, vs, eps)
            psi = psi0.copy()
            self._accumulate(psi, accum, measure_axes, op_a, 0)
            for k in range(1, n_cycles + 1):
                psi = block @ psi
                self._accumulate(psi, accum, measure_axes, op_a, k)

        norm = 1.0 / (self.phys.n_realizations * self.n)
        return {a: accum[a] * norm for a in measure_axes}

    def _accumulate(self, psi, accum, measure_axes, op_a, k) -> None:
        for a in measure_axes:
            s = 0.0
            ops = op_a[a]
            for i in range(self.n):
                s += float(np.real(np.vdot(psi, ops[i] @ psi)))
            accum[a][k] += s

    def _block_unitary(self, ops, h_s, es, vs, eps) -> np.ndarray:
        """One-cycle propagator for the base sequence under realization (h_s)."""
        u = np.eye(self.dim, dtype=complex)
        pulse_cache: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        for op in ops:
            if isinstance(op, _FreeOp):
                if op.duration_ns <= 0:
                    continue
                phase = np.exp(-1j * _TWO_PI_OVER_NS * es * op.duration_ns)
                u_op = (vs * phase) @ vs.conj().T
            else:
                u_op = self._pulse_unitary(op, h_s, eps, pulse_cache)
            u = u_op @ u
        return u

    def _pulse_unitary(self, op: _PulseOp, h_s, eps, cache) -> np.ndarray:
        key = (op.axis, op.sign)
        if key not in cache:
            omega = self.phys.rabi_mhz * (1.0 + eps)
            drive = self.drive_x if op.axis == "x" else self.drive_y
            h_p = h_s + op.sign * omega * drive
            cache[key] = np.linalg.eigh(h_p)
        ep, vp = cache[key]
        t_p = op.angle_deg / 360.0 * 1000.0 / self.phys.rabi_mhz
        phase = np.exp(-1j * _TWO_PI_OVER_NS * ep * t_p)
        return (vp * phase) @ vp.conj().T

    # ---- shot sampling for the wire surface ----

    def sample_sequence(
        self,
        *,
        init_axis: str,
        sequence: list,
        n_cycles: int,
        measure_axes: list[str],
        inject_error: bool,
        shots: int,
    ) -> list[list[str]]:
        """Per-cycle lists of per-shot collective-magnetization bitstrings.

        Outer index = cycle k (0..n_cycles). Each inner string has one char per
        measure axis: '1' for a +1 collective readout, '0' for -1. The mean over
        shots of (2*bit-1) estimates the (readout-confused) collective M_a.
        """
        exact = self.collective_magnetization(
            init_axis=init_axis,
            sequence=sequence,
            n_cycles=n_cycles,
            measure_axes=measure_axes,
            inject_error=inject_error,
        )
        p_pm = self.phys.readout_p_pm
        p_mp = self.phys.readout_p_mp
        out: list[list[str]] = []
        for k in range(n_cycles + 1):
            # Per-axis +1 probability after readout confusion.
            cols = []
            for a in measure_axes:
                m = float(np.clip(exact[a][k], -1.0, 1.0))
                p_plus_true = 0.5 * (1.0 + m)
                # apply asymmetric flip on the ideal +-1 outcome
                p_plus = p_plus_true * (1.0 - p_pm) + (1.0 - p_plus_true) * p_mp
                bits = (self.rng.random(shots) < p_plus).astype(int)
                cols.append(bits)
            strings = [
                "".join(str(cols[c][s]) for c in range(len(measure_axes))) for s in range(shots)
            ]
            out.append(strings)
        return out

    # ---- helpers for the hidden scorer (Stage A / C reference physics) ----

    def free_induction(self, times_ns: np.ndarray, axis: str = "x") -> np.ndarray:
        """Exact collective FID M_axis(t) from a +axis-polarized state (no pulses)."""
        init = "+" + axis
        out = np.zeros(len(times_ns))
        op_a = {"x": self.sx, "y": self.sy, "z": self.sz}[axis]
        psi0 = _product_state(init, self.n)
        for r in range(self.phys.n_realizations):
            h_s = self._static_h(self._h_fields[r])
            es, vs = np.linalg.eigh(h_s)
            c = vs.conj().T @ psi0
            for ti, t in enumerate(times_ns):
                phase = np.exp(-1j * _TWO_PI_OVER_NS * es * float(t))
                psi = vs @ (phase * c)
                s = 0.0
                for i in range(self.n):
                    s += float(np.real(np.vdot(psi, op_a[i] @ psi)))
                out[ti] += s
        return out / (self.phys.n_realizations * self.n)


def _normalize_ops(sequence: list) -> list:
    """Convert wire ops (or dicts) into engine ops."""
    ops: list = []
    for op in sequence:
        kind = getattr(op, "kind", None)
        if kind is None and isinstance(op, dict):
            kind = op.get("kind") or op.get("op")
        if kind == "free":
            dur = getattr(op, "duration_ns", None)
            if dur is None and isinstance(op, dict):
                dur = op["duration_ns"]
            ops.append(_FreeOp(float(dur)))
        elif kind == "pulse":
            axis = getattr(op, "axis", None) or op["axis"]
            sign_s = getattr(op, "sign", None) or (
                op.get("sign", "+") if isinstance(op, dict) else "+"
            )
            angle = getattr(op, "angle_deg", None)
            if angle is None and isinstance(op, dict):
                angle = op.get("angle_deg", 90.0)
            ops.append(
                _PulseOp(axis=axis, sign=(1 if sign_s == "+" else -1), angle_deg=float(angle))
            )
        else:
            raise ValueError(f"unknown op kind {kind!r}")
    return ops


def _product_state(init_axis: str, n: int) -> np.ndarray:
    if init_axis not in _AXIS_STATE:
        raise ValueError(f"unknown init_axis {init_axis!r}")
    single = _AXIS_STATE[init_axis]
    psi = single
    for _ in range(n - 1):
        psi = np.kron(psi, single)
    return psi.astype(complex)


__all__ = [
    "DipolarEnsembleEngine",
    "DipolarPhysics",
]
