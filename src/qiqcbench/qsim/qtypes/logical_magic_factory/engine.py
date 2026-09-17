"""Logical magic-state factory engine (logical_magic_factory qtype).

Black-box, **logical-effective** Monte Carlo: each drawn copy is a 2x2
density matrix over the logical {|H>, |Hbar>}-adjacent qubit space, not a
physical 7-qubit circuit simulation. This qtype models a fixed, already-built
black-box factory's infidelity.

Two properties are load-bearing for the task semantics (mirrors
``blackbox_analog_dynamics/engine.py``):

* **Budget persists across batches.** The scarce resource is total consumed
  magic states, so a single engine instance (and its accumulating evidence)
  lives for the whole run. The backend builds exactly one engine.
* **RNG is injected, never reseeded internally** (engine-contract invariant):
  the backend factory seeds it once from ``hidden.seed``.

Physics:

* An accepted (syndrome-clean) copy is
  ``rho_acc = D_lambda_dep(R_delta_rad |H><H| R_delta_rad^dagger)`` -- a
  residual coherent over-rotation about a Bloch axis perpendicular to |H>,
  followed by a depolarizing channel. A flagged copy is built the same way
  from the (worse) ``(delta_rej_rad, lambda_dep_rej)`` pair.
* **Twirling** dephases a state into the ``{|H>, |Hbar>}`` basis
  (``twirl()``); this preserves ``Tr[rho @ |H><H|]`` exactly (fidelity is
  unchanged), matching the source paper's Thm 2.
* The **two-copy joint circuit** applies CNOT(control=copy1, target=copy2)
  **then** the logical Hadamard on copy1, then measures both copies in the
  computational basis. This ordering (not the literal
  "CNOT.(H(x)I)" *operator-composition* reading, which is H-first and gives
  an a-independent, physically-wrong distribution -- verified by hand)
  reproduces ``P(outcome="11") = eps(1-eps)`` exactly when both copies are
  twirled and syndrome-clean: the general Bell-measurement-realizes-SWAP-test
  identity (Garcia-Escartin & Chamorro-Posada, 2013). Un-twirled or
  non-post-selected estimates are biased because they fall directly out of
  the same linear algebra applied to the untwirled/mixed density matrices --
  no separate bias formula is hardcoded anywhere in this module.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from qiqcbench.qsim.qtypes.logical_magic_factory.device import HiddenMagicFactoryConfig
from qiqcbench.qsim.qtypes.logical_magic_factory.wire import (
    JobMagicBenchmarkPoint,
    MagicBenchmarkPointRequest,
)

__all__ = [
    "MagicBenchmarkEvidence",
    "MagicFactoryEngine",
    "build_accepted_state",
    "build_rejected_state",
    "ideal_h_ket",
    "true_epsilon_from_params",
    "twirl",
]

_HADAMARD = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex)
_IDENTITY2 = np.eye(2, dtype=complex)

# Rotate the requested Pauli measurement axis onto computational Z.
_U_Y = (1.0 / np.sqrt(2.0)) * np.array([[1.0, -1.0j], [1.0, 1.0j]], dtype=complex)
_BASIS_ROTATION = {"z": _IDENTITY2, "x": _HADAMARD, "y": _U_Y}

# Fixed two-copy joint circuit: CNOT(control=copy1,target=copy2) first, then H
# on copy1. Basis order |q1,q2>, index = 2*q1 + q2.
_CNOT_1_TO_2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
_H_ON_COPY1 = np.kron(_HADAMARD, _IDENTITY2)
_JOINT_CIRCUIT = _H_ON_COPY1 @ _CNOT_1_TO_2


def ideal_h_ket() -> np.ndarray:
    """+1 eigenstate of the logical Hadamard: |H> = cos(pi/8)|0> + sin(pi/8)|1>."""
    return np.array([np.cos(np.pi / 8.0), np.sin(np.pi / 8.0)], dtype=complex)


def _h_eigenbasis() -> tuple[np.ndarray, np.ndarray]:
    """Return (|H>, |Hbar>), the +1/-1 eigenvectors of the logical Hadamard."""
    evals, evecs = np.linalg.eigh(_HADAMARD)
    # eigh returns ascending eigenvalues: [-1, +1].
    h_bar = evecs[:, 0].copy()
    h = evecs[:, 1].copy()
    if np.real(h[0]) < 0:
        h = -h
    if np.real(h_bar[1]) < 0:
        h_bar = -h_bar
    return h, h_bar


_H_KET, _HBAR_KET = _h_eigenbasis()
_P_H = np.outer(_H_KET, _H_KET.conj())
_P_HBAR = np.outer(_HBAR_KET, _HBAR_KET.conj())


def _rotation_perp(theta: float) -> np.ndarray:
    """Rotation about the Bloch axis perpendicular to |H> (within its great
    circle), by angle theta -- moves |H>'s Bloch vector by exactly theta."""
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _depolarize(rho: np.ndarray, lam: float) -> np.ndarray:
    return (1.0 - lam) * rho + lam * (_IDENTITY2 / 2.0)


def build_accepted_state(delta_rad: float, lambda_dep: float) -> np.ndarray:
    """rho_acc = D_lambda_dep(R_delta |H><H| R_delta^dagger)."""
    r = _rotation_perp(delta_rad)
    psi = r @ _H_KET.reshape(2, 1)
    rho_pure = psi @ psi.conj().T
    return _depolarize(rho_pure, lambda_dep)


def build_rejected_state(delta_rej_rad: float, lambda_dep_rej: float) -> np.ndarray:
    """rho_rej, built the same way from the (worse) flagged-copy parameters."""
    return build_accepted_state(delta_rej_rad, lambda_dep_rej)


def true_epsilon_from_params(delta_rad: float, lambda_dep: float) -> float:
    """Exact (not additive-approximation) infidelity of the accepted-copy
    state to the ideal |H>, computed from the constructed density matrix."""
    rho = build_accepted_state(delta_rad, lambda_dep)
    fidelity = float(np.real(np.trace(rho @ _P_H)))
    return 1.0 - fidelity


def twirl(rho: np.ndarray) -> np.ndarray:
    """Dephase into the {|H>, |Hbar>} basis. Preserves Tr[rho @ P_H] exactly."""
    f = float(np.real(np.trace(rho @ _P_H)))
    return f * _P_H + (1.0 - f) * _P_HBAR


def _readout_confusion_2x2(diag: tuple[float, float]) -> np.ndarray:
    """conf[i, j] = P(measure i | true j); columns sum to 1."""
    p00, p11 = diag
    return np.array([[p00, 1.0 - p11], [1.0 - p00, p11]], dtype=complex)


def _normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.real(p), 0.0, None)
    total = p.sum()
    if total <= 0.0:
        raise ValueError("degenerate probability vector")
    return p / total


@dataclass
class MagicBenchmarkEvidence:
    """Authoritative record of what the engine actually ran (cumulative).

    Budget counts accepted points only; rejected points are retained for
    verifier readback but never charged.
    """

    accepted_points: list[dict] = field(default_factory=list)
    rejected_points: list[dict] = field(default_factory=list)
    magic_states_consumed: int = 0


class MagicFactoryEngine:
    """Stateful hidden-dynamics engine for one run. RNG is injected."""

    def __init__(self, hidden: HiddenMagicFactoryConfig, rng: np.random.Generator):
        self._rng = rng
        self._a = hidden.syndrome_acceptance
        self._budget = hidden.magic_state_budget
        self._confusion = _readout_confusion_2x2(hidden.readout_confusion_diag)
        self._confusion4 = np.kron(self._confusion, self._confusion)

        self._rho_acc = build_accepted_state(hidden.delta_rad, hidden.lambda_dep)
        self._rho_rej = build_rejected_state(hidden.delta_rej_rad, hidden.lambda_dep_rej)
        self.true_epsilon = true_epsilon_from_params(hidden.delta_rad, hidden.lambda_dep)
        self.true_epsilon_rejected = true_epsilon_from_params(
            hidden.delta_rej_rad, hidden.lambda_dep_rej
        )

        self._evidence = MagicBenchmarkEvidence()
        self._lock = threading.Lock()

    @property
    def evidence(self) -> MagicBenchmarkEvidence:
        return self._evidence

    @property
    def budget(self) -> int:
        return self._budget

    # -- physics: single-copy measurement probabilities ----------------------

    def _copy_state(self, flagged_mask: np.ndarray, twirl_flag: bool) -> list[np.ndarray]:
        """rho for the 'accepted' branch and the 'flagged' branch, twirled if requested."""
        rho_acc = twirl(self._rho_acc) if twirl_flag else self._rho_acc
        rho_rej = twirl(self._rho_rej) if twirl_flag else self._rho_rej
        return [rho_acc, rho_rej]

    def _single_copy_probs(self, rho: np.ndarray, basis: str) -> np.ndarray:
        u = _BASIS_ROTATION[basis]
        rho_meas = u @ rho @ u.conj().T
        ideal = np.array([rho_meas[0, 0], rho_meas[1, 1]])
        return _normalize_probs(self._confusion @ ideal)

    def _joint_probs(self, rho1: np.ndarray, rho2: np.ndarray) -> np.ndarray:
        rho_joint = np.kron(rho1, rho2)
        rho_out = _JOINT_CIRCUIT @ rho_joint @ _JOINT_CIRCUIT.conj().T
        ideal = np.array([rho_out[i, i] for i in range(4)])
        return _normalize_probs(self._confusion4 @ ideal)

    # -- batch execution ------------------------------------------------------

    def run_batch(self, points: list[MagicBenchmarkPointRequest]) -> list[JobMagicBenchmarkPoint]:
        """Process one batch of points; mutate cumulative evidence atomically."""
        with self._lock:
            return [self._run_point(p) for p in points]

    def _reject_point(self, req: MagicBenchmarkPointRequest, reason: str) -> JobMagicBenchmarkPoint:
        self._evidence.rejected_points.append(
            {
                "n_copies": req.n_copies,
                "twirl": req.twirl,
                "basis": req.basis,
                "shots": req.shots,
                "reason": reason,
            }
        )
        return JobMagicBenchmarkPoint(
            n_copies=req.n_copies,
            twirl=req.twirl,
            basis=req.basis,
            shots=req.shots,
            rejected=True,
            reject_reason=reason,
            magic_states_consumed_this_point=0,
            magic_states_consumed_total=self._evidence.magic_states_consumed,
            budget_remaining=max(0, self._budget - self._evidence.magic_states_consumed),
        )

    def _run_point(self, req: MagicBenchmarkPointRequest) -> JobMagicBenchmarkPoint:
        cost = req.shots * req.n_copies
        if self._evidence.magic_states_consumed + cost > self._budget:
            return self._reject_point(req, "budget_exhausted")

        shots = req.shots
        if req.n_copies == 1:
            flagged, outcomes = self._sample_single(shots, req.twirl, req.basis)
            flagged2: list[int] = []
            outcomes2: list[int] = []
        else:
            flagged, flagged2, outcomes, outcomes2 = self._sample_joint(shots, req.twirl)

        self._evidence.magic_states_consumed += cost
        self._evidence.accepted_points.append(
            {
                "n_copies": req.n_copies,
                "twirl": req.twirl,
                "basis": req.basis,
                "shots": shots,
                "cost": cost,
            }
        )
        return JobMagicBenchmarkPoint(
            n_copies=req.n_copies,
            twirl=req.twirl,
            basis=req.basis,
            shots=shots,
            rejected=False,
            flagged=flagged,
            outcomes=outcomes,
            flagged2=flagged2,
            outcomes2=outcomes2,
            magic_states_consumed_this_point=cost,
            magic_states_consumed_total=self._evidence.magic_states_consumed,
            budget_remaining=max(0, self._budget - self._evidence.magic_states_consumed),
        )

    def _sample_single(
        self, shots: int, twirl_flag: bool, basis: str
    ) -> tuple[list[int], list[int]]:
        is_flagged = self._rng.random(shots) >= self._a
        rho_acc, rho_rej = self._copy_state(is_flagged, twirl_flag)
        p_acc = self._single_copy_probs(rho_acc, basis)
        p_rej = self._single_copy_probs(rho_rej, basis)

        outcomes = np.empty(shots, dtype=int)
        n_acc = int((~is_flagged).sum())
        n_rej = shots - n_acc
        if n_acc:
            outcomes[~is_flagged] = self._rng.choice(2, size=n_acc, p=p_acc)
        if n_rej:
            outcomes[is_flagged] = self._rng.choice(2, size=n_rej, p=p_rej)
        return is_flagged.astype(int).tolist(), outcomes.tolist()

    def _sample_joint(
        self, shots: int, twirl_flag: bool
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        flag1 = self._rng.random(shots) >= self._a
        flag2 = self._rng.random(shots) >= self._a
        rho_acc, rho_rej = self._copy_state(flag1, twirl_flag)

        combo_probs = {
            (False, False): self._joint_probs(rho_acc, rho_acc),
            (False, True): self._joint_probs(rho_acc, rho_rej),
            (True, False): self._joint_probs(rho_rej, rho_acc),
            (True, True): self._joint_probs(rho_rej, rho_rej),
        }

        m1 = np.empty(shots, dtype=int)
        m2 = np.empty(shots, dtype=int)
        for (f1, f2), probs in combo_probs.items():
            mask = (flag1 == f1) & (flag2 == f2)
            n = int(mask.sum())
            if n == 0:
                continue
            idx = self._rng.choice(4, size=n, p=probs)
            m1[mask] = idx // 2
            m2[mask] = idx % 2

        return (
            flag1.astype(int).tolist(),
            flag2.astype(int).tolist(),
            m1.tolist(),
            m2.tolist(),
        )
