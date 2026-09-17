"""Bosonic-cavity QEC engine: branch-tree open-system program executor.

Executes an agent-submitted **program** (ordered ops on the joint cavity (x)
ancilla density matrix) under the hidden open-system noise model, then returns
**raw per-shot measurement outcomes**. Because mid-circuit ``ancilla_measure`` +
``conditional`` make the program a fixed CPTP map (a sum over syndrome branches),
the engine propagates a **branch tree of sub-normalized density matrices** (each
branch's trace is its probability), prunes negligible branches, and **multinomially
samples** the requested shots from the resulting joint distribution over
(recorded syndromes, final readouts). This yields honest binomial shot statistics
while computing the exact channel -- and avoids ``shots`` separate trajectory loops.

Modeling notes (documented approximations, v1):
- Every op of finite (hidden) duration accrues cavity photon loss + cavity
  dephasing + ancilla T1/T2 decoherence (analytic Kraus channels).
- ``displace``/``snap``/``dispersive_wait`` apply the *exact ideal unitary* in the
  control frame; the real dispersive ``chi`` differs from the nominal value, so a
  parity wait tuned to the nominal ``chi`` is slightly off.
- During the two agent-timed waits (``idle``, ``dispersive_wait``) the storage
  mode also evolves under its own hidden Hamiltonian in the control frame --
  detuning ``cavity_detuning_khz`` and self-Kerr ``cavity_self_kerr_khz`` -- solved
  exactly together with loss and dephasing per coherence chain
  (``physics.cavity_free_evolution_propagators``); with both zero the closed-form
  Kraus path is used unchanged. Gates and readout keep their hidden durations and
  accrue no frame phase (the agent cannot know those durations).
- ``photon_number_measure`` is a terminal, non-collapsing readout (snapshot the
  cavity number distribution; sample per shot with a small assignment error). It
  is meant to be placed at the end of a tomography program.
- ``idle`` additionally accrues a random-displacement diffusion channel when the
  hidden config sets ``idle_displacement_variance_per_us > 0`` (Gauss-Hermite
  mixture of displacements applied after the op's loss/dephasing; the ordering
  is a documented second-order approximation).
- Low-probability branches are pruned (threshold + a hard branch cap); the
  surviving distribution is renormalized before sampling.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

from qiqcbench.qsim.qtypes.bosonic_cavity_qec import physics as P
from qiqcbench.qsim.qtypes.bosonic_cavity_qec.device import HiddenBosonicCavityConfig

_PRUNE = 1e-7
_MAX_BRANCHES = 4000
_MAX_DECO_CACHE_ENTRIES = 20
_MAX_DIFFUSION_CACHE_ENTRIES = 16
# Total negative population a number-basis diagonal may shed as float roundoff.
# The branch tree starts from a unit-trace state, so arithmetic roundoff is set
# by that O(1) scale (~1e-16 per entry, a few 1e-15 summed over the Fock cutoff)
# rather than by the branch's own, possibly tiny, weight -- hence an absolute
# bound, which a purely relative one could not give a small branch carrying
# inherited error. Bounding the SUM rather than the smallest entry is what makes
# it a bound on the mass this function is allowed to discard: a per-entry rule
# lets n_max entries each sit just inside the limit and quietly throw away n_max
# times as much.
_DIAG_NEG_MASS_TOL = 1e-9
# How far a stored snapshot's sum may drift from one and still be handed to
# ``choice`` untouched. ``Generator.choice`` accepts sqrt(float eps) ~ 1.5e-8,
# but it measures that against its own compensated sum, which is not the sum
# ``ndarray.sum()`` computes: a vector sitting on that boundary passes here and
# is still rejected there. A snapshot built by ``diag / diag.sum()`` lands within
# a few ulps of one, so this window is ~1e4x wider than well-formed snapshots
# need and ~1e4x tighter than numpy's threshold, clear of both boundaries.
_PMF_IDENTITY_TOL = 1e-12


def _number_distribution(diag: np.ndarray) -> np.ndarray:
    """Normalize a cavity number-basis population vector into a sampler-safe pmf.

    ``diag`` is read off a sub-normalized branch density matrix, so entries that
    are physically zero can come back as tiny negatives (the observed case was
    -1.5e-45). ``Generator.choice`` rejects any negative outright, which turned
    pure roundoff into a qsim infrastructure failure. Clip that roundoff away
    before renormalizing.

    Discarding more than ``_DIAG_NEG_MASS_TOL`` of negative population, a
    non-finite entry, or a diagonal with no positive mass left to normalize is
    not roundoff: the propagated state is not a density matrix any more. Those
    still raise, because substituting an invented distribution would hand the
    verifier fabricated shot evidence.
    """
    diag = np.asarray(diag, dtype=float)
    if not np.all(np.isfinite(diag)):
        raise RuntimeError("cavity number distribution contains a non-finite population")
    negative_mass = -float(diag[diag < 0.0].sum())
    if negative_mass > _DIAG_NEG_MASS_TOL:
        raise RuntimeError(
            f"cavity number distribution carries {negative_mass:.3e} of negative population, "
            f"beyond the {_DIAG_NEG_MASS_TOL:.0e} roundoff tolerance"
        )
    if negative_mass > 0.0:
        diag = np.maximum(diag, 0.0)
    s = float(diag.sum())
    if s <= 1e-15:
        # Unreachable for a branch that survives to sampling: branch weights sum
        # to one over at most _MAX_BRANCHES branches and _prune floors a live
        # branch at _PRUNE, while trace is non-increasing along a branch. Fail
        # rather than hand back a vector that is not a distribution.
        raise RuntimeError("cavity number distribution has no positive population to normalize")
    return diag / s


def _sampling_pmf(dist: np.ndarray) -> np.ndarray:
    """Return ``dist`` unchanged when it is safely a pmf, else re-normalize.

    The fast path is an exact identity, so a well-formed snapshot produces the
    same draws it always did; the slow path only runs on a vector that is at or
    near the edge of what ``choice`` accepts.
    """
    if dist.size and float(dist.min()) >= 0.0 and abs(float(dist.sum()) - 1.0) <= _PMF_IDENTITY_TOL:
        return dist
    return _number_distribution(dist)


@dataclass
class _Branch:
    rho: np.ndarray  # sub-normalized joint density matrix (trace == probability)
    # one entry per ancilla_measure op, in order: an int recorded bit (referenced
    # measures) or ("defer", true_outcome) for measures sampled at shot time.
    ancilla_outcomes: list = field(default_factory=list)
    # one cavity number-distribution snapshot per photon_number_measure op.
    number_snapshots: list = field(default_factory=list)

    @property
    def weight(self) -> float:
        return float(np.real(np.trace(self.rho)))


class CavityQecEngine:
    """Branch-tree executor for bosonic-cavity QEC programs (numpy only)."""

    def __init__(self, hidden: HiddenBosonicCavityConfig, rng: np.random.Generator) -> None:
        self.h = hidden
        self.rng = rng
        self.N = int(hidden.n_max)
        self._pg, self._pe = P.ancilla_projectors(self.N)
        self._reset_kraus = P.ancilla_reset_kraus(self.N)
        self._deco_cache: OrderedDict[float, tuple[list[list[np.ndarray]], np.ndarray]] = (
            OrderedDict()
        )
        self._diffusion_cache: OrderedDict[float, list[np.ndarray]] = OrderedDict()
        # The storage mode's own dynamics in the control frame (detuning + self-Kerr)
        # act during the two agent-timed waits (`idle`, `dispersive_wait`). Both zero
        # (every pre-frame device) keeps the closed-form loss/dephasing path bit-for-bit.
        self._frame = bool(hidden.cavity_detuning_khz != 0.0 or hidden.cavity_self_kerr_khz != 0.0)
        self._free_cache: dict[float, list[np.ndarray]] = {}

    # ---- decoherence ----

    def _deco_channels(self, duration_ns: float) -> tuple[list[list[np.ndarray]], np.ndarray]:
        key = round(float(duration_ns), 4)
        cached = self._deco_cache.get(key)
        if cached is not None:
            self._deco_cache.move_to_end(key)
            return cached
        dt = key / 1000.0  # us
        gamma = 1.0 - math.exp(-dt / self.h.cavity_t1_us)
        p_anc = 1.0 - math.exp(-dt / self.h.ancilla_t1_us)
        lam_a = 1.0 - math.exp(-2.0 * dt / self.h.ancilla_tphi_us)
        channels = [
            P.amplitude_damping_kraus(self.N, gamma),
            P.ancilla_amplitude_damping_kraus(self.N, p_anc),
            P.ancilla_dephasing_kraus(self.N, lam_a),
        ]
        mask = P.cavity_dephasing_mask(self.N, dt, self.h.cavity_tphi_us)
        self._deco_cache[key] = (channels, mask)
        if len(self._deco_cache) > _MAX_DECO_CACHE_ENTRIES:
            self._deco_cache.popitem(last=False)
        return channels, mask

    def _apply_deco(self, rho: np.ndarray, duration_ns: float) -> np.ndarray:
        channels, mask = self._deco_channels(duration_ns)
        for channel in channels:
            rho = P.apply_kraus(rho, channel)
        return rho * mask

    def _apply_wait_deco(self, rho: np.ndarray, duration_ns: float) -> np.ndarray:
        """Decoherence over an agent-timed wait (``idle`` / ``dispersive_wait``).

        With the frame parameters set, the cavity evolves under its own
        Hamiltonian in the control frame (detuning + self-Kerr) together with loss
        and dephasing, solved exactly per coherence chain
        (:func:`physics.cavity_free_evolution_propagators`); the ancilla channels
        are applied as in :meth:`_apply_deco`. With both parameters zero this is
        :meth:`_apply_deco` itself, so pre-frame devices are unchanged.
        """
        if not self._frame:
            return self._apply_deco(rho, duration_ns)
        key = round(float(duration_ns), 4)
        props = self._free_cache.get(key)
        if props is None:
            props = P.cavity_free_evolution_propagators(
                self.N,
                key / 1000.0,
                1.0 / self.h.cavity_t1_us,
                self.h.cavity_tphi_us,
                self.h.cavity_detuning_khz * 1e-3,
                self.h.cavity_self_kerr_khz * 1e-3,
            )
            self._free_cache[key] = props
        rho = P.apply_cavity_free_evolution(rho, props, self.N)
        channels, _mask = self._deco_channels(duration_ns)
        _cavity_loss, ancilla_t1, ancilla_dephasing = channels
        rho = P.apply_kraus(rho, ancilla_t1)
        return P.apply_kraus(rho, ancilla_dephasing)

    def _idle_diffusion_kraus(self, duration_ns: float) -> list[np.ndarray]:
        """Random-displacement diffusion accrued over one `idle` op (v1: applied
        after the loss/dephasing of the same op; the ordering difference is a
        documented second-order approximation)."""
        rate = self.h.idle_displacement_variance_per_us
        if rate <= 0.0:
            return []
        key = round(float(duration_ns), 4)
        cached = self._diffusion_cache.get(key)
        if cached is None:
            sigma = math.sqrt(rate * key / 1000.0)
            cached = P.gaussian_displacement_kraus(self.N, sigma)
            self._diffusion_cache[key] = cached
            if len(self._diffusion_cache) > _MAX_DIFFUSION_CACHE_ENTRIES:
                self._diffusion_cache.popitem(last=False)
        else:
            self._diffusion_cache.move_to_end(key)
        return cached

    # ---- single coherent op + its duration decoherence ----

    def _apply_coherent(
        self,
        branches: list[_Branch],
        u: np.ndarray,
        duration_ns: float,
        gate_deph: list[np.ndarray] | None,
        *,
        wait: bool = False,
    ) -> None:
        deco = self._apply_wait_deco if wait else self._apply_deco
        for b in branches:
            b.rho = P.apply_unitary(b.rho, u)
            if gate_deph is not None:
                b.rho = P.apply_kraus(b.rho, gate_deph)
            b.rho = deco(b.rho, duration_ns)

    # ---- ancilla measurement (branch on true outcome, + recorded bit if referenced) ----

    def _measure_ancilla(
        self, branches: list[_Branch], *, reset: bool, referenced: bool, record: bool = True
    ) -> list[_Branch]:
        peg = self.h.readout_p_e_given_g
        pge = self.h.readout_p_g_given_e
        out: list[_Branch] = []
        if not record:
            # Unrecorded measure-and-reset: the same projection (and reset), but
            # the outcome is discarded, so the two branches are merged at once --
            # the channel is identical, the branch tree does not grow.
            for b in branches:
                rho = self._apply_deco(b.rho, self.h.ancilla_measure_duration_ns)
                merged = np.zeros_like(rho)
                for t, proj in ((0, self._pg), (1, self._pe)):
                    rho_t = proj @ rho @ proj
                    if reset and t == 1:
                        rho_t = P.apply_kraus(rho_t, self._reset_kraus)
                    merged += rho_t
                b.rho = merged
                out.append(b)
            return self._prune(out)
        for b in branches:
            rho = self._apply_deco(b.rho, self.h.ancilla_measure_duration_ns)
            parts: list[np.ndarray] = []
            for t, proj in ((0, self._pg), (1, self._pe)):
                rho_t = proj @ rho @ proj
                if reset and t == 1:
                    rho_t = P.apply_kraus(rho_t, self._reset_kraus)
                parts.append(rho_t)
            if referenced:
                # Only the reported bit is available to later feedback. Merge
                # the two true-outcome contributions with the same reported bit;
                # linear subsequent channels preserve the exact joint law while
                # each recorded measurement grows the live tree by at most 2x.
                for r in (0, 1):
                    conf0 = peg if r == 1 else 1.0 - peg
                    conf1 = pge if r == 0 else 1.0 - pge
                    rho_r = conf0 * parts[0] + conf1 * parts[1]
                    if float(np.real(np.trace(rho_r))) < _PRUNE:
                        continue
                    out.append(
                        _Branch(
                            rho=rho_r,
                            ancilla_outcomes=[*b.ancilla_outcomes, int(r)],
                            number_snapshots=list(b.number_snapshots),
                        )
                    )
            else:
                for t, rho_t in enumerate(parts):
                    if float(np.real(np.trace(rho_t))) < _PRUNE:
                        continue
                    out.append(
                        _Branch(
                            rho=rho_t.copy(),
                            ancilla_outcomes=[*b.ancilla_outcomes, ("defer", t)],
                            number_snapshots=list(b.number_snapshots),
                        )
                    )
        return self._prune(out)

    def _prune(self, branches: list[_Branch]) -> list[_Branch]:
        live = [b for b in branches if b.weight >= _PRUNE]
        if len(live) > _MAX_BRANCHES:
            live.sort(key=lambda b: b.weight, reverse=True)
            live = live[:_MAX_BRANCHES]
        return live

    # ---- cavity number distribution (for photon_number_measure) ----

    def _cavity_number_dist(self, rho: np.ndarray) -> np.ndarray:
        r4 = rho.reshape(self.N, 2, self.N, 2)
        diag = np.real(np.einsum("nana->n", r4))
        return _number_distribution(diag)

    # ---- op dispatch ----

    def _coherent_unitary(self, op) -> tuple[np.ndarray, float, str | None]:
        """Return (joint_unitary, duration_ns, gate_subsystem) for a coherent op."""
        N = self.N
        if op.kind == "displace":
            u = P.lift_cavity(P.displace_unitary(complex(op.alpha_re, op.alpha_im), N), N)
            return u, self.h.displace_duration_ns, "cavity"
        if op.kind == "snap":
            u = P.lift_cavity(P.snap_unitary(op.thetas, N), N)
            return u, self.h.snap_duration_ns, "cavity"
        if op.kind == "ancilla_rotate":
            u = P.lift_ancilla(P.ancilla_rotation(op.theta, op.phi), N)
            return u, self.h.ancilla_rotate_duration_ns, "ancilla"
        if op.kind == "dispersive_wait":
            chi_t = 2.0 * math.pi * self.h.chi_mhz * (op.duration_ns / 1000.0)
            u = P.dispersive_phase_unitary(N, chi_t)
            return u, op.duration_ns, None
        raise ValueError(f"not a coherent op: {op.kind}")

    def _apply_op(self, branches: list[_Branch], op, *, referenced: bool) -> list[_Branch]:
        N = self.N
        deph = self.h.gate_depolarizing
        if op.kind in ("displace", "snap", "ancilla_rotate", "dispersive_wait"):
            u, dur, subsys = self._coherent_unitary(op)
            gate_deph = None
            if deph > 0.0 and subsys == "ancilla":
                gate_deph = P.ancilla_dephasing_kraus(N, deph)
            # cavity-gate coherent error is dominated by finite-duration decoherence;
            # a dispersive wait is agent-timed, so the mode's own evolution runs too.
            self._apply_coherent(branches, u, dur, gate_deph, wait=(op.kind == "dispersive_wait"))
            return branches
        if op.kind == "idle":
            diffusion = self._idle_diffusion_kraus(op.duration_ns)
            for b in branches:
                b.rho = self._apply_wait_deco(b.rho, op.duration_ns)
                if diffusion:
                    b.rho = P.apply_kraus(b.rho, diffusion)
            return branches
        if op.kind == "ancilla_measure":
            return self._measure_ancilla(
                branches, reset=op.reset, referenced=referenced, record=op.record
            )
        if op.kind == "photon_number_measure":
            for b in branches:
                b.number_snapshots.append(self._cavity_number_dist(b.rho))
            return branches
        raise ValueError(f"unexpected op kind {op.kind}")

    def _apply_conditional(self, branches: list[_Branch], cond) -> list[_Branch]:
        if cond.op.kind in {"ancilla_measure", "photon_number_measure"}:
            raise ValueError(
                "conditional measurement is unsupported until the result schema "
                "can represent path-dependent measurement records"
            )
        target_idx = cond.on_index
        # branches whose recorded syndrome #on_index == value receive the op.
        hit = []
        miss = []
        for b in branches:
            rec = b.ancilla_outcomes[target_idx] if target_idx < len(b.ancilla_outcomes) else None
            recorded = rec if isinstance(rec, int) else (rec[1] if isinstance(rec, tuple) else None)
            if recorded == cond.value:
                hit.append(b)
            else:
                miss.append(b)
        if hit:
            hit = self._apply_op(hit, cond.op, referenced=False)
        return hit + miss

    # ---- public drivers ----

    def channel_state(self, ops: list, max_branches: int = 256) -> np.ndarray:
        """Exact merged joint density matrix after ``ops`` -- no shot sampling.

        Runs the same CPTP channel as :meth:`run_program` and returns the sum of
        the surviving (sub-normalized) branch density matrices. Deterministic
        regardless of the injected rng (which only affects shot sampling). Used
        by hidden-dynamics verifiers to replay an agent-cited program under the
        true noise model.

        Branching discipline is tighter than :meth:`run_program` (which keeps
        per-outcome branches for honest shot statistics): unreferenced
        ``ancilla_measure`` ops do not branch at all (their outcome branches are
        summed immediately), and measures referenced by a ``conditional`` branch
        only on the *recorded* bit (the two true-outcome contributions per
        recorded bit are merged). The channel state is identical; the branch
        count is bounded by ``2^n_referenced``, further capped at
        ``max_branches`` by weight-ranked pruning (honest sBs / sharpen-trim
        programs stay far below the cap). ``record=False`` measures are merged
        exactly like unreferenced ones and do not advance the ``on_index``
        numbering (which counts recorded measures only).
        """
        referenced: set[int] = set()
        for op in ops:
            if getattr(op, "kind", None) == "conditional":
                referenced.add(int(op.on_index))

        rho0 = np.zeros((2 * self.N, 2 * self.N), dtype=complex)
        rho0[0, 0] = 1.0
        branches = [_Branch(rho=rho0)]
        peg = self.h.readout_p_e_given_g
        pge = self.h.readout_p_g_given_e
        a_counter = 0
        for op in ops:
            if op.kind == "ancilla_measure" and not op.record:
                branches = self._measure_ancilla(
                    branches, reset=op.reset, referenced=False, record=False
                )
            elif op.kind == "ancilla_measure":
                out: list[_Branch] = []
                for b in branches:
                    rho = self._apply_deco(b.rho, self.h.ancilla_measure_duration_ns)
                    parts = []
                    for t, proj in ((0, self._pg), (1, self._pe)):
                        rho_t = proj @ rho @ proj
                        if op.reset and t == 1:
                            rho_t = P.apply_kraus(rho_t, self._reset_kraus)
                        parts.append(rho_t)
                    if a_counter in referenced:
                        for r in (0, 1):
                            conf0 = peg if r == 1 else 1.0 - peg
                            conf1 = pge if r == 0 else 1.0 - pge
                            rho_r = conf0 * parts[0] + conf1 * parts[1]
                            if float(np.real(np.trace(rho_r))) < _PRUNE:
                                continue
                            out.append(
                                _Branch(
                                    rho=rho_r,
                                    ancilla_outcomes=[*b.ancilla_outcomes, int(r)],
                                )
                            )
                    else:
                        out.append(
                            _Branch(
                                rho=parts[0] + parts[1],
                                ancilla_outcomes=[*b.ancilla_outcomes, None],
                            )
                        )
                branches = [x for x in out if x.weight >= _PRUNE]
                if len(branches) > max_branches:
                    branches.sort(key=lambda x: x.weight, reverse=True)
                    branches = branches[:max_branches]
                a_counter += 1
            elif op.kind == "conditional":
                branches = self._apply_conditional(branches, op)
            else:
                branches = self._apply_op(branches, op, referenced=False)
        total = np.zeros((2 * self.N, 2 * self.N), dtype=complex)
        for b in branches:
            total += b.rho
        return total

    def run_program(self, ops: list, shots: int, sweep_value: float | None = None) -> dict:
        # which RECORDED ancilla-measure indices are referenced by a conditional?
        # (``on_index`` counts recorded measures only; unrecorded measure-and-reset
        # ops neither branch nor produce a result column.)
        referenced: set[int] = set()
        for op in ops:
            if getattr(op, "kind", None) == "conditional":
                referenced.add(int(op.on_index))

        rho0 = np.zeros((2 * self.N, 2 * self.N), dtype=complex)
        rho0[0, 0] = 1.0  # |0>_cav |g>_anc
        branches = [_Branch(rho=rho0)]

        # (kind, index-among-its-kind, original program-op index), in program order
        meas_order: list[tuple[str, int, int]] = []
        a_seen = 0
        n_seen = 0
        a_counter = 0
        for op_index, op in enumerate(ops):
            kind = op.kind
            if kind == "ancilla_measure" and not op.record:
                branches = self._measure_ancilla(
                    branches, reset=op.reset, referenced=False, record=False
                )
            elif kind == "ancilla_measure":
                meas_order.append(("ancilla", a_seen, op_index))
                branches = self._measure_ancilla(
                    branches, reset=op.reset, referenced=(a_counter in referenced)
                )
                a_seen += 1
                a_counter += 1
            elif kind == "photon_number_measure":
                meas_order.append(("number", n_seen, op_index))
                branches = self._apply_op(branches, op, referenced=False)
                n_seen += 1
            elif kind == "conditional":
                branches = self._apply_conditional(branches, op)
            else:
                branches = self._apply_op(branches, op, referenced=False)

        return self._sample(branches, shots, meas_order, sweep_value)

    def _sample(
        self,
        branches: list[_Branch],
        shots: int,
        meas_order: list[tuple[str, int, int]],
        sweep_value: float | None,
    ) -> dict:
        weights = np.array([max(b.weight, 0.0) for b in branches], dtype=float)
        total = weights.sum()
        kinds = [kind for kind, _, _ in meas_order]
        measurement_op_indices = [op_index for _, _, op_index in meas_order]
        if total <= 0.0 or not branches:
            return {
                "sweep_value": sweep_value,
                "measurement_kinds": kinds,
                "measurement_op_indices": measurement_op_indices,
                "outcomes": [[0 for _ in meas_order] for _ in range(shots)],
            }
        weights /= total
        counts = self.rng.multinomial(shots, weights)

        peg = self.h.readout_p_e_given_g
        pge = self.h.readout_p_g_given_e
        nerr = self.h.number_readout_error
        outcomes: list[list[int]] = []
        for b, c in zip(branches, counts, strict=True):
            if c == 0:
                continue
            # precompute branch-constant ancilla records and number cdfs
            snapshots = [_sampling_pmf(snap) for snap in b.number_snapshots]
            for _ in range(int(c)):
                row: list[int] = []
                for kind, idx, _ in meas_order:
                    if kind == "ancilla":
                        rec = b.ancilla_outcomes[idx]
                        if isinstance(rec, int):
                            row.append(rec)
                        else:  # ("defer", true_outcome) -> sample confusion
                            t = rec[1]
                            if t == 0:
                                row.append(1 if self.rng.random() < peg else 0)
                            else:
                                row.append(0 if self.rng.random() < pge else 1)
                    else:  # number
                        dist = snapshots[idx]
                        n = int(self.rng.choice(self.N, p=dist))
                        if nerr > 0.0 and self.rng.random() < nerr and self.N > 1:
                            if n == 0:
                                n = 1
                            elif n == self.N - 1:
                                n = self.N - 2
                            else:
                                n = n + (1 if self.rng.random() < 0.5 else -1)
                        row.append(n)
                outcomes.append(row)
        # shots were assigned per branch; outcomes length == shots
        return {
            "sweep_value": sweep_value,
            "measurement_kinds": kinds,
            "measurement_op_indices": measurement_op_indices,
            "outcomes": outcomes,
        }


def program_digest(ops: list) -> str:
    """Commit to every recursively validated operation and parameter."""
    payload = [op.model_dump(mode="json") for op in ops]
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


__all__ = ["CavityQecEngine", "program_digest"]
