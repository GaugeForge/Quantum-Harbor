"""Exact density-matrix branch-tree engine for ``neutral_atom_dm_processor``.

State model: a deterministic list of BRANCHES, one per observed mid-circuit
outcome history. Each branch carries a full density matrix as a ``(2,)*2n``
complex tensor (row axes ``0..n-1``, column axes ``n..2n-1``; atom k = row axis
k), its observed record bits, per-atom sites, per-atom pending idle time, and
per-atom accumulated loss probability. All hot kernels are bandwidth-bound
elementwise view arithmetic (validated against brute-force Kraus math): CZ is diagonal, 1q unitaries
are 2x2 linear combinations of half-array views, depolarizing is a scale +
partial-trace re-embed, and idle noise (exact amplitude damping + pure
dephasing over the accumulated duration) is flushed lazily when an atom is
next touched.

Measurement is an exact POVM: readout confusion and accumulated loss fold into
the OBSERVED-bit branch weights and the post-measurement mixtures, so
feedforward conditions on exactly what real hardware would observe. A maximal
trailing all-``measure`` suffix does not branch: terminal outcomes are sampled
classically from the confusion-transformed joint diagonal, so 10-atom
tomography costs no branch explosion and shots are nearly free.

Determinism (replay contract): branches split in op order, observed 0 before 1,
prune strictly below ``_WEIGHT_FLOOR`` preserving order, never reorder. The
verifier's exact expectations and any re-run of the same request produce the
identical branch list.

Memory (depth-first evaluation): one density matrix costs ``16 * 4**n`` bytes
(268 MB at 12 atoms), so the tree is never held breadth-first. Whenever an op
fans a branch out (a free mid-circuit measure), each child is carried through
the rest of the program -- in tree order -- before its sibling is touched, and a
finished branch is handed to its consumer (terminal statistics, the sampler, or
the verifier's state reducer), which drops the matrix at once. The matrices
alive at any moment are bounded by the fan-outs along ONE path, not by the tree
size; every branch's arithmetic is unchanged, so the branch list is what a
breadth-first sweep would produce, in the same tree order.

That bounds ONE program. Programs also run CONCURRENTLY (the job manager has two
workers) and their matrices are additive, which the per-request caps never
accounted for: two individually legal 14-atom programs together exceed
the container and kill the process. ``memory.ConcurrentStateBudget`` charges each
running program its projected full-layout matrix against a discovered budget, so
a program that does not fit alongside what is already running waits for it
instead of being admitted into an OOM.

Active atoms: a branch's tensor carries only the atoms some op has touched (an
untouched atom is a known product factor, tensored in on first touch), and a
whole-program evaluation (``final=True``) traces an atom out right after the
last op that touches it. Both are exact, and each retired atom shrinks every
later kernel by a factor of four -- a flagged measure-out whose blocks retire
one after another runs its 64-leaf tail on a handful of atoms.

Loss model (defined hidden truth, replay-identical): move loss accumulates a
classical forced-dark probability per atom; a "lost" component reads dark 0 and
leaves the quantum state untouched. Post-loss gate back-action is deliberately
not modeled.

Heralded abort (``measure`` with ``expect``): the shot aborts at the FIRST measured
atom whose observed bit differs from its expected bit. That observation is
recorded (executed = 1); every later record column -- later atoms of the same
op included -- is filler 0 with executed = 0, and the shot is marked in the
aborted mask. The engine parks the aborted branch (weight booked, density
matrix dropped) instead of evolving it, so expected-outcome readouts never
multiply the live branch count. Because a lost atom reads dark 0, ``expect: [1]``
heralds loss while ``expect: [0]`` does not. Parked branches follow the live
ones in every branch list (abort order), part of the determinism contract.
"""

from __future__ import annotations

import base64
import math
import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from qiqcbench.qsim.qtypes.neutral_atom_dm_processor.device import (
    HiddenNeutralAtomDmConfig,
    PublicNeutralAtomDmSpec,
)
from qiqcbench.qsim.qtypes.neutral_atom_dm_processor.evidence import (
    program_measurement_labels,
)
from qiqcbench.qsim.qtypes.neutral_atom_dm_processor.memory import (
    ConcurrentStateBudget,
    concurrent_state_budget_bytes,
    density_matrix_bytes,
)
from qiqcbench.qsim.qtypes.neutral_atom_dm_processor.wire import (
    AtomOp,
    AtomProgramRequest,
    AtomProgramSweepRequest,
    AtomSweepPointSummary,
    JobAtomDmShotData,
    JobAtomDmSweepData,
)

_WEIGHT_FLOOR = 1e-12


def terminal_suffix_start(ops: list[AtomOp]) -> int:
    """Index of the maximal trailing all-measure suffix (distinct atoms, no
    reset). Ops at/after this index are sampled classically without branching;
    record columns before it are the MID-CIRCUIT columns a decode may
    post-select on."""
    seen: set[int] = set()
    start = len(ops)
    for i in range(len(ops) - 1, -1, -1):
        op = ops[i]
        # An expected-outcome (heralded-abort) measure branches the evolution, so it
        # can never be sampled classically as part of the terminal block.
        if op.type != "measure" or op.reset or op.expect is not None:
            break
        targets = op.atoms or []
        if any(a in seen for a in targets):
            break
        seen.update(targets)
        start = i
    return start


# ----------------------------------------------------------------------
# peak working set: how many FULL-LAYOUT density matrices this
# engine has alive at once, counted from what the kernels below actually
# allocate.
#
# Every constant here is a count of simultaneously live full-size arrays, read
# off the code and then confirmed by measurement (peak numpy allocation via
# ``tracemalloc`` against ``np.lib.tracemalloc_domain``, at 9, 10 and 11
# atoms, in units of one density matrix):
#
#   shape                                     measured   accounted as
#   unitary ops + terminal readout only          1.250   rho + the one
#                                                        quarter-size temporary
#                                                        the noise kernels hold
#   one heralded (``expect``) mid-circuit
#     measure                                    1.250   max(1.25, 1 + 1/8)
#   one free (forking) mid-circuit measure       2.125   2**1 + 1/8
#   one measure op over 2 free atoms             4.125   2**2 + 1/8
#   one measure op over 3 free atoms             8.125   2**3 + 1/8
#   2/3/4 successive free measures               2.125   parked siblings are
#                                                        spilled, so 2**1 + 1/8
#
# Where it comes from. ``_apply_1q_unitary`` walks its half-views in eight
# chunks through two scratch buffers (1/8 of a matrix together);
# ``_flush_idle``, ``_reinit_atom``, ``_activate`` and ``_deactivate`` each
# hold one quarter-size temporary, which is the 0.25 in the unitary term
# (``_depolarize_1q`` is chunked to 1/8). ``_apply_measure_atom`` writes each
# surviving outcome's state block by block, the first into a fresh matrix and
# the last into the parent's own buffer, so a measure op holds exactly its
# ``2**free`` outcome states plus the largest scratch on that path (1/8: the
# reset X gate's two chunks, or the block kernel's product and loss chunks).
# Every product runs the same numpy loop the projector form ran (out of place,
# same operand order, no overlap); only additions are done in place. Every
# branch a fan-out parks is spilled to a file by
# ``descend`` until the walk reaches it, so EARLIER fan-outs add nothing; an
# atom whose last op was that fan-out is traced out of each branch only after
# the siblings are parked, so the trace's quarter-size destination never sits
# next to a second outcome matrix (measured: 2.125, not 2.25).
UNITARY_PEAK_COPIES = 1.25
MEASURE_PEAK_BASE_COPIES = 0.125

# Chunking depth of the scratch-buffer kernels: 2**depth chunks along the
# leading axes of the view they walk, so a scratch is that fraction of it.
_UNITARY_CHUNK_AXES = 3
_MEASURE_CHUNK_AXES = 2


def program_peak_matrix_copies(ops: list[AtomOp]) -> float:
    """Upper bound on full-layout density matrices alive at once for ``ops``.

    Exact for a program whose atoms never retire, and an over-estimate when
    they do (a matrix is charged at full layout even after an atom has been
    traced out of it). Over-estimating is the safe direction: this is what
    admission prices, and pricing low can cause a container OOM.
    """

    if not ops:
        return 0.0
    peak = UNITARY_PEAK_COPIES
    for op in ops[: terminal_suffix_start(ops)]:
        measured = op.then if op.type == "feedforward" and op.then is not None else op
        if measured.type != "measure":
            continue
        # A verified (``expect``) readout parks the contradicting branch instead
        # of building its state, so it fans out by one, not by two per atom.
        free_bits = 0 if measured.expect is not None else len(measured.atoms or [])
        peak = max(peak, MEASURE_PEAK_BASE_COPIES + 2.0**free_bits)
    return peak


def program_state_bytes(ops: list[AtomOp], n_atoms: int) -> int:
    """Projected peak resident density-matrix bytes for one program."""

    return math.ceil(program_peak_matrix_copies(ops) * density_matrix_bytes(n_atoms))


def memory_budget_reason(ops: list[AtomOp], n_atoms: int, capacity_bytes: int) -> str:
    """The ONE refusal text. Submission and ``validate_schedule`` both use it, so
    the free legality check can never disagree with what admission will do."""

    copies = program_peak_matrix_copies(ops)
    footprint = program_state_bytes(ops, n_atoms)
    return (
        f"program exceeds the qsim memory budget ({capacity_bytes // 2**20} MiB): "
        f"{n_atoms} atoms give a {density_matrix_bytes(n_atoms) // 2**20} MiB density "
        f"matrix and this schedule holds up to {copies:.4g} of them at once "
        f"({footprint // 2**20} MiB). Use fewer atoms, or fewer free mid-circuit "
        "measurement bits in any ONE measure op (split a wide free readout into "
        "one-atom ops, declare verification readouts with 'expect', or fold them "
        "into the terminal readout block)"
    )


def _contiguous(x: np.ndarray) -> np.ndarray:
    """C-contiguous copy of a view; keeps a 0-d array 0-d (``ascontiguousarray``
    would promote it to shape ``(1,)``, which the axis kernels cannot address)."""
    return np.array(x, order="C") if x.ndim == 0 else np.ascontiguousarray(x)


def _chunk_indices(view: np.ndarray, depth: int) -> list[tuple]:
    """Index tuples that split ``view`` along its first ``depth`` length-2 axes.

    The scratch-buffer kernels walk their operands in these chunks so a scratch
    is a fixed fraction of the matrix. Every element still sees exactly
    the same arithmetic, so the split changes nothing about the result.
    """
    axes = [a for a in range(view.ndim) if view.shape[a] == 2][:depth]
    idx: list = [slice(None)] * view.ndim
    out: list[tuple] = []
    for bits in np.ndindex(*(2,) * len(axes)):
        for a, b in zip(axes, bits, strict=True):
            idx[a] = b
        out.append(tuple(idx))
    return out


def _pack_bits(arr: np.ndarray) -> str:
    packed = np.packbits(np.ascontiguousarray(arr.reshape(-1), dtype=np.uint8))
    return base64.b64encode(packed.tobytes()).decode("ascii")


def _rot_unitary(name: str, theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    if name == "rx":
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
    if name == "ry":
        return np.array([[c, -s], [s, c]], dtype=complex)
    if name == "rz":
        return np.array([[c - 1j * s, 0.0], [0.0, c + 1j * s]], dtype=complex)
    raise ValueError(f"unknown rotation {name!r}")


_INIT_KET = {
    "0": np.array([1.0, 0.0], dtype=complex),
    "1": np.array([0.0, 1.0], dtype=complex),
    "+": np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0),
    "-": np.array([1.0, -1.0], dtype=complex) / math.sqrt(2.0),
}


@dataclass
class _Branch:
    weight: float
    # (2,)*2m tensor over the m ACTIVE atoms (row axes 0..m-1, column axes m..2m-1, in
    # ``active`` order); None once the branch is parked (aborted).
    rho: np.ndarray | None
    bits: list[int] = field(default_factory=list)
    executed: list[int] = field(default_factory=list)
    site_of_atom: list[int] = field(default_factory=list)
    pending_us: list[float] = field(default_factory=list)
    loss_prob: list[float] = field(default_factory=list)
    # Verifier-only exact path accounting. These counters follow the physical
    # branch, so conditional and post-abort ops contribute only where they run.
    elapsed_us: float = 0.0
    executed_moves: int = 0
    # Heralded abort: the index of the ``measure`` op whose ``expect`` the observed
    # bit contradicted. A parked branch never evolves again; its remaining record
    # columns are filler 0 with executed = 0, and its weight is booked classically.
    aborted_at: int | None = None
    # Atoms whose state the tensor carries, in axis order. An atom no op has touched
    # yet is a known product factor (it is activated on first touch); an atom no
    # later op will touch is traced out (``final`` evaluation). Atom indices stay the
    # program's; kernels address axes through ``active.index(atom)``.
    active: list[int] = field(default_factory=list)
    # While ``descend`` walks a sibling, a pending branch's matrix lives in this
    # file rather than in RAM; ``rho`` is None until it is reloaded.
    spill: str | None = None

    def fork(self) -> _Branch:
        return _Branch(
            weight=self.weight,
            rho=None if self.rho is None else self.rho.copy(),
            bits=list(self.bits),
            executed=list(self.executed),
            site_of_atom=list(self.site_of_atom),
            pending_us=list(self.pending_us),
            loss_prob=list(self.loss_prob),
            elapsed_us=self.elapsed_us,
            executed_moves=self.executed_moves,
            aborted_at=self.aborted_at,
            active=list(self.active),
        )


@dataclass
class _PrefixState:
    """The branch tree after the first ``n_done`` ops of a program (replay caching).

    ``continue_program`` resumes from here; ``fork=True`` copies the live branches so
    several readout suffixes can share one evolved preparation. Exactness: the
    per-op arithmetic depends only on the branch state, so splitting the op loop
    and copying arrays is bitwise identical to one uninterrupted replay.
    """

    layout: list[int]
    idle_scale: float
    noise_scale: float
    n_done: int
    live: list[_Branch]
    parked: list[_Branch]
    cost: float


class _Budget:
    __slots__ = ("used", "cap")

    def __init__(self, cap: int) -> None:
        self.used = 0
        self.cap = cap


class NeutralAtomDmEngine:
    """Run-long stateful engine; budgets accumulate across calls (one per run)."""

    def __init__(
        self,
        hidden: HiddenNeutralAtomDmConfig,
        public: PublicNeutralAtomDmSpec,
        rng: np.random.Generator,
        memory: ConcurrentStateBudget | None = None,
    ) -> None:
        self.rng = rng
        self.hidden = hidden
        self.public = public
        self.nz = hidden.noise
        self.dur = public.durations_us
        self.geom = public.layout
        self._lock = threading.Lock()
        self._budget = _Budget(public.budgets.shot_budget)
        self._experiment_calls = 0
        self._recorded_bits = 0
        # Working set. The per-request caps above admit each program on
        # its own shape; this one is what the program's own peak and the
        # ALREADY-RUNNING programs are charged against. The PUBLIC declared
        # budget is the contract the agent plans against, so it is authoritative;
        # a smaller deployment can only clamp it further (fail-safe), which on a
        # correctly sized bundle never happens because the two are bound by test.
        self.memory = memory or ConcurrentStateBudget(
            min(
                public.budgets.state_memory.budget_mib * 1024 * 1024,
                concurrent_state_budget_bytes(),
            )
        )

    # ------------------------------------------------------------------
    # kernels on one branch (n = atom count of the running program)
    # ------------------------------------------------------------------

    @staticmethod
    def _sl(n: int, axis_bits: dict[int, int]) -> tuple:
        idx: list = [slice(None)] * (2 * n)
        for axis, bit in axis_bits.items():
            idx[axis] = bit
        return tuple(idx)

    @staticmethod
    def _blk(n: int, k: int, r: int, c: int) -> tuple:
        """The (row bit r, column bit c) block of atom ``k`` as a VIEW of full rank
        (length-1 axes kept), so the block kernels never see a bare scalar."""
        idx: list = [slice(None)] * (2 * n)
        idx[k] = slice(r, r + 1)
        idx[n + k] = slice(c, c + 1)
        return tuple(idx)

    def _apply_1q_unitary(self, rho: np.ndarray, n: int, u: np.ndarray, k: int) -> None:
        # Exactly the products and sums of
        #     new0 = m00*a0 + m01*a1;  a1 = m10*a0 + m11*a1;  a0 = new0
        # in the same operand order, but through two scratch buffers that walk
        # the half-views in chunks: 1/8 of a matrix of transient instead of 1.5.
        s = t = None
        for axis, m in ((k, u), (n + k, u.conj())):
            a0 = rho[self._sl(n, {axis: 0})]
            a1 = rho[self._sl(n, {axis: 1})]
            chunks = _chunk_indices(a0, min(_UNITARY_CHUNK_AXES, a0.ndim - 1))
            if s is None:  # both axes' half-views chunk to the same shape
                s = np.empty(a0[chunks[0]].shape, dtype=rho.dtype)
                t = np.empty_like(s)
            for idx in chunks:
                c0, c1 = a0[idx], a1[idx]
                np.multiply(m[0, 0], c0, out=s)
                np.multiply(m[0, 1], c1, out=t)
                np.add(s, t, out=s)
                np.multiply(m[1, 0], c0, out=t)
                c0[...] = s
                np.multiply(m[1, 1], c1, out=s)
                np.add(t, s, out=c1)

    def _apply_cz_unitary(self, rho: np.ndarray, n: int, a: int, b: int) -> None:
        rho[self._sl(n, {a: 1, b: 1})] *= -1.0
        rho[self._sl(n, {n + a: 1, n + b: 1})] *= -1.0

    def _depolarize_1q(self, rho: np.ndarray, n: int, k: int, p: float) -> None:
        if p <= 0.0:
            return
        # Chunk by chunk: tr = rho00 + rho11 (before the scaling), rho *= 1-p,
        # rho00 += (p/2)*tr, rho11 += (p/2)*tr -- each product out of place into
        # a scratch, the same loop as the whole-array form, 1/8 of a matrix.
        b00, b11 = self._blk(n, k, 0, 0), self._blk(n, k, 1, 1)
        for idx in _chunk_indices(rho[b00], _MEASURE_CHUNK_AXES):
            tr = rho[b00][idx] + rho[b11][idx]
            rho[idx] *= 1.0 - p  # the chunk's leading indices select every block of it
            scaled = (p / 2.0) * tr
            rho[b00][idx] += scaled
            rho[b11][idx] += scaled

    def _depolarize_2q(self, rho: np.ndarray, n: int, a: int, b: int, p: float) -> None:
        if p <= 0.0:
            return
        tr = None
        for x in (0, 1):
            for y in (0, 1):
                blk = rho[self._sl(n, {a: x, b: y, n + a: x, n + b: y})]
                tr = blk.copy() if tr is None else tr + blk
        rho *= 1.0 - p
        for x in (0, 1):
            for y in (0, 1):
                rho[self._sl(n, {a: x, b: y, n + a: x, n + b: y})] += (p / 4.0) * tr

    def _dephase(self, rho: np.ndarray, n: int, k: int, p_flip: float) -> None:
        if p_flip <= 0.0:
            return
        f = 1.0 - 2.0 * p_flip
        rho[self._sl(n, {k: 0, n + k: 1})] *= f
        rho[self._sl(n, {k: 1, n + k: 0})] *= f

    # ------------------------------------------------------------------
    # active-atom bookkeeping (a branch carries only the atoms it needs)
    # ------------------------------------------------------------------

    def _inactive_factor(self, g: float) -> np.ndarray:
        """State of an atom no op has touched: |0><0| after the preparation error.

        Its idle noise since t = 0 is applied by ``_flush_idle`` on first touch, so
        activating late is exactly the state the atom would carry all along.
        """
        rho = np.zeros((2, 2), dtype=complex)
        rho[0, 0] = 1.0
        self._depolarize_1q(rho, 1, 0, min(self.nz.prep_error * g, 1.0))
        return rho

    def _activate(self, br: _Branch, k: int, g: float) -> int:
        """Axis of atom ``k`` in ``br.rho``; tensors the atom in on first touch."""
        if k in br.active:
            return br.active.index(k)
        m = len(br.active)
        rho = br.rho
        # grown[rows..., new row, cols..., new col] = rho[rows..., cols...] * factor,
        # written straight into that layout: the outer product's own axis order
        # (rows..., cols..., new row, new col) is a view of it, so no copy. The
        # product does not overlap its inputs and the view's inner stride is one
        # element, so numpy runs the same vector multiply loop it ran for the
        # outer product into a fresh array; only the outer iteration order differs.
        grown = np.empty(rho.shape[:m] + (2,) + rho.shape[m:] + (2,), dtype=rho.dtype)
        np.multiply.outer(rho, self._inactive_factor(g), out=np.moveaxis(grown, m, 2 * m))
        br.rho = grown
        br.active.append(k)
        return m

    def _deactivate(self, br: _Branch, k: int) -> None:
        """Trace atom ``k`` out (exact for an atom no later op touches: its pending
        idle noise is trace-preserving and its loss/site only matter when read)."""
        ax = br.active.index(k)
        m = len(br.active)
        br.rho = _contiguous(np.trace(br.rho, axis1=ax, axis2=m + ax))
        br.active.pop(ax)

    @staticmethod
    def _touched_atoms(op: AtomOp, n: int) -> list[int]:
        """Atoms whose state an op reads or writes (a global gate touches all)."""
        if op.type == "feedforward":
            return [] if op.then is None else NeutralAtomDmEngine._touched_atoms(op.then, n)
        if op.type == "gate":
            return list(range(n)) if op.scope == "global" else [int(a) for a in op.atoms or []]
        if op.type == "cz":
            return [int(op.pair[0]), int(op.pair[1])] if op.pair else []
        if op.atom is not None:
            return [int(op.atom)]
        return [int(a) for a in op.atoms or []]

    def _canonical_rho(self, br: _Branch, n: int, g: float) -> np.ndarray:
        """Full-layout density matrix in ATOM order (activates every atom): the
        brute-force comparison form used by tests and diagnostics."""
        for k in range(n):
            self._activate(br, k, g)
        order = [br.active.index(k) for k in range(n)]
        return _contiguous(np.transpose(br.rho, order + [n + ax for ax in order]))

    def _flush_idle(self, br: _Branch, k: int, g: float) -> int:
        """Apply atom ``k``'s accumulated idle noise; returns its axis (activating it)."""
        ax = self._activate(br, k, g)
        t = br.pending_us[k]
        if t <= 0.0:
            return ax
        br.pending_us[k] = 0.0
        m = len(br.active)
        gam = 1.0 - math.exp(-g * t / self.nz.t1_us)
        r_phi = 1.0 / self.nz.t2_star_us - 1.0 / (2.0 * self.nz.t1_us)
        f = math.sqrt(1.0 - gam) * math.exp(-g * t * max(r_phi, 0.0))
        rho = br.rho
        # Write through item assignment: with a single active atom the indexed
        # blocks are scalars, not views, so ``r00 += ...`` would update a copy.
        sl00 = self._sl(m, {ax: 0, m + ax: 0})
        sl11 = self._sl(m, {ax: 1, m + ax: 1})
        rho[sl00] += gam * rho[sl11]
        rho[sl11] *= 1.0 - gam
        rho[self._sl(m, {ax: 0, m + ax: 1})] *= f
        rho[self._sl(m, {ax: 1, m + ax: 0})] *= f
        return ax

    def _reinit_atom(self, rho: np.ndarray, n: int, k: int, state: str) -> None:
        """rho -> Tr_k(rho) (x) |s><s|_k, in place of axis k (well-defined anywhere).

        Written block by block into ``rho`` itself once the partial trace is
        taken: the one temporary is that quarter-size trace.
        """
        tr = rho[self._blk(n, k, 0, 0)] + rho[self._blk(n, k, 1, 1)]
        ket = _INIT_KET[state]
        for r in (0, 1):
            for c in (0, 1):
                amp = ket[r] * np.conj(ket[c])
                block = rho[self._blk(n, k, r, c)]
                if amp != 0:
                    np.multiply(amp, tr, out=block)
                else:
                    block[...] = 0.0

    def _diag(self, rho: np.ndarray, n: int) -> np.ndarray:
        subs = list(range(n)) * 2
        return np.real(np.einsum(rho, subs, list(range(n))))

    def _observed_bit_transfer(self, loss: float, g: float) -> np.ndarray:
        """T[o, t] = P(observed o | true t), folding confusion x g and loss."""
        p10 = min(self.nz.readout_p1_to_0 * g, 1.0)
        p01 = min(self.nz.readout_p0_to_1 * g, 1.0)
        keep = 1.0 - loss
        return np.array(
            [
                [keep * (1.0 - p01) + loss, keep * p10 + loss],
                [keep * p01, keep * (1.0 - p10)],
            ]
        )

    # ------------------------------------------------------------------
    # program execution
    # ------------------------------------------------------------------

    def _terminal_suffix_start(self, ops: list[AtomOp]) -> int:
        return terminal_suffix_start(ops)

    def _advance_all(self, br: _Branch, dt: float) -> None:
        if dt <= 0.0:
            return
        br.elapsed_us += dt
        for k in range(len(br.pending_us)):
            br.pending_us[k] += dt

    def _move_duration_us(self, br: _Branch, atom: int, to_site: int) -> float:
        dist = self.geom.move_distance_um(br.site_of_atom[atom], to_site)
        return self.dur.move_per_um * dist + self.dur.move_settle

    def _apply_measure_atom(
        self,
        branches: list[_Branch],
        n: int,
        k: int,
        g: float,
        record: bool,
        *,
        expect_bit: int | None = None,
        op_index: int | None = None,
    ) -> tuple[list[_Branch], list[_Branch]]:
        """Split every branch on the OBSERVED bit of atom k (POVM-exact).

        Returns ``(live, parked)``. With ``expect_bit`` set (heralded-abort
        verification), the branch whose observed bit differs is PARKED: its weight
        is booked classically, its density matrix is dropped, and it never evolves
        again. Only the expected outcome's post-measurement state is built.

        Each surviving outcome's state is ``proj0*c0 + proj1*c1 [+ rho*loss]``
        over the projectors onto atom k's bit, then ``/= w`` -- computed block by
        block by :meth:`_measured_state` with exactly that arithmetic, the first
        outcome into a fresh matrix and the last into the parent's own buffer, so
        the op holds its outcome states and a 1/16 scratch, never the projectors.
        """
        live: list[_Branch] = []
        parked: list[_Branch] = []
        p10 = min(self.nz.readout_p1_to_0 * g, 1.0)
        p01 = min(self.nz.readout_p0_to_1 * g, 1.0)
        for br in branches:
            ax = self._flush_idle(br, k, g)
            m = len(br.active)
            rho = br.rho
            d = self._diag(rho, m)
            p_true1 = float(np.clip(d[self._sl(m, {ax: 1})[:m]].sum(), 0.0, 1.0))
            p_true0 = 1.0 - p_true1
            t = self._observed_bit_transfer(br.loss_prob[k], g)
            keep = 1.0 - br.loss_prob[k]
            surviving: list[tuple[int, float]] = []
            for observed in (0, 1):
                w = t[observed, 0] * p_true0 + t[observed, 1] * p_true1
                if w < _WEIGHT_FLOOR:
                    continue
                if expect_bit is not None and observed != expect_bit:
                    parked.append(
                        _Branch(
                            weight=br.weight * w,
                            rho=None,
                            bits=br.bits + ([observed] if record else []),
                            executed=br.executed + ([1] if record else []),
                            site_of_atom=list(br.site_of_atom),
                            pending_us=list(br.pending_us),
                            loss_prob=list(br.loss_prob),
                            elapsed_us=br.elapsed_us,
                            executed_moves=br.executed_moves,
                            aborted_at=op_index,
                            active=list(br.active),
                        )
                    )
                    continue
                surviving.append((observed, w))
            for j, (observed, w) in enumerate(surviving):
                fresh = j < len(surviving) - 1  # the last outcome consumes the parent
                if observed == 0:
                    mix = self._measured_state(
                        rho, m, ax, keep * (1.0 - p01), keep * p10, br.loss_prob[k], w, fresh
                    )
                else:
                    mix = self._measured_state(
                        rho, m, ax, keep * p01, keep * (1.0 - p10), 0.0, w, fresh
                    )
                live.append(
                    _Branch(
                        weight=br.weight * w,
                        rho=mix,
                        bits=br.bits + ([observed] if record else []),
                        executed=br.executed + ([1] if record else []),
                        site_of_atom=list(br.site_of_atom),
                        pending_us=list(br.pending_us),
                        loss_prob=list(br.loss_prob),
                        elapsed_us=br.elapsed_us,
                        executed_moves=br.executed_moves,
                        active=list(br.active),
                    )
                )
            # The parent is replaced by its children (or parked): release its matrix
            # before the next branch is split.
            rho = None
            br.rho = None
        return live, parked

    def _measured_state(
        self,
        rho: np.ndarray,
        n: int,
        k: int,
        c0: float,
        c1: float,
        loss: float,
        w: float,
        fresh: bool,
    ) -> np.ndarray:
        """``(proj0*c0 + proj1*c1 [+ rho*loss]) / w`` for atom ``k``, block by block.

        Elementwise this is the projector form exactly: each product is rounded
        on its own, a projector's zero block is added as +0, the loss term comes
        last, and the division is over the whole matrix. Every product is taken
        out of place, as the projector form took it, so it runs the same numpy
        loop; when the blocks are written into ``rho`` itself (``fresh=False``)
        each chunk's product goes through a scratch first.
        """
        dst = np.empty_like(rho) if fresh else rho
        for r, scale in ((0, c0), (1, c1)):
            self._scaled_block(rho[self._blk(n, k, r, r)], dst[self._blk(n, k, r, r)], scale, loss)
        for r, c in ((0, 1), (1, 0)):
            src, out = rho[self._blk(n, k, r, c)], dst[self._blk(n, k, r, c)]
            if loss > 0.0:
                self._scaled_block(src, out, loss, 0.0)  # (+0) + rho*loss
            else:
                out[...] = 0.0  # (+0) + (+0)
        dst /= w
        return dst

    @staticmethod
    def _scaled_block(src: np.ndarray, out: np.ndarray, scale: float, loss: float) -> None:
        """``out = (src*scale + 0) [+ src*loss]`` in chunks, valid when ``out is src``.

        Each product is taken out of place into a chunk-size scratch -- the loop
        the projector form ran, never an in-place one -- and copied in, so writing
        into ``src``'s own buffer changes nothing about the arithmetic.
        """
        chunks = _chunk_indices(src, _MEASURE_CHUNK_AXES)
        shape = src[chunks[0]].shape
        product = np.empty(shape, dtype=src.dtype)
        lossy = np.empty(shape, dtype=src.dtype) if loss > 0.0 else None
        for idx in chunks:
            s, o = src[idx], out[idx]
            if lossy is not None:
                np.multiply(s, loss, out=lossy)  # rho*loss, from the unscaled block
            np.multiply(s, scale, out=product)  # proj*scale
            o[...] = product
            o += 0.0  # + the other projector's zero block
            if lossy is not None:
                o += lossy

    def _root_state(self, layout: list[int], g: float) -> _PrefixState:
        n = len(layout)
        # Every atom starts as the same untouched product factor (see
        # ``_inactive_factor``); the tensor grows as ops touch atoms.
        root = _Branch(
            weight=1.0,
            rho=np.ones((), dtype=complex),
            site_of_atom=list(layout),
            pending_us=[0.0] * n,
            loss_prob=[0.0] * n,
            active=[],
        )
        return _PrefixState(
            layout=list(layout),
            idle_scale=1.0,
            noise_scale=g,
            n_done=0,
            live=[root],
            parked=[],
            cost=0.0,
        )

    def execute_prefix(
        self,
        layout: list[int],
        ops: list[AtomOp],
        prefix_len: int,
        idle_scale: float,
        noise_scale: float,
        *,
        checkpoints: dict[int, Callable[[list[_Branch]], None]] | None = None,
        on_final: Callable[[_Branch], None] | None = None,
        final: bool = False,
    ) -> _PrefixState | str:
        """Evolve the branch tree through ``ops[:prefix_len]`` (all as mid-circuit).

        Returns the resumable state or a clean model-level rejection. See
        :meth:`_continue` for the depth-first evaluation order, the ``checkpoints``
        reducer contract, the streaming ``on_final`` sink and ``final``.
        """
        state = self._root_state(layout, noise_scale)
        state.idle_scale = idle_scale
        return self._continue(
            state, ops, prefix_len, checkpoints=checkpoints, on_final=on_final, final=final
        )

    def continue_program(
        self,
        prefix: _PrefixState,
        ops: list[AtomOp],
        stop: int,
        *,
        fork: bool,
        on_final: Callable[[_Branch], None] | None = None,
        final: bool = False,
    ) -> _PrefixState | str:
        """Resume ``prefix`` through ``ops[prefix.n_done:stop]``.

        ``fork=True`` leaves ``prefix`` untouched (live branches are copied) so
        several suffixes can continue from one evolved preparation; ``fork=False``
        consumes the prefix's live branches and must be its last use. ``final``
        declares ``ops`` the whole future of these branches (see :meth:`_continue`).
        """
        if fork:
            state = _PrefixState(
                layout=list(prefix.layout),
                idle_scale=prefix.idle_scale,
                noise_scale=prefix.noise_scale,
                n_done=prefix.n_done,
                live=[br.fork() for br in prefix.live],
                parked=[br.fork() for br in prefix.parked],
                cost=prefix.cost,
            )
        else:
            state = prefix
        return self._continue(state, ops, stop, on_final=on_final, final=final)

    def _continue(
        self,
        state: _PrefixState,
        ops: list[AtomOp],
        stop: int,
        *,
        checkpoints: dict[int, Callable[[list[_Branch]], None]] | None = None,
        on_final: Callable[[_Branch], None] | None = None,
        final: bool = False,
    ) -> _PrefixState | str:
        """Evolve ``state`` through ``ops[state.n_done:stop]`` DEPTH-FIRST.

        A density matrix of ``n`` atoms costs ``16 * 4**n`` bytes (268 MB at 12
        atoms), so the tree is never materialised breadth-first: whenever an op
        fans the live set out (a free mid-circuit measure), each branch is carried
        through the remaining ops -- in tree order -- before the next one is
        touched, and the siblings still pending are spilled to files in a
        directory of this call's own until the walk reaches them. The
        matrices in RAM at once are therefore bounded by the fan-out of ONE op,
        not by the fan-outs along a path or the size of the tree. Per-branch
        arithmetic is identical to a breadth-first sweep (a spilled matrix is
        reloaded byte for byte), so every branch's weight, record and state are
        the same and the final order is tree order.

        ``checkpoints`` maps an op index ``i`` to a REDUCER called with the live
        branches of the current tree frame just before op ``i`` is applied (after
        ``stop`` ops when ``i == stop``). Depth-first evaluation presents the
        branches alive at op ``i`` as a sequence of disjoint subsets -- one call per
        frame that passes ``i``, in tree order -- so a callback must accumulate
        over calls and must not mutate the branches (work on copies).

        ``on_final`` receives every branch that reaches ``stop`` (tree order); the
        branch is NOT retained in ``state.live``, so the consumer can reduce it and
        drop its density matrix at once. Without it the final branches are
        collected in ``state.live`` in the same order. Parked (aborted) branches
        carry no density matrix; they are collected in ``state.parked`` ordered by
        the op that aborted them (tree order within an op) and padded to the full
        mid-circuit record width (filler 0, executed 0). Returns the state or a
        clean model-level rejection string.

        ``final=True`` declares the whole ``ops`` list (terminal suffix included) the
        complete future of these branches, so an atom is traced out right after the
        last op that touches it -- exact, and the branch shrinks by a factor of four
        per retired atom. A prefix that will be continued with other ops must not
        set it. Atoms untouched so far are carried as a product factor either way.
        """
        n = len(state.layout)
        lam, g = state.idle_scale, state.noise_scale
        dead_after: dict[int, list[int]] = {}
        if final:
            last_use: dict[int, int] = {}
            for index, op in enumerate(ops):
                for k in self._touched_atoms(op, n):
                    last_use[k] = index
            for k, index in last_use.items():
                dead_after.setdefault(index, []).append(k)
            stale = [k for k in range(n) if last_use.get(k, -1) < state.n_done]
            for br in state.live:
                for k in stale:
                    if k in br.active:
                        self._deactivate(br, k)
        budgets = self.public.budgets
        cost_cap = budgets.max_program_cost_units
        max_branches = budgets.max_branches
        state_units = float(4**n)
        stop = min(stop, len(ops))
        checkpoints = checkpoints or {}
        finals: list[_Branch] = []
        sink = on_final if on_final is not None else finals.append
        # Branches passing each op index (== the breadth-first live count there);
        # index ``stop`` counts the finished branches.
        passes = [0] * (stop + 1)

        spill: dict = {"dir": None, "count": 0}

        def park(br: _Branch) -> None:
            """Move a pending sibling's matrix out of RAM until the walk reaches it.
            ``np.save`` writes the buffer as it is and ``np.load`` allocates exactly
            one array back, so the round trip is byte-exact and copy-free."""
            spill["count"] += 1
            path = os.path.join(spill["dir"], f"sibling_{spill['count']:08d}.npy")
            np.save(path, br.rho)
            br.rho = None
            br.spill = path

        def unpark(br: _Branch) -> None:
            if br.spill is not None:
                br.rho = np.load(br.spill)
                os.unlink(br.spill)
                br.spill = None

        def retire(br: _Branch, atoms: list[int]) -> None:
            for k in atoms:
                if k in br.active:
                    self._deactivate(br, k)

        def descend(live: list[_Branch], start: int, dead: list[int] | None = None) -> str | None:
            """``dead``: atoms whose last op was ``start - 1``, still to be traced out
            of each branch. With several branches they are traced only once the
            siblings are parked, so the trace's quarter-size destination never sits
            in RAM next to a second outcome matrix (the fan-out stays 2**free)."""
            if len(live) > 1:
                pending: list[_Branch | None] = list(live)
                live.clear()
                for sibling in pending[1:]:
                    park(sibling)
                for j in range(len(pending)):
                    branch = pending[j]
                    pending[j] = None  # release the sibling once it is consumed
                    unpark(branch)
                    if dead:
                        retire(branch, dead)
                    err = descend([branch], start)
                    if err is not None:
                        return err
                return None
            if dead:
                retire(live[0], dead)
            for i in range(start, stop):
                passes[i] += len(live)
                if passes[i] > max_branches:
                    return (
                        f"mid-circuit branching exceeds max_branches {max_branches} "
                        f"at op {i}; condition on fewer bits"
                    )
                if i in checkpoints:
                    checkpoints[i](live)
                state.cost += len(live) * state_units
                if state.cost > cost_cap:
                    return (
                        f"program exceeds max_program_cost_units ({cost_cap:.3g}) at op {i}: "
                        f"4^{n} state units per branch-op; use fewer atoms, fewer "
                        "mid-circuit measurements, or a shallower schedule"
                    )
                err = self._apply_op(
                    live, n, i, ops[i], lam, g, conditional=False, parked=state.parked
                )
                if err is not None:
                    return err
                live[:] = [br for br in live if br.weight >= _WEIGHT_FLOOR]
                if not live:
                    return None  # this frame aborted or underflowed entirely
                if len(live) > 1:
                    return descend(live, i + 1, dead_after.get(i))
                if i in dead_after:
                    retire(live[0], dead_after[i])
            passes[stop] += len(live)
            if passes[stop] > max_branches:
                return (
                    f"mid-circuit branching exceeds max_branches {max_branches}; "
                    "condition on fewer bits"
                )
            if stop in checkpoints:
                checkpoints[stop](live)
            for br in live:
                sink(br)
            live.clear()
            return None

        # The siblings' directory is this call's own, removed on every exit. It is
        # the process's temporary directory (TMPDIR), never /qsim_logs (agent- and
        # ferry-visible); the shipped bundle declares no /tmp volume or tmpfs, so
        # it is the container's writable layer (a tmpfs would count its pages
        # against the very cgroup the spill exists to stay inside).
        with tempfile.TemporaryDirectory(prefix="qiqcbench-atom-siblings-") as spill_dir:
            spill["dir"] = spill_dir
            err = descend(state.live, state.n_done)
        if err is not None:
            return err
        state.live = finals
        state.n_done = stop
        parked = [br for br in state.parked if br.weight >= _WEIGHT_FLOOR]
        parked.sort(key=lambda br: -1 if br.aborted_at is None else br.aborted_at)
        state.parked = parked
        if passes[stop] == 0 and not parked:
            return "all measurement branches vanished (numerical weight underflow)"
        # Parked branches carry the columns recorded up to their abort; pad them to the
        # full mid-circuit record width (filler 0, executed 0).
        n_mid = len(program_measurement_labels(ops[:stop]))
        for br in parked:
            missing = n_mid - len(br.bits)
            if missing > 0:
                br.bits = br.bits + [0] * missing
                br.executed = br.executed + [0] * missing
        return state

    def _apply_op(
        self,
        branches: list[_Branch],
        n: int,
        op_index: int,
        op: AtomOp,
        lam: float,
        g: float,
        conditional: bool,
        parked: list[_Branch] | None = None,
    ) -> str | None:
        """Apply one op across all branches. Returns an error string on illegality.

        ``parked`` collects the branches an expected-outcome measure aborts; a
        caller that passes none discards their weight (internal probes only).
        """
        nz = self.nz
        if op.type == "init":
            targets = [int(op.atom)] if op.atom is not None else [int(a) for a in op.atoms or []]
            if not targets:
                return f"op {op_index}: init requires 'atom' or 'atoms'"
            for k in targets:
                if not (0 <= k < n):
                    return f"op {op_index}: init atom {k} out of range [0,{n})"
            for br in branches:
                for k in targets:
                    ax = self._flush_idle(br, k, g)
                    m = len(br.active)
                    self._reinit_atom(br.rho, m, ax, op.state or "0")
                    self._depolarize_1q(br.rho, m, ax, min(nz.prep_error * g, 1.0))
            return None

        if op.type == "gate":
            targets = list(range(n)) if op.scope == "global" else [int(a) for a in op.atoms or []]
            for k in targets:
                if not (0 <= k < n):
                    return f"op {op_index}: gate atom {k} out of range [0,{n})"
            u = _rot_unitary(op.gate or "rx", op.angle)
            dt = (
                self.dur.rotation_global if op.scope == "global" else self.dur.rotation_local
            ) * lam
            for br in branches:
                for k in targets:
                    ax = self._flush_idle(br, k, g)
                    m = len(br.active)
                    self._apply_1q_unitary(br.rho, m, u, ax)
                    self._depolarize_1q(br.rho, m, ax, min(nz.rotation_depolarizing * g, 1.0))
                self._advance_all(br, dt)
            return None

        if op.type == "cz":
            a, b = int(op.pair[0]), int(op.pair[1])
            if not (0 <= a < n and 0 <= b < n) or a == b:
                return f"op {op_index}: bad cz pair {op.pair}"
            dt = self.dur.cz * lam
            for br in branches:
                slot_a = self.geom.slot_of(br.site_of_atom[a])
                slot_b = self.geom.slot_of(br.site_of_atom[b])
                if slot_a is None or slot_a != slot_b:
                    return (
                        f"op {op_index}: cz pair {op.pair} is not co-located in one "
                        f"gate-zone slot (sites {br.site_of_atom[a]}, {br.site_of_atom[b]})"
                    )
                ax_a = self._flush_idle(br, a, g)
                ax_b = self._flush_idle(br, b, g)
                m = len(br.active)
                self._apply_cz_unitary(br.rho, m, ax_a, ax_b)
                self._depolarize_2q(br.rho, m, ax_a, ax_b, min(nz.cz_depolarizing * g, 1.0))
                self._advance_all(br, dt)
            return None

        if op.type == "move":
            k = int(op.atom) if op.atom is not None else -1
            if not (0 <= k < n):
                return f"op {op_index}: move atom {op.atom!r} out of range [0,{n})"
            to = op.to_site
            if to is None or not (0 <= to < self.geom.n_sites):
                return f"op {op_index}: move to_site {to!r} out of range [0,{self.geom.n_sites})"
            for br in branches:
                occupied = set(br.site_of_atom)
                if to in occupied and br.site_of_atom[k] != to:
                    return f"op {op_index}: move target site {to} is occupied"
                dt = self._move_duration_us(br, k, to) * lam
                br.site_of_atom[k] = to
                ax = self._flush_idle(br, k, g)
                self._dephase(br.rho, len(br.active), ax, min(nz.move_dephasing * g, 0.5))
                br.loss_prob[k] = 1.0 - (1.0 - br.loss_prob[k]) * (1.0 - nz.move_loss * g)
                self._advance_all(br, dt)
                br.executed_moves += 1
            return None

        if op.type == "measure":
            targets = [int(a) for a in op.atoms or []]
            for k in targets:
                if not (0 <= k < n):
                    return f"op {op_index}: measure atom {k} out of range [0,{n})"
            dt = self.dur.readout * lam
            new_branches = branches[:]
            sink: list[_Branch] = parked if parked is not None else []
            for j, k in enumerate(targets):
                expect_bit = None if op.expect is None else int(op.expect[j])
                new_branches, aborted = self._apply_measure_atom(
                    new_branches, n, k, g, record=True, expect_bit=expect_bit, op_index=op_index
                )
                # The abort takes effect at the first contradicted atom: that observation
                # is recorded, every later column (same op included) is filler.
                for br in aborted:
                    # The readout that revealed the contradiction did execute. The
                    # branch is parked without a density matrix, so charge its elapsed
                    # path directly rather than advancing pending idle time.
                    br.elapsed_us += dt
                sink.extend(aborted)
                if not new_branches:
                    # every branch aborted or underflowed at this atom; whether anything
                    # survives elsewhere in the tree is the caller's judgement
                    break
            branches[:] = new_branches
            for br in branches:
                self._advance_all(br, dt)
                if op.reset:
                    # reset acts on the OBSERVED bit(s): X where observed 1
                    for j, k in enumerate(targets):
                        observed = br.bits[len(br.bits) - len(targets) + j]
                        if observed == 1:
                            self._apply_1q_unitary(
                                br.rho,
                                len(br.active),
                                np.array([[0, 1], [1, 0]], dtype=complex),
                                br.active.index(k),
                            )
                    self._advance_all(br, self.dur.reset * lam)
            return None

        if op.type == "feedforward":
            idx = int(op.on_bit) if op.on_bit is not None else -1
            then = op.then
            if then is None:
                return f"op {op_index}: feedforward requires 'then'"
            # Split branches into matching / non-matching; apply `then` to the
            # matching set. A conditional measure ALWAYS records: non-matching
            # branches append a filler 0 with executed=0.
            matching: list[_Branch] = []
            rest: list[_Branch] = []
            for br in branches:
                if not (0 <= idx < len(br.bits)):
                    return (
                        f"op {op_index}: feedforward on_bit {idx} references a bit not yet "
                        f"recorded ({len(br.bits)} available)"
                    )
                (matching if br.bits[idx] == int(op.value) else rest).append(br)
            result: list[_Branch] = []
            if matching:
                err = self._apply_op(
                    matching, n, op_index, then, lam, g, conditional=True, parked=parked
                )
                if err is not None:
                    return err
            if then.type == "measure":
                targets = then.atoms or []
                for br in rest:
                    br.bits.extend([0] * len(targets))
                    br.executed.extend([0] * len(targets))
                    # the readout window is still scheduled for everyone
                    self._advance_all(br, self.dur.readout * lam)
            result.extend(matching)
            result.extend(rest)
            # deterministic partition order: matching branches first, then the rest
            branches[:] = result
            return None

        return f"op {op_index}: unknown op type {op.type!r}"

    # ------------------------------------------------------------------
    # terminal statistics + sampling
    # ------------------------------------------------------------------

    @staticmethod
    def _terminal_atoms(terminal_ops: list[AtomOp]) -> list[int]:
        atoms_in_order: list[int] = []
        for op in terminal_ops:
            atoms_in_order.extend(int(a) for a in op.atoms or [])
        return atoms_in_order

    @staticmethod
    def _prune_terminal_pmf(br: _Branch, pmf: np.ndarray) -> np.ndarray | None:
        """Apply the replay outcome and retained-branch floor to one terminal row.

        The floor applies to each absolute terminal leaf, not merely its
        conditional probability inside this prefix branch. Retained leaf mass is
        transferred into the row's branch weight before the surviving outcomes
        are normalized for classical sampling.
        """
        pmf = np.asarray(pmf, dtype=float)
        kept = np.where(br.weight * pmf < _WEIGHT_FLOOR, 0.0, pmf)
        retained_mass = float(kept.sum())
        retained_weight = br.weight * retained_mass
        if not math.isfinite(retained_mass) or not math.isfinite(retained_weight):
            raise ValueError("terminal measurement produced a non-finite probability")
        if retained_mass <= 0.0 or retained_weight < _WEIGHT_FLOOR:
            return None
        br.weight = retained_weight
        return kept / retained_mass

    def _terminal_row(
        self,
        br: _Branch,
        terminal_ops: list[AtomOp],
        n: int,
        g: float,
        idle_scale: float,
    ) -> tuple[_Branch, list[int], np.ndarray] | None:
        """(branch, terminal atom order, joint observed-bit pmf) of one LIVE branch."""
        del n
        atoms_in_order = self._terminal_atoms(terminal_ops)
        if not atoms_in_order:
            return br, atoms_in_order, np.ones(1, dtype=float)
        # Distinct reset-free terminal Z measurements commute with local idle
        # channels on atoms read in later windows. Flush one simultaneous target
        # group, advance exactly one readout window, then flush the next group.
        # This preserves the vectorized joint-PMF path without a density branch tree.
        for op in terminal_ops:
            for k in op.atoms or []:
                self._flush_idle(br, int(k), g)
            self._advance_all(br, self.dur.readout * idle_scale)
        m = len(br.active)
        d = self._diag(br.rho, m)
        d = np.clip(d, 0.0, None)
        total = d.sum()
        if total > 0:
            d = d / total
        # marginalize to the measured atoms (in tensor-axis order), then apply
        # per-atom observed-bit transfers and reorder to record order.
        axis_of = {k: br.active.index(k) for k in atoms_in_order}
        keep_axes = sorted(axis_of[k] for k in atoms_in_order)
        drop_axes = [ax for ax in range(m) if ax not in keep_axes]
        marg = d.sum(axis=tuple(drop_axes)) if drop_axes else d
        # marg axes correspond to keep_axes (ascending). Apply transfers:
        for pos, ax in enumerate(keep_axes):
            t = self._observed_bit_transfer(br.loss_prob[br.active[ax]], g)
            marg = np.moveaxis(np.tensordot(t, np.moveaxis(marg, pos, 0), axes=([1], [0])), 0, pos)
        # reorder axes into record order
        perm = [keep_axes.index(axis_of[k]) for k in atoms_in_order]
        marg = np.transpose(marg, perm)
        pmf = self._prune_terminal_pmf(br, marg.reshape(-1))
        return None if pmf is None else (br, atoms_in_order, pmf)

    @staticmethod
    def _parked_row(
        br: _Branch, atoms_in_order: list[int]
    ) -> tuple[_Branch, list[int], np.ndarray]:
        """A parked (aborted) branch's terminal columns are filler 0: a one-hot pmf
        at outcome 0 (never an all-zero pmf, which the sampler could not draw from)."""
        one_hot = np.zeros(2 ** len(atoms_in_order), dtype=float)
        one_hot[0] = 1.0
        return br, atoms_in_order, one_hot

    def _terminal_stats(
        self,
        branches: list[_Branch],
        terminal_ops: list[AtomOp],
        n: int,
        g: float,
        parked: list[_Branch] | None = None,
    ) -> list[tuple[_Branch, list[int], np.ndarray]]:
        """Per branch: (branch, terminal atom order, joint observed-bit pmf).

        Live branches first (tree order), then parked (aborted) branches.
        """
        atoms_in_order = self._terminal_atoms(terminal_ops)
        out = []
        for br in branches:
            row = self._terminal_row(br, terminal_ops, n, g, 1.0)
            if row is not None:
                out.append(row)
        out.extend(self._parked_row(br, atoms_in_order) for br in parked or [])
        return out

    def _stream_terminal_rows(
        self,
        run: Callable[[Callable[[_Branch], None]], _PrefixState | str],
        ops: list[AtomOp],
        suffix: int,
        n: int,
        g: float,
        *,
        idle_scale: float = 1.0,
    ) -> list[tuple[_Branch, list[int], np.ndarray]] | str:
        """Evolve to the terminal suffix through ``run(sink)``, reducing each finished
        branch to its terminal pmf as it arrives (its density matrix is dropped at
        once), then append the parked branches. Rows follow the branch-list order
        of the determinism contract."""
        terminal_ops = list(ops[suffix:])
        atoms_in_order = self._terminal_atoms(terminal_ops)
        rows: list[tuple[_Branch, list[int], np.ndarray]] = []

        def sink(br: _Branch) -> None:
            row = self._terminal_row(br, terminal_ops, n, g, idle_scale)
            if row is not None:
                rows.append(row)
            br.rho = None

        state = run(sink)
        if isinstance(state, str):
            return state
        rows.extend(self._parked_row(br, atoms_in_order) for br in state.parked)
        if not rows:
            return "all terminal measurement outcomes vanished (numerical weight underflow)"
        return rows

    def _sample_records(
        self,
        stats: list[tuple[_Branch, list[int], np.ndarray]],
        shots: int,
        n_mid: int,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample (shots, n_recorded) observed bits + executed mask + aborted mask.

        ``rng`` is the job's own generator (see :meth:`_job_rng`); the engine's
        shared generator is used only by callers that run one job at a time.
        """
        rng = self.rng if rng is None else rng
        weights = np.array([max(br.weight, 0.0) for br, _, _ in stats])
        wsum = weights.sum()
        if wsum <= 0:
            raise ValueError("no surviving branches to sample")
        weights = weights / wsum
        n_term = len(stats[0][1]) if stats else 0
        n_rec = n_mid + n_term
        bits = np.zeros((shots, n_rec), dtype=np.uint8)
        executed = np.ones((shots, n_rec), dtype=np.uint8)
        aborted = np.zeros(shots, dtype=np.uint8)
        branch_choice = rng.choice(len(stats), size=shots, p=weights)
        for b_idx, (br, _, pmf) in enumerate(stats):
            mask = branch_choice == b_idx
            count = int(mask.sum())
            if count == 0:
                continue
            if n_mid:
                bits[mask, :n_mid] = np.array(br.bits, dtype=np.uint8)[None, :]
                executed[mask, :n_mid] = np.array(br.executed, dtype=np.uint8)[None, :]
            if br.aborted_at is not None:
                aborted[mask] = 1
                if n_term:
                    executed[mask, n_mid:] = 0
                continue
            if n_term:
                pmf_norm = pmf / pmf.sum() if pmf.sum() > 0 else pmf
                outcomes = rng.choice(len(pmf_norm), size=count, p=pmf_norm)
                for j in range(n_term):
                    shift = n_term - 1 - j
                    bits[mask, n_mid + j] = (outcomes >> shift) & 1
        return bits, executed, aborted

    # ------------------------------------------------------------------
    # budgets + public entry points
    # ------------------------------------------------------------------

    @staticmethod
    def _program_state_bytes(request: AtomProgramRequest | AtomProgramSweepRequest) -> int:
        """Projected PEAK resident state of one in-flight program.

        The measured peak-copy model above times one full-layout density
        matrix. A sweep evolves its points one after another, so its footprint
        is a single point's, not the sum.
        """

        return program_state_bytes(request.ops, len(request.layout))

    def _reserve_locked(self, request: AtomProgramRequest | AtomProgramSweepRequest) -> str | None:
        b = self.public.budgets
        points = len(request.sweep_values) if isinstance(request, AtomProgramSweepRequest) else 1
        if isinstance(request, AtomProgramSweepRequest) and points > b.max_sweep_points:
            return f"sweep has {points} points > max_sweep_points {b.max_sweep_points}"
        if len(request.ops) > b.max_ops:
            return f"program has {len(request.ops)} ops > max_ops {b.max_ops}"
        if len(request.layout) > b.max_atoms:
            return f"layout has {len(request.layout)} atoms > max_atoms {b.max_atoms}"
        footprint = self._program_state_bytes(request)
        if self.memory.exceeds_capacity(footprint):
            # Refused rather than queued: no amount of waiting frees capacity
            # this program could ever fit in.
            return memory_budget_reason(
                request.ops, len(request.layout), self.memory.capacity_bytes
            )
        if any(not (0 <= s < self.geom.n_sites) for s in request.layout):
            return f"layout site out of range [0,{self.geom.n_sites})"
        if request.shots > b.max_shots_per_call:
            return f"shots {request.shots} exceeds max_shots_per_call {b.max_shots_per_call}"
        total_shots = request.shots * points
        if self._budget.used + total_shots > self._budget.cap:
            return (
                f"shot budget exhausted: {self._budget.used}+{total_shots} > cap {self._budget.cap}"
            )
        if self._experiment_calls + 1 > b.max_experiment_calls:
            return (
                "experiment-call budget exhausted: "
                f"{self._experiment_calls}+1 > cap {b.max_experiment_calls}"
            )
        # Every schema-valid backend attempt consumes the public call quota,
        # including attempts later rejected by semantic validation.
        self._experiment_calls += 1
        recorded = request.shots * points * len(program_measurement_labels(request.ops))
        if recorded > b.max_recorded_bits_per_call:
            return (
                "raw result exceeds max_recorded_bits_per_call: "
                f"{recorded} > {b.max_recorded_bits_per_call}; split the experiment"
            )
        if self._recorded_bits + recorded > b.max_recorded_bits_per_run:
            return (
                "run-wide raw-result budget exhausted: "
                f"{self._recorded_bits}+{recorded} > cap {b.max_recorded_bits_per_run}"
            )
        self._recorded_bits += recorded
        self._budget.used += total_shots
        return None

    def reserve_program(self, request: AtomProgramRequest | AtomProgramSweepRequest) -> str | None:
        """Atomically preflight and charge a request before async enqueue."""
        with self._lock:
            return self._reserve_locked(request)

    def _run_single(
        self,
        layout: list[int],
        ops: list[AtomOp],
        shots: int,
        idle_scale: float,
        noise_scale: float,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]] | str:
        suffix = self._terminal_suffix_start(ops)
        rows = self._stream_terminal_rows(
            lambda sink: self.execute_prefix(
                layout, ops, suffix, idle_scale, noise_scale, on_final=sink, final=True
            ),
            ops,
            suffix,
            len(layout),
            noise_scale,
            idle_scale=idle_scale,
        )
        if isinstance(rows, str):
            return rows
        n_mid = len(program_measurement_labels(ops[:suffix]))
        labels = program_measurement_labels(ops)
        try:
            bits, executed, aborted = self._sample_records(rows, shots, n_mid, rng)
        except ValueError as exc:
            return str(exc)
        if bits.shape[1] != len(labels):
            return (
                f"recorded {bits.shape[1]} bits but the label contract expects "
                f"{len(labels)} (internal record mismatch)"
            )
        return bits, executed, aborted, labels

    @staticmethod
    def _declares_expect(ops: list[AtomOp]) -> bool:
        for op in ops:
            measured = op.then if op.type == "feedforward" and op.then is not None else op
            if measured.type == "measure" and measured.expect is not None:
                return True
        return False

    def _job_rng(self) -> np.random.Generator:
        """A child generator for one job, drawn from the engine's generator.

        Jobs compute OUTSIDE the engine lock (a 12-atom job runs for minutes, and
        every new submission's budget reservation must not wait behind it), so
        each job samples from its own generator: no shared-state races, and the
        child seeds are deterministic in submission order for a given engine seed.
        Callers must hold ``self._lock``.
        """
        return np.random.default_rng(int(self.rng.integers(0, 2**63 - 1)))

    def run_atom_program(
        self, request: AtomProgramRequest, *, preadmitted: bool = False
    ) -> JobAtomDmShotData | str:
        with self._lock:
            if not preadmitted:
                error = self._reserve_locked(request)
                if error is not None:
                    return error
            rng = self._job_rng()
        # Outside ``self._lock`` on purpose: the gate holds for the whole
        # evolution, and new submissions must keep reserving budget meanwhile.
        with self.memory.hold(self._program_state_bytes(request)):
            out = self._run_single(
                request.layout,
                request.ops,
                request.shots,
                request.idle_scale,
                request.noise_scale,
                rng,
            )
        if isinstance(out, str):
            return out
        bits, executed, aborted, labels = out
        return JobAtomDmShotData(
            shots=request.shots,
            n_recorded=len(labels),
            measure_labels=labels,
            measure_bits_b64=_pack_bits(bits),
            executed_b64=_pack_bits(executed),
            idle_scale=request.idle_scale,
            noise_scale=request.noise_scale,
            aborted_b64=_pack_bits(aborted) if self._declares_expect(request.ops) else None,
        )

    def run_atom_program_sweep(
        self, request: AtomProgramSweepRequest, *, preadmitted: bool = False
    ) -> JobAtomDmSweepData | str:
        with self._lock:
            if not preadmitted:
                error = self._reserve_locked(request)
                if error is not None:
                    return error
            rng = self._job_rng()
        labels = program_measurement_labels(request.ops)
        declares_expect = self._declares_expect(request.ops)
        point_bits: list[str] = []
        point_executed: list[str] = []
        point_aborted: list[str] = []
        summaries: list[AtomSweepPointSummary] = []
        # One hold for the whole sweep: the points are evolved one after another,
        # so the resident state is a single point's, and re-acquiring per point
        # would only let a second large program interleave into the gap.
        with self.memory.hold(self._program_state_bytes(request)):
            for value in request.sweep_values:
                lam = value if request.sweep_parameter == "idle_scale" else request.idle_scale
                g = value if request.sweep_parameter == "noise_scale" else request.noise_scale
                out = self._run_single(request.layout, request.ops, request.shots, lam, g, rng)
                if isinstance(out, str):
                    return f"sweep point {value}: {out}"
                bits, executed, aborted, run_labels = out
                if run_labels != labels:
                    return "sweep points disagree on the record contract (internal)"
                point_bits.append(_pack_bits(bits))
                point_executed.append(_pack_bits(executed))
                point_aborted.append(_pack_bits(aborted))
                summaries.append(
                    AtomSweepPointSummary(value=value, shots=request.shots, n_recorded=len(labels))
                )
        return JobAtomDmSweepData(
            sweep_parameter=request.sweep_parameter,
            points=summaries,
            measure_labels=labels,
            n_recorded=len(labels),
            idle_scale=request.idle_scale,
            noise_scale=request.noise_scale,
            point_bits_b64=point_bits,
            point_executed_b64=point_executed,
            point_aborted_b64=point_aborted if declares_expect else None,
        )

    # ------------------------------------------------------------------
    # exact statistics for the verifier (no shot noise, no budget charge)
    # ------------------------------------------------------------------

    def exact_terminal_statistics(
        self,
        layout: list[int],
        ops: list[AtomOp],
        idle_scale: float,
        noise_scale: float,
    ) -> dict | str:
        """Deterministic branch statistics for replay scoring.

        Returns {labels, n_mid, branches: [{weight, bits, executed, aborted,
        elapsed_us, executed_moves, pmf}]} where pmf is the confusion-transformed
        joint observed-bit distribution over the terminal columns (record order).
        Aborted (parked) branches are listed after the live ones with ``aborted =
        True`` and a one-hot pmf at outcome 0. Never charges budgets.
        """
        suffix = self._terminal_suffix_start(ops)
        rows = self._stream_terminal_rows(
            lambda sink: self.execute_prefix(
                layout, ops, suffix, idle_scale, noise_scale, on_final=sink, final=True
            ),
            ops,
            suffix,
            len(layout),
            noise_scale,
            idle_scale=idle_scale,
        )
        if isinstance(rows, str):
            return rows
        return self._terminal_statistics_table(rows, ops, suffix)

    def exact_terminal_statistics_from_prefix(
        self, prefix: _PrefixState, ops: list[AtomOp], suffix: int, *, fork: bool = True
    ) -> dict | str:
        """As :meth:`exact_terminal_statistics`, resuming from a cached prefix.

        With ``fork=True`` ``prefix`` is left untouched (the continuation copies its
        branches), which costs one extra full-layout matrix for as long as the
        prefix must survive. ``fork=False`` consumes the prefix and must be its
        last use; a caller holding the prefix's state elsewhere (a spill file, say)
        uses it to keep the replay footprint down to the execution footprint.
        ``suffix`` is the index where the classically sampled terminal
        block begins; it must be at least ``prefix.n_done`` and at most
        ``len(ops)``.
        """
        if suffix < prefix.n_done:
            return "terminal suffix cannot start inside the cached prefix"
        rows = self._stream_terminal_rows(
            lambda sink: self.continue_program(
                prefix, ops, suffix, fork=fork, on_final=sink, final=True
            ),
            ops,
            suffix,
            len(prefix.layout),
            prefix.noise_scale,
            idle_scale=prefix.idle_scale,
        )
        if isinstance(rows, str):
            return rows
        return self._terminal_statistics_table(rows, ops, suffix)

    @staticmethod
    def _terminal_statistics_table(
        rows: list[tuple[_Branch, list[int], np.ndarray]], ops: list[AtomOp], suffix: int
    ) -> dict:
        return {
            "labels": program_measurement_labels(ops),
            "n_mid": len(program_measurement_labels(ops[:suffix])),
            "branches": [
                {
                    "weight": br.weight,
                    "bits": list(br.bits),
                    "executed": list(br.executed),
                    "aborted": br.aborted_at is not None,
                    "elapsed_us": br.elapsed_us,
                    "executed_moves": br.executed_moves,
                    "pmf": pmf,
                }
                for br, _, pmf in rows
            ],
        }

    def exact_prepared_state(
        self,
        layout: list[int],
        ops: list[AtomOp],
        idle_scale: float,
        noise_scale: float,
        *,
        prefix_len: int,
        keep_atoms: list[int],
        accept_entries: list[tuple[list[int], int]] | None = None,
    ) -> tuple[float, list[tuple[float, np.ndarray]]] | str:
        """Exact reduced state of ``keep_atoms`` after the first ``prefix_len`` ops.

        Replays the shared preparation prefix deterministically and keeps the
        branches whose recorded bits satisfy ``accept_entries`` (the prep's own
        verification post-selection, as GF(2) parity constraints over recorded
        columns). This is the state the protocol actually prepared, so a hidden
        verifier can evaluate code-level observables (logical operators of the
        declared code blocks) on it instead of trusting an agent-declared decode.

        Returns ``(accepted_weight_fraction, branch_states)``, where each entry of
        ``branch_states`` is ``(normalized weight, reduced density matrix of
        keep_atoms in the given ORDER)``. Branches are kept SEPARATE because a
        recorded outcome is classical information the protocol has: a schedule that
        tracks its Pauli frame per shot prepares a definite state on each branch,
        and averaging the branches first would misread that as a mixture. Callers
        that want the unconditional state can average the entries themselves.
        Returns an error string on a clean model-level rejection.
        """
        prefix = self.execute_prefix(layout, ops, prefix_len, idle_scale, noise_scale)
        if isinstance(prefix, str):
            return prefix
        return self.exact_prepared_state_from_prefix(prefix, keep_atoms, accept_entries)

    def exact_prepared_state_from_prefix(
        self,
        prefix: _PrefixState,
        keep_atoms: list[int],
        accept_entries: list[tuple[list[int], int]] | None = None,
        *,
        fork: bool = True,
        include_records: bool = False,
    ) -> tuple[float, list[tuple]] | str:
        """As :meth:`exact_prepared_state`, reading a cached prefix.

        With ``fork=True`` (default) the prefix's branches are left untouched (their
        pending idle noise is flushed on copies); ``fork=False`` flushes in place and
        must be the prefix's last use. ``include_records=True`` appends immutable
        copies of each branch's ``bits`` and ``executed`` columns to its return row;
        this verifier-only form lets a task resolve a declared Pauli frame without
        choosing a best frame from hidden state truth.
        """
        branches = prefix.live
        parked = prefix.parked
        n = len(prefix.layout)
        noise_scale = prefix.noise_scale
        if any(not (0 <= k < n) for k in keep_atoms) or len(set(keep_atoms)) != len(keep_atoms):
            return "keep_atoms must be distinct atom indices of the running program"

        entries = accept_entries or []
        keep_sorted = sorted(keep_atoms)
        # Reorder the kept axes from ascending atom index into the requested order.
        order = [keep_sorted.index(k) for k in keep_atoms]
        n_keep = len(keep_sorted)
        perm = order + [n_keep + i for i in order]

        total_weight = 0.0
        accepted_weight = 0.0
        states: list[tuple] = []
        # Aborted shots count in the denominator and never in the prepared ensemble.
        for br in parked:
            total_weight += br.weight
        for br in branches:
            total_weight += br.weight
            if not _branch_satisfies(br.bits, entries):
                continue
            if fork:
                br = br.fork()
            # Kept atoms: activate + flush their idle noise. Traced atoms need no
            # flush -- a local trace-preserving map does not change the partial trace.
            for k in keep_sorted:
                self._flush_idle(br, k, noise_scale)
            m = len(br.active)
            axis_of = {k: br.active.index(k) for k in keep_sorted}
            kept_axes = set(axis_of.values())
            # Partial-trace subscripts: a traced axis shares one label between its row
            # and column position (contracted); a kept axis keeps two distinct labels.
            row_subs = list(range(m))
            col_subs: list[int] = []
            next_label = m
            for ax in range(m):
                if ax in kept_axes:
                    col_subs.append(next_label)
                    next_label += 1
                else:
                    col_subs.append(row_subs[ax])
            out_subs = [row_subs[axis_of[k]] for k in keep_sorted] + [
                col_subs[axis_of[k]] for k in keep_sorted
            ]
            reduced = np.einsum(br.rho, row_subs + col_subs, out_subs)
            reduced = np.transpose(reduced, perm).reshape(2**n_keep, 2**n_keep)
            trace = float(np.real(np.trace(reduced)))
            if trace > 0:
                reduced = reduced / trace
            if include_records:
                states.append((br.weight, reduced, list(br.bits), list(br.executed)))
            else:
                states.append((br.weight, reduced))
            accepted_weight += br.weight
        if accepted_weight <= 0.0 or total_weight <= 0.0:
            return "no preparation branch survives the declared post-selection"
        normalized = [(row[0] / accepted_weight, *row[1:]) for row in states]
        return accepted_weight / total_weight, normalized


def _branch_satisfies(bits: list[int], entries: list[tuple[list[int], int]]) -> bool:
    for indices, target in entries:
        parity = 0
        for index in indices:
            if index >= len(bits):
                return False
            parity ^= int(bits[index])
        if parity != target:
            return False
    return True


__all__ = ["NeutralAtomDmEngine", "_PrefixState", "terminal_suffix_start"]
