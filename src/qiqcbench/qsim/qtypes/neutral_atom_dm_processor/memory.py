"""Working-set budget for the ``neutral_atom_dm_processor`` engine.

Every admission check in the engine is PER REQUEST. Two individually
legal programs could therefore both be admitted and both start executing, and
their density matrices are additive: a 14-atom matrix is ``16 * 4**14`` = 4 GiB,
so two of them exceeded what the sidecar was sized for and the process was
killed. A killed qsim never recovers inside a trial -- the agent then polls a
dead MCP transport until its whole budget is gone -- so the working set has to be
bounded before the jobs run, not discovered afterwards.

What is charged: the PEAK number of full-layout density matrices a program holds
at once (``engine.program_peak_matrix_copies``, every constant there measured),
times ``16 * 4**n_atoms``. It is deliberately NOT the engine's
``max_program_cost_units``: that cap is ``sum over ops of live_branches *
4**n_atoms``, a total-work quantity that grows with program DEPTH while the
resident working set does not -- a 4000-op and a 4-op program on the same layout
hold the same matrices. Cost units and bytes are different quantities and neither
bounds the other, so memory gets its own model.

Charging one bare matrix per program, as the first cut of this fix did, is an
accounting estimate rather than an upper bound: a free mid-circuit readout
holds one matrix per outcome state it builds, and the gate and noise kernels
add a scratch on top. The model in ``engine.py`` counts those.

Budget discovery, so the bound tracks the deployment instead of drifting from it:

1. ``QIQCBENCH_QSIM_MEMORY_BUDGET_MB`` -- explicit operator/task declaration.
2. the container's own cgroup limit, when one is applied.
3. :data:`DECLARED_CONTAINER_MEMORY_MB`, the declared qsim container budget.

The smallest applicable bound wins. Note the sizing-versus-enforcement
distinction: Harbor's resource override writes
``services.main`` only, so ``[environment] memory_mb`` in ``task.toml`` sizes the
AGENT container and is not enforced on the qsim sidecar. A qsim ``mem_limit``
makes (2) a real cgroup limit at runtime;
(3) is what bounds a bare ``docker compose`` run or a developer host where no
cgroup limit exists.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

__all__ = [
    "ConcurrentStateBudget",
    "DECLARED_CONTAINER_MEMORY_MB",
    "MEMORY_BUDGET_MB_ENV",
    "RUNTIME_RESERVE_MB",
    "concurrent_state_budget_bytes",
    "container_memory_limit_bytes",
    "density_matrix_bytes",
    "max_concurrent_programs",
]

MEMORY_BUDGET_MB_ENV = "QIQCBENCH_QSIM_MEMORY_BUDGET_MB"

# The memory the shipped neutral-atom bundle sizes its qsim sidecar for.
DECLARED_CONTAINER_MEMORY_MB = 11264

# Everything in the container that is NOT a density matrix: the interpreter and
# numpy import (measured 62 MiB resident before any program runs), the MCP
# transport, and the shot-record buffers, whose size the public budgets already
# cap -- ``max_recorded_bits_per_call`` = 5e7 bits is a 47.7 MiB uint8 array
# before packing, and the sampler holds a few of those alongside their packed
# forms. Also the kernels' fixed strided-iterator buffers (under 1 MiB), the
# state report's 10-atom reduced-state temporaries (up to three x 16 MiB), and
# the terminal pmf rows retained until sampling (``8 * 2**n_atoms`` bytes per
# leaf: 128 KiB at 14 atoms, at most ``max_branches`` of them, 32 MiB). Fixed
# rather than fractional because none of it scales with the matrix, and sized
# at roughly 4x the largest of those record buffers.
#
# Honest limit of this figure: it is an allowance for the allocations named
# above, not a demonstrated bound on resident set size. Peak RSS on the macOS
# development host runs 1.27-1.36x the peak numpy allocation, which is attributed
# to that allocator retaining freed blocks; glibc returns the multi-GiB mmap
# blocks these programs use, so the container is expected to track the traced
# peak, but that was not measured (no Docker in this worktree). At the largest
# admissible program the reserve plus the model's own slack is what stands
# between the two.
RUNTIME_RESERVE_MB = 512

_CGROUP_V2_MAX = Path("/sys/fs/cgroup/memory.max")
_CGROUP_V1_MAX = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")

# Anything at or above this is a "no limit" sentinel rather than a real cap
# (cgroup v1 reports PAGE_COUNTER_MAX, which is petabyte-scale).
_UNLIMITED_FLOOR = 1 << 53


def density_matrix_bytes(n_atoms: int) -> int:
    """Bytes of ONE complex128 density matrix over ``n_atoms`` atoms."""

    return 16 * 4**n_atoms


def container_memory_limit_bytes() -> int | None:
    """The cgroup memory limit of this container, or ``None`` when unlimited.

    Returns ``None`` on every platform that does not expose a cgroup limit
    (macOS development hosts) and whenever the limit reads as unlimited.
    """

    for path in (_CGROUP_V2_MAX, _CGROUP_V1_MAX):
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not raw or raw == "max":
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if value <= 0 or value >= _UNLIMITED_FLOOR:
            continue
        return value
    return None


def _declared_budget_bytes() -> int:
    raw = os.environ.get(MEMORY_BUDGET_MB_ENV)
    if raw is not None:
        try:
            declared = int(raw)
        except ValueError:
            declared = 0
        if declared > 0:
            return declared * 1024 * 1024
    return DECLARED_CONTAINER_MEMORY_MB * 1024 * 1024


def concurrent_state_budget_bytes() -> int:
    """Bytes of density-matrix state the engine may keep alive at once."""

    budget = _declared_budget_bytes()
    limit = container_memory_limit_bytes()
    if limit is not None:
        budget = min(budget, limit)
    return max(1, budget - RUNTIME_RESERVE_MB * 1024 * 1024)


def max_concurrent_programs(peak_bytes: int, budget_bytes: int | None = None) -> int:
    """How many programs of ``peak_bytes`` may run at once (at least one).

    Used by the verifier's replay pool, whose workers are separate PROCESSES and
    so cannot share the in-process gate below; sizing the pool is the only place
    their combined working set can be bounded.
    """

    if budget_bytes is None:
        budget_bytes = concurrent_state_budget_bytes()
    return max(1, budget_bytes // max(1, peak_bytes))


class ConcurrentStateBudget:
    """Admission gate for the density-matrix state alive in this process.

    A program holds its projected peak for as long as it is evolving and
    releases it at the end. A program that does not fit alongside what is
    already running WAITS -- it was already accepted as a job and the agent is
    already polling it, so waiting keeps the async job model intact (invariant
    4) where blocking the submission itself would not. A program that cannot
    fit even on an empty engine is refused at admission instead, by
    :meth:`exceeds_capacity`, so nothing ever waits for capacity that will
    never exist.
    """

    __slots__ = ("_capacity", "_condition", "_reserved")

    def __init__(self, capacity_bytes: int) -> None:
        self._capacity = max(1, int(capacity_bytes))
        self._reserved = 0
        self._condition = threading.Condition()

    @property
    def capacity_bytes(self) -> int:
        return self._capacity

    @property
    def reserved_bytes(self) -> int:
        with self._condition:
            return self._reserved

    def exceeds_capacity(self, nbytes: int) -> bool:
        """Whether ``nbytes`` could never fit, even with nothing else running."""

        return nbytes > self._capacity

    @contextmanager
    def hold(self, nbytes: int) -> Iterator[None]:
        """Block until ``nbytes`` fits alongside the running programs."""

        # Clamping to the capacity is the forward-progress escape: a request
        # that admission should have refused still runs, alone, instead of
        # waiting for a release that can never be enough.
        need = max(0, min(int(nbytes), self._capacity))
        with self._condition:
            while self._reserved + need > self._capacity:
                self._condition.wait()
            self._reserved += need
        try:
            yield
        finally:
            with self._condition:
                self._reserved -= need
                self._condition.notify_all()
