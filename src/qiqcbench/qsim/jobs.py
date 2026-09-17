"""In-memory job manager.

Tools return a job_id immediately; client polls get_job_result.
For the MVP this runs jobs synchronously in a thread pool — fast enough that
the "queued" state is barely observed, but the interface stays async-shaped
so we can add latency/queueing later without changing the wire protocol.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

from qiqcbench.qsim.core.wire import JobResult

# Admission bound on jobs that have been accepted but whose worker has not yet
# started. Without it a single agent can flood the two-worker pool faster than
# it drains: a stored harper trial admitted 77 sweeps in ~3.5 minutes and 37 of
# them never ran at all. The bound is a backpressure signal, not a
# science limit -- it is sized above every stored trial's realized concurrent
# demand (agents submit serially over MCP, so depth builds only when individual
# jobs are slow) and the rejection is a typed, agent-visible error telling the
# agent to poll and resubmit rather than a silent drop.
DEFAULT_MAX_QUEUED_JOBS = 64

JOB_QUEUE_FULL_ERROR = (
    "qsim job queue is full: {queued} submitted jobs are already waiting for a "
    "worker (limit {limit}). This submission was NOT accepted and has no job_id. "
    "Poll get_job_result on your outstanding jobs and resubmit after some of "
    "them finish; submitting faster than results return only starves the queue."
)


class JobQueueFullError(ValueError):
    """Typed admission rejection: the submit was refused, nothing was enqueued."""


# A qsim worker cannot be preempted, and the pool is only two slots wide, so an
# admitted job that runs far longer than any real experiment holds its slot and
# starves every job queued behind it.  The agent then polls ``running`` forever,
# gives up, and answers without data -- a reward-0 trial that reads as a science
# failure but is an infrastructure failure.  Give such a job a terminal, typed
# disposition instead: once it has held a slot past this deadline the poll
# returns ``failed`` with a diagnosable reason.  The worker thread is abandoned
# rather than killed (Python cannot cancel a running thread), so this bounds
# what the agent waits for, not what the container computes.
#
# Queue wait and worker execution each receive this full window. A job queued
# behind a wedged worker therefore still gets a terminal answer, while a job
# that reaches a worker in time is not charged its queue wait again as execution
# time. This is a last-resort net, not a performance policy.
DEFAULT_JOB_DEADLINE_S = 5400.0

JOB_DEADLINE_ERROR = (
    "qsim abandoned this job after {deadline:.0f}s without a result. It was admitted "
    "but did not finish, so no data will ever be available for this job_id -- stop "
    "polling it. Resubmitting a smaller experiment is more likely to return than "
    "resubmitting the same one."
)
JOB_EXECUTION_ERROR = "qsim internal job execution failed"


def is_internal_job_failure(error: str | None) -> bool:
    """Return whether a failed result represents qsim-owned execution failure."""

    return bool(
        error
        and (error == JOB_EXECUTION_ERROR or error.startswith("qsim abandoned this job after "))
    )


class ModelRequestError(ValueError):
    """A MODEL-owned request fault raised from inside a running job.

    Agent input is normally rejected before a job runs, by the wire schema or by
    the qtype backend's validator, and those rejections are already model-owned:
    they surface as a failed ``JobResult`` with free-form text and no
    ``failure_kind`` stamp. A few checks cannot run there because the offending
    value does not exist yet at admission -- a sweep is validated with
    placeholders in place, so the value each point resolves to is never
    bound-checked. Those checks fire inside the engine, and the exception they
    raise is a model mistake even though it is raised where qsim-owned faults
    are raised too.

    Raising this type marks that difference so the engine can catch exactly the
    model-owned cause and let every other exception propagate to ``JobManager``,
    which types it as ``JOB_EXECUTION_ERROR`` and earns the ``qsim_internal``
    stamp. Deleting the engine's blanket catch without this distinction
    would hand a model mistake a free infrastructure rerun; keeping the blanket
    catch destroys the marker for genuine faults. It subclasses ``ValueError``
    so existing callers that catch ``ValueError`` are unaffected.
    """


def new_job_id() -> str:
    # 48 random bits. A within-trial collision would let the later job's polls
    # return the earlier job's cached/persisted delivery;
    # IDs are qsim-generated, never caller-supplied, so at ~2e-7 for even a
    # 10k-job trial this is accepted risk rather than an enforced-uniqueness
    # seam.
    return f"job_{uuid.uuid4().hex[:12]}"


class JobManager:
    def __init__(
        self,
        max_workers: int = 2,
        deadline_s: float = DEFAULT_JOB_DEADLINE_S,
        max_queued_jobs: int = DEFAULT_MAX_QUEUED_JOBS,
    ):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="qsim-job")
        self._results: dict[str, JobResult] = {}
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._deadline_s = deadline_s
        self._max_queued_jobs = max_queued_jobs
        # job_id -> monotonic start of the current queued/running phase, for
        # jobs whose runner has not returned. The worker resets this timestamp
        # when it changes a queued job to running.
        self._unreturned: dict[str, float] = {}
        self._abandoned: set[str] = set()
        # Jobs admitted but whose worker has not started yet.
        self._queued: set[str] = set()
        # Jobs whose raw result data was evicted after the delivery was
        # persisted and cached (see actions/common.py); polls must rebuild
        # from the persisted artifact, never from the slimmed stored result.
        self._data_evicted: set[str] = set()
        # Invoked from the worker thread with a completed result BEFORE it is
        # stored, so expensive delivery materialization never runs on the
        # transport event loop and a poll that sees "complete" finds the warm
        # cache. Returns True when the delivery was cached (raw data may then
        # be evicted). Failures surface via the poll path, which rebuilds
        # inline.
        self.on_terminal_result: Callable[[str, JobResult], bool] | None = None

    def reserve_admission(self, job_id: str) -> str:
        """Atomically claim one queue slot for ``job_id`` before any evidence.

        The reservation IS the queued entry: it counts toward the admission
        cap and polls as a truthful ``queued`` placeholder, so a concurrent
        submit can never fill the queue between an action's admission check
        and its ``submit`` call -- ``submit`` consumes the reservation without
        re-checking capacity. If a later pre-submit step (cumulative budget
        reservation, poll-budget registration, submission-event publication)
        refuses, call :meth:`release_admission` so the slot and the budgets
        succeed or fail together.

        Raises :class:`JobQueueFullError` with no side effects when the queue
        is full."""
        with self._lock:
            if job_id in self._results:
                raise ValueError(f"job admission already exists for {job_id!r}")
            self._check_capacity_locked()
            self._results[job_id] = JobResult(job_id=job_id, device_id="", status="queued")
            self._unreturned[job_id] = time.monotonic()
            self._queued.add(job_id)
        return job_id

    def release_admission(self, job_id: str) -> None:
        """Roll back an unconsumed admission reservation.

        A no-op once a runner was attached (the worker owns the entry then) or
        when nothing is reserved, so failure paths may call it unconditionally.
        """
        with self._lock:
            if job_id in self._futures:
                return
            result = self._results.get(job_id)
            if result is None or result.status != "queued":
                return
            self._queued.discard(job_id)
            self._unreturned.pop(job_id, None)
            self._results.pop(job_id, None)

    def _check_capacity_locked(self) -> None:
        if len(self._queued) < self._max_queued_jobs:
            return
        # Before refusing, sweep queued entries past the deadline whose worker
        # never started: a bare reservation whose owner crashed between
        # reserve and submit is never polled, so without this sweep it would
        # consume a queue slot forever. Each swept entry gets
        # the same terminal abandoned disposition a deadline poll delivers.
        self._abandon_expired_queued_locked()
        if len(self._queued) >= self._max_queued_jobs:
            raise JobQueueFullError(
                JOB_QUEUE_FULL_ERROR.format(queued=len(self._queued), limit=self._max_queued_jobs)
            )

    def _abandon_expired_queued_locked(self) -> None:
        now = time.monotonic()
        for job_id in list(self._queued):
            submitted = self._unreturned.get(job_id)
            if submitted is None or (now - submitted) <= self._deadline_s:
                continue
            future = self._futures.get(job_id)
            if future is not None and not future.cancel():
                # The worker actually started; it owns the entry now.
                continue
            self._queued.discard(job_id)
            self._unreturned.pop(job_id, None)
            self._abandoned.add(job_id)
            previous = self._results.get(job_id)
            self._results[job_id] = JobResult(
                job_id=job_id,
                device_id=previous.device_id if previous is not None else "",
                status="failed",
                shots=previous.shots if previous is not None else 0,
                error=JOB_DEADLINE_ERROR.format(deadline=self._deadline_s),
            )

    def submit(self, runner: Callable[[str], JobResult], job_id: str | None = None) -> str:
        """Enqueue ``runner``. Callers may pre-allocate ``job_id`` (via
        ``new_job_id``) so a submission event can be logged BEFORE the worker
        starts — otherwise a fast worker's result event can precede the
        submission event in the evidence log.

        Raises :class:`JobQueueFullError` without side effects when too many
        admitted jobs are still waiting for a worker."""
        if job_id is None:
            job_id = new_job_id()
        # Pre-record a "queued" result so polls before the worker starts see the
        # truthful documented status; the worker flips it to "running". A prior
        # reserve_admission for this job_id is consumed as-is (no re-check), so
        # a concurrent submit cannot refuse an already-granted slot.
        with self._lock:
            existing = self._results.get(job_id)
            reserved = (
                job_id in self._queued
                and job_id not in self._futures
                and existing is not None
                and existing.status == "queued"
            )
            if not reserved:
                self._check_capacity_locked()
                self._results[job_id] = JobResult(job_id=job_id, device_id="", status="queued")
                self._unreturned[job_id] = time.monotonic()
                self._queued.add(job_id)
        try:
            future = self._executor.submit(self._run_and_store, job_id, runner)
        except BaseException:
            # Executor refusal must not strand an orphan queued entry that no
            # worker will ever run: roll the whole admission back.
            self.release_admission(job_id)
            raise
        with self._lock:
            self._futures[job_id] = future
        return job_id

    def _run_and_store(self, job_id: str, runner: Callable[[str], JobResult]) -> None:
        with self._lock:
            current = self._results.get(job_id)
            started = time.monotonic()
            queued_since = self._unreturned.get(job_id)
            if (
                job_id not in self._abandoned
                and current is not None
                and current.status == "queued"
                and queued_since is not None
                and (started - queued_since) > self._deadline_s
            ):
                # No poll or later admission swept this entry before a worker
                # became available. Enforce the queue liveness window here so
                # an already-expired job never consumes compute.
                self._queued.discard(job_id)
                self._unreturned.pop(job_id, None)
                self._abandoned.add(job_id)
                self._results[job_id] = JobResult(
                    job_id=job_id,
                    device_id=current.device_id,
                    status="failed",
                    shots=current.shots,
                    error=JOB_DEADLINE_ERROR.format(deadline=self._deadline_s),
                )
                return
            self._queued.discard(job_id)
            if job_id not in self._abandoned and current is not None and current.status == "queued":
                self._unreturned[job_id] = started
                self._results[job_id] = JobResult(job_id=job_id, device_id="", status="running")
        try:
            result = runner(job_id)
        except Exception:  # pragma: no cover
            # Runner exceptions are qsim-owned failures, not agent protocol
            # rejections. Keep the public result typed and do not expose private
            # exception strings or paths through the agent-facing JobResult.
            result = JobResult(
                job_id=job_id,
                device_id="",
                status="failed",
                error=JOB_EXECUTION_ERROR,
            )
        # Materialize the delivery BEFORE publishing the terminal status: a poll
        # that observes "complete" is then guaranteed a warm cache, so it never
        # duplicates the serialization inline on the transport event loop while
        # this thread is still building (measured: the race roughly doubled the
        # loop's tail latency). Until the delivery is ready the job truthfully
        # polls "running" -- preparing the result is part of the job.
        precomputed = False
        hook = self.on_terminal_result
        if hook is not None and result.status == "complete":
            try:
                precomputed = bool(hook(job_id, result))
            except Exception:
                # Delivery precompute is an optimization: the poll path rebuilds
                # inline and surfaces persistence failures with its existing
                # typed qsim-internal disposition.
                precomputed = False
        stored = self.set_result(job_id, result)
        if stored and precomputed:
            self.evict_result_data(job_id)

    def set_result(self, job_id: str, result: JobResult) -> bool:
        with self._lock:
            self._unreturned.pop(job_id, None)
            self._queued.discard(job_id)
            if job_id in self._abandoned:
                # A terminal disposition was already delivered for this job.
                # A late result must not resurrect it.
                return False
            self._results[job_id] = result
            self._data_evicted.discard(job_id)
            return True

    def evict_result_data(self, job_id: str) -> None:
        """Drop a completed result's raw ``data`` from the retained store.

        Call only after the full delivery artifact has been durably persisted
        (delivery spool or published public file) and cached; later polls for
        an evicted job are answered from the cache or rebuilt from that
        artifact. Keeping every completed JobResult forever held multi-GB of
        dead heap in one container."""
        with self._lock:
            result = self._results.get(job_id)
            if (
                result is None
                or job_id in self._abandoned
                or result.status != "complete"
                or result.data is None
            ):
                return
            self._results[job_id] = result.model_copy(update={"data": None})
            self._data_evicted.add(job_id)

    def result_data_evicted(self, job_id: str) -> bool:
        with self._lock:
            return job_id in self._data_evicted

    def peek(self, job_id: str) -> JobResult | None:
        """Return the current record without advancing deadline state."""

        with self._lock:
            return self._results.get(job_id)

    def get(self, job_id: str) -> JobResult | None:
        with self._lock:
            result = self._results.get(job_id)
            if result is None or job_id in self._abandoned:
                return result
            phase_started = self._unreturned.get(job_id)
            # A runner that has already returned is out of the pool -- e.g. a
            # live-provider submission whose result is completed by the backend's
            # own poll -- and is not subject to this deadline.
            if phase_started is None or (time.monotonic() - phase_started) <= self._deadline_s:
                return result
            self._abandoned.add(job_id)
            # A job whose worker never started can be cancelled outright:
            # letting it start later would burn a worker slot computing a
            # result nobody can receive. An already-started future ignores
            # this ``cancel`` (Python cannot preempt a running thread). A
            # missing future is a bare admission reservation that was never
            # submitted (its owner crashed between reserve and submit); its
            # slot must be freed here or it consumes queue capacity forever
            # unless explicitly released.
            future = self._futures.get(job_id)
            if future is None or future.cancel():
                self._queued.discard(job_id)
                self._unreturned.pop(job_id, None)
            abandoned = JobResult(
                job_id=job_id,
                device_id=result.device_id,
                status="failed",
                shots=result.shots,
                error=JOB_DEADLINE_ERROR.format(deadline=self._deadline_s),
            )
            self._results[job_id] = abandoned
            return abandoned
