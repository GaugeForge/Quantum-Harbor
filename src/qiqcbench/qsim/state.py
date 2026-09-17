"""Runtime state shared by qsim MCP and local-code sidecar surfaces."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from ruamel.yaml import YAML

from qiqcbench.qsim.backends import CircuitBackend, PulseBackend
from qiqcbench.qsim.core.device import HiddenDeviceConfig, PublicDeviceSpec
from qiqcbench.qsim.devices import AnyLabNotebook
from qiqcbench.qsim.execution_context import (
    EXECUTION_CONTEXT_SCHEMA_VERSION,
    validate_execution_context_id,
    validate_hidden_commitment_secret,
)
from qiqcbench.qsim.hidden_commitment import (
    hidden_device_model_commitment,
    public_device_model_commitment,
)
from qiqcbench.qsim.jobs import JobManager

_yaml = YAML()
_yaml.default_flow_style = False


class QsimState:
    """Container for hidden truth, runtime metadata, logging, and jobs."""

    def __init__(
        self,
        hidden: HiddenDeviceConfig,
        public: PublicDeviceSpec,
        notebook: AnyLabNotebook,
        log_dir: Path | None,
        backend: PulseBackend | CircuitBackend,
        backend_mode: str = "simulator",
        task_id: str | None = None,
        snapshot_hidden_truth: bool = True,
        active_surfaces: tuple[str, ...] = ("mcp",),
        active_capabilities: tuple[str, ...] = ("basic_measurement",),
        surface_manifest: dict[str, object] | None = None,
        execution_context_id: str | None = None,
        hidden_commitment_secret: str | None = None,
    ):
        self.hidden = hidden
        self.public = public
        self.notebook = notebook
        self.backend = backend
        self.backend_mode = backend_mode
        self.task_id = task_id
        self.snapshot_hidden_truth = snapshot_hidden_truth
        self.active_surfaces = tuple(active_surfaces)
        self.active_capabilities = tuple(active_capabilities)
        self.surface_manifest: dict[str, object] = (
            dict(surface_manifest) if surface_manifest is not None else {}
        )
        self.jobs = JobManager(max_workers=2)
        # Materialize + persist + cache each completed delivery in the worker
        # thread that produced it, before the terminal status is published, so
        # the transport event loop never serializes a raw result payload and a
        # poll that sees "complete" always finds the warm cache.
        self.jobs.on_terminal_result = self._precompute_job_result_delivery
        self._execution_context_id = (
            None
            if execution_context_id is None
            else validate_execution_context_id(execution_context_id)
        )
        self._hidden_commitment_secret = hidden_commitment_secret
        if self._execution_context_id is not None and self._hidden_commitment_secret is None:
            raise ValueError("an execution context requires a hidden commitment secret")
        if self._execution_context_id is None and self._hidden_commitment_secret is not None:
            raise ValueError("a hidden commitment secret requires an execution context")
        if self._execution_context_id is not None and not self.task_id:
            raise ValueError("an execution context requires an active task ID")
        if self._hidden_commitment_secret is not None:
            self._hidden_commitment_secret = validate_hidden_commitment_secret(
                self._hidden_commitment_secret
            )
        self._salt = 0
        self._salt_lock = threading.Lock()
        self._action_budget_lock = threading.Lock()
        self._action_budget_usage: dict[str, tuple[int, int]] = {}
        self._job_result_poll_budget_lock = threading.Lock()
        self._job_result_poll_budgets: dict[str, tuple[str, int]] = {}
        self._job_result_delivery_lock = threading.Lock()
        # job_id -> canonical JSON string of the terminal delivery
        # (response_payload, offload pointer, data summary). Immutable entries;
        # byte-budget LRU bounded by actions.common.JOB_RESULT_DELIVERY_CACHE_MAX_BYTES.
        self._job_result_delivery_cache: dict[str, str] = {}
        self._log_lock = threading.Lock()
        self._final_answer_lock = threading.Lock()
        self.log_dir = log_dir
        if self.execution_context_id is not None and log_dir is None:
            raise ValueError("an execution context requires log_dir")
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = log_dir / "experiment_log.jsonl"
            if self.execution_context_id is not None and (
                self._log_path.is_symlink()
                or (
                    self._log_path.exists()
                    and (not self._log_path.is_file() or self._log_path.stat().st_size != 0)
                )
            ):
                raise ValueError("an execution context requires an empty experiment log path")
            public_job_results = log_dir / "public_job_results"
            if public_job_results.is_symlink() or (
                public_job_results.exists() and not public_job_results.is_dir()
            ):
                raise ValueError("public_job_results must be a real directory")
            public_job_results.mkdir(mode=0o755, exist_ok=True)
            public_job_results.chmod(0o755)
            if snapshot_hidden_truth:
                # Snapshot hidden truth ONCE per run.
                snap = log_dir / "hidden_truth_snapshot.yaml"
                if not snap.exists():
                    with snap.open("w") as f:
                        _yaml.dump(hidden.model_dump(), f)
        else:
            self._log_path = None
        # THE INVARIANT: experiment_log.jsonl exists if and only if this
        # constructor completed. Verifiers treat a missing evidence log as
        # infrastructure failure (reserved exit, no score) because qsim owns the
        # file and the agent's mount is read-only, while an empty one means the
        # agent drove nothing and scores 0. A half-initialized qsim that leaves an
        # empty log behind therefore reads as "the agent drove nothing" instead of
        # "qsim failed", and an agent that never calls a tool must not be able to
        # reach the infrastructure disposition by doing nothing.
        #
        # Exactly two statements below can create the file, and BOTH are covered
        # here. Ordering alone is not sufficient and was not: QsimState.log opens
        # the path in append mode, which creates it *before* the write is
        # attempted, so a failure during the bootstrap write strands an empty log
        # without ever reaching the touch. Hence the cleanup, not just the order.
        # Only a file this constructor created is removed -- a pre-existing
        # development log is never touched.
        log_preexisted = self._log_path is not None and self._log_path.exists()
        try:
            if self.execution_context_id is not None:
                self.log(
                    {
                        "action": "qsim_bootstrap_context",
                        "context_schema_version": EXECUTION_CONTEXT_SCHEMA_VERSION,
                        "public_device_model_commitment": public_device_model_commitment(
                            self.public,
                            task_id=self.task_id,
                        ),
                        "hidden_device_model_commitment": hidden_device_model_commitment(
                            self.hidden,
                            task_id=self.task_id,
                            commitment_secret=self.hidden_commitment_secret,
                        ),
                    }
                )
            if self._log_path is not None:
                # touch() creates without truncating, so an execution context whose
                # bootstrap event was just written keeps it.
                self._log_path.touch()
        except BaseException:
            if self._log_path is not None and not log_preexisted:
                self._log_path.unlink(missing_ok=True)
            raise

    def _precompute_job_result_delivery(self, job_id: str, result) -> bool:
        """Worker-thread hook: build and cache the delivery before publication."""

        # Deferred so this module stays importable independent of the actions
        # package (mirrors the deferred-import idiom in actions.common).
        from qiqcbench.qsim.actions.common import precompute_job_result_delivery

        return precompute_job_result_delivery(self, job_id=job_id, result=result)

    @property
    def execution_context_id(self) -> str | None:
        """Opaque immutable execution binding, when a run is execution-bound."""

        return self._execution_context_id

    @property
    def hidden_commitment_secret(self) -> str | None:
        """Private immutable HMAC key; never serialized into qsim evidence."""

        return self._hidden_commitment_secret

    def next_salt(self) -> int:
        with self._salt_lock:
            self._salt += 1
            return self._salt

    def reserve_action_budgets(
        self,
        *,
        reservations: dict[str, tuple[int, int]],
    ) -> dict[str, tuple[int, int]]:
        """Atomically reserve run-long action budgets before job allocation.

        Each value is ``(amount, limit)``.  The returned values are
        ``(used_before, used_after)``.  All reservations are validated before
        any is committed, so a rejected multi-resource request consumes none.
        A limit change for an existing key indicates public-runtime drift and
        fails closed instead of silently resetting the meter.
        """

        with self._action_budget_lock:
            updates: dict[str, tuple[int, int]] = {}
            evidence: dict[str, tuple[int, int]] = {}
            for budget_key, (amount, limit) in reservations.items():
                if (
                    not isinstance(budget_key, str)
                    or not budget_key
                    or isinstance(amount, bool)
                    or not isinstance(amount, int)
                    or amount < 0
                    or isinstance(limit, bool)
                    or not isinstance(limit, int)
                    or limit < 0
                ):
                    raise ValueError(
                        "action budget keys must be non-empty strings and amounts/limits "
                        "must be non-negative integers"
                    )
                used_before, bound_limit = self._action_budget_usage.get(budget_key, (0, limit))
                if bound_limit != limit:
                    raise RuntimeError(
                        f"action budget {budget_key!r} changed during the run: "
                        f"{bound_limit} -> {limit}"
                    )
                used_after = used_before + amount
                if used_after > limit:
                    raise ValueError(
                        f"action budget {budget_key!r} exhausted: {used_before}+{amount} > {limit}"
                    )
                updates[budget_key] = (used_after, limit)
                evidence[budget_key] = (used_before, used_after)
            self._action_budget_usage.update(updates)
            return evidence

    def release_action_budgets(self, *, reservations: dict[str, int]) -> None:
        """Roll back exact previously reserved amounts after an admission refusal.

        Only for the window before any logged evidence references the
        reservation: a queue-slot refusal or an executor-submit failure that
        follows a successful ``reserve_action_budgets``, so slot and budgets
        succeed or fail together. Usage clamps at zero and the
        bound limit is untouched; releasing an unknown key is a no-op.
        """

        with self._action_budget_lock:
            for budget_key, amount in reservations.items():
                if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
                    raise ValueError("action budget release amounts must be non-negative integers")
                usage = self._action_budget_usage.get(budget_key)
                if usage is None:
                    continue
                used, bound_limit = usage
                self._action_budget_usage[budget_key] = (max(0, used - amount), bound_limit)

    def register_job_result_poll_budget(self, *, job_id: str, budget_key: str, limit: int) -> None:
        """Bind one accepted job to a cumulative task-local poll meter."""

        if (
            not isinstance(job_id, str)
            or not job_id
            or not isinstance(budget_key, str)
            or not budget_key
            or isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 0
        ):
            raise ValueError("job poll budget requires non-empty IDs and a non-negative limit")
        with self._job_result_poll_budget_lock:
            previous = self._job_result_poll_budgets.get(job_id)
            binding = (budget_key, limit)
            if previous is not None and previous != binding:
                raise RuntimeError(f"job {job_id!r} poll budget changed during the run")
            self._job_result_poll_budgets[job_id] = binding

    def unregister_job_result_poll_budget(self, *, job_id: str) -> None:
        """Drop a poll-meter binding when submission admission is rolled back."""

        with self._job_result_poll_budget_lock:
            self._job_result_poll_budgets.pop(job_id, None)

    def reserve_job_result_poll(self, job_id: str) -> tuple[int, int, int] | None:
        """Reserve one poll when ``job_id`` belongs to a capped action family."""

        with self._job_result_poll_budget_lock:
            binding = self._job_result_poll_budgets.get(job_id)
        if binding is None:
            return None
        budget_key, limit = binding
        used_before, used_after = self.reserve_action_budgets(
            reservations={budget_key: (1, limit)}
        )[budget_key]
        return used_before, used_after, limit

    def log(self, event: dict) -> None:
        if self._log_path is None:
            return
        stamped_event = {"ts": datetime.now(UTC).isoformat(), **event}
        if self.execution_context_id is not None:
            # The state-owned value is applied last so action callers cannot
            # replace the execution binding, even accidentally.
            stamped_event["execution_context_id"] = self.execution_context_id
        # Some run_* jobs log from JobManager worker threads; serialize appends.
        with self._log_lock, self._log_path.open("a") as f:
            f.write(json.dumps(stamped_event) + "\n")
