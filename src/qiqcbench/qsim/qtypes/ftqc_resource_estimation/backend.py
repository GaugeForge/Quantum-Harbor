"""Synchronous evaluator backend for the ``ftqc_resource_estimation`` qtype.

There is no engine, shots, or async job. Each task owns its static instance and
pure-Python evaluator under ``hidden_dynamics.<task_id>.instance``; this shared
backend adds only dispatch and a lock-guarded per-run call cap.
"""

from __future__ import annotations

import importlib
import threading
from typing import Any

from qiqcbench.qsim.qtypes.ftqc_resource_estimation.device import (
    HiddenFtqcConfig,
    PublicFtqcSpec,
)

__all__ = [
    "GenericFtqcEvaluatorBackend",
    "build_ftqc_simulator_backend",
]


class GenericFtqcEvaluatorBackend:
    """Task-agnostic evaluator: resolves the task's ``hidden_dynamics`` evaluator.

    The task package (``hidden_dynamics.<task_id>.instance``) owns the held-out
    instance + the pure-Python cost model and exposes ``build_evaluator(hidden,
    public)`` returning an object with ``public_instance() -> dict`` and
    ``evaluate(**design) -> dict`` (minimal output). This backend adds only the
    shared lock-guarded call cap, mirroring ``qccd_ion_compiler``.
    """

    def __init__(self, hidden: HiddenFtqcConfig, public: PublicFtqcSpec) -> None:
        self._public = public
        mod = importlib.import_module(f"qiqcbench.qsim.hidden_dynamics.{public.task_id}.instance")
        self._evaluator = mod.build_evaluator(hidden, public)
        self._call_cap = public.evaluator_call_cap
        self._calls = 0
        self._lock = threading.Lock()

    # --- get_algorithm_instance ---
    def algorithm_instance(self) -> dict[str, Any]:
        payload = self._evaluator.public_instance()
        payload["device_id"] = self._public.device_id
        payload["task_id"] = self._public.task_id
        payload["evaluator_call_cap"] = self._call_cap
        return payload

    def mps_qpe_instance(self) -> dict[str, Any]:
        """Return public MPS-QPE case handles from a task-owned evaluator."""
        payload = self._evaluator.public_instance()
        payload["device_id"] = self._public.device_id
        payload["task_id"] = self._public.task_id
        payload["evaluator_call_cap"] = self._call_cap
        return payload

    # --- evaluate_factory_design ---
    def evaluate_design(
        self,
        *,
        distillation_l1_d: int,
        distillation_l2_d: int,
        n_factories: int,
        data_block_d: int,
    ) -> dict[str, Any]:
        with self._lock:
            if self._calls >= self._call_cap:
                return {
                    "accepted": False,
                    "reason": "evaluator_call_cap_exhausted",
                    "evaluator_calls_used": self._calls,
                    "evaluator_call_cap": self._call_cap,
                }
            self._calls += 1
            calls_used = self._calls

        result = self._evaluator.evaluate(
            distillation_l1_d=int(distillation_l1_d),
            distillation_l2_d=int(distillation_l2_d),
            n_factories=int(n_factories),
            data_block_d=int(data_block_d),
        )
        result["evaluator_calls_used"] = calls_used
        result["evaluator_calls_remaining"] = self._call_cap - calls_used
        return result

    def evaluate_mps_qpe_plan(
        self,
        *,
        selected_block_encoding_id: str,
        candidate_methods: list[str],
        mps_descriptor_ids: list[str],
    ) -> dict[str, Any]:
        """Evaluate one compact MPS-QPE method/descriptor selection."""
        with self._lock:
            if self._calls >= self._call_cap:
                return {
                    "accepted": False,
                    "reason": "evaluator_call_cap_exhausted",
                    "evaluator_calls_used": self._calls,
                    "evaluator_call_cap": self._call_cap,
                }
            self._calls += 1
            calls_used = self._calls
        result = self._evaluator.evaluate(
            selected_block_encoding_id=selected_block_encoding_id,
            candidate_methods=candidate_methods,
            mps_descriptor_ids=mps_descriptor_ids,
        )
        result["evaluator_calls_used"] = calls_used
        result["evaluator_calls_remaining"] = self._call_cap - calls_used
        return result

    @property
    def calls_used(self) -> int:
        return self._calls


def build_ftqc_simulator_backend(
    hidden: HiddenFtqcConfig, public: PublicFtqcSpec
) -> GenericFtqcEvaluatorBackend:
    """Build the task-owned static evaluator behind the shared call-cap seam."""
    return GenericFtqcEvaluatorBackend(hidden, public)
