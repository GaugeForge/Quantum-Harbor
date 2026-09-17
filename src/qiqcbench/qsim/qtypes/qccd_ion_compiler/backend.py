"""Synchronous calculator backend for the ``qccd_ion_compiler`` qtype.

No engine, no shots, no async job. Holds the per-run QAOA instance (a
deterministic function of ``QIQCBENCH_INSTANCE_SEED``) and serves two synchronous
calculators:

* :meth:`compilation_instance` — the target QAOA circuit + a machine-readable copy
  of the public cost-model coefficients.
* :meth:`evaluate` — the deterministic cost oracle (legality + correctness +, only
  when both pass, the total infidelity), call-capped per run.

The instance generator and the cost model are resolved by ``public.task_id`` (the
task owns its hidden_dynamics package), so the qtype stays task-agnostic.
"""

from __future__ import annotations

import importlib
import os
import threading
from typing import Any

from qiqcbench.qsim.qtypes.qccd_ion_compiler.device import HiddenQccdConfig, PublicQccdSpec

# Public cost-model coefficient groups echoed into the compilation-instance payload.
_COST_MODEL_FIELDS = {
    "topology",
    "modes",
    "ms_error_model",
    "transport",
    "anomalous_heating",
    "recool",
    "idle_dephasing",
    "single_qubit",
    "readout",
}


class QccdEvaluatorBackend:
    """Per-run state for the ``qccd_ion_compiler`` synchronous tools."""

    def __init__(self, hidden: HiddenQccdConfig, public: PublicQccdSpec) -> None:
        self._public = public
        seed_env = os.environ.get("QIQCBENCH_INSTANCE_SEED")
        self._seed = int(seed_env) if seed_env not in (None, "") else hidden.seed
        base = f"qiqcbench.qsim.hidden_dynamics.{public.task_id}"
        self._instance_mod = importlib.import_module(f"{base}.instance")
        self._cost_mod = importlib.import_module(f"{base}.cost_model")
        self._instance = self._instance_mod.build_instance(self._seed)
        self._call_cap = public.evaluator_call_cap
        self._calls = 0
        self._lock = threading.Lock()

    # ---- get_compilation_instance ----

    def compilation_instance(self) -> dict[str, Any]:
        payload = self._instance.to_public_dict()
        payload["device_id"] = self._public.device_id
        payload["task_id"] = self._public.task_id
        payload["evaluator_call_cap"] = self._call_cap
        payload["cost_model"] = self._public.model_dump(include=_COST_MODEL_FIELDS)
        return payload

    # ---- evaluate_schedule ----

    def evaluate(self, schedule: Any) -> dict[str, Any]:
        with self._lock:
            if self._calls >= self._call_cap:
                return {
                    "accepted": False,
                    "reason": "evaluator_call_cap_exhausted",
                    "evaluator_calls_used": self._calls,
                    "evaluator_call_cap": self._call_cap,
                }
            self._calls += 1
            used = self._calls

        if not isinstance(schedule, dict):
            return {
                "accepted": True,
                "legal": False,
                "legality_errors": [
                    "schedule must be an object with 'ion_placement' + 'op_schedule'"
                ],
                "circuit_correct": False,
                "correctness_errors": [],
                "total_infidelity": None,
                "evaluator_calls_used": used,
                "evaluator_calls_remaining": self._call_cap - used,
            }

        result = self._cost_mod.evaluate(schedule, self._instance, self._public)
        out = result.oracle_dict()
        out["accepted"] = True
        out["evaluator_calls_used"] = used
        out["evaluator_calls_remaining"] = self._call_cap - used
        return out


def build_qccd_simulator_backend(
    hidden: HiddenQccdConfig, public: PublicQccdSpec
) -> QccdEvaluatorBackend:
    return QccdEvaluatorBackend(hidden, public)
