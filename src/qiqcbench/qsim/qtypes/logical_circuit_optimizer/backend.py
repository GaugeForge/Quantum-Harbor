"""Synchronous instance-serving backend for ``logical_circuit_optimizer``.

No engine, no shots, no async job (device-less, like ``ftqc_resource_estimation``).
The backend loads the frozen public materials and serves the opaque input circuit +
objective for ``get_optimization_instance``. It holds **no** answer key — the agent
submits an optimized circuit, and the separate-mode Harbor verifier re-simulates it.
"""

from __future__ import annotations

from typing import Any

from qiqcbench.qsim.qtypes.logical_circuit_optimizer.device import (
    HiddenLogicalOptConfig,
    PublicLogicalOptSpec,
)
from qiqcbench.qsim.qtypes.logical_circuit_optimizer.instance import load_public_instance

__all__ = ["LogicalOptBackend", "build_logical_opt_simulator_backend"]


class LogicalOptBackend:
    """Serves the frozen opaque circuit + objective (synchronous calculator)."""

    def __init__(self, hidden: HiddenLogicalOptConfig, public: PublicLogicalOptSpec) -> None:
        self._public = public
        self._instance = load_public_instance(public.task_id)

    def optimization_instance(self) -> dict[str, Any]:
        p = self._public
        payload = dict(self._instance["spec"])
        payload.update(
            {
                "device_id": p.device_id,
                "task_id": p.task_id,
                "qtype": p.qtype,
                "n_qubits": p.n_qubits,
                "gate_set": list(p.gate_set),
                "circuit_format": p.circuit_format,
                "max_total_qubits": p.max_total_qubits,
                "max_gate_entries": p.max_gate_entries,
                "submission_semantics": p.submission_semantics,
                "admission_rule": p.admission_rule,
                "equivalence_check": p.equivalence_check.model_dump(),
                "objective": p.objective,
                "rating_mode": p.rating_mode,
                "race_metric": p.race_metric,
                "initial_t_count": p.initial_t_count,
                # the full opaque circuit to optimize (also mounted at /task_materials)
                "circuit": self._instance["circuit"],
            }
        )
        return payload


def build_logical_opt_simulator_backend(
    hidden: HiddenLogicalOptConfig, public: PublicLogicalOptSpec
) -> LogicalOptBackend:
    return LogicalOptBackend(hidden, public)
