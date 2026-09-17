"""Feasibility-gate adapter over the production analog engine.

The black-box probe physics, validation, sampling, and budget accounting live
once in the reusable qtype engine
(:mod:`qiqcbench.qsim.qtypes.blackbox_analog_dynamics.engine`). This module is a
thin, **test/gate-facing** adapter: :class:`Oracle` wraps an
:class:`AnalogEngine` built from the fixed hidden instance, so the §9h
feasibility gate and the real agent device path execute the *same* engine — no
forked second copy of the dynamics to drift.

It keeps the gate's stable :class:`ProbeRow` / :class:`ProbeResult` shapes and
re-exports :class:`ProbeEvidence` (the scorer's evidence type) so existing
callers (reference solver, ``_hamlearn_support``, scorer) are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qiqcbench.qsim.core.wire import ProbeRowOp
from qiqcbench.qsim.hidden_dynamics.time_budgeted_hamlearn_10q.construction import (
    MAX_ROWS_PER_BATCH,
    HamLearnInstance,
)
from qiqcbench.qsim.hidden_dynamics.time_budgeted_hamlearn_10q.device_configs import (
    analog_configs_from_instance,
)
from qiqcbench.qsim.qtypes.blackbox_analog_dynamics.engine import (
    AnalogEngine,
    ProbeEvidence,
)

__all__ = ["Oracle", "ProbeEvidence", "ProbeResult", "ProbeRow"]


@dataclass(frozen=True)
class ProbeRow:
    """One requested probe."""

    initial_state: list[str]
    evolve_time_us: float
    observable_pauli: str


@dataclass(frozen=True)
class ProbeResult:
    """One probe outcome row returned to the agent."""

    observable_pauli: str
    status: str  # "accepted" | "rejected"
    accepted_evolve_time_us: float | None = None
    raw_pauli_outcomes: list[int] | None = None
    num_internal_repetitions: int | None = None
    reject_reason: str | None = None


class Oracle:
    """Hidden-dynamics engine for the gate. RNG is injected (engine contract)."""

    def __init__(self, instance: HamLearnInstance, rng: np.random.Generator):
        public, hidden = analog_configs_from_instance(instance)
        self._engine = AnalogEngine(hidden, public, rng)

    def evidence(self) -> ProbeEvidence:
        return self._engine.evidence

    def _exact_expectation(self, labels: list[str], obs: str, t: float) -> float:
        return self._engine._exact_expectation(labels, obs, t)

    def run_probe_batch(self, rows: list[ProbeRow]) -> list[ProbeResult]:
        if len(rows) > MAX_ROWS_PER_BATCH:
            raise ValueError(f"probe batch has {len(rows)} rows; max is {MAX_ROWS_PER_BATCH}")
        op_rows = [
            ProbeRowOp(
                initial_state=row.initial_state,
                evolve_time_us=row.evolve_time_us,
                observable_pauli=row.observable_pauli,
            )
            for row in rows
        ]
        data = self._engine.run_probe_batch(op_rows)
        return [
            ProbeResult(
                observable_pauli=out.observable_pauli,
                status=out.status,
                accepted_evolve_time_us=out.accepted_evolve_time_us,
                raw_pauli_outcomes=out.raw_pauli_outcomes,
                num_internal_repetitions=out.num_internal_repetitions,
                reject_reason=out.reject_reason,
            )
            for out in data.rows
        ]
