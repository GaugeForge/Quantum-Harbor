"""Run-long stateful engine for the ``transmon_repeater_array`` qtype.

One engine per run (the pair budget accumulates across batch calls). Lock-guarded
budget meter + per-row rejection. Each row runs the agent's designed local-unitary
circuit (gate list per end) for ``depth`` purification rounds via the shared ``physics``
core, samples ``shots`` raw bitstrings of the surviving pair in the requested basis, and
charges the realistic streaming pair cost (shots * 2^depth / success_prob). RNG injected
once by the backend (engine never reseeds). A party's circuit can only touch that
party's two qubits, so nothing acts across the two ends by construction.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import numpy as np

from qiqcbench.qsim.qtypes.transmon_repeater_array import physics
from qiqcbench.qsim.qtypes.transmon_repeater_array.device import (
    HiddenRepeaterConfig,
    PublicRepeaterSpec,
)
from qiqcbench.qsim.qtypes.transmon_repeater_array.wire import (
    JobPurificationData,
    LocalGate,
    PurificationBatchRequest,
    PurificationRow,
    PurificationRowResult,
    _BudgetView,
)


def _to_ops(circuit: list[LocalGate]) -> list[tuple]:
    ops: list[tuple] = []
    for g in circuit:
        if g.gate == "cnot":
            ops.append(("cnot", g.control, g.target))
        elif g.gate in ("rx", "ry", "rz"):
            ops.append((g.gate, g.q, math.radians(g.angle_deg or 0.0)))
        else:
            ops.append((g.gate, g.q, 0.0))
    return ops


@dataclass
class _Evidence:
    pairs_used: int = 0
    rows_used: int = 0


def params_from_hidden(
    hidden: HiddenRepeaterConfig, bias_axis: tuple[float, float, float] | None = None
) -> physics.RepeaterParams:
    """The physics parameters of a hidden device config plus the realized transport-bias
    axis of this attempt (None only when the config has no ``transport_bias``). Shared by
    the engine and the verifier's re-run, so both see the same instance."""
    tb = hidden.transport_bias
    if tb is None:
        p_bias, axis = 0.0, (0.0, 0.0, 1.0)
    else:
        if bias_axis is None:
            raise ValueError("transport_bias is configured but no realized bias axis was given")
        p_bias, axis = tb.p_bias, tuple(float(v) for v in bias_axis)
    return physics.RepeaterParams(
        p_gen=hidden.p_gen,
        n_swap_per_arm=hidden.n_swap_per_arm,
        p_swap=hidden.p_swap,
        t_swap_ns=hidden.t_swap_ns,
        t1_us=hidden.t1_us,
        t2_us=hidden.t2_us,
        p_cnot=hidden.p_cnot,
        p_1q=hidden.p_1q,
        readout_p01=hidden.readout_p01,
        readout_p10=hidden.readout_p10,
        t_mem_per_round_ns=hidden.t_mem_per_round_ns,
        dwell_growth=hidden.dwell_growth,
        p_bias=p_bias,
        bias_axis=axis,
    )


class RepeaterEngine:
    def __init__(
        self,
        hidden: HiddenRepeaterConfig,
        public: PublicRepeaterSpec,
        rng: np.random.Generator,
        bias_axis: tuple[float, float, float] | None = None,
    ) -> None:
        self._rng = rng
        self._lock = threading.Lock()
        self._params = params_from_hidden(hidden, bias_axis)
        b = public.budgets
        self._pairs_cap = b.pairs_cap
        self._rows_cap = b.experiment_rows
        self._max_shots = b.max_shots_per_row
        self._max_depth = b.max_depth
        self._ev = _Evidence()

    def _budget_view(self) -> _BudgetView:
        return _BudgetView(
            pairs_used=self._ev.pairs_used,
            pairs_cap=self._pairs_cap,
            rows_used=self._ev.rows_used,
            rows_cap=self._rows_cap,
        )

    @staticmethod
    def _reject(row: PurificationRow, reason: str) -> PurificationRowResult:
        return PurificationRowResult(
            depth=row.depth,
            measure_basis=row.measure_basis,
            status="rejected",
            reject_reason=reason,
        )

    def _row(self, row: PurificationRow) -> PurificationRowResult:
        if self._ev.rows_used >= self._rows_cap:
            return self._reject(row, "row_cap_exhausted")
        if row.shots > self._max_shots:
            return self._reject(row, "shots_exceed_max")
        if row.depth > self._max_depth:
            return self._reject(row, "depth_exceeds_max")

        alice_ops = _to_ops(row.alice_circuit)
        bob_ops = _to_ops(row.bob_circuit)
        states, keeps = physics.recurrence_states(self._params, row.depth, alice_ops, bob_ops)
        rho = states[row.depth]
        success_prob = 1.0
        for level in range(1, row.depth + 1):
            success_prob *= keeps[level]
        success_prob = max(success_prob, 1e-3)
        pairs_charged = int(math.ceil(row.shots * (2**row.depth) / success_prob))

        if self._ev.pairs_used + pairs_charged > self._pairs_cap:
            return self._reject(row, "pairs_budget_exhausted")

        counts = physics.measure_counts(rho, row.measure_basis, row.shots, self._params, self._rng)
        self._ev.pairs_used += pairs_charged
        self._ev.rows_used += 1
        return PurificationRowResult(
            depth=row.depth,
            measure_basis=row.measure_basis,
            status="accepted",
            counts=counts,
            shots=row.shots,
            success_prob=round(success_prob, 5),
            pairs_charged=pairs_charged,
        )

    def run_purification_batch(self, request: PurificationBatchRequest) -> JobPurificationData:
        with self._lock:
            rows = [self._row(r) for r in request.rows]
            return JobPurificationData(rows=rows, budget=self._budget_view())
