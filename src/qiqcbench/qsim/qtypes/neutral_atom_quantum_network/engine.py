"""Run-long stateful engine for the ``neutral_atom_quantum_network`` qtype.

Holds the hidden instance (true association, noise floor) for the whole run and accumulates a
single **communication-budget** meter across every estimation call. Each
``run_distributed_estimation`` charges the deterministic cost of the chosen primitive (failing
closed if the budget would be exceeded) and returns the noisy estimate sampled from the
primitive's faithful statistical model under the hidden floor. The engine never returns the
true association, the floor, or the communication budget gate.
"""

from __future__ import annotations

import threading

import numpy as np

from qiqcbench.qsim.qtypes.neutral_atom_quantum_network import physics as P
from qiqcbench.qsim.qtypes.neutral_atom_quantum_network.device import (
    HiddenNetworkConfig,
    PublicNetworkSpec,
)
from qiqcbench.qsim.qtypes.neutral_atom_quantum_network.wire import (
    DistributedEstimationRequest,
    JobEstimationData,
    _CommBudgetView,
)


class NetworkEngine:
    def __init__(
        self,
        hidden: HiddenNetworkConfig,
        public: PublicNetworkSpec,
        rng: np.random.Generator,
    ):
        self._rng = rng
        self._lock = threading.Lock()
        # The production association is returned only at the full index register; a reduced
        # register returns the SEPARATE calibration sub-instance (different value, same
        # floor/scaling), so cheap small-register probes cannot steal the production answer.
        self._c = float(hidden.instance.correlation)
        self._c_calib = float(hidden.instance.calibration_correlation)
        self._full_index_bits = hidden.n_index_bits
        self._floor = float(hidden.instance.noise_floor_true)
        self._comm_budget = hidden.comm_budget
        self._max_index_bits = public.budgets.max_index_bits
        self._comm_used = 0

    def _budget_view(self) -> _CommBudgetView:
        return _CommBudgetView(comm_used=self._comm_used, comm_budget=self._comm_budget)

    def run_distributed_estimation(
        self, request: DistributedEstimationRequest
    ) -> JobEstimationData | str:
        """Returns JobEstimationData, or an error string (budget exhausted / bad request)."""
        with self._lock:
            try:
                precision = P.validate_precision(request.estimator, request.precision)
            except P.EstimatorError as exc:
                return str(exc)
            if request.n_index_bits > self._max_index_bits:
                return f"n_index_bits {request.n_index_bits} exceeds max {self._max_index_bits}"
            charged = P.cost(request.estimator, precision, request.n_index_bits)
            if self._comm_used + charged > self._comm_budget:
                return (
                    f"communication budget exhausted: {self._comm_used}+{charged} "
                    f"> budget {self._comm_budget}"
                )
            # full register -> production instance; reduced register -> calibration sub-instance
            true_c = self._c if request.n_index_bits >= self._full_index_bits else self._c_calib
            estimate = P.draw_estimate(
                true_c,
                request.estimator,
                precision,
                request.n_index_bits,
                self._floor,
                self._rng,
            )
            self._comm_used += charged
            return JobEstimationData(
                estimator=request.estimator,
                channel=P.channel_of(request.estimator),
                precision=precision,
                n_index_bits=request.n_index_bits,
                estimate=estimate,
                communication_units_charged=charged,
                budget=self._budget_view(),
            )
