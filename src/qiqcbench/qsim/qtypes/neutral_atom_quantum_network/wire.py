"""neutral_atom_quantum_network control + result wire schemas (qtype-local).

One experiment primitive on the async job model: ``run_distributed_estimation`` — choose a
distributed-estimation primitive + precision and estimate the single cross-node association
over the photonic link, paying the link's communication cost. It returns the raw estimate
(noisy evidence) plus the exact communication charged and the cumulative budget. A single
communication budget accumulates across every call in the run.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from qiqcbench.qsim.core.wire import SCHEMA_VERSION, _Strict, _StrictRequest


class DistributedEstimationRequest(_StrictRequest):
    schema_version: int = SCHEMA_VERSION
    estimator: Literal["estimator_a", "estimator_b", "estimator_c"]
    precision: int = Field(..., ge=1)
    n_index_bits: int = Field(18, ge=1, le=18)


class _CommBudgetView(_Strict):
    comm_used: int
    comm_budget: int


class JobEstimationData(_Strict):
    kind: Literal["distributed_estimation"] = "distributed_estimation"
    estimator: Literal["estimator_a", "estimator_b", "estimator_c"]
    channel: Literal["quantum", "classical"]
    precision: int
    n_index_bits: int
    # One noisy estimate of the cross-node association in [-1, 1].
    estimate: float
    # Communication units charged for THIS call.
    communication_units_charged: int
    budget: _CommBudgetView


__all__ = [
    "DistributedEstimationRequest",
    "JobEstimationData",
    "_CommBudgetView",
]
