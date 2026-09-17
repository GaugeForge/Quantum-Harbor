"""Result-data model for the ``ftqc_resource_estimation`` qtype.

This qtype is **device-less / synchronous** (documented exception to
the async-job invariant): its calculators return result dicts directly, never
via ``get_job_result``. It therefore emits no ``JobResult`` at runtime.

The self-registering qtype contract (``core/wire.py::_build_job_data``) still requires every qtype to declare at
least one ``result_data_model`` carrying a ``kind`` discriminator so the dynamic
``JobData`` union is well-formed. The retired logical-evaluation shape remains
declared solely to avoid silently reshaping that versioned union.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class FtqcEvaluationData(BaseModel):
    """Legacy synchronous-evaluator result retained for wire compatibility.

    No active capability emits this shape.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["ftqc_evaluation"] = "ftqc_evaluation"
    total_t_count: int
    valid: bool


class FtqcPhysicalEvaluationData(BaseModel):
    """Structured result of one ``evaluate_factory_design`` cost-oracle call.

    Delivered synchronously (no ``job_id``). ``spacetime_volume_qubit_seconds`` is
    the objective; ``valid`` is the single feasibility boolean. Footprint, failure
    probability, and duration stay held out.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["ftqc_physical_evaluation"] = "ftqc_physical_evaluation"
    spacetime_volume_qubit_seconds: float
    valid: bool


__all__ = ["FtqcEvaluationData", "FtqcPhysicalEvaluationData"]
