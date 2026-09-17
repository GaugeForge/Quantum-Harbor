"""Result-data model for the ``qccd_ion_compiler`` qtype.

This qtype is **device-less / synchronous** (documented exception to
the async-job invariant): ``get_compilation_instance`` and ``evaluate_schedule``
are synchronous calculators that return their result dict directly, never via
``get_job_result``. No ``JobResult`` is emitted at runtime.

The self-registering qtype contract (``core/wire.py::_build_job_data``) still requires every qtype to declare at least
one ``result_data_model`` carrying a ``kind`` discriminator so the dynamic
``JobData`` union is well-formed. ``QccdEvaluationData`` documents the
``evaluate_schedule`` oracle's structured result shape and fulfils that contract
without reintroducing an async job for a task that has none.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class QccdEvaluationData(BaseModel):
    """Structured result of one ``evaluate_schedule`` oracle call.

    Delivered synchronously by the MCP tool (no ``job_id``); declared as this
    qtype's ``result_data_model`` so the registry-derived ``JobData`` union is
    well-formed. ``total_infidelity`` is the headline metric (only populated when
    the schedule is both legal and circuit-correct); ``legal`` / ``circuit_correct``
    are the hard-gate booleans.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["qccd_schedule_evaluation"] = "qccd_schedule_evaluation"
    legal: bool
    circuit_correct: bool
    total_infidelity: float | None = None


__all__ = ["QccdEvaluationData"]
