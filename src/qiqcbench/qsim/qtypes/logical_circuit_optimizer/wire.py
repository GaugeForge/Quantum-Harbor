"""Result-data model for the ``logical_circuit_optimizer`` qtype.

Device-less / synchronous (like ``ftqc_resource_estimation``):
``get_optimization_instance`` returns its payload directly, never via
``get_job_result``. The self-registering qtype contract still requires at least one
``result_data_model`` with a unique ``kind`` discriminator so the dynamic ``JobData``
union is well-formed; this model documents the instance-serving payload shape and
fulfils that contract without an async job.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class LogicalOptInstanceData(BaseModel):
    """Structured result of ``get_optimization_instance`` (delivered synchronously)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["logical_circuit_instance"] = "logical_circuit_instance"
    n_qubits: int
    initial_t_count: int


__all__ = ["LogicalOptInstanceData"]
