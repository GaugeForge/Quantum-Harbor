"""Result-data model for the ``neutral_atom_ftqc_compiler`` qtype.

Device-less / synchronous: ``get_target_circuit`` is a synchronous
calculator that returns its payload directly, never via ``get_job_result``. This
qtype emits no ``JobResult`` at runtime.

The self-registering qtype contract (``core/wire.py::_build_job_data``) still requires every qtype to declare ≥1
``result_data_model`` carrying a ``kind`` discriminator so the dynamic
``JobData`` union is well-formed. ``NeutralAtomTargetData`` documents the
synchronous target-circuit payload shape and fulfils that contract without
reintroducing an async job for a task that has none.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class NeutralAtomTargetData(BaseModel):
    """Structured summary of one ``get_target_circuit`` call.

    Delivered synchronously by the MCP tool (no ``job_id``); declared as this
    qtype's ``result_data_model`` so the registry-derived ``JobData`` union is
    well-formed. The full netlist (gates + DAG) is returned in the tool payload;
    this carries the summary counts that pin the instance.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["neutral_atom_target"] = "neutral_atom_target"
    instance_seed: int
    n_data_qubits: int
    t_count: int
    n_ref_cycles: int


__all__ = ["NeutralAtomTargetData"]
