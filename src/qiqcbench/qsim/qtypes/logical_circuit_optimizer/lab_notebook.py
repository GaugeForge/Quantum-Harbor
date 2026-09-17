"""Lab notebook for the ``logical_circuit_optimizer`` qtype (``get_lab_notebook``)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.logical_circuit_optimizer.device import HiddenLogicalOptConfig


class LogicalOptLabNotebook(BaseModel):
    """Stale, agent-visible lab notebook. Its actionable advice is a trap (§6)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    device_id: str
    qtype: Literal["logical_circuit_optimizer"] = "logical_circuit_optimizer"
    method_advice: str
    ancilla_advice: str
    notes: str
    reliability: str


def build_logical_opt_lab_notebook(hidden: HiddenLogicalOptConfig) -> LogicalOptLabNotebook:
    nb = hidden.stale_lab_notebook
    return LogicalOptLabNotebook(
        device_id=hidden.device_id,
        method_advice=nb.method_advice,
        ancilla_advice=nb.ancilla_advice,
        notes=nb.notes,
        reliability=nb.reliability,
    )


__all__ = ["LogicalOptLabNotebook", "build_logical_opt_lab_notebook"]
