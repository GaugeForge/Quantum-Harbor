"""Versioned lab-notebook surface for the blackbox bounded-gate qtype."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.device import (
    HiddenBoundedgateConfig,
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BoundedgateLabNotebook(_Strict):
    schema_version: int = 1
    device_id: str
    qtype: Literal["blackbox_boundedgate_circuit"] = "blackbox_boundedgate_circuit"
    previous_fit_summary: str | None = None
    suggested_degrees: list[int] | None = None
    suggested_protocol: str | None = None
    last_characterized: str | None = None
    notes: str | None = None


def build_boundedgate_lab_notebook(hidden: HiddenBoundedgateConfig) -> BoundedgateLabNotebook:
    notebook = hidden.stale_lab_notebook
    return BoundedgateLabNotebook(
        device_id=hidden.device_id,
        previous_fit_summary=notebook.previous_fit_summary,
        suggested_degrees=notebook.suggested_degrees,
        suggested_protocol=notebook.suggested_protocol,
        last_characterized=notebook.last_characterized,
        notes=notebook.notes,
    )


__all__ = ["BoundedgateLabNotebook", "build_boundedgate_lab_notebook"]
