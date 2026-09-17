"""Stale notebook surface for scheduled native-gate transmon devices."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.scheduled_transmon_gate_model.device import (
    HiddenScheduledTransmonConfig,
)


class ScheduledTransmonLabNotebook(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    qtype: Literal["scheduled_transmon_gate_model"] = "scheduled_transmon_gate_model"
    device_id: str
    last_full_calibration: str
    claimed_ecr_fidelity: float
    claimed_readout_fidelity: float
    simultaneous_gate_penalty: str
    recommended_fom: str
    recommended_candidate_id: str
    drift_note: str
    note: str
    reliability: str = "stale aggregate prior; experimentally re-rank candidates"


def build_scheduled_transmon_lab_notebook(
    hidden: HiddenScheduledTransmonConfig,
) -> ScheduledTransmonLabNotebook:
    notebook = hidden.stale_lab_notebook
    kwargs: dict[str, object] = dict(
        device_id=hidden.device_id,
        last_full_calibration=notebook.last_full_calibration,
        claimed_ecr_fidelity=notebook.claimed_ecr_fidelity,
        claimed_readout_fidelity=notebook.claimed_readout_fidelity,
        simultaneous_gate_penalty=notebook.simultaneous_gate_penalty,
        recommended_fom=notebook.recommended_fom,
        recommended_candidate_id=notebook.recommended_candidate_id,
        drift_note=notebook.drift_note,
        note=notebook.note,
    )
    if notebook.reliability:
        kwargs["reliability"] = notebook.reliability
    return ScheduledTransmonLabNotebook(**kwargs)


__all__ = ["ScheduledTransmonLabNotebook", "build_scheduled_transmon_lab_notebook"]
