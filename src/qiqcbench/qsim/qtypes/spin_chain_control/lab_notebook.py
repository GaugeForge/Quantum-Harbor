"""Stale notebook surface for structured spin-chain control."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.spin_chain_control.device import HiddenSpinChainControlConfig


class SpinChainControlLabNotebook(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    qtype: Literal["spin_chain_control"] = "spin_chain_control"
    device_id: str
    last_characterization: str
    recommended_controller_id: Literal["A", "B", "C", "D"]
    claimed_nominal_success: dict[str, float]
    claimed_local_sensitivity: dict[str, float]
    recommendation_method: str
    drift_note: str
    note: str
    reliability: str = "local characterization recorded at the last scheduled scan"


def build_spin_chain_control_lab_notebook(
    hidden: HiddenSpinChainControlConfig,
) -> SpinChainControlLabNotebook:
    notebook = hidden.stale_lab_notebook
    return SpinChainControlLabNotebook(
        device_id=hidden.device_id,
        last_characterization=notebook.last_characterization,
        recommended_controller_id=notebook.recommended_controller_id,
        claimed_nominal_success=notebook.claimed_nominal_success,
        claimed_local_sensitivity=notebook.claimed_local_sensitivity,
        recommendation_method=notebook.recommendation_method,
        drift_note=notebook.drift_note,
        note=notebook.note,
    )


__all__ = ["SpinChainControlLabNotebook", "build_spin_chain_control_lab_notebook"]
