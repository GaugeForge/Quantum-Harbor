"""Lab notebook for ``composite_measurement_estimation`` (stale, deceptive)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from qiqcbench.qsim.qtypes.composite_measurement_estimation.device import HiddenCmeConfig


class CmeLabNotebook(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["composite_measurement_estimation"] = "composite_measurement_estimation"
    device_id: str
    warm_start_v_haar_claim: float = 602.0
    stale_energy_hartree: float = -74.80
    uncertainty_model: str = ""
    control_variates: str = ""
    reuse_track_a_scheme_for_production: bool = True
    recommendations: list[str] = Field(default_factory=list)
    note: str = ""
    reliability: str = "stale"


def build_composite_measurement_lab_notebook(hidden: HiddenCmeConfig) -> CmeLabNotebook:
    nb = hidden.stale_lab_notebook
    return CmeLabNotebook(
        device_id=hidden.device_id,
        warm_start_v_haar_claim=nb.warm_start_v_haar_claim,
        stale_energy_hartree=nb.stale_energy_hartree,
        uncertainty_model=nb.uncertainty_model,
        control_variates=nb.control_variates,
        reuse_track_a_scheme_for_production=nb.reuse_track_a_scheme_for_production,
        recommendations=[
            "A 16-component warm-start C-LBCS scheme is in the public materials "
            "(stale_notebook.json -> track_a_warm_start); use it to initialize Track A.",
        ],
        note=nb.note,
        reliability=nb.reliability,
    )
