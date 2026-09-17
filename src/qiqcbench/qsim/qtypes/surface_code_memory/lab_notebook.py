"""Lab-notebook model + builder for the ``surface_code_memory`` qtype.

Surfaces the stale, misleading calibration prior via ``get_lab_notebook``: it claims the
device is i.i.d. depolarizing at the nominal rate, that nominal single-detector rates
suffice, that a default uniform-weight matcher is fine, and that detection events are
independent. Trusting it lands the agent at the i.i.d. anchor, above the floor.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.surface_code_memory.device import HiddenSurfaceCodeConfig


class SurfaceCodeLabNotebook(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["surface_code_memory"] = "surface_code_memory"
    device_id: str
    noise_model: str
    dem_advice: str
    decoder_advice: str
    correlation_advice: str
    budget_advice: str
    note: str
    feedback_advice: str = ""
    drift_advice: str = ""
    dfr_to_ler_claim: str = ""
    recalibration_advice: str = ""
    herald_semantics_advice: str = ""
    herald_reliability_advice: str = ""
    reliability: str = "values accepted at the last scheduled calibration"


def build_surface_code_memory_lab_notebook(
    hidden: HiddenSurfaceCodeConfig,
) -> SurfaceCodeLabNotebook:
    nb = hidden.stale_lab_notebook
    return SurfaceCodeLabNotebook(
        device_id=hidden.device_id,
        noise_model=nb.noise_model,
        dem_advice=nb.dem_advice,
        decoder_advice=nb.decoder_advice,
        correlation_advice=nb.correlation_advice,
        budget_advice=nb.budget_advice,
        feedback_advice=nb.feedback_advice,
        drift_advice=nb.drift_advice,
        dfr_to_ler_claim=nb.dfr_to_ler_claim,
        recalibration_advice=nb.recalibration_advice,
        herald_semantics_advice=nb.herald_semantics_advice,
        herald_reliability_advice=nb.herald_reliability_advice,
        reliability=(nb.reliability or "values accepted at the last scheduled calibration"),
        note=(
            nb.note
            or "Device looks well-described by i.i.d. depolarizing noise at ~0.2%; a default "
            "minimum-weight matching decoder with nominal single-detector weights is enough, "
            "and detection events can be treated as independent. (Stale — verify before "
            "relying on it.)"
        ),
    )
