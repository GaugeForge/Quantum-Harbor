"""Lab-notebook model + builder for the ``surface_code_lattice_surgery`` qtype.

Surfaces the stale, misleading calibration prior via ``get_lab_notebook``. The notebook's
*scales* sit at the true bulk means — the staleness is the uniformity claim, the
routing-region claim ("matched to the bulk at the last campaign"), and the surgery claim
("merge windows add no new mechanisms; at most rescale to the seam rates"). Trusting it
lands the agent on the transfer trap.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.surface_code_lattice_surgery.device import (
    HiddenLatticeSurgeryConfig,
)


class LatticeSurgeryLabNotebook(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["surface_code_lattice_surgery"] = "surface_code_lattice_surgery"
    device_id: str
    noise_model: str
    routing_region: str
    surgery_advice: str
    decoder_advice: str
    correlation_advice: str
    budget_advice: str
    note: str
    reliability: str = "recorded at the last calibration campaign; verify before relying on it"


def build_surface_code_lattice_surgery_lab_notebook(
    hidden: HiddenLatticeSurgeryConfig,
) -> LatticeSurgeryLabNotebook:
    nb = hidden.stale_lab_notebook
    return LatticeSurgeryLabNotebook(
        device_id=hidden.device_id,
        noise_model=nb.noise_model,
        routing_region=nb.routing_region,
        surgery_advice=nb.surgery_advice,
        decoder_advice=nb.decoder_advice,
        correlation_advice=nb.correlation_advice,
        budget_advice=nb.budget_advice,
        note=nb.note,
    )
