"""Lab-notebook model + builder for the ``qccd_ion_compiler`` qtype.

Surfaces the stale, abstraction-trusting recommendations (§6a) via
``get_lab_notebook``. The advice is plausible but every actionable line is a trap
(trust the all-to-all API, swap-heavy routing, decompose to full gates, fast
transport, recool-once, COM-only, single zone, no DD). The notebook never carries
the scoring anchors or the optimal schedule.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.qccd_ion_compiler.device import HiddenQccdConfig


class QccdLabNotebook(BaseModel):
    """Stale lab notebook for the QCCD ion-compiler qtype."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["qccd_ion_compiler"] = "qccd_ion_compiler"
    device_id: str
    connectivity_model: str = ""
    placement_advice: str = ""
    routing_advice: str = ""
    gate_decomposition: str = ""
    transport_speed: str = ""
    heating_advice: str = ""
    recool_advice: str = ""
    mode_advice: str = ""
    zone_advice: str = ""
    dd_advice: str = ""
    notes: str = ""
    reliability: str = "operator notes carried over from a prior compilation study"


def build_qccd_lab_notebook(hidden: HiddenQccdConfig) -> QccdLabNotebook:
    """Build the QCCD lab-notebook view from a hidden config's stale prior."""
    nb = hidden.stale_lab_notebook
    return QccdLabNotebook(
        device_id=hidden.device_id,
        connectivity_model=nb.connectivity_model,
        placement_advice=nb.placement_advice,
        routing_advice=nb.routing_advice,
        gate_decomposition=nb.gate_decomposition,
        transport_speed=nb.transport_speed,
        heating_advice=nb.heating_advice,
        recool_advice=nb.recool_advice,
        mode_advice=nb.mode_advice,
        zone_advice=nb.zone_advice,
        dd_advice=nb.dd_advice,
        notes=nb.notes,
        reliability=nb.reliability,
    )
