"""Lab-notebook model and builder for the NV sensor-network qtype.

The notebook is intentionally optimistic/stale: it homogenizes the
heterogeneous per-node T2* and link fidelities, reports symmetric readout, and
recommends "GHZ always beats product states". Trusting it leads an agent to
allocate interrogation time uniformly, treat entanglement distribution as free,
and apply a uniform GHZ to every target.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.nv_sensor_network.device import HiddenNvSensorNetworkConfig


class NvSensorNetworkLabNotebook(BaseModel):
    """Stale lab-notebook view for the NV sensor-network qtype."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["nv_sensor_network"] = "nv_sensor_network"
    device_id: str
    last_updated: str = "stale"
    t2star_us_claim: float | None = None
    link_fidelity_claim: float | None = None
    readout_fidelity_claim: float | None = None
    recommended_strategy: str | None = None
    notes: str = ""


def build_nv_sensor_network_lab_notebook(
    hidden: HiddenNvSensorNetworkConfig,
) -> NvSensorNetworkLabNotebook:
    """Build the agent-facing lab notebook from the hidden stale entry."""
    nb = hidden.stale_lab_notebook
    return NvSensorNetworkLabNotebook(
        device_id=hidden.device_id,
        last_updated=nb.last_calibrated or "stale",
        t2star_us_claim=nb.t2star_us_claim,
        link_fidelity_claim=nb.link_fidelity_claim,
        readout_fidelity_claim=nb.readout_fidelity_claim,
        recommended_strategy=nb.recommended_strategy,
        notes=nb.notes,
    )


__all__ = ["NvSensorNetworkLabNotebook", "build_nv_sensor_network_lab_notebook"]
