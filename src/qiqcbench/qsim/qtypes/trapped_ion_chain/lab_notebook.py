"""Lab-notebook builder for trapped_ion_chain.

Renders HiddenTrappedIonChainConfig.stale_lab_notebook into an
IonChainLabNotebook Pydantic model that the agent can read via
get_lab_notebook MCP tool.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.trapped_ion_chain.device import (
    HiddenTrappedIonChainConfig,
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IonChainLabNotebook(_Strict):
    schema_version: int = 1
    device_id: str
    qtype: Literal["trapped_ion_chain"] = "trapped_ion_chain"
    decay_model_claim: Literal["exponential", "gaussian", "stretched_exponential"] | None = None
    t2_star_us_claim: float | None = None
    ms_fidelity_claim: float | None = None
    readout_fidelity_claim: float | None = None
    readout_symmetric_p_claim: float | None = None
    notes: str = ""


def build_ion_chain_lab_notebook(
    hidden: HiddenTrappedIonChainConfig,
) -> IonChainLabNotebook:
    stale = hidden.stale_lab_notebook
    return IonChainLabNotebook(
        device_id=hidden.device_id,
        decay_model_claim=stale.claimed_decay_model,
        t2_star_us_claim=stale.claimed_t2_star_us,
        ms_fidelity_claim=stale.claimed_ms_fidelity,
        readout_fidelity_claim=stale.claimed_readout_fidelity,
        readout_symmetric_p_claim=stale.claimed_readout_symmetric_p,
        notes=stale.advisory_notes or "Calibrations may be stale.",
    )


__all__ = ["IonChainLabNotebook", "build_ion_chain_lab_notebook"]
