"""Lab-notebook model + builder for the Kitaev-chain qtype.

Intentionally optimistic/stale: it advertises a *uniform* sweet spot with the
design protection exponent and negligible defects. Trusting it leads an agent to
report the advertised exponent and never localize or classify the real defect --
the failure mode the localization task probes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.kitaev_chain.device import HiddenKitaevChainConfig


class KitaevChainLabNotebook(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["kitaev_chain"] = "kitaev_chain"
    device_id: str
    last_updated: str = "stale"
    uniform_sweet_spot_claim: bool | None = None
    coupling_ueV_claim: float | None = None
    protection_exponent_claim: float | None = None
    parity_lifetime_ms_claim: float | None = None
    drive_rate_ueV_per_amp_claim: float | None = None
    bulk_gap_ueV_claim: float | None = None
    notes: str = ""


def build_kitaev_chain_lab_notebook(
    hidden: HiddenKitaevChainConfig,
) -> KitaevChainLabNotebook:
    nb = hidden.stale_lab_notebook
    return KitaevChainLabNotebook(
        device_id=hidden.device_id,
        last_updated=nb.last_calibrated or "stale",
        uniform_sweet_spot_claim=nb.uniform_sweet_spot_claim,
        coupling_ueV_claim=nb.coupling_ueV_claim,
        protection_exponent_claim=nb.protection_exponent_claim,
        parity_lifetime_ms_claim=nb.parity_lifetime_ms_claim,
        drive_rate_ueV_per_amp_claim=nb.drive_rate_ueV_per_amp_claim,
        bulk_gap_ueV_claim=nb.bulk_gap_ueV_claim,
        notes=nb.notes,
    )


__all__ = ["KitaevChainLabNotebook", "build_kitaev_chain_lab_notebook"]
