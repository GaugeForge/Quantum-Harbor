"""Lab-notebook model and builder for the Bose-Hubbard chain qtype.

Notebook values retain their calibration-time provenance and may differ from a
later realized device. They are public observations, not current hidden truth.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.bose_hubbard_chain.device import HiddenBoseHubbardConfig


class BoseHubbardLabNotebook(BaseModel):
    """Stale lab-notebook view for the Bose-Hubbard chain qtype."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    qtype: Literal["bose_hubbard_chain"] = "bose_hubbard_chain"
    device_id: str
    last_updated: str = "stale"
    coherence_t2_ns_claim: float | None = None
    spectral_linewidth_mhz_claim: float | None = None
    expected_two_photon_energies_mhz: list[float] | None = None
    notes: str = ""


def build_bose_hubbard_lab_notebook(
    hidden: HiddenBoseHubbardConfig,
) -> BoseHubbardLabNotebook:
    """Build the agent-facing lab notebook from the hidden stale entry."""
    nb = hidden.stale_lab_notebook
    return BoseHubbardLabNotebook(
        device_id=hidden.device_id,
        last_updated=nb.last_calibrated or "stale",
        coherence_t2_ns_claim=nb.coherence_t2_ns_claim,
        spectral_linewidth_mhz_claim=nb.spectral_linewidth_mhz_claim,
        expected_two_photon_energies_mhz=nb.expected_two_photon_energies_mhz,
        notes=nb.notes,
    )


__all__ = ["BoseHubbardLabNotebook", "build_bose_hubbard_lab_notebook"]
