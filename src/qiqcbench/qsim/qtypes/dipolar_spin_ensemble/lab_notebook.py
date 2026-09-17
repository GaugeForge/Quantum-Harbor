"""Lab-notebook model and builder for the dipolar-spin-ensemble qtype.

The notebook is intentionally stale/optimistic: it under-reports the disorder
``W``, over-reports the interaction ``J``, claims near-ideal hard pulses, and
recommends textbook NMR sequences. Trusting it leads an agent to under-decouple
the disorder, ignore finite-pulse and rotation-error robustness, and import a
sequence tuned for the wrong (interaction-dominated) regime — the failure modes
that this qtype can expose.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.dipolar_spin_ensemble.device import HiddenDipolarConfig


class DipolarLabNotebook(BaseModel):
    """Stale lab-notebook view for the dipolar-spin-ensemble qtype."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["dipolar_spin_ensemble"] = "dipolar_spin_ensemble"
    device_id: str
    last_updated: str = "unrecorded"
    w_mhz_claim: float | None = None
    j_khz_claim: float | None = None
    pulse_quality_claim: str | None = None
    recommended_sequence: str | None = None
    notes: str = ""


def build_dipolar_lab_notebook(hidden: HiddenDipolarConfig) -> DipolarLabNotebook:
    """Build the agent-facing lab notebook from the hidden stale entry."""
    nb = hidden.stale_lab_notebook
    return DipolarLabNotebook(
        device_id=hidden.device_id,
        last_updated=nb.last_calibrated or "unrecorded",
        w_mhz_claim=nb.w_mhz_claim,
        j_khz_claim=nb.j_khz_claim,
        pulse_quality_claim=nb.pulse_quality_claim,
        recommended_sequence=nb.recommended_sequence,
        notes=nb.notes,
    )


__all__ = ["DipolarLabNotebook", "build_dipolar_lab_notebook"]
