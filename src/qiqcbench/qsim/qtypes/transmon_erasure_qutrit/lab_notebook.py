"""Lab-notebook model and builder for the g-f erasure-qutrit qtype.

The notebook is intentionally stale: it quotes a "logical lifetime" of ~40 us
that predates the erasure-detection setup (a coarse, pre-post-selection number).
Trusting it - or measuring the bare |1_L> decay - lands ~20x below the true
post-selected code-space bit-flip lifetime.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.transmon_erasure_qutrit.device import (
    HiddenErasureQutritConfig,
)


class ErasureQutritLabNotebook(BaseModel):
    """Stale lab-notebook view for the g-f erasure-qutrit qtype."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["transmon_erasure_qutrit"] = "transmon_erasure_qutrit"
    device_id: str
    last_updated: str = "stale"
    logical_lifetime_us: float | None = None
    t1_ge_us: float | None = None
    t1_ef_us: float | None = None
    recommended_cycle_time_us: float | None = None
    false_negative_rate: float | None = None
    expected_per_cycle_error: float | None = None
    readout_fidelity_claim: float | None = None
    notes: str = ""


def build_erasure_qutrit_lab_notebook(
    hidden: HiddenErasureQutritConfig,
) -> ErasureQutritLabNotebook:
    """Build the agent-facing lab notebook from the hidden stale entry."""
    nb = hidden.stale_lab_notebook
    return ErasureQutritLabNotebook(
        device_id=hidden.device_id,
        last_updated=nb.last_calibrated or "stale",
        logical_lifetime_us=nb.logical_lifetime_us,
        t1_ge_us=nb.t1_ge_us,
        t1_ef_us=nb.t1_ef_us,
        recommended_cycle_time_us=nb.recommended_cycle_time_us,
        false_negative_rate=nb.false_negative_rate,
        expected_per_cycle_error=nb.expected_per_cycle_error,
        readout_fidelity_claim=nb.readout_fidelity_claim,
        notes=nb.notes,
    )


__all__ = ["ErasureQutritLabNotebook", "build_erasure_qutrit_lab_notebook"]
