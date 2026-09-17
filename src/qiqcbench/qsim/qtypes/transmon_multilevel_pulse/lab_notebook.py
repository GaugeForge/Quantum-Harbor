"""Lab-notebook model + builder for the multilevel-transmon qtype.

The notebook is intentionally stale/optimistic: it advertises an anharmonicity
and amplitude calibration copied from an earlier configuration, the textbook
``beta = 0.5`` DRAG weight, a "detuning negligible" claim, and a "3 levels are
enough" model claim. Trusting it leads an agent into the documented DRAG traps.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.transmon_multilevel_pulse.device import (
    HiddenTransmonMultilevelConfig,
)


class TransmonMultilevelLabNotebook(BaseModel):
    """Stale lab-notebook view for the multilevel-transmon qtype."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["transmon_multilevel_pulse"] = "transmon_multilevel_pulse"
    device_id: str
    last_updated: str = "stale"
    anharmonicity_mhz_claim: float | None = None
    rabi_mhz_per_dac_claim: float | None = None
    amplitude_linear_claim: bool | None = None
    drag_beta_claim: float | None = None
    detuning_claim: str | None = None
    level_model_claim: str | None = None
    notes: str = ""


def build_transmon_multilevel_lab_notebook(
    hidden: HiddenTransmonMultilevelConfig,
) -> TransmonMultilevelLabNotebook:
    """Build the agent-facing stale lab notebook from the hidden stale entry."""
    nb = hidden.stale_lab_notebook
    return TransmonMultilevelLabNotebook(
        device_id=hidden.device_id,
        last_updated=nb.last_calibrated or "stale",
        anharmonicity_mhz_claim=nb.anharmonicity_mhz_claim,
        rabi_mhz_per_dac_claim=nb.rabi_mhz_per_dac_claim,
        amplitude_linear_claim=nb.amplitude_linear_claim,
        drag_beta_claim=nb.drag_beta_claim,
        detuning_claim=nb.detuning_claim,
        level_model_claim=nb.level_model_claim,
        notes=nb.notes,
    )


__all__ = ["TransmonMultilevelLabNotebook", "build_transmon_multilevel_lab_notebook"]
