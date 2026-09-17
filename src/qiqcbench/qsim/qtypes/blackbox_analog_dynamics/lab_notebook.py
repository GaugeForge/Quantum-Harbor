"""Lab-notebook model and builder for the blackbox analog-dynamics qtype.

Surfaces the stale prior (a weak, possibly-wrong coefficient guess) via
``get_lab_notebook``. The notebook is public by construction; it never carries
the true hidden coefficients.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.blackbox_analog_dynamics.device import HiddenAnalogConfig


class AnalogLabNotebook(BaseModel):
    """Stale lab notebook for the analog-dynamics qtype."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["blackbox_analog_dynamics"] = "blackbox_analog_dynamics"
    device_id: str
    coefficients_rad_per_us: dict[str, float] = {}
    note: str = ""
    reliability: str = "stale; use only as a weak prior"


def build_analog_lab_notebook(hidden: HiddenAnalogConfig) -> AnalogLabNotebook:
    """Build the analog lab-notebook view from a hidden config's stale prior."""
    nb = hidden.stale_lab_notebook
    return AnalogLabNotebook(
        device_id=hidden.device_id,
        coefficients_rad_per_us=dict(nb.coefficients_rad_per_us),
        note=(
            nb.note
            or "Older scans suggested a sparse nearest-neighbor model. These "
            "coefficients are stale and may be wrong; verify before relying on them."
        ),
        reliability=nb.reliability,
    )


__all__ = ["AnalogLabNotebook", "build_analog_lab_notebook"]
