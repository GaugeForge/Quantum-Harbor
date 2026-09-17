"""Lab-notebook model and builder for the blackbox Lindblad-dynamics qtype.

Surfaces the stale prior (a weak, possibly-wrong coherent guess plus an
incomplete Pauli-diagonal/unital noise model) via ``get_lab_notebook``. The
notebook is public by construction; it never carries the true hidden
coefficients.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.blackbox_lindblad_dynamics.device import HiddenLindbladConfig


class LindbladLabNotebook(BaseModel):
    """Stale lab notebook for the Lindblad-dynamics qtype."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["blackbox_lindblad_dynamics"] = "blackbox_lindblad_dynamics"
    device_id: str
    coherent_rad_per_us: dict[str, float] = {}
    dissipative_per_us: dict[str, float] = {}
    note: str = ""
    reliability: str = "stale; use only as a weak prior"


def build_lindblad_lab_notebook(hidden: HiddenLindbladConfig) -> LindbladLabNotebook:
    """Build the Lindblad lab-notebook view from a hidden config's stale prior."""
    nb = hidden.stale_lab_notebook
    return LindbladLabNotebook(
        device_id=hidden.device_id,
        coherent_rad_per_us=dict(nb.coherent_rad_per_us),
        dissipative_per_us=dict(nb.dissipative_per_us),
        note=(
            nb.note
            or "Older scans suggested a sparse nearest-neighbor coherent model and a "
            "Pauli-diagonal, unital noise model. These are stale and may be wrong or "
            "incomplete; verify before relying on them."
        ),
        reliability=nb.reliability,
    )


__all__ = ["LindbladLabNotebook", "build_lindblad_lab_notebook"]
