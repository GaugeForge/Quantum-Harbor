"""Lab-notebook model and builder for the bosonic_cavity_qec qtype.

The notebook is intentionally stale/optimistic. Its exact device-keyed public
content is loaded from tracked material rather than derived from hidden truth.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.bosonic_cavity_qec.device import HiddenBosonicCavityConfig


class BosonicCavityLabNotebook(BaseModel):
    """Stale lab-notebook view for the bosonic_cavity_qec qtype."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    qtype: Literal["bosonic_cavity_qec"] = "bosonic_cavity_qec"
    device_id: str
    last_updated: str = "stale"
    storage_lifetime_us: float | None = None
    process_fidelity: float | None = None
    lifetime_gain: str | None = None
    recommended_qec: str | None = None
    nominal_chi_mhz: float | None = None
    notes: str = ""


def build_bosonic_cavity_lab_notebook(
    hidden: HiddenBosonicCavityConfig,
) -> BosonicCavityLabNotebook:
    """Serve versioned public material rather than deriving it from hidden truth."""

    # Local import breaks the public-surface/model cycle while keeping the
    # descriptor's existing one-argument notebook-builder protocol.
    from qiqcbench.qsim.qtypes.bosonic_cavity_qec.public_surface import (
        load_bosonic_cavity_device_notebook,
    )

    return load_bosonic_cavity_device_notebook(hidden.device_id)


__all__ = ["BosonicCavityLabNotebook", "build_bosonic_cavity_lab_notebook"]
