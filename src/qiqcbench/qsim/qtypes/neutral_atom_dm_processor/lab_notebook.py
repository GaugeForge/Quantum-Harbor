"""Lab-notebook model + builder for the ``neutral_atom_dm_processor`` qtype.

Surfaces device-keyed historical calibration claims via ``get_lab_notebook``.
Misinformation is scoped to optimistic/stale numbers; protocol facts stay
honest. Reliability wording does not reveal the direction of current drift.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class NeutralAtomDmLabNotebook(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["neutral_atom_dm_processor"] = "neutral_atom_dm_processor"
    device_id: str
    last_calibrated: str
    cz_fidelity_claim: float
    rotation_fidelity_claim: float
    readout_fidelity_claim: float
    transport_note: str
    coherence_note: str
    note: str
    reliability: str = (
        "stale; current drift direction and magnitude are unverified -- re-measure the "
        "primitives used by your protocol"
    )


def build_neutral_atom_dm_lab_notebook(hidden) -> NeutralAtomDmLabNotebook:
    """Serve versioned public material rather than deriving public data from hidden truth."""

    # Local import breaks the public-surface/model cycle while keeping the
    # descriptor's existing one-argument notebook-builder protocol.
    from qiqcbench.qsim.qtypes.neutral_atom_dm_processor.public_surface import (
        load_neutral_atom_dm_device_notebook,
    )

    return load_neutral_atom_dm_device_notebook(hidden.device_id)


__all__ = ["NeutralAtomDmLabNotebook", "build_neutral_atom_dm_lab_notebook"]
