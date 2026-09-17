"""Lab-notebook model + builder for the ``neutral_atom_logical_processor`` qtype.

Surfaces device-keyed historical calibration claims via ``get_lab_notebook``. Reliability
wording does not reveal the direction of current drift; task-specific protocol guidance stays
in the active task instruction and reusable gate identities stay in the public device spec.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class NeutralAtomLabNotebook(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["neutral_atom_logical_processor"] = "neutral_atom_logical_processor"
    device_id: str
    last_calibrated: str
    cz_fidelity_claim: float
    single_atom_fidelity_claim: float
    readout_fidelity_claim: float
    transport_note: str
    note: str
    reliability: str = "values accepted at the last scheduled calibration"


def build_neutral_atom_lab_notebook(hidden) -> NeutralAtomLabNotebook:
    """Serve versioned public material rather than deriving public data from hidden truth."""

    # Local import breaks the public-surface/model cycle while keeping the
    # descriptor's existing one-argument notebook-builder protocol.
    from qiqcbench.qsim.qtypes.neutral_atom_logical_processor.public_surface import (
        load_neutral_atom_device_notebook,
    )

    return load_neutral_atom_device_notebook(hidden.device_id)


__all__ = ["NeutralAtomLabNotebook", "build_neutral_atom_lab_notebook"]
