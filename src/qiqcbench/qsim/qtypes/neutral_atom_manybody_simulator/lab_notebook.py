"""Public notebook view for the Rydberg many-body simulator."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.neutral_atom_manybody_simulator.device import (
    HiddenNeutralAtomManyBodyConfig,
)


class NeutralAtomManyBodyLabNotebook(BaseModel):
    model_config = ConfigDict(extra="forbid")
    last_calibrated: str
    loss_note: str
    correlation_note: str
    readout_note: str


def build_neutral_atom_manybody_lab_notebook(
    hidden: HiddenNeutralAtomManyBodyConfig,
) -> NeutralAtomManyBodyLabNotebook:
    return NeutralAtomManyBodyLabNotebook(**hidden.stale_lab_notebook.model_dump())


__all__ = ["NeutralAtomManyBodyLabNotebook", "build_neutral_atom_manybody_lab_notebook"]
