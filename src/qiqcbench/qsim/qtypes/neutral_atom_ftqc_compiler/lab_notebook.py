"""Lab-notebook model + builder for the ``neutral_atom_ftqc_compiler`` qtype.

Surfaces the stale compilation recommendations (§6) via ``get_lab_notebook``.
The recipe is plausible (and what a textbook / superconducting-style compiler
would do), but every actionable recommendation is a trap: lattice surgery
everywhere (``~d×`` slow), a conservative ``d=9`` (``~d³`` waste), one serial
factory (T-gate stalls), and SWAP-style routing. It never carries the achievable
optimum.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.neutral_atom_ftqc_compiler.device import HiddenNeutralAtomConfig


class NeutralAtomLabNotebook(BaseModel):
    """Stale lab notebook for the neutral-atom FTQC compiler qtype."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["neutral_atom_ftqc_compiler"] = "neutral_atom_ftqc_compiler"
    device_id: str
    cnot_recipe: str = ""
    code_distance_advice: int = 9
    factory_advice: str = ""
    routing_advice: str = ""
    movement_advice: str = ""
    note: str = ""
    reliability: str = "operator notes carried over from a prior compilation study"


def build_neutral_atom_lab_notebook(hidden: HiddenNeutralAtomConfig) -> NeutralAtomLabNotebook:
    """Build the neutral-atom lab-notebook view from a hidden config's stale prior."""
    nb = hidden.stale_lab_notebook
    return NeutralAtomLabNotebook(
        device_id=hidden.device_id,
        cnot_recipe=nb.cnot_recipe,
        code_distance_advice=nb.code_distance_advice,
        factory_advice=nb.factory_advice,
        routing_advice=nb.routing_advice,
        movement_advice=nb.movement_advice,
        note=nb.note,
        reliability=nb.reliability,
    )


__all__ = ["NeutralAtomLabNotebook", "build_neutral_atom_lab_notebook"]
