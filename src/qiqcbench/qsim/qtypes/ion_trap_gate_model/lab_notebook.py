"""Lab-notebook builder for ion_trap_gate_model.

Renders the (deliberately thin / honest) stale notebook entry into an agent-
facing model. The notebook carries no misleading parameters.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.ion_trap_gate_model.device import (
    HiddenIonTrapGateModelConfig,
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IonTrapLabNotebook(_Strict):
    schema_version: int = 1
    device_id: str
    qtype: Literal["ion_trap_gate_model"] = "ion_trap_gate_model"
    last_calibrated: str | None = None
    two_qubit_fidelity_claim: float | None = None
    notes: str = ""


def build_ion_trap_lab_notebook(
    hidden: HiddenIonTrapGateModelConfig,
) -> IonTrapLabNotebook:
    stale = hidden.stale_lab_notebook
    return IonTrapLabNotebook(
        device_id=hidden.device_id,
        last_calibrated=stale.last_calibrated,
        two_qubit_fidelity_claim=stale.claimed_two_qubit_fidelity,
        notes=stale.advisory_notes
        or "Daily-calibrated device; claimed fidelities are nominal and may drift.",
    )


__all__ = ["IonTrapLabNotebook", "build_ion_trap_lab_notebook"]
