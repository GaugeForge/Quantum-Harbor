"""Lab-notebook model and builder for the digital (gate-model) qtype.

The canonical ``DigitalLabNotebook`` lives here; ``qsim.core.lab_notebook``
re-exports it for backward compatibility and exposes the cross-qtype union
alias.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.digital_gate_model.device import HiddenDigitalConfig


class DigitalLabNotebook(BaseModel):
    """Stale lab notebook entry for the digital (gate-model) Qtype."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["digital_gate_model"] = "digital_gate_model"
    device_id: str
    last_updated: str = "stale"
    one_qubit_fidelity_claim: float | None = None
    two_qubit_cx_fidelity_claim: float | None = None
    readout_fidelity_claim: float | None = None
    notes: str = ""


def build_digital_lab_notebook(hidden: HiddenDigitalConfig) -> DigitalLabNotebook:
    """Build the digital lab-notebook view from a hidden config."""
    nb = hidden.stale_lab_notebook
    return DigitalLabNotebook(
        device_id=hidden.device_id,
        last_updated=nb.last_calibrated or "stale",
        one_qubit_fidelity_claim=nb.one_qubit_fidelity_claim,
        two_qubit_cx_fidelity_claim=nb.two_qubit_cx_fidelity_claim,
        readout_fidelity_claim=nb.readout_fidelity_claim,
        notes=(
            nb.notes
            or "Last calibration is dated; gate fidelities and readout "
            "may have drifted. Verify before relying on these values."
        ),
    )


__all__ = ["DigitalLabNotebook", "build_digital_lab_notebook"]
