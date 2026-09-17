"""Public + hidden device config types for the digital (gate-model) qtype.

Mirrors the structural public/hidden split in qsim/core/device.py: the public
spec carries no fields that could leak hidden truth.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Public spec (visible to agent) ----------


class PublicDigitalQubit(_Strict):
    id: int = Field(..., ge=0)


class GateSet(_Strict):
    one_qubit: list[str]
    two_qubit: list[str]


class ClaimedFidelities(_Strict):
    one_qubit_avg: float = Field(..., gt=0, le=1)
    two_qubit_cx: float = Field(..., gt=0, le=1)
    readout: float = Field(..., gt=0, le=1)


class PublicDigitalSpec(_Strict):
    """Hardware-manual-style public spec for the digital qtype."""

    schema_version: int = 1
    device_id: str
    qtype: Literal["digital_gate_model"] = "digital_gate_model"
    qubits: list[PublicDigitalQubit] = Field(..., min_length=1)
    gate_set: GateSet
    connectivity: list[tuple[int, int]] = Field(default_factory=list)
    claimed_fidelities: ClaimedFidelities
    max_shots: int = Field(100_000, gt=0)
    notes: str = "Calibrations may be stale; verify before relying on them."


# ---------- Hidden truth (only inside qsim) ----------


class HiddenReadoutQubit(_Strict):
    id: int = Field(..., ge=0)
    p_0_to_1: float = Field(..., ge=0, le=1)
    p_1_to_0: float = Field(..., ge=0, le=1)


class HiddenGateErrors(_Strict):
    one_qubit_depolarizing: float = Field(..., ge=0, le=1)
    two_qubit_cx_depolarizing: float = Field(..., ge=0, le=1)


class StaleDigitalNotebookEntry(_Strict):
    """Stale calibration claims surfaced via get_lab_notebook."""

    last_calibrated: str | None = None
    one_qubit_fidelity_claim: float | None = None
    two_qubit_cx_fidelity_claim: float | None = None
    readout_fidelity_claim: float | None = None
    notes: str | None = None


class HiddenDigitalConfig(_Strict):
    """Hidden truth for the digital qtype. Lives only inside the qsim container."""

    schema_version: int = 1
    device_id: str
    qtype: Literal["digital_gate_model"] = "digital_gate_model"
    seed: int
    gate_errors: HiddenGateErrors
    readout: list[HiddenReadoutQubit] = Field(..., min_length=1)
    stale_lab_notebook: StaleDigitalNotebookEntry = StaleDigitalNotebookEntry()
