"""Control + result wire schemas for the tunable-coupler CZ qtype (qtype-local).

The agent programs DAC samples for ``q1.flux`` on the AWG grid, sets a constant
coupler bias, and may bracket the flux pulse with single-qubit microwave rotations
(state prep + pre-measure tomography). One async tool ``run_flux_pulse`` covers
the whole experiment family: spectroscopy/chevron (prep |11>, scan the pulse),
conditional-phase Ramsey (prep |+>, post -|+>), idle-ZZ (zero-flux idle), cryoscope
(truncated pulse + Ramsey), and CZ verification (computational prep + tomography).

The result kind ``JobPairLevelOutcomeData`` returns raw per-shot, per-qubit level
outcomes (0/1/2 for q1,q2), defined here and registered into ``JobData`` via the
descriptor (``core/wire.py`` is never edited).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from qiqcbench.qsim.core.wire import SCHEMA_VERSION, _Strict, _StrictRequest


class PairRotationOp(_StrictRequest):
    """Single-qubit microwave rotation for state prep / pre-measure tomography."""

    qubit: Literal[1, 2]
    axis: Literal["x", "y", "z"]
    angle_rad: float


class FluxPulseRequest(_StrictRequest):
    """Program a q1.flux DAC waveform at a fixed coupler bias and read out.

    ``programmed_flux_q1`` is the DAC waveform on the AWG grid; the qubit sees the
    flux-line-distorted version. ``coupler_flux`` sets J_eff (and the static ZZ).
    ``prep_ops`` run before the pulse, ``post_ops`` after (tomography basis).
    """

    schema_version: int = SCHEMA_VERSION
    shots: int = Field(..., gt=0, le=1_000_000)
    coupler_flux: float = Field(..., description="Coupler bias in Phi_0 (constant).")
    programmed_flux_q1: list[float] = Field(
        default_factory=list, description="DAC samples for q1.flux (Phi_0) on the AWG grid."
    )
    sample_dt_ns: float = Field(0.4, gt=0)
    prep_ops: list[PairRotationOp] = Field(default_factory=list)
    post_ops: list[PairRotationOp] = Field(default_factory=list)
    idle_ns: float = Field(
        0.0, ge=0, description="Idle evolution (no flux pulse) for static-ZZ measurement."
    )
    measure_qubits: list[Literal[1, 2]] = Field(default_factory=lambda: [1, 2])


class JobPairLevelOutcomeData(_Strict):
    """Per-shot, per-qubit level-resolved outcomes (q1,q2 in {0,1,2})."""

    kind: Literal["pair_level_outcome"] = "pair_level_outcome"
    outcomes: list[list[str]]  # outcomes[point][shot] = e.g. "12" (q1=1,q2=2)
    n_levels: int
    measure_qubits: list[int]


__all__ = [
    "FluxPulseRequest",
    "JobPairLevelOutcomeData",
    "PairRotationOp",
]
