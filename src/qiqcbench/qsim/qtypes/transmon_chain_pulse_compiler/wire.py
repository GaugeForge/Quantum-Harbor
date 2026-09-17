"""Chain pulse-compiler control + result wire schemas (qtype-local).

The agent's compilation surface is a flat *scheduled gate program*: an ordered
list of gate ops ``{type, qubits, angle, virtual, layer}`` where ``type`` is one
of the native gates ``rz`` / ``rx`` / ``ry`` / ``cphase``. The same gate-program
shape is what the agent runs on the device (``run_compiled_circuit``) and what it
submits in the final answer (and the verifier replays).

The scored decoherence cost is each qubit's busy time recomputed from the gate
list (never self-reported); ``layer`` adds a validity constraint (no two
same-layer gates may share a qubit). ``rz`` with ``virtual=True`` is a free frame
change (zero duration, ~zero error); ``rz`` with ``virtual=False`` is charged as a
finite physical pulse.

The result kind ``JobChainLevelOutcomeData`` returns raw per-shot, per-qubit level
outcomes (``'0'``/``'1'``/``'2'``) so the agent chooses its own readout
discrimination / mitigation. ``JobChainLevelOutcomeData`` is registered into the
shared ``JobData`` union via this qtype's ``DESCRIPTOR.result_data_models`` — it is
defined here, NOT in ``core/wire.py``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from qiqcbench.qsim.core.wire import SCHEMA_VERSION, _Strict, _StrictRequest

# Native gate set; ``cphase`` is the only entangling gate (one pulse per
# controlled-phase). A CNOT decomposition or a physical SWAP network is expressed
# as extra ``cphase`` + rotation ops, so the levers show up as gate count.
ChainGateType = Literal["rz", "rx", "ry", "cphase"]


class ChainGateOp(_StrictRequest):
    """One scheduled gate in the compiled program."""

    type: ChainGateType
    qubits: list[int] = Field(..., min_length=1, max_length=2)
    angle: float = Field(0.0, description="Rotation angle / conditional phase (rad).")
    virtual: bool = Field(
        False,
        description="rz only: True = virtual-Z frame change (free); False = physical pulse.",
    )
    layer: int = Field(
        0, ge=0, description="Schedule layer annotation; no two same-layer gates share a qubit."
    )


class ChainEntanglingCalibration(_StrictRequest):
    """The agent's recalibration of the native CPhase pulse.

    Identity (both 1.0) replays the shipped (stale) pulse. The realized conditional
    phase is ``shipped_over_rotation * over_rotation_correction * intended``; an agent
    that recovered the +5% over-rotation sets ``over_rotation_correction ~ 1/1.05``.
    ``leakage_correction`` scales the excess leakage toward the well-tuned floor.
    """

    over_rotation_correction: float = Field(
        1.0, gt=0, description="Multiplies the realized conditional phase (1.0 = shipped)."
    )
    leakage_correction: float = Field(
        1.0, ge=0, description="Scales the excess leakage factor toward 1.0 (the floor)."
    )


class ChainCircuitRequest(_StrictRequest):
    """Run a scheduled gate program from |0...0> and read out level-resolved."""

    schema_version: int = SCHEMA_VERSION
    shots: int = Field(..., gt=0, le=1_000_000)
    circuit: list[ChainGateOp] = Field(..., min_length=1)
    calibration: ChainEntanglingCalibration | None = Field(
        None, description="Optional CPhase recalibration; None = shipped (stale) pulse."
    )
    measure_qubits: list[int] | None = Field(
        None, description="Qubits to read out (None = all). Always computational (Z) basis."
    )


class JobChainLevelOutcomeData(_Strict):
    """Per-shot, per-qubit level-resolved readout for the chain compiler.

    ``outcomes`` outer index is the run point (length 1 for a single run); inner
    index is the shot. Each string has one character per measured qubit (in
    ``measured_qubits`` order), the reported level ``'0'``/``'1'``/``'2'`` after the
    hidden readout confusion. The agent aggregates these into populations and
    chooses any readout mitigation.
    """

    kind: Literal["chain_level_outcome"] = "chain_level_outcome"
    outcomes: list[list[str]]
    n_levels: int
    measured_qubits: list[int]


__all__ = [
    "ChainCircuitRequest",
    "ChainEntanglingCalibration",
    "ChainGateOp",
    "ChainGateType",
    "JobChainLevelOutcomeData",
]
