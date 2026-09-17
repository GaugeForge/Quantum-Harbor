"""Wire schemas for the ``transmon_repeater_array`` qtype.

One batched experiment tool: ``run_purification_experiment``. Each row designs the
**local unitary applied at each end** (a small gate list per party, party-local qubit
indices 0/1) for one purification round; the engine applies them, measures the second
pair, keeps the first pair when the two outcomes agree (post-selection), iterates to
``depth``, and measures the surviving pair in ``measure_basis``. Returns raw 2-bit
counts + the success probability + a run-long pair budget. Reuses ``SCHEMA_VERSION`` /
``_Strict`` from core/wire.py. There is no cross-end operation by construction: a
party's circuit can only touch that party's two qubits.

A gate carries exactly the fields its kind uses: rotations take ``q`` +
``angle_deg``, the other 1-qubit gates take ``q`` only, ``cnot`` takes ``control`` +
``target`` only. Fields the engine would ignore are rejected rather than logged, so one
physical circuit has one canonical logged form -- the verifier matches the submitted
design against the log literally.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from qiqcbench.qsim.core.wire import SCHEMA_VERSION, _Strict, _StrictRequest

GATE_NAMES = ("rx", "ry", "rz", "h", "s", "sdg", "x", "y", "z", "cnot")
ROTATION_GATES = ("rx", "ry", "rz")
MAX_CIRCUIT_GATES = 16
MAX_DEPTH = 7  # = transmon_repeater_16q_v0 public budgets.max_depth
# Measurement settings: first letter Alice's basis, second Bob's -- the nine Pauli pairs
# (full two-qubit state tomography of the surviving / delivered pair).
MEASUREMENT_SETTINGS = ("XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ")


class LocalGate(_StrictRequest):
    """One local gate inside a party's circuit. Party-local qubit indices are 0 or 1
    (qubit 0 = that party's half of pair-1 = the kept pair; qubit 1 = its half of
    pair-2 = the measured pair)."""

    gate: Literal["rx", "ry", "rz", "h", "s", "sdg", "x", "y", "z", "cnot"]
    q: int | None = Field(default=None, ge=0, le=1)  # 1-qubit gate target
    control: int | None = Field(default=None, ge=0, le=1)  # cnot control
    target: int | None = Field(default=None, ge=0, le=1)  # cnot target
    angle_deg: float | None = Field(default=None, allow_inf_nan=False)  # rx/ry/rz angle

    @model_validator(mode="after")
    def _check(self) -> LocalGate:
        if self.gate == "cnot":
            if self.control is None or self.target is None or self.control == self.target:
                raise ValueError("cnot needs distinct control and target in {0,1}")
            if self.q is not None or self.angle_deg is not None:
                raise ValueError("cnot takes only control and target (no q, no angle_deg)")
        else:
            if self.q is None:
                raise ValueError(f"{self.gate} needs a target qubit q in {{0,1}}")
            if self.control is not None or self.target is not None:
                raise ValueError(f"{self.gate} takes q only (no control/target)")
            if self.gate in ROTATION_GATES:
                if self.angle_deg is None:
                    raise ValueError(f"{self.gate} needs angle_deg")
            elif self.angle_deg is not None:
                raise ValueError(f"{self.gate} takes no angle_deg")
        return self


class PurificationRow(_StrictRequest):
    depth: int = Field(..., ge=0, le=MAX_DEPTH)  # purification rounds (2^depth input pairs)
    alice_circuit: list[LocalGate] = Field(default_factory=list, max_length=MAX_CIRCUIT_GATES)
    bob_circuit: list[LocalGate] = Field(default_factory=list, max_length=MAX_CIRCUIT_GATES)
    measure_basis: Literal["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"]
    shots: int = Field(..., ge=64, le=100000)


class PurificationBatchRequest(_StrictRequest):
    schema_version: int = SCHEMA_VERSION
    rows: list[PurificationRow] = Field(..., min_length=1, max_length=64)


class _BudgetView(_Strict):
    pairs_used: int
    pairs_cap: int
    rows_used: int
    rows_cap: int


class PurificationRowResult(_Strict):
    depth: int
    measure_basis: str
    status: Literal["accepted", "rejected"]
    counts: dict[str, int] | None = None  # raw 2-bit outcomes "ab" (a=Alice, b=Bob)
    shots: int | None = None
    success_prob: float | None = None  # product of per-round post-selection probabilities
    pairs_charged: int | None = None
    reject_reason: str | None = None


class JobPurificationData(_Strict):
    kind: Literal["purification_counts"] = "purification_counts"
    rows: list[PurificationRowResult]
    budget: _BudgetView
