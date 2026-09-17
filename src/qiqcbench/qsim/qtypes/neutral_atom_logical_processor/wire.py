"""Control + result wire schemas for ``neutral_atom_logical_processor`` (qtype-local).

The agent submits a *scheduled atom program*: an ordered list of ops the engine
executes per-shot under the hidden neutral-atom noise model. Ops:
  - ``init``       {atom, state in 0/1/+/-/H}           prepare one atom
  - ``gate``       {atoms[], gate, angle}               single-atom gate (h/s/sdg/x/y/z/rx/ry/rz)
  - ``cz``         {atoms:[a,b]}                          Rydberg-blockade CZ
  - ``move``       {atom, to_site}                        AOD transport (timing/loss, not state)
  - ``measure``    {atoms[], basis in z/x/y, reset}      fluorescence readout; records bits
  - ``feedforward``{on_bit, value, then}                 conditional nested op (available, unused here)

``JobAtomShotData`` returns the raw per-shot, per-measure-op outcome bits (base64
packbits) so the verifier Hamming-decodes them. Registered into the shared ``JobData``
union via ``DESCRIPTOR.result_data_models`` — defined here, NOT in ``core/wire.py``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from qiqcbench.qsim.core.wire import SCHEMA_VERSION, _Strict, _StrictRequest

AtomOpType = Literal["init", "gate", "cz", "move", "measure", "feedforward"]


class AtomOp(_StrictRequest):
    """One scheduled op. Only the fields relevant to ``type`` are set (extra forbidden)."""

    type: AtomOpType
    # init
    atom: int | None = None
    state: Literal["0", "1", "+", "-", "H"] | None = None
    # gate
    atoms: list[int] | None = Field(None, description="target atoms (gate/measure)")
    gate: str | None = Field(None, description="h/s/sdg/x/y/z/rx/ry/rz")
    angle: float = 0.0
    # cz / move
    pair: list[int] | None = Field(None, min_length=2, max_length=2)
    to_site: int | None = None
    # measure
    basis: Literal["z", "x", "y"] = "z"
    reset: bool = False
    # feedforward
    on_bit: int | None = None
    value: int | None = None
    then: AtomOp | None = None

    @model_validator(mode="after")
    def _measurement_targets_are_physical(self) -> AtomOp:
        if self.type != "measure":
            return self
        targets = self.atoms or []
        if not targets:
            raise ValueError("measure requires at least one atom")
        if len(targets) != len(set(targets)):
            raise ValueError("one measure op cannot repeat an atom target")
        return self


class AtomProgramRequest(_StrictRequest):
    """Run a scheduled atom program from |0...0> for ``shots`` per-shot trajectories."""

    schema_version: int = SCHEMA_VERSION
    shots: int = Field(..., gt=0, le=1_000_000)
    ops: list[AtomOp] = Field(..., min_length=1)
    layout: list[int] | None = Field(
        None,
        description="initial site index per atom (None = identity sites). For timing/legality.",
    )


class JobAtomShotData(_Strict):
    """Raw per-shot measurement record. ``measure_bits_b64`` packs a (shots, n_recorded)
    0/1 array (the concatenated outcomes of all measure ops, in op order); ``measure_labels``
    names each recorded column ``"<op_index>:<atom>:<basis>"``. ``n_recorded`` is the per-shot
    bit count. ``lost_b64`` packs the matching (shots, n_recorded) atom-lost mask (1 = the
    measured atom was lost, so its bit is a forced dark "0")."""

    kind: Literal["atom_shot_record"] = "atom_shot_record"
    shots: int
    n_recorded: int
    measure_labels: list[str]
    measure_bits_b64: str
    lost_b64: str
    accepted_b64: str | None = None  # optional post-selection survival mask (engine leaves None)


AtomOp.model_rebuild()

__all__ = ["AtomOp", "AtomOpType", "AtomProgramRequest", "JobAtomShotData"]
