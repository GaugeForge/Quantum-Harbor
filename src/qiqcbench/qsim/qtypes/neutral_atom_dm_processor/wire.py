"""Control + result wire schemas for ``neutral_atom_dm_processor`` (qtype-local).

The agent submits a *scheduled atom program* over a zoned neutral-atom array
(storage row + Rydberg gate zone), executed by an exact density-matrix
branch-tree engine. Ops:
  - ``init``        {atom|atoms, state in 0/1/+/-}      prepare atom(s)
  - ``gate``        {atoms, gate in rx/ry/rz, angle, scope}  rotations (the ONLY 1q gates);
                    ``scope="global"`` hits every atom at the global duration
  - ``cz``          {pair:[a,b]}     Rydberg-blockade CZ; both atoms must sit in
                    the two positions of one gate-zone slot
  - ``move``        {atom, to_site}  AOD transport (duration = distance x rate + settle)
  - ``measure``     {atoms, reset, expect?}  mid-circuit fluorescence readout, Z basis ONLY
                    (basis changes are the agent's own noisy rotations, deliberately);
                    ``expect`` turns it into a heralded-abort verification readout (a
                    shot whose observed bits differ aborts; see the engine docstring)
  - ``feedforward`` {on_bit, value, then}  conditional op on an earlier recorded bit

Two request parameters are part of the physics: ``idle_scale`` multiplies every
schedule duration; ``noise_scale`` multiplies every hidden error rate together
(the conventional dial for error-order scaling studies). Both are bound into
the evidence digest — evidence gathered at one noise_scale cannot be cited as
another (plan-review adjustment #2).

Always-record feedforward semantics (plan-review adjustment #3): a conditional
``measure`` inside ``then`` emits its record column in EVERY shot; when the
condition is unmet the bit records 0 and the matching ``executed`` mask bit is
0. The record stays rectangular, so decode bit indices are always well-defined.
``then`` may not itself be a ``feedforward`` (single-level nesting).

``JobAtomDmShotData`` returns raw per-shot bits inline (packbits b64).
``JobAtomDmSweepData`` summarizes per-point counts inline and carries the full
per-point raw bits ONLY transiently (``point_bits_b64``); the action layer
moves them to a public raw file under ``/qsim_logs`` and nulls the inline copy
(plan-review adjustment #4) so a k-point sweep cannot flood agent context.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, FiniteFloat, model_validator

from qiqcbench.qsim.core.wire import SCHEMA_VERSION, _Strict, _StrictRequest

AtomOpType = Literal["init", "gate", "cz", "move", "measure", "feedforward"]
InitState = Literal["0", "1", "+", "-"]
RotationGate = Literal["rx", "ry", "rz"]
SweepParameter = Literal["noise_scale", "idle_scale"]

NOISE_SCALE_RANGE = (0.25, 2.0)
IDLE_SCALE_RANGE = (0.25, 4.0)


class AtomOp(_StrictRequest):
    """One scheduled op. Only the fields relevant to ``type`` are set (extra forbidden)."""

    type: AtomOpType
    # init / move
    atom: int | None = None
    state: InitState | None = None
    # gate / measure targets
    atoms: list[int] | None = Field(None, description="target atoms (gate/measure/init batch)")
    gate: RotationGate | None = None
    angle: FiniteFloat = 0.0
    scope: Literal["local", "global"] = "local"
    # cz
    pair: list[int] | None = Field(None, min_length=2, max_length=2)
    # move
    to_site: int | None = None
    # measure
    reset: bool = False
    # Heralded-abort verification: the expected observed bit per measured atom. A shot
    # whose observation differs ABORTS -- no later op executes, every later record
    # column reads 0 with executed = 0, and the shot is marked in the aborted mask.
    # Expected-outcome bits do not split the exact simulation (the engine follows the
    # accepted branch and books the aborted weight classically), so they do not count
    # toward the mid-circuit branch cap.
    expect: list[int] | None = None
    # feedforward
    on_bit: int | None = None
    value: int | None = None
    then: AtomOp | None = None

    @model_validator(mode="after")
    def _op_shape(self) -> AtomOp:
        if self.type == "measure":
            targets = self.atoms or []
            if not targets:
                raise ValueError("measure requires at least one atom")
            if len(targets) != len(set(targets)):
                raise ValueError("one measure op cannot repeat an atom target")
            if self.expect is not None:
                if len(self.expect) != len(targets):
                    raise ValueError("measure 'expect' must list one expected bit per atom")
                if any(bit not in (0, 1) for bit in self.expect):
                    raise ValueError("measure 'expect' bits must be 0 or 1")
        elif self.expect is not None:
            raise ValueError("'expect' is only valid on a measure op")
        if self.type == "gate":
            if self.gate is None:
                raise ValueError("gate op requires 'gate' (rx/ry/rz)")
            if self.scope == "global" and self.atoms:
                raise ValueError("a global gate addresses every atom; omit 'atoms'")
            if self.scope == "local" and not self.atoms:
                raise ValueError("a local gate requires 'atoms'")
        if self.type == "feedforward":
            if self.on_bit is None or self.value is None or self.then is None:
                raise ValueError("feedforward requires 'on_bit', 'value', and 'then'")
            if self.then.type == "feedforward":
                raise ValueError("feedforward cannot nest another feedforward")
            if self.then.type == "measure" and self.then.expect is not None:
                raise ValueError(
                    "a feedforward-conditioned measure cannot carry 'expect' (an unexecuted "
                    "conditional measure records a filler 0, which is not an observation)"
                )
        return self


class AtomProgramRequest(_StrictRequest):
    """Run a scheduled atom program for ``shots`` samples.

    ``layout`` is REQUIRED and sets both the atom count (= its length; the
    density-matrix dimension is 4^n) and each atom's initial site. Atoms are
    indexed 0..len(layout)-1 in every op.
    """

    schema_version: int = SCHEMA_VERSION
    shots: int = Field(..., gt=0, le=1_000_000)
    layout: list[int] = Field(..., min_length=1)
    ops: list[AtomOp] = Field(..., min_length=1)
    idle_scale: FiniteFloat = Field(1.0, ge=IDLE_SCALE_RANGE[0], le=IDLE_SCALE_RANGE[1])
    noise_scale: FiniteFloat = Field(1.0, ge=NOISE_SCALE_RANGE[0], le=NOISE_SCALE_RANGE[1])

    @model_validator(mode="after")
    def _unique_sites(self) -> AtomProgramRequest:
        if len(self.layout) != len(set(self.layout)):
            raise ValueError("layout assigns two atoms to one site")
        return self


class AtomProgramSweepRequest(_StrictRequest):
    """Run the same ops at several values of one run parameter (one job).

    ``shots`` applies PER POINT; budgets charge len(sweep_values) x shots in a
    single reservation. The swept parameter's base-field value is ignored at
    each point (replaced by the point value); the other parameter keeps its
    base value.
    """

    schema_version: int = SCHEMA_VERSION
    shots: int = Field(..., gt=0, le=1_000_000)
    layout: list[int] = Field(..., min_length=1)
    ops: list[AtomOp] = Field(..., min_length=1)
    idle_scale: FiniteFloat = Field(1.0, ge=IDLE_SCALE_RANGE[0], le=IDLE_SCALE_RANGE[1])
    noise_scale: FiniteFloat = Field(1.0, ge=NOISE_SCALE_RANGE[0], le=NOISE_SCALE_RANGE[1])
    sweep_parameter: SweepParameter
    sweep_values: list[FiniteFloat] = Field(..., min_length=1, max_length=16)

    @model_validator(mode="after")
    def _values_in_range(self) -> AtomProgramSweepRequest:
        if len(self.layout) != len(set(self.layout)):
            raise ValueError("layout assigns two atoms to one site")
        lo, hi = NOISE_SCALE_RANGE if self.sweep_parameter == "noise_scale" else IDLE_SCALE_RANGE
        for v in self.sweep_values:
            if not (lo <= v <= hi):
                raise ValueError(
                    f"sweep value {v} outside the public {self.sweep_parameter} range [{lo}, {hi}]"
                )
        if len(self.sweep_values) != len(set(self.sweep_values)):
            raise ValueError("sweep values must be distinct")
        return self


class JobAtomDmShotData(_Strict):
    """Raw per-shot measurement record for one program.

    ``measure_bits_b64`` packs a (shots, n_recorded) 0/1 array — the outcomes of
    all measure ops in op order (conditional measures included; see the module
    docstring). ``executed_b64`` packs the matching executed mask (0 = the
    conditional measure did not run that shot, its bit is a filler 0).
    ``measure_labels`` names each column ``"<op_index>:<atom>"``. Atom loss is
    physically undetectable here: a lost atom simply reads dark 0 — no loss
    mask is exposed.
    """

    kind: Literal["atom_dm_shot_record"] = "atom_dm_shot_record"
    shots: int
    n_recorded: int
    measure_labels: list[str]
    measure_bits_b64: str
    executed_b64: str
    idle_scale: float
    noise_scale: float
    # Heralded-abort mask (one packed bit per shot; 1 = the shot aborted at a
    # measure whose observed bits differed from its ``expect``). Every record
    # column after the abort reads 0 with ``executed = 0``. ``None`` means the
    # program declared no expected outcomes (pre-review records are read the
    # same way: no shot aborted).
    aborted_b64: str | None = None


class AtomSweepPointSummary(_Strict):
    value: float
    shots: int
    n_recorded: int


class JobAtomDmSweepData(_Strict):
    """Sweep result: inline per-point summaries + a raw-file pointer.

    ``point_bits_b64``/``point_executed_b64`` (one packed record per point, in
    ``sweep_values`` order) are filled by the engine and moved by the action
    layer into ``raw_data_file`` (a JSON file below ``/qsim_logs``); the
    agent-facing copy carries only the pointer. Inspect the file with a local
    script rather than printing it.
    """

    kind: Literal["atom_dm_sweep_record"] = "atom_dm_sweep_record"
    sweep_parameter: SweepParameter
    points: list[AtomSweepPointSummary]
    measure_labels: list[str]
    n_recorded: int
    idle_scale: float
    noise_scale: float
    point_bits_b64: list[str] | None = None
    point_executed_b64: list[str] | None = None
    # Per-point heralded-abort masks (see ``JobAtomDmShotData.aborted_b64``);
    # offloaded together with the raw bits.
    point_aborted_b64: list[str] | None = None
    raw_data_file: str | None = None


AtomOp.model_rebuild()

__all__ = [
    "IDLE_SCALE_RANGE",
    "NOISE_SCALE_RANGE",
    "AtomOp",
    "AtomOpType",
    "AtomProgramRequest",
    "AtomProgramSweepRequest",
    "AtomSweepPointSummary",
    "InitState",
    "JobAtomDmShotData",
    "JobAtomDmSweepData",
    "RotationGate",
    "SweepParameter",
]
