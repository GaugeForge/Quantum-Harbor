"""Wire schemas (qtype-local) for the bosonic_cavity_qec qtype.

The agent submits a **program**: an ordered list of primitive ops on the
cavity + ancilla, optionally with mid-circuit ancilla measurements and
classical feedback (``conditional``). The engine executes it under the hidden
open-system model and returns **raw per-shot measurement outcomes**.

``SCHEMA_VERSION`` and ``_Strict`` come from ``core/wire.py``; the result model
joins the dynamic ``JobData`` union via ``QtypeDescriptor.result_data_models``,
so no edit to ``core/wire.py`` is needed.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import AfterValidator, Field, FiniteFloat, JsonValue

from qiqcbench.qsim.core.wire import SCHEMA_VERSION, _Strict, _StrictRequest

# ---------- Primitive ops ----------


class DisplaceOp(_StrictRequest):
    """Cavity displacement ``D(alpha)``, ``alpha = alpha_re + i alpha_im``."""

    kind: Literal["displace"] = "displace"
    alpha_re: FiniteFloat = Field(..., ge=-5.0, le=5.0)
    alpha_im: FiniteFloat = Field(..., ge=-5.0, le=5.0)


class SnapOp(_StrictRequest):
    """SNAP gate ``diag(exp(i theta_n))`` (zero-padded above ``len(thetas)``)."""

    kind: Literal["snap"] = "snap"
    thetas: list[FiniteFloat] = Field(..., min_length=1, max_length=64)


class AncillaRotateOp(_StrictRequest):
    """Ancilla rotation ``R(theta, phi)`` about an equatorial axis."""

    kind: Literal["ancilla_rotate"] = "ancilla_rotate"
    theta: FiniteFloat
    phi: FiniteFloat = 0.0


class DispersiveWaitOp(_StrictRequest):
    """Joint dispersive evolution for ``duration_ns`` (the parity-map primitive)."""

    kind: Literal["dispersive_wait"] = "dispersive_wait"
    duration_ns: FiniteFloat = Field(..., gt=0)


class AncillaMeasureOp(_StrictRequest):
    """Projective ancilla readout; records a syndrome bit. Optionally resets to g.

    ``record=False`` (additive optional field, wire version unchanged) performs the
    same projection and optional reset but returns no bit: the op contributes no
    result column, is not counted by ``conditional.on_index``, and cannot drive
    feedback. The engine merges its two outcome branches immediately, so an
    autonomous (measure-and-reset) correction suffix does not multiply the cost
    of every later tomography point by two per measurement.
    """

    kind: Literal["ancilla_measure"] = "ancilla_measure"
    reset: bool = True
    record: bool = True


class PhotonNumberMeasureOp(_StrictRequest):
    """Ancilla-assisted photon-number-resolved cavity readout (tomography aid)."""

    kind: Literal["photon_number_measure"] = "photon_number_measure"


class IdleOp(_StrictRequest):
    """Free storage: photon loss on the cavity for ``duration_ns`` (no control)."""

    kind: Literal["idle"] = "idle"
    duration_ns: FiniteFloat = Field(..., gt=0)


# Schema-v3 shapes that may appear inside a conditional (no nested conditionals).
# The runtime explicitly rejects measurement variants until results can represent
# branch-local measurement columns; retaining them here preserves wire compatibility.
NonConditionalOp = Annotated[
    DisplaceOp
    | SnapOp
    | AncillaRotateOp
    | DispersiveWaitOp
    | AncillaMeasureOp
    | PhotonNumberMeasureOp
    | IdleOp,
    Field(discriminator="kind"),
]


class ConditionalOp(_StrictRequest):
    """Apply ``op`` only on shots whose recorded syndrome ``#on_index == value``."""

    kind: Literal["conditional"] = "conditional"
    on_index: int = Field(..., ge=0)
    value: int = Field(0, ge=0, le=1)
    op: NonConditionalOp


BosonicOp = Annotated[
    DisplaceOp
    | SnapOp
    | AncillaRotateOp
    | DispersiveWaitOp
    | AncillaMeasureOp
    | PhotonNumberMeasureOp
    | IdleOp
    | ConditionalOp,
    Field(discriminator="kind"),
]

MAX_EVIDENCE_TAG_BYTES = 16 * 1024
MAX_ANCILLA_MEASUREMENTS_PER_CALL = 16
MAX_FEEDBACK_SYNDROMES_PER_CALL = 16
MAX_RECORDED_OUTCOMES_PER_CALL = 2_000_000


def _stable_evidence_tag(value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Require a bounded, recursively JSON-safe tag with no non-finite numbers."""
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("evidence_tag must contain only finite JSON values") from exc
    if len(encoded) > MAX_EVIDENCE_TAG_BYTES:
        raise ValueError(f"evidence_tag exceeds the {MAX_EVIDENCE_TAG_BYTES}-byte logging limit")
    return value


EvidenceTag = Annotated[dict[str, JsonValue], AfterValidator(_stable_evidence_tag)]


# ---------- Control (requests) ----------


class BosonicProgramRequest(_StrictRequest):
    """Run one bosonic program for ``shots`` shots; returns raw per-shot outcomes.

    ``evidence_tag`` is an optional task-owned pre-execution semantic commitment;
    it does not change execution. Tasks that score tagged evidence define their
    exact tag shape in ``instruction.md``.
    """

    schema_version: int = SCHEMA_VERSION
    shots: int = Field(..., gt=0, le=100_000)
    ops: list[BosonicOp] = Field(..., min_length=1, max_length=512)
    evidence_tag: EvidenceTag | None = None


class BosonicProgramSweepRequest(_StrictRequest):
    """Run a program once per sweep value, overriding one scalar op field.

    ``sweep_field`` is the name of a float field on ``ops[sweep_op_index]`` (e.g.
    ``duration_ns`` for an ``idle``/``dispersive_wait`` op, or ``alpha_re``). One
    result point is produced per entry of ``sweep_values``, in order.
    ``evidence_tag`` has the same optional task-owned semantics as on a single
    program; the exact sweep coordinates remain independently committed.
    """

    schema_version: int = SCHEMA_VERSION
    shots: int = Field(..., gt=0, le=100_000)
    ops: list[BosonicOp] = Field(..., min_length=1, max_length=512)
    sweep_op_index: int = Field(..., ge=0)
    sweep_field: str = Field(..., min_length=1, max_length=64)
    sweep_values: list[FiniteFloat] = Field(..., min_length=1, max_length=64)
    evidence_tag: EvidenceTag | None = None


# ---------- Result (joins the dynamic JobData union) ----------


class BosonicProgramPoint(_Strict):
    """Raw per-shot outcomes for one program execution (one sweep value).

    ``outcomes[s]`` is the ordered list of recorded measurement values for shot
    ``s`` (one entry per ``ancilla_measure`` / ``photon_number_measure`` op, in
    program order). ``measurement_kinds`` / ``measurement_op_indices`` label them.
    """

    sweep_value: float | None = None
    measurement_kinds: list[str] = Field(default_factory=list)
    measurement_op_indices: list[int] = Field(default_factory=list)
    outcomes: list[list[int]] = Field(default_factory=list)


class JobBosonicProgramData(_Strict):
    """Raw per-shot bosonic-program records grouped by sweep point."""

    kind: Literal["bosonic_program"] = "bosonic_program"
    points: list[BosonicProgramPoint] = Field(..., min_length=1)
    n_max: int
    program_digest: str


__all__ = [
    "AncillaMeasureOp",
    "AncillaRotateOp",
    "BosonicOp",
    "BosonicProgramPoint",
    "BosonicProgramRequest",
    "BosonicProgramSweepRequest",
    "ConditionalOp",
    "DispersiveWaitOp",
    "DisplaceOp",
    "EvidenceTag",
    "IdleOp",
    "JobBosonicProgramData",
    "MAX_ANCILLA_MEASUREMENTS_PER_CALL",
    "MAX_EVIDENCE_TAG_BYTES",
    "MAX_FEEDBACK_SYNDROMES_PER_CALL",
    "MAX_RECORDED_OUTCOMES_PER_CALL",
    "NonConditionalOp",
    "PhotonNumberMeasureOp",
    "SnapOp",
]
