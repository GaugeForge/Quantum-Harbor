"""Dipolar-spin-ensemble control wire schemas (qtype-local).

Mirrors how ``gmon_ring_3q/wire.py`` defines its own discriminated op union
rather than widening the shared ``SequenceOp``. These ops are dipolar-specific
and never mix into the transmon/digital/gmon op unions; the result type
(``JobResult``/``JobBitstringData``) is imported from ``core/wire.py``.

Control model
-------------
The device exposes ONE primitive: run a periodic global pulse sequence
stroboscopically. A *base sequence* is an ordered list of two op kinds:

- ``free``  — free evolution for ``duration_ns`` under the native Hamiltonian
  (on-site disorder + traceless secular dipolar interaction).
- ``pulse`` — a global rotation about ``+-x`` or ``+-y`` by ``angle_deg``
  (default ``90``, i.e. a ``pi/2`` pulse). Finite pulse duration and (on
  request) a systematic rotation-angle error are applied by the engine.

The base sequence is repeated ``n_cycles`` times; the engine reports the
collective magnetization on the requested ``measure_axes`` *stroboscopically*
after each completed cycle (cycles ``0..n_cycles``, where cycle ``0`` is the
freshly-prepared state). One base ``free`` window recovers a free-induction
decay; an echo / XY-8 base recovers the disorder-refocused (interaction-limited)
decay; an engineered toggling-frame base realizes a target average Hamiltonian.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from qiqcbench.qsim.core.wire import SCHEMA_VERSION, _StrictRequest

# ---------- Control ops ----------


class DipolarFreeOp(_StrictRequest):
    """Free evolution under the native Hamiltonian for ``duration_ns``."""

    kind: Literal["free"] = "free"
    duration_ns: float = Field(..., ge=0)


class DipolarPulseOp(_StrictRequest):
    """Global rotation about ``+-x`` or ``+-y`` by ``angle_deg`` (default pi/2).

    ``sign`` selects the rotation sense (``+`` = about ``+axis``, ``-`` = about
    ``-axis``). The physical pulse has a finite duration set by the device Rabi
    frequency and the requested ``angle_deg``; the engine propagates the native
    Hamiltonian *during* the pulse and (on ``inject_rotation_error``) applies the
    hidden systematic over-rotation.
    """

    kind: Literal["pulse"] = "pulse"
    axis: Literal["x", "y"]
    sign: Literal["+", "-"] = "+"
    angle_deg: float = Field(90.0, gt=0, le=360)


DipolarSequenceOp = Annotated[
    DipolarFreeOp | DipolarPulseOp,
    Field(discriminator="kind"),
]


# ---------- run_pulse_train request ----------


class DipolarSequenceRequest(_StrictRequest):
    """Run a periodic global pulse sequence and read out stroboscopically."""

    schema_version: int = SCHEMA_VERSION
    shots: int = Field(..., gt=0, le=1_000_000)
    init_axis: Literal["+x", "-x", "+y", "-y", "+z", "-z"] = "+x"
    sequence: list[DipolarSequenceOp] = Field(..., min_length=1)
    n_cycles: int = Field(1, ge=1, le=4096)
    measure_axes: list[Literal["x", "y", "z"]] = Field(..., min_length=1, max_length=3)
    inject_rotation_error: bool = False


__all__ = [
    "DipolarFreeOp",
    "DipolarPulseOp",
    "DipolarSequenceOp",
    "DipolarSequenceRequest",
]
