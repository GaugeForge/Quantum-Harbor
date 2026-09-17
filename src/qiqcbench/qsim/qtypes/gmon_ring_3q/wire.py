"""Gmon-ring control wire schemas (qtype-local).

Mirrors how the digital qtype defined its own ``CircuitOp`` sibling-union in
``core/wire.py`` rather than widening the shared ``SequenceOp``. These ops are
gmon-specific and never mix into the transmon/digital op unions; the result
types (``JobResult``/``JobIQData``) are imported from ``core/wire.py``.

Control model
-------------
Ring modulation is physically *concurrent*: an adiabatic ground-state prep drives
all three couplers together under one shared ramp envelope. A flat per-coupler
serial op list would misrepresent that, so the unit of time evolution is an
``evolve`` segment carrying the full set of concurrent coupler settings plus a
ramp envelope. Single-qubit XY rotations (state prep + pre-measure tomography),
idle delays, and measurement are the other ops.

Each coupler is modulated as ``g_jk(t) = amp * env(t) * cos(2*pi*freq_hz*t + phase_rad)``
where ``amp`` is normalized to [0, 1] (the realized hopping at amp=1.0 is the hidden
calibration ``g0_hz``). Resonant hopping between detuned qubits emerges only when
``freq_hz ~= |omega_j - omega_k|``; the effective synthetic flux is the DIRECTED ring
sum ``phi12 + phi23 - phi31`` (the only gauge invariant). See ``engine.py`` for why
CP31 enters conjugated.

Sweepable numeric fields accept ``float | str``; a string like ``"$phi"`` is a
placeholder bound by ``GmonSweepRequest.sweep``. Concrete (non-sweep) requests must
carry only floats — the runner/backend rejects unresolved placeholders.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from qiqcbench.qsim.core.wire import SCHEMA_VERSION, _StrictRequest

# ---------- Control ops ----------


class GmonRotationOp(_StrictRequest):
    """Single-qubit XY(Z) rotation for state prep and pre-measure tomography."""

    kind: Literal["xy_rot"] = "xy_rot"
    qubit: str
    axis: Literal["x", "y", "z"]
    angle_rad: float


class CouplerSetting(_StrictRequest):
    """One coupler's parametric modulation during an ``evolve`` segment."""

    coupler: str
    amp: float | str = Field(
        ..., description="Normalized modulation amplitude in [0, 1] (or '$name' placeholder)."
    )
    freq_hz: float | str = Field(
        ...,
        description="Modulation frequency f_jk in Hz (or '$name'). Resonance at |omega_j-omega_k|.",
    )
    phase_rad: float | str = Field(0.0, description="Modulation phase phi_jk (or '$name').")


class GmonEvolveOp(_StrictRequest):
    """Evolve the joint ring state under concurrent coupler modulations.

    Couplers absent from ``couplers`` are off (amp=0) for this segment.
    """

    kind: Literal["evolve"] = "evolve"
    duration_ns: float | str = Field(..., description="Segment duration in ns (or '$name').")
    couplers: list[CouplerSetting] = Field(default_factory=list)
    envelope: Literal["constant", "raised_cosine"] = "constant"
    ramp_ns: float | None = Field(
        None,
        ge=0,
        description=(
            "Raised-cosine ramp-up/down time applied to every coupler amplitude in this "
            "segment (the adiabatic ramp). Ignored for the 'constant' envelope."
        ),
    )


class GmonDelayOp(_StrictRequest):
    """Idle evolution with all couplers off (captures T1/T2 decoherence)."""

    kind: Literal["delay"] = "delay"
    duration_ns: float | str = Field(..., description="Idle duration in ns (or '$name').")


class GmonMeasureOp(_StrictRequest):
    """Measure the listed qubits in the computational (Z) basis.

    Pauli-basis tomography is expressed by preceding ``xy_rot`` ops; the simulator
    returns raw correlated IQ shots so the agent reconstructs the correlators.
    """

    kind: Literal["gmon_measure"] = "gmon_measure"
    qubits: list[str] = Field(..., min_length=1)


GmonSequenceOp = Annotated[
    GmonRotationOp | GmonEvolveOp | GmonDelayOp | GmonMeasureOp,
    Field(discriminator="kind"),
]


# ---------- run_gmon_sequence / run_gmon_sweep requests ----------


class GmonSequenceRequest(_StrictRequest):
    schema_version: int = SCHEMA_VERSION
    shots: int = Field(..., gt=0, le=1_000_000)
    sequence: list[GmonSequenceOp] = Field(..., min_length=1)


class GmonSweepRequest(_StrictRequest):
    schema_version: int = SCHEMA_VERSION
    shots: int = Field(..., gt=0, le=1_000_000)
    template_sequence: list[GmonSequenceOp] = Field(..., min_length=1)
    sweep: dict[str, list[float]] = Field(
        ...,
        description=(
            "Map of placeholder name (e.g. 'phi') to list of values. Placeholders appear "
            "in the template as the string '$<name>' in sweepable numeric fields."
        ),
    )
    mode: Literal["product", "zip"] = "product"


__all__ = [
    "CouplerSetting",
    "GmonDelayOp",
    "GmonEvolveOp",
    "GmonMeasureOp",
    "GmonRotationOp",
    "GmonSequenceOp",
    "GmonSequenceRequest",
    "GmonSweepRequest",
]
