"""Driven-dissipative transmon-array control wire schemas (qtype-local).

Mirrors how the gmon qtype defined its own sibling wire family in
``gmon_ring_3q/wire.py`` rather than widening the shared ``core/wire.py`` op
unions. The result payload reuses ``JobBitstringData`` (kind="bitstring") from
``core/wire.py`` — per-shot bitstrings over the measured sites — so no new
``JobData`` variant and no ``SCHEMA_VERSION`` bump is needed.

Control model
-------------
The agent picks one adjacent ``pair`` (e.g. "q0,q1"; site i is qubit q{i}),
prepares the pair in one of {gg, ge, eg, ee} (spectators in |g>), turns on local
energy-selective pump/loss reservoirs (pump detuning ``delta_s``, loss detuning
``delta_d``, both relative to the pair's single-excitation resonance; couplings
``g_s``, ``g_d``), lets the open system relax for ``duration_us``, then reads the
requested sites out in the X/Y/Z analysis basis. Detunings/couplings are in MHz
(ordinary frequency); durations are in microseconds.

``StabilizeSweepRequest`` optionally sweeps the stabilization ``duration`` on a
fixed (pair, initial state, reservoir setting, measurement). The task verifier
does not require this transport shape; a single endpoint job is equally valid.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from qiqcbench.qsim.core.wire import SCHEMA_VERSION, _StrictRequest


class MeasureSetting(_StrictRequest):
    """One site to read out, with the single-qubit analysis basis to apply.

    ``basis="z"`` is the bare computational measurement; ``basis="x"`` applies a
    pre-rotation so the returned bit carries sigma^X; ``basis="y"`` carries
    sigma^Y. Outcomes are per-shot 0/1. (Core's ``MeasureSetting`` is x|y only;
    this qtype-local one adds z for the singlet ZZ correlator.)
    """

    site: int = Field(..., ge=0)
    basis: Literal["x", "y", "z"]


class StabilizeRequest(_StrictRequest):
    """Prepare a pair, drive energy-selective reservoirs for one duration, read out."""

    schema_version: int = SCHEMA_VERSION
    shots: int = Field(..., gt=0, le=1_000_000)
    pair: str = Field(..., description='Adjacent pair, e.g. "q0,q1"; validated against the device.')
    initial_state: Literal["gg", "ge", "eg", "ee"] = Field(
        ..., description="Pair prep; first char is the lower-index site, spectators start in |g>."
    )
    delta_s_mhz: float = Field(..., description="Pump (blue-sideband) detuning / 2pi in MHz.")
    delta_d_mhz: float = Field(..., description="Loss (red-sideband) detuning / 2pi in MHz.")
    g_s_mhz: float = Field(..., ge=0, description="Pump reservoir coupling / 2pi in MHz.")
    g_d_mhz: float = Field(..., ge=0, description="Loss reservoir coupling / 2pi in MHz.")
    duration_us: float = Field(..., ge=0, description="Stabilization duration in microseconds.")
    measure: list[MeasureSetting] = Field(..., min_length=1)


class StabilizeSweepRequest(_StrictRequest):
    """Same as StabilizeRequest but over a grid of stabilization durations.

    One measurement record is produced per duration in ``duration_grid_us``, in
    order; the per-point durations are echoed in the job metadata as
    ``sweep_coords["duration_us"]``.
    """

    schema_version: int = SCHEMA_VERSION
    shots: int = Field(..., gt=0, le=1_000_000)
    pair: str
    initial_state: Literal["gg", "ge", "eg", "ee"]
    delta_s_mhz: float
    delta_d_mhz: float
    g_s_mhz: float = Field(..., ge=0)
    g_d_mhz: float = Field(..., ge=0)
    duration_grid_us: list[float] = Field(..., min_length=1)
    measure: list[MeasureSetting] = Field(..., min_length=1)


__all__ = [
    "MeasureSetting",
    "StabilizeRequest",
    "StabilizeSweepRequest",
]
