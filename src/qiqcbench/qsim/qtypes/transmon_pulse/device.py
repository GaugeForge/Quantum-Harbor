"""Public + hidden device config types for the transmon (pulse-level) qtype.

Public and hidden are STRUCTURALLY separated: the public spec carries no
fields that could leak hidden truth. Loading a hidden
YAML never produces a public spec and vice versa.

These are the canonical definitions; ``qsim.core.device`` re-exports them and
adds the cross-qtype discriminated-union aliases.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Public spec (visible to agent) ----------


class PublicChannel(_Strict):
    id: str
    kind: Literal["drive", "measure"]


class PublicQubit(_Strict):
    id: str
    frequency_range_hz: tuple[float, float] = Field(
        ..., description="Rough advertised frequency range."
    )


class PulseLimits(_Strict):
    min_duration_ns: float = 4.0
    max_duration_ns: float = 10_000.0
    max_amp: float = 1.0
    timing_resolution_ns: float = 1.0
    supported_shapes: list[str] = ["gaussian", "square"]


class PublicTransmonSpec(_Strict):
    """Hardware-manual-style public spec for the transmon (pulse-level) qtype."""

    schema_version: int = 2
    device_id: str
    qtype: Literal["transmon_pulse"] = "transmon_pulse"
    qubits: list[PublicQubit]
    channels: list[PublicChannel]
    pulse_limits: PulseLimits = PulseLimits()
    measurement_return: Literal["iq"] = "iq"
    provider_measurement_return: Literal["bitstring"] = "bitstring"
    max_shots: int = 100_000
    notes: str = "Calibrations may be stale; verify before relying on them."


# ---------- Hidden truth (only inside qsim) ----------


class HiddenReadout(_Strict):
    p_0_to_1: float = Field(..., ge=0, le=1)
    p_1_to_0: float = Field(..., ge=0, le=1)
    iq_0_mean: tuple[float, float]
    iq_1_mean: tuple[float, float]
    iq_sigma: float = Field(..., gt=0)


class HiddenQubit(_Strict):
    id: str
    frequency_hz: float
    anharmonicity_hz: float
    t1_s: float = Field(..., gt=0)
    t2_s: float = Field(..., gt=0)


class StaleNotebookEntry(_Strict):
    """Optional stale calibration values written into the lab notebook."""

    t1_s: float | None = None
    x180_amp: float | None = None
    x180_duration_ns: float | None = None
    x180_sigma_ns: float | None = None
    readout_threshold_i: float | None = None


class HiddenTransmonConfig(_Strict):
    """Hidden truth for the transmon qtype. Lives only inside the qsim container."""

    schema_version: int = 2
    device_id: str
    qtype: Literal["transmon_pulse"] = "transmon_pulse"
    seed: int
    qubits: list[HiddenQubit]
    readout: HiddenReadout
    stale_lab_notebook: StaleNotebookEntry = StaleNotebookEntry()


__all__ = [
    "HiddenQubit",
    "HiddenReadout",
    "HiddenTransmonConfig",
    "PublicChannel",
    "PublicQubit",
    "PublicTransmonSpec",
    "PulseLimits",
    "StaleNotebookEntry",
]
