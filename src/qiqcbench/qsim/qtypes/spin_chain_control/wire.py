"""Capability request models for structured spin-chain control probes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    # allow_inf_nan=False: NaN/Infinity refused on every request float.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ControlTransferRequest(_Strict):
    schema_version: Literal[1] = 1
    controller_id: Literal["A", "B", "C", "D"]
    x_drive_scale_fraction: float = Field(..., ge=-0.12, le=0.12)
    shots: int = Field(..., gt=0)


class ControlReadoutReferenceRequest(_Strict):
    schema_version: Literal[1] = 1
    prepared_bitstring: Literal["000", "100", "010", "001"]
    shots: int = Field(..., gt=0)


__all__ = ["ControlReadoutReferenceRequest", "ControlTransferRequest"]
