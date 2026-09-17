"""Capability request schemas for scheduled candidate evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    # allow_inf_nan=False: NaN/Infinity refused on every request float.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CompilationMirrorRequest(_Strict):
    schema_version: Literal[1] = 1
    candidate_id: str = Field(..., pattern=r"^cand_[0-9]{2}$")
    mirror_seed: int = Field(..., ge=0)
    shots: int = Field(..., gt=0)


class CompilationReadoutReferenceRequest(_Strict):
    schema_version: Literal[1] = 1
    prepared_bitstring: str = Field(..., pattern=r"^[01]+$")
    shots: int = Field(..., gt=0)


__all__ = ["CompilationMirrorRequest", "CompilationReadoutReferenceRequest"]
