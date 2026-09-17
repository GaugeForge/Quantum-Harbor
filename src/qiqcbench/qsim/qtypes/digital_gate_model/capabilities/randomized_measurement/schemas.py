"""Wire result models for local randomized-measurement jobs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LocalRandomCircuitBitstrings(_Strict):
    circuit_index: int = Field(..., ge=0)
    basis_bitstrings: list[list[str]] = Field(..., min_length=1)


class JobLocalRandomizedMeasurementData(_Strict):
    """Raw full-register shots from one fresh circuit-ensemble realization."""

    kind: Literal["local_randomized_measurement"] = "local_randomized_measurement"
    depth: int = Field(..., ge=1)
    measured_qubits: list[int] = Field(..., min_length=1)
    n_random_circuits: int = Field(..., ge=1)
    n_measurement_bases: int = Field(..., ge=1)
    shots_per_basis: int = Field(..., ge=2)
    readout_calibration_shots_per_preparation: int = Field(..., ge=2)
    readout_calibration_zero_bitstrings: list[str] = Field(..., min_length=2)
    readout_calibration_one_bitstrings: list[str] = Field(..., min_length=2)
    circuits: list[LocalRandomCircuitBitstrings] = Field(..., min_length=1)
