"""Strict request and raw-measurement schemas for the Rydberg instrument."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from qiqcbench.qsim.core.wire import SCHEMA_VERSION, _Strict, _StrictRequest

_PACKED_BITS = (
    "C-order flatten, NumPy packbits/unpackbits with bitorder='big', then Base64; "
    "truncate to product(shape) and require zero trailing padding bits"
)


class Cz2CharacterizationRequest(_StrictRequest):
    schema_version: int = SCHEMA_VERSION
    echo_protocol: Literal["duration_echo", "phase_echo"]
    repetitions: int = Field(..., ge=1, le=16)
    duration_scale: float = Field(..., ge=0.90, le=1.10)
    target_phase_compensation_rad: float = Field(..., ge=-0.30, le=0.30)
    shots: int = Field(..., ge=1, le=100_000)


class MultitargetCheckPartition(_StrictRequest):
    check_id: str
    target_role_pairs: list[list[str]] = Field(..., min_length=2, max_length=2)


class MultitargetStabilizerCycle(_StrictRequest):
    mode: Literal["cz2_depth_reduced"]
    bulk_check_partitions: list[MultitargetCheckPartition] = Field(..., min_length=4, max_length=4)


class MultitargetMemoryRequest(_StrictRequest):
    schema_version: int = SCHEMA_VERSION
    decoded_basis: Literal["Z", "X"]
    cycle: MultitargetStabilizerCycle
    duration_scale: float = Field(..., ge=0.90, le=1.10)
    target_phase_compensation_rad: float = Field(..., ge=-0.30, le=0.30)
    rounds: int = Field(..., ge=3, le=32)
    shots: int = Field(..., ge=1, le=100_000)


class _ShotBudgetView(_Strict):
    shots_used: int = Field(..., ge=0)
    shots_cap: int = Field(..., ge=1)


class JobCz2CharacterizationData(_Strict):
    kind: Literal["cz2_echo_characterization"] = "cz2_echo_characterization"
    echo_protocol: Literal["duration_echo", "phase_echo"]
    repetitions: int
    duration_scale: float
    target_phase_compensation_rad: float
    shots: int
    readout_bits_b64: str = Field(
        ...,
        description=(
            "Packed per-shot fluorescence bits with shape [shot, role] and role order "
            "[control, target0, target1]; " + _PACKED_BITS
        ),
    )
    readout_shape: list[int]
    budget: _ShotBudgetView

    @model_validator(mode="after")
    def _shape_matches_shots(self) -> JobCz2CharacterizationData:
        if self.readout_shape != [self.shots, 3]:
            raise ValueError("readout_shape must be [shots, 3]")
        return self


class JobMultitargetMemoryData(_Strict):
    kind: Literal["multitarget_stabilizer_memory"] = "multitarget_stabilizer_memory"
    decoded_basis: Literal["Z", "X"]
    rounds: int
    shots: int
    cycle_digest: str
    cycle_duration_us: float
    duration_scale: float
    target_phase_compensation_rad: float
    syndrome_measurement_bits_b64: str = Field(
        ...,
        description=(
            "Packed raw ancilla measurement bitstrings with shape [shot, round, ancilla] "
            "and ancilla order m0..m11; " + _PACKED_BITS
        ),
    )
    syndrome_measurement_shape: list[int]
    final_data_measurement_bits_b64: str = Field(
        ...,
        description=(
            "Packed raw final data measurement bitstrings with shape [shot, data] and data "
            "order d0..d12; " + _PACKED_BITS
        ),
    )
    final_data_measurement_shape: list[int]
    budget: _ShotBudgetView

    @model_validator(mode="after")
    def _shapes_match_request(self) -> JobMultitargetMemoryData:
        if self.syndrome_measurement_shape != [self.shots, self.rounds, 12]:
            raise ValueError("syndrome_measurement_shape must be [shots, rounds, 12]")
        if self.final_data_measurement_shape != [self.shots, 13]:
            raise ValueError("final_data_measurement_shape must be [shots, 13]")
        return self


__all__ = [
    "Cz2CharacterizationRequest",
    "JobCz2CharacterizationData",
    "JobMultitargetMemoryData",
    "MultitargetCheckPartition",
    "MultitargetMemoryRequest",
    "MultitargetStabilizerCycle",
]
