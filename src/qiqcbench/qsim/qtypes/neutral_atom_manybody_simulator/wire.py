"""Qtype-local wire contracts for stroboscopic Rydberg trajectories."""

from __future__ import annotations

import base64
from typing import Literal

import numpy as np
from pydantic import Field

from qiqcbench.qsim.core.wire import SCHEMA_VERSION, _Strict, _StrictRequest


class RydbergTrajectoryRequest(_StrictRequest):
    schema_version: int = SCHEMA_VERSION
    protocol_id: Literal["depth6_rydberg_ising_v2"] = "depth6_rydberg_ising_v2"
    system_size: int = Field(..., ge=2, le=121)
    shots: int = Field(..., ge=1, le=50_000)


class JobRydbergTrajectoryData(_Strict):
    kind: Literal["rydberg_stroboscopic_trajectory"] = "rydberg_stroboscopic_trajectory"
    protocol_id: str
    system_size: int
    shots: int
    bit_order: Literal["little"] = "little"
    #: Absolute indices of the two atoms whose state is read out, in the order their bits
    #: appear in ``target_z_readout_b64``.
    target_sites: list[int] = Field(default_factory=list)
    # Small engine-only uses may keep arrays inline. Run-backed jobs externalize
    # them into the public qsim-log volume and return only ``raw_data_file``.
    #: ``(shots, system_size)`` occupancy after loading, before the sequence.
    load_occupancy_b64: str | None = None
    #: ``(shots, system_size)`` occupancy from the mid-sequence erasure image.
    midsequence_occupancy_b64: str | None = None
    #: ``(shots, system_size)`` occupancy after the sequence.
    final_occupancy_b64: str | None = None
    #: ``(shots, 2)`` raw fluorescence bits for the two target atoms.
    target_z_readout_b64: str | None = None
    raw_data_file: str | None = None


def pack_binary(values: np.ndarray) -> str:
    packed = np.packbits(values.astype(np.uint8).reshape(-1), bitorder="little")
    return base64.b64encode(packed.tobytes()).decode()


def unpack_binary(encoded: str, *, shots: int, width: int) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(encoded.encode()), dtype=np.uint8)
    bits = np.unpackbits(raw, bitorder="little")[: shots * width]
    return bits.reshape(shots, width).astype(bool)


__all__ = ["JobRydbergTrajectoryData", "RydbergTrajectoryRequest", "pack_binary", "unpack_binary"]
