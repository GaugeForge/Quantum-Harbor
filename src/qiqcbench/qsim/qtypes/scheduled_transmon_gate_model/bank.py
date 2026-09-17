"""Public scheduled-candidate schemas and canonical semantic-digest helpers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScheduledNativeGate(_Strict):
    """One native gate with its explicit layer and wall-clock placement."""

    name: Literal["rz", "sx", "x", "ecr"]
    qubits: list[int] = Field(..., min_length=1, max_length=2)
    params: list[float] = Field(default_factory=list, max_length=1)
    layer_index: int = Field(..., ge=0)
    start_ns: int = Field(..., ge=0)
    duration_ns: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _shape_matches_gate(self) -> ScheduledNativeGate:
        if self.name == "ecr":
            if len(self.qubits) != 2 or self.qubits[0] == self.qubits[1] or self.params:
                raise ValueError("ecr requires two distinct qubits and no params")
        elif len(self.qubits) != 1:
            raise ValueError(f"{self.name} requires exactly one qubit")
        elif self.name == "rz" and len(self.params) != 1:
            raise ValueError("rz requires one angle parameter")
        elif self.name != "rz" and self.params:
            raise ValueError(f"{self.name} does not accept params")
        return self


class ScheduledLayer(_Strict):
    layer_index: int = Field(..., ge=0)
    start_ns: int = Field(..., ge=0)
    duration_ns: int = Field(..., ge=0)
    gate_indices: list[int] = Field(default_factory=list)


class CandidateStructure(_Strict):
    rz_count: int = Field(..., ge=0)
    sx_count: int = Field(..., ge=0)
    x_count: int = Field(..., ge=0)
    ecr_count: int = Field(..., ge=0)
    ecr_depth: int = Field(..., ge=0)
    scheduled_depth: int = Field(..., ge=0)
    duration_ns: int = Field(..., ge=0)
    stale_estimated_success: float = Field(..., gt=0.0, le=1.0)


class CompiledCandidate(_Strict):
    candidate_id: str = Field(..., pattern=r"^cand_[0-9]{2}$")
    initial_logical_to_physical: list[int] = Field(..., min_length=1)
    output_logical_to_physical: list[int] = Field(..., min_length=1)
    gates: list[ScheduledNativeGate] = Field(..., min_length=1)
    layers: list[ScheduledLayer] = Field(..., min_length=1)
    structure: CandidateStructure
    candidate_digest: str

    @model_validator(mode="after")
    def _schedule_is_closed(self) -> CompiledCandidate:
        n_gates = len(self.gates)
        seen: list[int] = []
        previous_start = -1
        for expected, layer in enumerate(self.layers):
            if layer.layer_index != expected:
                raise ValueError("candidate layers must be contiguous from zero")
            if layer.start_ns < previous_start:
                raise ValueError("candidate layer start times must be monotone")
            previous_start = layer.start_ns
            seen.extend(layer.gate_indices)
            for gate_index in layer.gate_indices:
                if gate_index < 0 or gate_index >= n_gates:
                    raise ValueError("candidate layer references an invalid gate index")
                gate = self.gates[gate_index]
                if (
                    gate.layer_index != layer.layer_index
                    or gate.start_ns != layer.start_ns
                    or gate.duration_ns > layer.duration_ns
                ):
                    raise ValueError("candidate gate placement disagrees with its layer")
        if sorted(seen) != list(range(n_gates)):
            raise ValueError("candidate layers must reference every gate exactly once")
        if len(set(self.initial_logical_to_physical)) != len(self.initial_logical_to_physical):
            raise ValueError("initial layout must not repeat physical qubits")
        if len(set(self.output_logical_to_physical)) != len(self.output_logical_to_physical):
            raise ValueError("output layout must not repeat physical qubits")
        if len(self.initial_logical_to_physical) != len(self.output_logical_to_physical):
            raise ValueError("initial and output layouts must have equal width")
        return self


class CandidateBank(_Strict):
    schema_version: Literal[1] = 1
    target_name: str
    target_clifford_sha256: str
    qiskit_version: str
    native_gate_order: list[str]
    measured_physical_qubits: list[int] = Field(..., min_length=1)
    candidates: list[CompiledCandidate] = Field(..., min_length=1)
    bank_digest: str

    @model_validator(mode="after")
    def _candidate_ids_are_unique(self) -> CandidateBank:
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate IDs must be unique")
        if len(set(self.measured_physical_qubits)) != len(self.measured_physical_qubits):
            raise ValueError("measured physical qubits must be unique")
        return self

    def by_id(self) -> dict[str, CompiledCandidate]:
        return {candidate.candidate_id: candidate for candidate in self.candidates}


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def candidate_digest(candidate: CompiledCandidate | dict) -> str:
    """Hash canonical candidate JSON after removing its self-digest field."""

    payload = (
        candidate.model_dump(mode="json")
        if isinstance(candidate, CompiledCandidate)
        else dict(candidate)
    )
    payload.pop("candidate_digest", None)
    return sha256_json(payload)


def bank_digest(bank: CandidateBank | dict) -> str:
    """Hash canonical bank JSON after removing its self-digest field.

    This semantic commitment is intentionally not a digest of serialized file bytes.
    """

    payload = bank.model_dump(mode="json") if isinstance(bank, CandidateBank) else dict(bank)
    payload.pop("bank_digest", None)
    return sha256_json(payload)


def validate_bank_digests(bank: CandidateBank) -> None:
    if _SHA256.fullmatch(bank.target_clifford_sha256) is None:
        raise ValueError("target_clifford_sha256 is not a lowercase SHA-256 digest")
    for candidate in bank.candidates:
        if _SHA256.fullmatch(candidate.candidate_digest) is None:
            raise ValueError(f"{candidate.candidate_id} has an invalid candidate digest")
        actual = candidate_digest(candidate)
        if actual != candidate.candidate_digest:
            raise ValueError(
                f"{candidate.candidate_id} digest mismatch: {actual} != "
                f"{candidate.candidate_digest}"
            )
    actual_bank = bank_digest(bank)
    if actual_bank != bank.bank_digest:
        raise ValueError(f"candidate bank digest mismatch: {actual_bank} != {bank.bank_digest}")


def load_candidate_bank(path: str | Path, *, expected_digest: str | None = None) -> CandidateBank:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    bank = CandidateBank.model_validate(payload)
    validate_bank_digests(bank)
    if expected_digest is not None and bank.bank_digest != expected_digest:
        raise ValueError(
            f"candidate bank does not match public device commitment: "
            f"{bank.bank_digest} != {expected_digest}"
        )
    return bank


__all__ = [
    "CandidateBank",
    "CandidateStructure",
    "CompiledCandidate",
    "ScheduledLayer",
    "ScheduledNativeGate",
    "bank_digest",
    "candidate_digest",
    "canonical_json_bytes",
    "load_candidate_bank",
    "sha256_json",
    "validate_bank_digests",
]
