"""Canonical, versioned evidence commitments for bosonic-cavity runs."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel

BOSONIC_EVIDENCE_CONTRACT = "bosonic_program_evidence_v2"
BOSONIC_EVIDENCE_SCHEMA_VERSION = 2
BOSONIC_EXECUTION_ARTIFACT_KIND = "bosonic_program_execution_v1"
BOSONIC_EXECUTION_ARTIFACT_MEDIA_TYPE = (
    "application/vnd.qiqcbench.bosonic-program-execution+json;version=1"
)
PUBLIC_DEVICE_MODEL_COMMITMENT_NAME = "bosonic_public_device_model_v1"
SWEEP_COORDINATES_COMMITMENT_NAME = "bosonic_program_sweep_coordinates_v1"


def canonical_json_bytes(payload: Any) -> bytes:
    """Return the canonical JSON encoding used by bosonic evidence contracts."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    """Return SHA-256 over the canonical JSON encoding used by this contract."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def public_device_model_commitment(public: BaseModel) -> dict[str, Any]:
    """Commit to the validated public model that the running qsim actually uses."""
    envelope = {
        "name": PUBLIC_DEVICE_MODEL_COMMITMENT_NAME,
        "schema_version": 1,
        "public_device": public.model_dump(mode="json"),
    }
    return {
        "name": PUBLIC_DEVICE_MODEL_COMMITMENT_NAME,
        "schema_version": 1,
        "sha256": canonical_json_sha256(envelope),
    }


def sweep_coordinates_commitment(
    *, sweep_op_index: int, sweep_field: str, sweep_values: list[float]
) -> dict[str, Any]:
    """Expose exact executed sweep coordinates plus their canonical commitment."""
    payload = {
        "name": SWEEP_COORDINATES_COMMITMENT_NAME,
        "schema_version": 1,
        "sweep_op_index": int(sweep_op_index),
        "sweep_field": str(sweep_field),
        "sweep_values": [float(value) for value in sweep_values],
    }
    return {**payload, "sha256": canonical_json_sha256(payload)}


__all__ = [
    "BOSONIC_EXECUTION_ARTIFACT_KIND",
    "BOSONIC_EXECUTION_ARTIFACT_MEDIA_TYPE",
    "BOSONIC_EVIDENCE_CONTRACT",
    "BOSONIC_EVIDENCE_SCHEMA_VERSION",
    "PUBLIC_DEVICE_MODEL_COMMITMENT_NAME",
    "SWEEP_COORDINATES_COMMITMENT_NAME",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "public_device_model_commitment",
    "sweep_coordinates_commitment",
]
