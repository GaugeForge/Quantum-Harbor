"""Run-level equality proofs for the qsim device models used in one execution."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from qiqcbench.qsim.execution_context import validate_hidden_commitment_secret

HIDDEN_DEVICE_MODEL_COMMITMENT_NAME = "qiqcbench_hidden_device_model_hmac_v1"
PUBLIC_DEVICE_MODEL_COMMITMENT_NAME = "qiqcbench_public_device_model_sha256_v1"


class PublicDeviceModelCommitment(BaseModel):
    """Public, task-bound digest of the validated qsim device model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["qiqcbench_public_device_model_sha256_v1"] = PUBLIC_DEVICE_MODEL_COMMITMENT_NAME
    schema_version: Literal[1] = 1
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class HiddenDeviceModelCommitment(BaseModel):
    """Opaque equality proof; it reveals neither hidden bytes nor a stable digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["qiqcbench_hidden_device_model_hmac_v1"] = HIDDEN_DEVICE_MODEL_COMMITMENT_NAME
    schema_version: Literal[1] = 1
    hmac_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _model_payload(model: Any) -> Any:
    try:
        return model.model_dump(mode="json")
    except TypeError:
        # Small protocol fakes in state tests may expose model_dump() without
        # Pydantic's mode keyword; production configs are Pydantic models.
        return model.model_dump()


def public_device_model_commitment(
    public: Any,
    *,
    task_id: str,
) -> dict[str, Any]:
    """Commit to one validated public model for one task execution."""

    if not isinstance(task_id, str) or not task_id:
        raise ValueError("public device commitment requires a task ID")
    envelope = {
        "name": PUBLIC_DEVICE_MODEL_COMMITMENT_NAME,
        "schema_version": 1,
        "task_id": task_id,
        "public_device": _model_payload(public),
    }
    digest = hashlib.sha256(_canonical_json_bytes(envelope)).hexdigest()
    return PublicDeviceModelCommitment(sha256=digest).model_dump(mode="json")


def hidden_device_model_commitment(
    hidden: Any,
    *,
    task_id: str,
    commitment_secret: str,
) -> dict[str, Any]:
    """Commit to one validated hidden model under an execution-scoped HMAC key."""

    if not isinstance(task_id, str) or not task_id:
        raise ValueError("hidden device commitment requires a task ID")
    secret = validate_hidden_commitment_secret(commitment_secret)
    envelope = {
        "name": HIDDEN_DEVICE_MODEL_COMMITMENT_NAME,
        "schema_version": 1,
        "task_id": task_id,
        "hidden_device": _model_payload(hidden),
    }
    digest = hmac.new(
        bytes.fromhex(secret),
        _canonical_json_bytes(envelope),
        hashlib.sha256,
    ).hexdigest()
    return HiddenDeviceModelCommitment(hmac_sha256=digest).model_dump(mode="json")


__all__ = [
    "HIDDEN_DEVICE_MODEL_COMMITMENT_NAME",
    "PUBLIC_DEVICE_MODEL_COMMITMENT_NAME",
    "HiddenDeviceModelCommitment",
    "PublicDeviceModelCommitment",
    "hidden_device_model_commitment",
    "public_device_model_commitment",
]
