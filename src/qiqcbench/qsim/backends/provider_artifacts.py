"""Provider artifact schemas and deterministic replay fixture helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderStatusEvent(_Strict):
    """One provider job status observation for artifact replay/audit."""

    state: str
    ts: str


class ProviderArtifactV1(_Strict):
    """Versioned sidecar artifact for replay and live provider runs."""

    schema_version: Literal[1] = 1
    task_id: str
    device_id: str
    backend_mode: Literal["provider_replay", "live_provider"]
    provider_name: str
    backend_name: str
    qiqcbench_job_id: str
    provider_job_ids: list[str] = Field(default_factory=list)
    requested_shots: int
    returned_shots: int
    measured_qubits: list[int]
    bitstrings: list[list[str]] | None = None
    counts: list[dict[str, int]] | None = None
    status_timeline: list[ProviderStatusEvent]
    calibration_snapshot: dict[str, Any] | None = None
    usage_metadata: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None


def canonical_request_hash(request: BaseModel) -> str:
    """Hash the validated request JSON payload used by replay fixtures."""

    payload = request.model_dump(mode="json")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def replay_fixture_path(
    replay_root: str | Path,
    task_id: str,
    device_id: str,
    request: BaseModel,
    *,
    must_exist: bool = False,
) -> Path:
    """Return the whole-request replay fixture path for ``request``.

    ``must_exist=True`` gives replay backends a fail-closed read path while
    keeping the default useful for fixture authoring and deterministic tests.
    """

    path = (
        Path(replay_root)
        / task_id
        / device_id
        / f"{canonical_request_hash(request)}-shots{request.shots}.json"
    )
    if must_exist and not path.exists():
        raise FileNotFoundError(f"replay fixture not found: {path}")
    return path
