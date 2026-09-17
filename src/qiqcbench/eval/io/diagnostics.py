"""Public-safe structured diagnostics for evaluation artifact scans."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import field_validator

from qiqcbench.eval.contracts.base import Digest, Identifier, NonEmptyStr, StrictModel


class ArtifactScanDiagnostic(StrictModel):
    """One terminal decision for a trial-relative artifact claim."""

    schema_version: Literal[1] = 1
    stage: Literal["discovery", "legacy_projection", "semantic_validation"]
    subject_reference: NonEmptyStr
    artifact_kind: Identifier
    artifact_role: Identifier | None = None
    decision: Literal["included", "development_only", "excluded", "quarantined"]
    reason_code: Identifier
    verification_id: Identifier | None = None
    digest: Digest | None = None

    @field_validator("subject_reference")
    @classmethod
    def _subject_is_safe_relative(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            value.startswith(("/", "~"))
            or "\\" in value
            or "://" in value
            or path.is_absolute()
            or ".." in path.parts
            or path in {PurePosixPath("."), PurePosixPath("..")}
        ):
            raise ValueError("scan diagnostic subject must be a safe relative reference")
        return value


__all__ = ["ArtifactScanDiagnostic"]
