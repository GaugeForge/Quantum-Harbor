"""Canonical neutral-atom evidence records shared by actions and scorers."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from qiqcbench.qsim.hidden_commitment import (
    HIDDEN_DEVICE_MODEL_COMMITMENT_NAME,
    HiddenDeviceModelCommitment,
    hidden_device_model_commitment,
)
from qiqcbench.qsim.qtypes.neutral_atom_logical_processor.wire import (
    AtomOp,
    AtomProgramRequest,
    JobAtomShotData,
)

NEUTRAL_ATOM_EVIDENCE_CONTRACT = "neutral_atom_program_evidence_v1"
NEUTRAL_ATOM_EVIDENCE_SCHEMA_VERSION = 1
PUBLIC_DEVICE_MODEL_COMMITMENT_NAME = "neutral_atom_public_device_model_v1"
NEUTRAL_ATOM_RESULT_ARTIFACT_KIND = "neutral_atom_program_result_v1"
NEUTRAL_ATOM_RESULT_REF_KIND = "neutral_atom_program_result_ref_v1"
NEUTRAL_ATOM_RESULT_ARTIFACT_SCHEMA_VERSION = 1
MAX_NEUTRAL_ATOM_RESULT_ARTIFACT_BYTES = 256 * 1024 * 1024
NEUTRAL_ATOM_INFRASTRUCTURE_FAILURE = "neutral_atom_infrastructure_failure_v1"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NeutralAtomResultArtifact(_StrictModel):
    """Lossless qsim-owned request/result record for one completed atom program."""

    artifact_kind: Literal["neutral_atom_program_result_v1"] = NEUTRAL_ATOM_RESULT_ARTIFACT_KIND
    schema_version: Literal[1] = NEUTRAL_ATOM_RESULT_ARTIFACT_SCHEMA_VERSION
    evidence_contract: Literal["neutral_atom_program_evidence_v1"] = NEUTRAL_ATOM_EVIDENCE_CONTRACT
    evidence_schema_version: Literal[1] = NEUTRAL_ATOM_EVIDENCE_SCHEMA_VERSION
    job_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    tool: Literal["run_atom_program"] = "run_atom_program"
    program_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_device_model_commitment: dict[str, Any]
    hidden_device_model_commitment: HiddenDeviceModelCommitment | None = None
    request: AtomProgramRequest
    result_status: Literal["complete"] = "complete"
    result: JobAtomShotData

    @model_validator(mode="after")
    def _request_and_result_agree(self) -> NeutralAtomResultArtifact:
        if self.request.shots != self.result.shots:
            raise ValueError("atom result artifact request/result shots differ")
        if self.program_digest != program_digest(self.request.ops):
            raise ValueError("atom result artifact program digest does not match request")
        expected_labels = program_measurement_labels(self.request.ops)
        if self.result.measure_labels != expected_labels:
            raise ValueError("atom result artifact measurement labels do not match request")
        if self.result.n_recorded != len(expected_labels):
            raise ValueError("atom result artifact n_recorded does not match request")
        return self


class NeutralAtomResultArtifactRef(_StrictModel):
    """Content-addressed reference written into the bounded JSONL event."""

    kind: Literal["neutral_atom_program_result_ref_v1"] = NEUTRAL_ATOM_RESULT_REF_KIND
    schema_version: Literal[1] = NEUTRAL_ATOM_RESULT_ARTIFACT_SCHEMA_VERSION
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0, le=MAX_NEUTRAL_ATOM_RESULT_ARTIFACT_BYTES)

    @model_validator(mode="after")
    def _path_is_the_content_address(self) -> NeutralAtomResultArtifactRef:
        expected = f"neutral_atom_evidence/{self.sha256}.json"
        if self.relative_path != expected:
            raise ValueError("neutral-atom evidence path must match its content digest")
        return self


def canonical_json_sha256(payload: Any) -> str:
    """Return SHA-256 over this evidence contract's canonical JSON encoding."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    """Return the exact canonical bytes used for content-addressed artifacts."""
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def program_digest(ops: list[dict] | list[AtomOp]) -> str:
    """Return the stable digest used to bind atom programs to logged evidence."""
    canonical = [
        (op if isinstance(op, AtomOp) else AtomOp.model_validate(op)).model_dump(
            mode="json", exclude_none=True
        )
        for op in ops
    ]
    return canonical_json_sha256(canonical)


def program_measurement_labels(ops: list[dict] | list[AtomOp]) -> list[str]:
    """Derive the fixed result-column contract from a validated program."""
    validated = [op if isinstance(op, AtomOp) else AtomOp.model_validate(op) for op in ops]
    labels: list[str] = []
    for index, op in enumerate(validated):
        measured = op
        if op.type == "feedforward" and op.then is not None:
            measured = op.then
        if measured.type != "measure":
            continue
        labels.extend(f"{index}:{atom}:{measured.basis}" for atom in measured.atoms or [])
    return labels


def public_device_model_commitment(public: BaseModel) -> dict[str, Any]:
    """Commit to the validated public model used by the running qsim or verifier."""
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


def persist_result_artifact(
    log_dir: Path,
    artifact: NeutralAtomResultArtifact,
) -> NeutralAtomResultArtifactRef:
    """Atomically publish one verifier-readable artifact below the qsim log root."""
    root = Path(log_dir).resolve()
    artifact_dir = root / "neutral_atom_evidence"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    resolved_dir = artifact_dir.resolve()
    try:
        resolved_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("neutral-atom evidence directory escapes qsim log root") from exc

    payload = canonical_json_bytes(artifact.model_dump(mode="json"))
    if len(payload) > MAX_NEUTRAL_ATOM_RESULT_ARTIFACT_BYTES:
        raise ValueError(
            "neutral-atom result artifact exceeds the per-job evidence limit; "
            "split the experiment into smaller jobs"
        )
    digest = hashlib.sha256(payload).hexdigest()
    target = resolved_dir / f"{digest}.json"
    if target.is_symlink():
        raise ValueError("neutral-atom evidence target must not be a symlink")
    if target.exists():
        if target.read_bytes() != payload:
            raise ValueError("content-addressed neutral-atom evidence collision")
    else:
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                "wb",
                dir=resolved_dir,
                prefix=".neutral_atom_evidence.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
            temporary.chmod(0o644)
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    target.chmod(0o644)
    relative_path = target.relative_to(root).as_posix()
    return NeutralAtomResultArtifactRef(
        relative_path=relative_path,
        sha256=digest,
        size_bytes=len(payload),
    )


def load_result_artifact(
    log_root: Path,
    raw_ref: dict[str, Any],
) -> NeutralAtomResultArtifact:
    """Resolve, verify, and strictly parse one qsim-owned result artifact."""
    ref = NeutralAtomResultArtifactRef.model_validate(raw_ref)
    root = Path(log_root).resolve()
    relative = Path(ref.relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("neutral-atom evidence reference must be a confined relative path")
    candidate = root / relative
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("neutral-atom evidence path must not contain symlinks")
    path = candidate.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("neutral-atom evidence reference escapes qsim log root") from exc
    try:
        metadata = path.stat()
    except FileNotFoundError as exc:
        raise ValueError("neutral-atom evidence artifact is missing") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("neutral-atom evidence artifact is missing")
    if metadata.st_size != ref.size_bytes:
        raise ValueError("neutral-atom evidence artifact size does not match reference")
    with path.open("rb") as handle:
        payload = handle.read(ref.size_bytes + 1)
    if len(payload) != ref.size_bytes:
        raise ValueError("neutral-atom evidence artifact size changed while reading")
    if hashlib.sha256(payload).hexdigest() != ref.sha256:
        raise ValueError("neutral-atom evidence artifact digest does not match reference")
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("neutral-atom evidence artifact is not canonical JSON") from exc
    artifact = NeutralAtomResultArtifact.model_validate(raw)
    if canonical_json_bytes(artifact.model_dump(mode="json")) != payload:
        raise ValueError("neutral-atom evidence artifact is not canonically encoded")
    return artifact


__all__ = [
    "NEUTRAL_ATOM_EVIDENCE_CONTRACT",
    "NEUTRAL_ATOM_EVIDENCE_SCHEMA_VERSION",
    "NEUTRAL_ATOM_INFRASTRUCTURE_FAILURE",
    "NEUTRAL_ATOM_RESULT_ARTIFACT_KIND",
    "NEUTRAL_ATOM_RESULT_ARTIFACT_SCHEMA_VERSION",
    "NEUTRAL_ATOM_RESULT_REF_KIND",
    "MAX_NEUTRAL_ATOM_RESULT_ARTIFACT_BYTES",
    "HIDDEN_DEVICE_MODEL_COMMITMENT_NAME",
    "HiddenDeviceModelCommitment",
    "NeutralAtomResultArtifact",
    "NeutralAtomResultArtifactRef",
    "PUBLIC_DEVICE_MODEL_COMMITMENT_NAME",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "load_result_artifact",
    "hidden_device_model_commitment",
    "persist_result_artifact",
    "program_digest",
    "program_measurement_labels",
    "public_device_model_commitment",
]
