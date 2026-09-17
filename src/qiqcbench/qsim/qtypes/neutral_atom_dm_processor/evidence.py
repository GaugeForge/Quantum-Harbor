"""Canonical evidence records for ``neutral_atom_dm_processor`` (actions + scorers).

Differences from the sibling ``neutral_atom_logical_processor`` contract, per
the implementation-plan review:
  - The binding digest is over the FULL request (ops + layout + idle_scale +
    noise_scale + sweep parameter/values), not ops alone — evidence gathered at
    one noise_scale must not be citable as another.
  - The artifact is a union over the single-program and sweep tools; the sweep
    artifact keeps the lossless per-point raw bits even though the agent-facing
    job result carries only a raw-file pointer.
"""

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
from qiqcbench.qsim.qtypes.neutral_atom_dm_processor.wire import (
    AtomOp,
    AtomProgramRequest,
    AtomProgramSweepRequest,
    JobAtomDmShotData,
    JobAtomDmSweepData,
)

NEUTRAL_ATOM_DM_EVIDENCE_CONTRACT = "neutral_atom_dm_program_evidence_v1"
# Schema 2: ``measure.expect`` heralded-abort verification and
# the per-shot ``aborted`` masks in the result records. Version 1 evidence is not read by
# the schema-2 scorer (qsim and verifier images must be built from the same commit).
NEUTRAL_ATOM_DM_EVIDENCE_SCHEMA_VERSION = 2
PUBLIC_DEVICE_MODEL_COMMITMENT_NAME = "neutral_atom_dm_public_device_model_v1"
NEUTRAL_ATOM_DM_RESULT_ARTIFACT_KIND = "neutral_atom_dm_program_result_v1"
NEUTRAL_ATOM_DM_RESULT_REF_KIND = "neutral_atom_dm_program_result_ref_v1"
NEUTRAL_ATOM_DM_RESULT_ARTIFACT_SCHEMA_VERSION = 2
MAX_NEUTRAL_ATOM_DM_RESULT_ARTIFACT_BYTES = 256 * 1024 * 1024
NEUTRAL_ATOM_DM_INFRASTRUCTURE_FAILURE = "neutral_atom_dm_infrastructure_failure_v1"
EVIDENCE_DIRNAME = "neutral_atom_dm_evidence"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class NeutralAtomDmResultArtifact(_StrictModel):
    """Lossless qsim-owned request/result record for one completed program or sweep."""

    artifact_kind: Literal["neutral_atom_dm_program_result_v1"] = (
        NEUTRAL_ATOM_DM_RESULT_ARTIFACT_KIND
    )
    schema_version: Literal[2] = NEUTRAL_ATOM_DM_RESULT_ARTIFACT_SCHEMA_VERSION
    evidence_contract: Literal["neutral_atom_dm_program_evidence_v1"] = (
        NEUTRAL_ATOM_DM_EVIDENCE_CONTRACT
    )
    evidence_schema_version: Literal[2] = NEUTRAL_ATOM_DM_EVIDENCE_SCHEMA_VERSION
    job_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    tool: Literal["run_atom_program", "run_atom_program_sweep"]
    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    public_device_model_commitment: dict[str, Any]
    hidden_device_model_commitment: HiddenDeviceModelCommitment | None = None
    request: AtomProgramSweepRequest | AtomProgramRequest
    result_status: Literal["complete"] = "complete"
    result: JobAtomDmSweepData | JobAtomDmShotData

    @model_validator(mode="after")
    def _request_and_result_agree(self) -> NeutralAtomDmResultArtifact:
        if self.request_digest != request_digest(self.request):
            raise ValueError("atom-dm result artifact request digest does not match request")
        expected_labels = program_measurement_labels(self.request.ops)
        declares_expect = program_declares_expect(self.request.ops)
        if isinstance(self.request, AtomProgramSweepRequest):
            if self.tool != "run_atom_program_sweep":
                raise ValueError("sweep request must be recorded by the sweep tool")
            if not isinstance(self.result, JobAtomDmSweepData):
                raise ValueError("sweep request requires a sweep result")
            if len(self.result.points) != len(self.request.sweep_values):
                raise ValueError("sweep artifact point count does not match sweep values")
            if self.result.point_bits_b64 is None or self.result.point_executed_b64 is None:
                raise ValueError("sweep artifact must keep the lossless per-point raw bits")
            if len(self.result.point_bits_b64) != len(self.request.sweep_values):
                raise ValueError("sweep artifact raw point count does not match sweep values")
            if declares_expect and (
                self.result.point_aborted_b64 is None
                or len(self.result.point_aborted_b64) != len(self.request.sweep_values)
            ):
                raise ValueError(
                    "sweep artifact of a program with expected-outcome readouts must keep "
                    "the per-point aborted masks"
                )
            for point, value in zip(self.result.points, self.request.sweep_values, strict=True):
                if point.value != value or point.shots != self.request.shots:
                    raise ValueError("sweep artifact point summary does not match request")
        else:
            if self.tool != "run_atom_program":
                raise ValueError("single-program request must be recorded by run_atom_program")
            if not isinstance(self.result, JobAtomDmShotData):
                raise ValueError("single-program request requires a shot-record result")
            if self.request.shots != self.result.shots:
                raise ValueError("atom-dm result artifact request/result shots differ")
            if declares_expect and self.result.aborted_b64 is None:
                raise ValueError(
                    "artifact of a program with expected-outcome readouts must keep the "
                    "aborted mask"
                )
        if self.result.measure_labels != expected_labels:
            raise ValueError("atom-dm result artifact measurement labels do not match request")
        if self.result.n_recorded != len(expected_labels):
            raise ValueError("atom-dm result artifact n_recorded does not match request")
        return self


class NeutralAtomDmResultArtifactRef(_StrictModel):
    """Content-addressed reference written into the bounded JSONL event."""

    kind: Literal["neutral_atom_dm_program_result_ref_v1"] = NEUTRAL_ATOM_DM_RESULT_REF_KIND
    schema_version: Literal[2] = NEUTRAL_ATOM_DM_RESULT_ARTIFACT_SCHEMA_VERSION
    relative_path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0, le=MAX_NEUTRAL_ATOM_DM_RESULT_ARTIFACT_BYTES)

    @model_validator(mode="after")
    def _path_is_the_content_address(self) -> NeutralAtomDmResultArtifactRef:
        expected = f"{EVIDENCE_DIRNAME}/{self.sha256}.json"
        if self.relative_path != expected:
            raise ValueError("neutral-atom-dm evidence path must match its content digest")
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


def request_digest(request: AtomProgramRequest | AtomProgramSweepRequest) -> str:
    """Digest binding evidence to the FULL physics-bearing request.

    Includes ops, layout, idle_scale, noise_scale, and (for sweeps) the swept
    parameter and values. Excludes ``shots`` deliberately: shot count changes
    statistics, not physics, and the artifact separately records it.
    """
    payload = request.model_dump(mode="json", exclude_none=True)
    payload.pop("shots", None)
    payload.pop("schema_version", None)
    payload["request_kind"] = "sweep" if isinstance(request, AtomProgramSweepRequest) else "single"
    return canonical_json_sha256(payload)


def program_declares_expect(ops: list[dict] | list[AtomOp]) -> bool:
    """True when any measure (conditioned ones included) carries an expected outcome."""
    for op in ops:
        validated = op if isinstance(op, AtomOp) else AtomOp.model_validate(op)
        measured = (
            validated.then
            if validated.type == "feedforward" and validated.then is not None
            else validated
        )
        if measured.type == "measure" and measured.expect is not None:
            return True
    return False


def program_measurement_labels(ops: list[dict] | list[AtomOp]) -> list[str]:
    """Derive the fixed rectangular result-column contract from a program.

    Conditional (feedforward-nested) measures ALWAYS contribute their columns —
    the always-record semantics that keeps decode bit indices well-defined.
    Labels are ``"<op_index>:<atom>"`` (Z basis only).
    """
    validated = [op if isinstance(op, AtomOp) else AtomOp.model_validate(op) for op in ops]
    labels: list[str] = []
    for index, op in enumerate(validated):
        measured = op
        if op.type == "feedforward" and op.then is not None:
            measured = op.then
        if measured.type != "measure":
            continue
        labels.extend(f"{index}:{atom}" for atom in measured.atoms or [])
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
    artifact: NeutralAtomDmResultArtifact,
) -> NeutralAtomDmResultArtifactRef:
    """Atomically publish one verifier-readable artifact below the qsim log root."""
    root = Path(log_dir).resolve()
    artifact_dir = root / EVIDENCE_DIRNAME
    artifact_dir.mkdir(parents=True, exist_ok=True)
    resolved_dir = artifact_dir.resolve()
    try:
        resolved_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("neutral-atom-dm evidence directory escapes qsim log root") from exc

    payload = canonical_json_bytes(artifact.model_dump(mode="json"))
    if len(payload) > MAX_NEUTRAL_ATOM_DM_RESULT_ARTIFACT_BYTES:
        raise ValueError(
            "neutral-atom-dm result artifact exceeds the per-job evidence limit; "
            "split the experiment into smaller jobs"
        )
    digest = hashlib.sha256(payload).hexdigest()
    target = resolved_dir / f"{digest}.json"
    if target.is_symlink():
        raise ValueError("neutral-atom-dm evidence target must not be a symlink")
    if target.exists():
        if target.read_bytes() != payload:
            raise ValueError("content-addressed neutral-atom-dm evidence collision")
    else:
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                "wb",
                dir=resolved_dir,
                prefix=".neutral_atom_dm_evidence.",
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
    return NeutralAtomDmResultArtifactRef(
        relative_path=relative_path,
        sha256=digest,
        size_bytes=len(payload),
    )


def load_result_artifact(
    log_root: Path,
    raw_ref: dict[str, Any],
) -> NeutralAtomDmResultArtifact:
    """Resolve, verify, and strictly parse one qsim-owned result artifact."""
    ref = NeutralAtomDmResultArtifactRef.model_validate(raw_ref)
    root = Path(log_root).resolve()
    relative = Path(ref.relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("neutral-atom-dm evidence reference must be a confined relative path")
    candidate = root / relative
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("neutral-atom-dm evidence path must not contain symlinks")
    path = candidate.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("neutral-atom-dm evidence reference escapes qsim log root") from exc
    try:
        metadata = path.stat()
    except FileNotFoundError as exc:
        raise ValueError("neutral-atom-dm evidence artifact is missing") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("neutral-atom-dm evidence artifact is missing")
    if metadata.st_size != ref.size_bytes:
        raise ValueError("neutral-atom-dm evidence artifact size does not match reference")
    with path.open("rb") as handle:
        payload = handle.read(ref.size_bytes + 1)
    if len(payload) != ref.size_bytes:
        raise ValueError("neutral-atom-dm evidence artifact size changed while reading")
    if hashlib.sha256(payload).hexdigest() != ref.sha256:
        raise ValueError("neutral-atom-dm evidence artifact digest does not match reference")
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("neutral-atom-dm evidence artifact is not canonical JSON") from exc
    artifact = NeutralAtomDmResultArtifact.model_validate(raw)
    if canonical_json_bytes(artifact.model_dump(mode="json")) != payload:
        raise ValueError("neutral-atom-dm evidence artifact is not canonically encoded")
    return artifact


__all__ = [
    "EVIDENCE_DIRNAME",
    "HIDDEN_DEVICE_MODEL_COMMITMENT_NAME",
    "HiddenDeviceModelCommitment",
    "MAX_NEUTRAL_ATOM_DM_RESULT_ARTIFACT_BYTES",
    "NEUTRAL_ATOM_DM_EVIDENCE_CONTRACT",
    "NEUTRAL_ATOM_DM_EVIDENCE_SCHEMA_VERSION",
    "NEUTRAL_ATOM_DM_INFRASTRUCTURE_FAILURE",
    "NEUTRAL_ATOM_DM_RESULT_ARTIFACT_KIND",
    "NEUTRAL_ATOM_DM_RESULT_ARTIFACT_SCHEMA_VERSION",
    "NEUTRAL_ATOM_DM_RESULT_REF_KIND",
    "NeutralAtomDmResultArtifact",
    "NeutralAtomDmResultArtifactRef",
    "PUBLIC_DEVICE_MODEL_COMMITMENT_NAME",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "hidden_device_model_commitment",
    "load_result_artifact",
    "persist_result_artifact",
    "program_declares_expect",
    "program_measurement_labels",
    "public_device_model_commitment",
    "request_digest",
]
