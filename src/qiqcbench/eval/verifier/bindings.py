"""Shared identity/evidence binding for legacy verifier dual-write adapters."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from qiqcbench.eval.contracts.identity import (
    TaskEditionManifest,
    VerifierRevisionManifest,
)
from qiqcbench.eval.contracts.outcome import (
    EvidenceInput,
    VerificationManifest,
    verification_manifest_id,
)
from qiqcbench.eval.io.canonical import sha256_bytes
from qiqcbench.eval.io.config import (
    load_task_edition,
    load_verifier_revision,
)
from qiqcbench.eval.verifier.context import VerifierRunContext

_QSIM_BOOTSTRAP_ACTION = "qsim_bootstrap_context"
_QSIM_CONTEXT_SCHEMA_VERSION = 1


class EvidenceIntegrityError(ValueError):
    """Preserved execution evidence contradicts its declared runtime binding."""


@dataclass(frozen=True, slots=True)
class BoundVerification:
    """Catalog manifests and one input-derived verification identity."""

    task_edition: TaskEditionManifest
    verifier_revision: VerifierRevisionManifest
    verification: VerificationManifest


def existing_trial_reference(
    path: Path,
    *,
    trial_root: Path,
    role: str,
) -> tuple[str, bytes]:
    try:
        resolved_root = trial_root.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
        reference = resolved_path.relative_to(resolved_root).as_posix()
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise ValueError(f"{role} must be an existing file below trial_root") from exc
    if not resolved_path.is_file() or path.is_symlink():
        raise ValueError(f"{role} must be a non-symlink regular file below trial_root")
    return reference, resolved_path.read_bytes()


def target_trial_reference(
    path: Path,
    *,
    trial_root: Path,
    role: str,
) -> str:
    try:
        root = trial_root.absolute()
        target = path.absolute()
        reference = target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{role} must be below trial_root") from exc
    if reference == Path(".") or not reference.parts:
        raise ValueError(f"{role} must identify a file below trial_root")
    return reference.as_posix()


def bind_verification(
    *,
    context: VerifierRunContext,
    trial_root: Path,
    evidence_paths: Mapping[str, str | os.PathLike[str]],
    config_root: str | os.PathLike[str] | None,
    evidence_provenance: str = "execution_evidence",
) -> BoundVerification:
    """Bind catalog policy and exact preserved execution evidence once.

    Hidden construction/config/scorer inputs are not execution evidence.  They
    must remain in private ``InstanceBinding`` or audit state and are rejected
    here so low-entropy hidden digests cannot influence a public verification
    identity.
    """

    if evidence_provenance != "execution_evidence":
        raise ValueError("verification identity accepts only execution_evidence provenance")

    edition = load_task_edition(
        context.task_id,
        context.task_edition_name,
        root=config_root,
    )
    if edition.task_edition_id != context.attempt.task_edition_id:
        raise ValueError("verifier context TaskEdition ID does not match targeted catalog entry")
    revision = load_verifier_revision(
        context.task_id,
        context.verifier_revision_name,
        root=config_root,
        task_edition=edition,
    )
    if revision.verifier_revision_id != context.verifier_revision_id:
        raise ValueError("verifier context revision ID does not match targeted catalog entry")
    evidence_snapshots: list[tuple[str, str, bytes]] = []
    for role, path in sorted(evidence_paths.items()):
        try:
            reference, content = existing_trial_reference(
                Path(path),
                trial_root=trial_root,
                role=f"evidence {role!r}",
            )
        except ValueError as exc:
            if (
                context.evidence_binding_mode == "qsim_context_v1"
                and role == context.qsim_log_evidence_role
            ):
                raise EvidenceIntegrityError(
                    "declared qsim log role must reference preserved execution evidence"
                ) from exc
            raise
        evidence_snapshots.append((role, reference, content))

    if context.evidence_binding_mode == "qsim_context_v1":
        qsim_log_payload = next(
            (
                content
                for role, _reference, content in evidence_snapshots
                if role == context.qsim_log_evidence_role
            ),
            None,
        )
        if qsim_log_payload is None:
            raise EvidenceIntegrityError(
                "strict evidence binding requires the declared qsim log role"
            )
        _validate_qsim_context_log(
            qsim_log_payload,
            expected_context_id=context.execution.qsim_evidence_nonce,
        )

    evidence_inputs: list[EvidenceInput] = []
    for role, reference, content in evidence_snapshots:
        evidence_inputs.append(
            EvidenceInput(
                role=role,
                digest=sha256_bytes(content),
                reference=reference,
                provenance="execution_evidence",
            )
        )
    if not evidence_inputs:
        raise ValueError("v2 verification requires at least one preserved scoring input")
    evidence = tuple(evidence_inputs)
    verification_id = verification_manifest_id(
        execution_id=context.execution.execution_id,
        task_edition_id=edition.task_edition_id,
        verifier_revision_id=revision.verifier_revision_id,
        reason=context.reason,
        predecessor_verification_id=context.predecessor_verification_id,
        evidence_inputs=evidence,
    )
    verification = VerificationManifest(
        verification_id=verification_id,
        execution_id=context.execution.execution_id,
        task_edition_id=edition.task_edition_id,
        verifier_revision_id=revision.verifier_revision_id,
        reason=context.reason,
        predecessor_verification_id=context.predecessor_verification_id,
        evidence_inputs=evidence,
    )
    return BoundVerification(
        task_edition=edition,
        verifier_revision=revision,
        verification=verification,
    )


def _duplicate_checked_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _validate_qsim_context_log(
    payload: bytes,
    *,
    expected_context_id: str,
) -> None:
    """Fail closed unless every qsim event binds one bootstrap context."""

    if not payload:
        raise EvidenceIntegrityError("qsim context evidence must be non-empty")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceIntegrityError("qsim context evidence must be UTF-8 JSONL") from exc

    events: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            event = json.loads(
                line,
                object_pairs_hook=_duplicate_checked_object,
                parse_constant=_reject_constant,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise EvidenceIntegrityError(
                f"qsim event line {line_number} must be a strict JSON object"
            ) from exc
        if not isinstance(event, dict):
            raise EvidenceIntegrityError(
                f"qsim event line {line_number} must be a strict JSON object"
            )
        events.append(event)

    if not events:
        raise EvidenceIntegrityError("qsim context evidence must be non-empty")
    first = events[0]
    if first.get("action") != _QSIM_BOOTSTRAP_ACTION:
        raise EvidenceIntegrityError("first qsim event must bind the bootstrap context")
    context_schema_version = first.get("context_schema_version")
    if (
        type(context_schema_version) is not int
        or context_schema_version != _QSIM_CONTEXT_SCHEMA_VERSION
    ):
        raise EvidenceIntegrityError("first qsim event has an unsupported context schema version")
    for line_number, event in enumerate(events, start=1):
        if event.get("execution_context_id") != expected_context_id:
            raise EvidenceIntegrityError(
                f"qsim event line {line_number} has an execution context mismatch"
            )


def resolve_predecessor_verification(
    *,
    context: VerifierRunContext,
    trial_root: Path,
    supplied: VerificationManifest | None = None,
) -> VerificationManifest | None:
    """Resolve a rescore predecessor from its immutable private artifact set."""

    if context.reason == "initial":
        if supplied is not None:
            raise ValueError("initial verification cannot accept a predecessor")
        return None
    predecessor_id = context.predecessor_verification_id
    if predecessor_id is None:
        raise ValueError("rescore context requires predecessor_verification_id")
    if supplied is not None:
        if supplied.verification_id != predecessor_id:
            raise ValueError("supplied predecessor does not match rescore context")
        return supplied

    from qiqcbench.eval.io.artifacts import read_private_verification_artifacts

    return read_private_verification_artifacts(
        trial_root,
        predecessor_id,
    ).verification


__all__ = [
    "BoundVerification",
    "EvidenceIntegrityError",
    "bind_verification",
    "existing_trial_reference",
    "resolve_predecessor_verification",
    "target_trial_reference",
]
