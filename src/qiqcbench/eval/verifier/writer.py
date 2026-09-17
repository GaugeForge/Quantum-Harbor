"""Canonical, immutable output writer for evaluation-v2 verifiers.

The writer binds already-computed verifier facts to exact output bytes.  It is
deliberately task-agnostic: task physics and hidden scoring remain in each
verifier, while this module validates identities, projects the declared
admission gate to Harbor reward bytes, and publishes immutable artifacts.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TypeVar

from pydantic import BaseModel

from qiqcbench.eval.contracts.campaign import AttemptManifest, ExecutionManifest
from qiqcbench.eval.contracts.identity import (
    TaskEditionManifest,
    VerifierRevisionManifest,
    validate_task_edition_manifest,
)
from qiqcbench.eval.contracts.outcome import (
    ArtifactBinding,
    ArtifactRole,
    ArtifactSetScope,
    VerificationArtifactSet,
    VerificationAuditRecord,
    VerificationManifest,
    VerifiedOutcome,
    project_public_verified_outcome,
    validate_verification_artifact_set_bindings,
    validate_verified_outcome_bindings,
)
from qiqcbench.eval.io.canonical import canonical_json_bytes, sha256_bytes

_PRIVATE_DIRECTORY = "private"
_SUPPORTED_PRIVATE_DIRECTORIES = frozenset(
    {
        _PRIVATE_DIRECTORY,
        "verifier/artifacts/private",
    }
)
_ARTIFACT_SET_FILENAME = "verification_artifact_set.json"
_VERIFICATION_FILENAME = "verification_manifest.json"
_OUTCOME_FILENAME = "verified_outcome.json"
_REWARD_FILENAME = "reward.txt"
_AUDIT_FILENAME = "verification_audit.json"

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ArtifactPayload:
    """Exact optional report bytes and their verification-local filename."""

    reference: str
    content: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.reference, str) or not self.reference:
            raise TypeError("artifact payload reference must be a non-empty string")
        if not isinstance(self.content, bytes):
            raise TypeError("artifact payload content must be exact bytes")


@dataclass(frozen=True, slots=True)
class VerificationWriteResult:
    """Paths and manifests produced by one successful immutable write."""

    trial_root: Path
    namespace: Path
    verification_manifest_path: Path
    artifact_set: VerificationArtifactSet
    artifact_set_path: Path
    compatibility_artifact_set: VerificationArtifactSet | None = None
    compatibility_verification_manifest_path: Path | None = None
    compatibility_artifact_set_path: Path | None = None

    def path_for(self, role: ArtifactRole) -> Path:
        """Resolve one canonical private artifact path by role."""

        return _path_for_role(self.trial_root, self.artifact_set, role)

    def compatibility_path_for(self, role: ArtifactRole) -> Path:
        """Resolve one initial Harbor compatibility artifact path by role."""

        if self.compatibility_artifact_set is None:
            raise KeyError("no initial Harbor compatibility artifact set was written")
        return _path_for_role(self.trial_root, self.compatibility_artifact_set, role)


def _path_for_role(
    trial_root: Path,
    artifact_set: VerificationArtifactSet,
    role: ArtifactRole,
) -> Path:
    for artifact in artifact_set.artifacts:
        if artifact.role == role:
            return trial_root.joinpath(*PurePosixPath(artifact.reference).parts)
    raise KeyError(f"artifact role was not written: {role}")


def _revalidate(model: ModelT, model_type: type[ModelT], *, name: str) -> ModelT:
    if not isinstance(model, model_type):
        raise TypeError(f"{name} must be a {model_type.__name__}")
    return model_type.model_validate(model.model_dump(mode="python"))


def _trial_relative_reference(value: str | os.PathLike[str], *, name: str) -> str:
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{name} must be a non-empty trial-root-relative reference")
    path = PurePosixPath(raw)
    if (
        raw.startswith(("/", "~"))
        or "\\" in raw
        or "://" in raw
        or path.is_absolute()
        or ".." in path.parts
        or path in {PurePosixPath("."), PurePosixPath("..")}
    ):
        raise ValueError(f"{name} must be a safe trial-root-relative reference")
    return path.as_posix()


def _private_reference(
    private_directory_reference: str,
    verification_id: str,
    local_reference: str,
) -> str:
    return (
        PurePosixPath(private_directory_reference) / verification_id / local_reference
    ).as_posix()


def _reward_projection(outcome: VerifiedOutcome, edition: TaskEditionManifest) -> bytes:
    admission = next(
        (gate for gate in outcome.gates if gate.gate_id == edition.admission_gate_id),
        None,
    )
    if outcome.disposition == "model_failure":
        return b"0\n"
    if admission is None:
        raise ValueError("scored outcome requires the TaskEdition admission gate for reward")
    return b"1\n" if admission.passed else b"0\n"


def _artifact_payloads(
    *,
    outcome: VerifiedOutcome,
    task_edition: TaskEditionManifest,
    task_report: ArtifactPayload | None,
    legacy_elo_report: ArtifactPayload | None,
    audit: VerificationAuditRecord | None = None,
) -> dict[ArtifactRole, tuple[str, bytes]]:
    payloads: dict[ArtifactRole, tuple[str, bytes]] = {
        "verified_outcome": (_OUTCOME_FILENAME, canonical_json_bytes(outcome)),
        "reward_projection": (_REWARD_FILENAME, _reward_projection(outcome, task_edition)),
    }
    for role, payload in (
        ("task_score_report", task_report),
        ("legacy_elo_report", legacy_elo_report),
    ):
        if payload is None:
            continue
        reference = _trial_relative_reference(
            payload.reference,
            name=f"{role} artifact reference",
        )
        if reference == _ARTIFACT_SET_FILENAME:
            raise ValueError(f"{role} artifact cannot replace {_ARTIFACT_SET_FILENAME}")
        payloads[role] = (reference, payload.content)  # type: ignore[index]
    if audit is not None:
        payloads["verification_audit"] = (
            _AUDIT_FILENAME,
            canonical_json_bytes(audit),
        )
    return payloads


def _artifact_set_for_references(
    *,
    verification: VerificationManifest,
    outcome: VerifiedOutcome,
    payloads: Mapping[ArtifactRole, tuple[str, bytes]],
    references: Mapping[ArtifactRole, str],
    scope: ArtifactSetScope,
) -> VerificationArtifactSet:
    artifact_set = VerificationArtifactSet(
        scope=scope,
        verification_id=verification.verification_id,
        artifacts=tuple(
            ArtifactBinding(
                role=role,
                reference=references[role],
                digest=sha256_bytes(content),
            )
            for role, (_, content) in payloads.items()
        ),
    )
    validate_verification_artifact_set_bindings(
        artifact_set,
        verification=verification,
        outcome=outcome,
    )
    return artifact_set


def _merge_path_payload(
    expected: dict[str, bytes],
    reference: str,
    payload: bytes,
) -> None:
    previous = expected.setdefault(reference, payload)
    if previous != payload:
        raise ValueError(f"one artifact reference cannot bind multiple byte payloads: {reference}")


def _assert_no_symlink_components(root: Path, destination: Path) -> None:
    current = root
    if current.is_symlink():
        raise FileExistsError(f"artifact root cannot be a symlink: {current}")
    relative = destination.relative_to(root)
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise FileExistsError(f"artifact path cannot traverse a symlink: {current}")


def _assert_exact_file(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise FileExistsError(f"immutable artifact cannot be a symlink: {path}")
    if not path.is_file():
        raise FileExistsError(f"immutable artifact path is not a regular file: {path}")
    if path.read_bytes() != payload:
        raise FileExistsError(f"immutable artifact contains different bytes: {path}")


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _write_immutable_bytes(
    root: Path,
    reference: str,
    payload: bytes,
    *,
    trusted_root: Path | None = None,
) -> Path:
    destination = root.joinpath(*PurePosixPath(reference).parts)
    safety_root = root if trusted_root is None else trusted_root
    _assert_no_symlink_components(safety_root, destination)
    if _lexists(destination):
        _assert_exact_file(destination, payload)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(safety_root, destination)
    descriptor, raw_temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            _assert_exact_file(destination, payload)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _expected_directories(references: Mapping[str, bytes]) -> set[str]:
    directories: set[str] = set()
    for reference in references:
        parent = PurePosixPath(reference).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def _preflight_private_namespace(
    namespace: Path,
    expected_local_files: Mapping[str, bytes],
) -> None:
    if not _lexists(namespace):
        return
    if namespace.is_symlink() or not namespace.is_dir():
        raise FileExistsError(f"verification namespace is not an immutable directory: {namespace}")

    expected_directories = _expected_directories(expected_local_files)
    seen_files: set[str] = set()
    for path in namespace.rglob("*"):
        reference = path.relative_to(namespace).as_posix()
        if path.is_symlink():
            raise FileExistsError(f"verification namespace contains a symlink: {reference}")
        if path.is_dir():
            if reference not in expected_directories:
                raise FileExistsError(
                    f"verification namespace contains unexpected directory: {reference}"
                )
            continue
        if not path.is_file() or reference not in expected_local_files:
            raise FileExistsError(f"verification namespace contains unexpected file: {reference}")
        _assert_exact_file(path, expected_local_files[reference])
        seen_files.add(reference)

    if _ARTIFACT_SET_FILENAME in seen_files:
        missing = set(expected_local_files).difference(seen_files)
        if missing:
            raise FileExistsError(
                "completed verification namespace is missing immutable artifact(s): "
                + ", ".join(sorted(missing))
            )


def _preflight_claimed_files(
    trial_root: Path,
    expected: Mapping[str, bytes],
    *,
    manifest_reference: str,
) -> None:
    manifest_exists = _lexists(trial_root.joinpath(*PurePosixPath(manifest_reference).parts))
    missing: list[str] = []
    for reference, payload in expected.items():
        path = trial_root.joinpath(*PurePosixPath(reference).parts)
        _assert_no_symlink_components(trial_root, path)
        if _lexists(path):
            _assert_exact_file(path, payload)
        else:
            missing.append(reference)
    if manifest_exists and missing:
        raise FileExistsError(
            "completed compatibility artifact set is missing immutable artifact(s): "
            + ", ".join(sorted(missing))
        )


def write_verification_artifacts(
    *,
    trial_root: str | os.PathLike[str],
    verification: VerificationManifest,
    outcome: VerifiedOutcome,
    execution: ExecutionManifest,
    attempt: AttemptManifest,
    task_edition: TaskEditionManifest,
    verifier_revision: VerifierRevisionManifest,
    predecessor_verification: VerificationManifest | None = None,
    task_report: ArtifactPayload | None = None,
    legacy_elo_report: ArtifactPayload | None = None,
    initial_compatibility_paths: Mapping[ArtifactRole, str | os.PathLike[str]] | None = None,
    initial_artifact_set_reference: str | os.PathLike[str] | None = None,
    private_directory_reference: str | os.PathLike[str] = _PRIVATE_DIRECTORY,
) -> VerificationWriteResult:
    """Validate and immutably publish one verification's exact output bytes.

    Canonical outputs live under one of the reader-supported per-verification
    private roots and bind trial-root-relative references. An initial
    verification may also request a complete Harbor-root compatibility copy
    plus a second artifact-set manifest whose references bind those root copies.
    Rescores are private-only.
    """

    verification = _revalidate(
        verification,
        VerificationManifest,
        name="verification",
    )
    outcome = _revalidate(outcome, VerifiedOutcome, name="outcome")
    execution = _revalidate(execution, ExecutionManifest, name="execution")
    attempt = _revalidate(attempt, AttemptManifest, name="attempt")
    if not isinstance(task_edition, TaskEditionManifest):
        raise TypeError("task_edition must be a versioned TaskEdition")
    task_edition = validate_task_edition_manifest(task_edition)
    verifier_revision = _revalidate(
        verifier_revision,
        VerifierRevisionManifest,
        name="verifier_revision",
    )

    compatibility_requested = (
        initial_compatibility_paths is not None or initial_artifact_set_reference is not None
    )
    if verification.reason != "initial" and compatibility_requested:
        raise ValueError("Harbor root compatibility output is valid only for initial verification")

    if verification.reason == "rescore":
        if predecessor_verification is None:
            raise ValueError("rescore writer requires predecessor VerificationManifest")
        predecessor = _revalidate(
            predecessor_verification,
            VerificationManifest,
            name="predecessor_verification",
        )
        verification.assert_rescore_predecessor(predecessor)
    elif predecessor_verification is not None:
        raise ValueError("initial verification cannot accept a predecessor VerificationManifest")

    validate_verified_outcome_bindings(
        outcome,
        verification=verification,
        execution=execution,
        attempt=attempt,
        task_edition=task_edition,
        verifier_revision=verifier_revision,
    )

    public_outcome, audit = project_public_verified_outcome(outcome)
    validate_verified_outcome_bindings(
        public_outcome,
        verification=verification,
        execution=execution,
        attempt=attempt,
        task_edition=task_edition,
        verifier_revision=verifier_revision,
    )
    public_payloads = _artifact_payloads(
        outcome=public_outcome,
        task_edition=task_edition,
        task_report=task_report,
        legacy_elo_report=legacy_elo_report,
    )
    private_payloads = _artifact_payloads(
        outcome=public_outcome,
        task_edition=task_edition,
        task_report=task_report,
        legacy_elo_report=legacy_elo_report,
        audit=audit,
    )
    private_directory = _trial_relative_reference(
        private_directory_reference,
        name="private artifact directory",
    )
    if private_directory not in _SUPPORTED_PRIVATE_DIRECTORIES:
        raise ValueError("private artifact directory is not reader-supported")
    private_references = {
        role: _private_reference(
            private_directory,
            verification.verification_id,
            local_reference,
        )
        for role, (local_reference, _) in private_payloads.items()
    }
    private_artifact_set = _artifact_set_for_references(
        verification=verification,
        outcome=public_outcome,
        payloads=private_payloads,
        references=private_references,
        scope="private_canonical",
    )
    private_manifest_bytes = canonical_json_bytes(private_artifact_set)

    compatibility_artifact_set: VerificationArtifactSet | None = None
    compatibility_manifest_reference: str | None = None
    compatibility_expected: dict[str, bytes] = {}
    if compatibility_requested:
        if initial_compatibility_paths is None or initial_artifact_set_reference is None:
            raise ValueError(
                "initial compatibility paths and artifact-set reference must be provided together"
            )
        compatibility_references = {
            role: _trial_relative_reference(
                reference,
                name=f"{role} compatibility path",
            )
            for role, reference in dict(initial_compatibility_paths).items()
        }
        if set(compatibility_references) != set(public_payloads):
            raise ValueError("initial compatibility paths must bind the complete role inventory")
        compatibility_manifest_reference = _trial_relative_reference(
            initial_artifact_set_reference,
            name="initial artifact-set reference",
        )
        if compatibility_manifest_reference in compatibility_references.values():
            raise ValueError("initial artifact-set reference cannot replace an artifact role")
        compatibility_artifact_set = _artifact_set_for_references(
            verification=verification,
            outcome=public_outcome,
            payloads=public_payloads,
            references=compatibility_references,
            scope="public_compatibility",
        )
        for role, (_, content) in public_payloads.items():
            _merge_path_payload(
                compatibility_expected,
                compatibility_references[role],
                content,
            )
        _merge_path_payload(
            compatibility_expected,
            compatibility_manifest_reference,
            canonical_json_bytes(compatibility_artifact_set),
        )

    root = Path(trial_root)
    namespace = root.joinpath(*PurePosixPath(private_directory).parts) / (
        verification.verification_id
    )
    _assert_no_symlink_components(root, namespace / _ARTIFACT_SET_FILENAME)
    private_expected: dict[str, bytes] = {}
    for local_reference, content in private_payloads.values():
        _merge_path_payload(private_expected, local_reference, content)
    _merge_path_payload(
        private_expected,
        _VERIFICATION_FILENAME,
        canonical_json_bytes(verification),
    )
    _merge_path_payload(
        private_expected,
        _ARTIFACT_SET_FILENAME,
        private_manifest_bytes,
    )

    _preflight_private_namespace(namespace, private_expected)
    if compatibility_manifest_reference is not None:
        _preflight_claimed_files(
            root,
            compatibility_expected,
            manifest_reference=compatibility_manifest_reference,
        )

    for reference, content in private_expected.items():
        if reference == _ARTIFACT_SET_FILENAME:
            continue
        _write_immutable_bytes(
            namespace,
            reference,
            content,
            trusted_root=root,
        )
    verification_manifest_path = namespace / _VERIFICATION_FILENAME
    artifact_set_path = _write_immutable_bytes(
        namespace,
        _ARTIFACT_SET_FILENAME,
        private_manifest_bytes,
        trusted_root=root,
    )
    _preflight_private_namespace(namespace, private_expected)

    compatibility_artifact_set_path: Path | None = None
    compatibility_verification_manifest_path: Path | None = None
    if compatibility_manifest_reference is not None:
        for reference, content in compatibility_expected.items():
            if reference == compatibility_manifest_reference:
                continue
            _write_immutable_bytes(root, reference, content)
        compatibility_artifact_set_path = _write_immutable_bytes(
            root,
            compatibility_manifest_reference,
            compatibility_expected[compatibility_manifest_reference],
        )
        _preflight_claimed_files(
            root,
            compatibility_expected,
            manifest_reference=compatibility_manifest_reference,
        )

    return VerificationWriteResult(
        trial_root=root,
        namespace=namespace,
        verification_manifest_path=verification_manifest_path,
        artifact_set=private_artifact_set,
        artifact_set_path=artifact_set_path,
        compatibility_artifact_set=compatibility_artifact_set,
        compatibility_verification_manifest_path=compatibility_verification_manifest_path,
        compatibility_artifact_set_path=compatibility_artifact_set_path,
    )


__all__ = [
    "ArtifactPayload",
    "VerificationWriteResult",
    "write_verification_artifacts",
]
