"""Strict, task-agnostic discovery of verification artifact sets.

Harbor has used three verifier output layouts over time: the shared
``artifacts/`` directory, ``verifier/``, and ``verifier/artifacts/``.  This
reader treats those locations as copies, not as a priority list.  A copy is
accepted only after every manifest claim and every same-verification role has
reached byte consensus.

Initial compatibility artifacts may remain in those Harbor roots. Immutable
claims live either under ``private/<verification_id>/`` or under the verifier
artifact collection root used by separate Harbor verifiers. This module never
writes either location; writers are responsible for no-clobber publication,
while the digest checks here make later mutation fail closed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import ValidationError

from qiqcbench.eval.contracts.campaign import AttemptManifest, ExecutionManifest
from qiqcbench.eval.contracts.identity import (
    TaskEditionManifest,
    VerifierRevisionManifest,
)
from qiqcbench.eval.contracts.outcome import (
    ArtifactBinding,
    ArtifactRole,
    VerificationArtifactSet,
    VerificationAuditRecord,
    VerificationManifest,
    VerifiedOutcome,
    validate_public_verified_outcome,
    validate_verification_artifact_set_bindings,
    validate_verification_audit_bindings,
    validate_verified_outcome_bindings,
)
from qiqcbench.eval.io.canonical import canonical_json_bytes, sha256_bytes
from qiqcbench.eval.io.diagnostics import ArtifactScanDiagnostic

ArtifactLayout = Literal[
    "shared",
    "verifier",
    "verifier_artifacts",
    "private_verification",
]

_MANIFEST_NAME = "verification_artifact_set.json"
_VERIFICATION_MANIFEST_NAME = "verification_manifest.json"
_ROOT_LAYOUTS: tuple[tuple[ArtifactLayout, PurePosixPath], ...] = (
    ("shared", PurePosixPath("artifacts")),
    ("verifier", PurePosixPath("verifier")),
    ("verifier_artifacts", PurePosixPath("verifier/artifacts")),
)
_PRIVATE_LAYOUT_ROOTS: tuple[PurePosixPath, ...] = (
    PurePosixPath("private"),
    PurePosixPath("verifier/artifacts/private"),
)
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


class ArtifactDiscoveryError(ValueError):
    """Base error for an unusable artifact discovery request."""


class ArtifactQuarantineError(ArtifactDiscoveryError):
    """An artifact claim failed integrity checks and must be quarantined."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "artifact_integrity_failure",
        subject_reference: str | None = None,
        diagnostics: tuple[ArtifactScanDiagnostic, ...] = (),
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.subject_reference = subject_reference
        self.diagnostics = diagnostics


class ArtifactSetNotFoundError(ArtifactDiscoveryError):
    """No discovered artifact set carries the requested verification ID."""


@dataclass(frozen=True, slots=True)
class VerificationArtifactSetCopy:
    """One exact serialized copy of a validated artifact-set manifest."""

    layout: ArtifactLayout
    relative_path: str
    path: Path
    payload: bytes
    artifact_set: VerificationArtifactSet


@dataclass(frozen=True, slots=True)
class VerificationManifestCopy:
    """One canonical serialized copy of a verification input manifest."""

    layout: ArtifactLayout
    relative_path: str
    path: Path
    payload: bytes
    verification: VerificationManifest


@dataclass(frozen=True, slots=True)
class ArtifactCopy:
    """One exact file copy cited by an artifact-set manifest."""

    layout: ArtifactLayout
    role: ArtifactRole
    reference: str
    path: Path
    digest: str
    payload: bytes


@dataclass(frozen=True, slots=True)
class ResolvedArtifact:
    """Consensus bytes for one role plus every validated physical copy."""

    role: ArtifactRole
    digest: str
    payload: bytes
    copies: tuple[ArtifactCopy, ...]


@dataclass(frozen=True, slots=True)
class DiscoveredVerificationArtifacts:
    """All exact manifests and consensus role bytes for one verification."""

    verification_id: str
    trial_root: Path
    manifests: tuple[VerificationArtifactSetCopy, ...]
    verification_manifests: tuple[VerificationManifestCopy, ...]
    artifacts: tuple[ResolvedArtifact, ...]

    @property
    def verification(self) -> VerificationManifest:
        """Return the byte-consensus verification manifest."""

        return self.verification_manifests[0].verification

    def artifact(self, role: ArtifactRole) -> ResolvedArtifact:
        """Return the consensus artifact for ``role`` or raise ``KeyError``."""

        for artifact in self.artifacts:
            if artifact.role == role:
                return artifact
        raise KeyError(role)


@dataclass(frozen=True, slots=True)
class ArtifactScanResult:
    """Tolerant scan result; canonical callers must invoke :meth:`require_clean`."""

    verifications: tuple[DiscoveredVerificationArtifacts, ...]
    diagnostics: tuple[ArtifactScanDiagnostic, ...]
    quarantine_errors: tuple[ArtifactQuarantineError, ...] = ()

    def require_clean(self) -> tuple[DiscoveredVerificationArtifacts, ...]:
        quarantined = tuple(
            diagnostic for diagnostic in self.diagnostics if diagnostic.decision == "quarantined"
        )
        if quarantined:
            if self.quarantine_errors:
                first = self.quarantine_errors[0]
                message = str(first)
                reason_code = first.reason_code
            else:
                reason_codes = sorted({item.reason_code for item in quarantined})
                message = (
                    "artifact scan quarantined "
                    f"{len(quarantined)} claim(s): {', '.join(reason_codes)}"
                )
                reason_code = "artifact_scan_quarantined"
            raise ArtifactQuarantineError(
                message,
                reason_code=reason_code,
                diagnostics=self.diagnostics,
            )
        return self.verifications


@dataclass(frozen=True, slots=True)
class SemanticValidationScanResult:
    """Public-safe terminal result for one verification semantic check."""

    outcome: VerifiedOutcome | None
    diagnostic: ArtifactScanDiagnostic

    def require_clean(self) -> VerifiedOutcome:
        """Return the outcome or raise a sanitized quarantine error."""

        if self.outcome is None:
            raise ArtifactQuarantineError(
                "verification artifact semantic validation failed",
                reason_code=self.diagnostic.reason_code,
                subject_reference=self.diagnostic.subject_reference,
                diagnostics=(self.diagnostic,),
            )
        return self.outcome


@dataclass(frozen=True, slots=True)
class _ManifestCandidate:
    layout: ArtifactLayout
    relative_path: PurePosixPath
    private_namespace: str | None = None
    private_root: PurePosixPath | None = None


@dataclass(frozen=True, slots=True)
class _LoadedManifest:
    manifest: VerificationArtifactSetCopy
    verification_manifest: VerificationManifestCopy | None
    artifacts: tuple[ArtifactCopy, ...]


class _DuplicateJsonKey(ValueError):
    pass


def _safe_component(value: str, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_COMPONENT.fullmatch(value) is None:
        raise ValueError(f"{field} must be a safe filesystem component")
    return value


def _duplicate_checked_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _validate_json_syntax(payload: bytes) -> None:
    try:
        decoded = payload.decode("utf-8")
        json.loads(
            decoded,
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=_reject_json_constant,
        )
    except _DuplicateJsonKey:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError("invalid JSON") from exc


def _path_claim_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _read_relative_file(
    trial_root: Path,
    relative_path: PurePosixPath,
    *,
    description: str,
    frozen_inputs: Mapping[str, bytes] | None = None,
) -> tuple[Path, bytes]:
    reference = relative_path.as_posix()
    if relative_path.is_absolute() or ".." in relative_path.parts or reference in {"", "."}:
        raise ArtifactQuarantineError(
            f"unsafe {description} reference: {reference}",
            reason_code="path_escape",
            subject_reference=reference,
        )
    candidate = trial_root.joinpath(*relative_path.parts)
    if frozen_inputs is not None:
        try:
            payload = frozen_inputs[reference]
        except KeyError as exc:
            raise ArtifactQuarantineError(
                f"missing {description}: {reference}",
                reason_code="missing_claimed_file",
                subject_reference=reference,
            ) from exc
        if not isinstance(payload, bytes):
            raise TypeError("frozen verification inputs must contain bytes")
        return candidate, payload
    if trial_root.is_symlink():
        raise ArtifactQuarantineError(
            "trial root cannot be a symlink",
            reason_code="symlink_traversal",
            subject_reference=relative_path.as_posix(),
        )
    current = trial_root
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            raise ArtifactQuarantineError(
                f"{description} cannot traverse a symlink: {relative_path.as_posix()}",
                reason_code="symlink_traversal",
                subject_reference=relative_path.as_posix(),
            )
    try:
        resolved_root = trial_root.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ArtifactQuarantineError(
            f"missing {description}: {relative_path.as_posix()}",
            reason_code="missing_claimed_file",
            subject_reference=relative_path.as_posix(),
        ) from exc
    except OSError as exc:
        raise ArtifactQuarantineError(
            f"cannot resolve {description}: {relative_path.as_posix()}",
            reason_code="unreadable_claimed_file",
            subject_reference=relative_path.as_posix(),
        ) from exc

    if not resolved.is_relative_to(resolved_root):
        raise ArtifactQuarantineError(
            f"{description} escapes the trial root: {relative_path.as_posix()}",
            reason_code="path_escape",
            subject_reference=relative_path.as_posix(),
        )
    if not resolved.is_file():
        raise ArtifactQuarantineError(
            f"{description} is not a regular file: {relative_path.as_posix()}",
            reason_code="non_regular_file",
            subject_reference=relative_path.as_posix(),
        )
    try:
        return resolved, resolved.read_bytes()
    except OSError as exc:
        raise ArtifactQuarantineError(
            f"cannot read {description}: {relative_path.as_posix()}",
            reason_code="unreadable_claimed_file",
            subject_reference=relative_path.as_posix(),
        ) from exc


def _manifest_candidates(
    trial_root: Path,
    *,
    frozen_inputs: Mapping[str, bytes] | None = None,
) -> tuple[_ManifestCandidate, ...]:
    candidates: list[_ManifestCandidate] = []
    for layout, root in _ROOT_LAYOUTS:
        relative = root / _MANIFEST_NAME
        if (
            relative.as_posix() in frozen_inputs
            if frozen_inputs is not None
            else _path_claim_exists(trial_root.joinpath(*relative.parts))
        ):
            candidates.append(_ManifestCandidate(layout=layout, relative_path=relative))

    if frozen_inputs is not None:
        for private_root in _PRIVATE_LAYOUT_ROOTS:
            prefix = private_root.parts
            private_manifests = sorted(
                PurePosixPath(reference)
                for reference in frozen_inputs
                if len(PurePosixPath(reference).parts) == len(prefix) + 2
                and PurePosixPath(reference).parts[: len(prefix)] == prefix
                and PurePosixPath(reference).name == _MANIFEST_NAME
            )
            candidates.extend(
                _ManifestCandidate(
                    layout="private_verification",
                    relative_path=relative,
                    private_namespace=relative.parts[len(prefix)],
                    private_root=private_root,
                )
                for relative in private_manifests
            )
        return tuple(sorted(candidates, key=lambda item: item.relative_path.as_posix()))

    for private_reference in _PRIVATE_LAYOUT_ROOTS:
        private_root = trial_root.joinpath(*private_reference.parts)
        if not _path_claim_exists(private_root):
            continue
        if private_root.is_symlink() or not private_root.is_dir():
            raise ArtifactQuarantineError(
                "private verification namespace must be a non-symlink directory",
                reason_code="invalid_private_namespace",
                subject_reference=private_reference.as_posix(),
            )
        try:
            resolved_root = trial_root.resolve(strict=True)
            resolved_private = private_root.resolve(strict=True)
            if not resolved_private.is_relative_to(resolved_root):
                raise ArtifactQuarantineError(
                    "private verification namespace escapes trial root",
                    reason_code="path_escape",
                    subject_reference=private_reference.as_posix(),
                )
            if not resolved_private.is_dir():
                raise ArtifactQuarantineError(
                    "private verification namespace is not a directory",
                    reason_code="invalid_private_namespace",
                    subject_reference=private_reference.as_posix(),
                )
            entries = sorted(resolved_private.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise ArtifactQuarantineError(
                "cannot scan private verification namespace",
                reason_code="unreadable_private_namespace",
                subject_reference=private_reference.as_posix(),
            ) from exc
        for entry in entries:
            if entry.is_symlink() or not entry.is_dir():
                entry_reference = private_reference / entry.name
                raise ArtifactQuarantineError(
                    "private verification namespace contains a non-directory entry",
                    reason_code="invalid_private_namespace",
                    subject_reference=entry_reference.as_posix(),
                )
            relative = private_reference / entry.name / _MANIFEST_NAME
            if _path_claim_exists(trial_root.joinpath(*relative.parts)):
                candidates.append(
                    _ManifestCandidate(
                        layout="private_verification",
                        relative_path=relative,
                        private_namespace=entry.name,
                        private_root=private_reference,
                    )
                )
            else:
                raise ArtifactQuarantineError(
                    f"incomplete private verification namespace: {entry.name}",
                    reason_code="incomplete_private_namespace",
                    subject_reference=(private_reference / entry.name).as_posix(),
                )
    return tuple(sorted(candidates, key=lambda item: item.relative_path.as_posix()))


def _parse_manifest(
    payload: bytes,
    *,
    relative_path: PurePosixPath,
) -> VerificationArtifactSet:
    try:
        _validate_json_syntax(payload)
        artifact_set = VerificationArtifactSet.model_validate_json(payload)
    except _DuplicateJsonKey as exc:
        raise ArtifactQuarantineError(
            f"duplicate JSON key in {relative_path.as_posix()}: {exc}",
            reason_code="invalid_artifact_set_manifest",
            subject_reference=relative_path.as_posix(),
        ) from exc
    except (ValidationError, ValueError) as exc:
        raise ArtifactQuarantineError(
            f"invalid VerificationArtifactSet at {relative_path.as_posix()}",
            reason_code="invalid_artifact_set_manifest",
            subject_reference=relative_path.as_posix(),
        ) from exc
    if payload != canonical_json_bytes(artifact_set):
        raise ArtifactQuarantineError(
            f"non-canonical VerificationArtifactSet bytes at {relative_path.as_posix()}",
            reason_code="noncanonical_artifact_set_manifest",
            subject_reference=relative_path.as_posix(),
        )
    return artifact_set


def _parse_verification_manifest(
    payload: bytes,
    *,
    relative_path: PurePosixPath,
) -> VerificationManifest:
    try:
        _validate_json_syntax(payload)
        verification = VerificationManifest.model_validate_json(payload)
    except _DuplicateJsonKey as exc:
        raise ArtifactQuarantineError(
            f"duplicate JSON key in {relative_path.as_posix()}: {exc}",
            reason_code="invalid_verification_manifest",
            subject_reference=relative_path.as_posix(),
        ) from exc
    except (ValidationError, ValueError) as exc:
        raise ArtifactQuarantineError(
            f"invalid VerificationManifest at {relative_path.as_posix()}",
            reason_code="invalid_verification_manifest",
            subject_reference=relative_path.as_posix(),
        ) from exc
    if payload != canonical_json_bytes(verification):
        raise ArtifactQuarantineError(
            f"non-canonical VerificationManifest bytes at {relative_path.as_posix()}",
            reason_code="noncanonical_verification_manifest",
            subject_reference=relative_path.as_posix(),
        )
    return verification


def _load_candidate(
    trial_root: Path,
    candidate: _ManifestCandidate,
    *,
    expected_initial_verification_id: str | None,
    frozen_inputs: Mapping[str, bytes] | None = None,
) -> _LoadedManifest:
    manifest_path, manifest_payload = _read_relative_file(
        trial_root,
        candidate.relative_path,
        description="VerificationArtifactSet manifest",
        frozen_inputs=frozen_inputs,
    )
    artifact_set = _parse_manifest(manifest_payload, relative_path=candidate.relative_path)
    is_private = candidate.layout == "private_verification"
    expected_scope = "private_canonical" if is_private else "public_compatibility"
    if artifact_set.scope != expected_scope:
        raise ArtifactQuarantineError(
            f"{candidate.layout} artifact set must use scope {expected_scope!r}",
            reason_code="artifact_set_scope_mismatch",
            subject_reference=candidate.relative_path.as_posix(),
        )
    verification_copy: VerificationManifestCopy | None = None
    if is_private:
        verification_relative = candidate.relative_path.parent / _VERIFICATION_MANIFEST_NAME
        verification_path, verification_payload = _read_relative_file(
            trial_root,
            verification_relative,
            description="VerificationManifest",
            frozen_inputs=frozen_inputs,
        )
        verification = _parse_verification_manifest(
            verification_payload,
            relative_path=verification_relative,
        )
        if verification.verification_id != artifact_set.verification_id:
            raise ArtifactQuarantineError(
                "VerificationManifest verification_id does not match VerificationArtifactSet"
            )
        verification_copy = VerificationManifestCopy(
            layout=candidate.layout,
            relative_path=verification_relative.as_posix(),
            path=verification_path,
            payload=verification_payload,
            verification=verification,
        )
    try:
        verification_id = _safe_component(
            artifact_set.verification_id,
            field="verification_id",
        )
    except ValueError as exc:
        raise ArtifactQuarantineError(str(exc)) from exc

    if is_private and candidate.private_namespace != verification_id:
        raise ArtifactQuarantineError(
            "VerificationArtifactSet verification_id does not match its private namespace"
        )
    if (
        candidate.layout != "private_verification"
        and expected_initial_verification_id is not None
        and verification_id != expected_initial_verification_id
    ):
        raise ArtifactQuarantineError(
            "root compatibility artifact set does not match the expected initial verification_id"
        )

    copies: list[ArtifactCopy] = []
    for binding in artifact_set.artifacts:
        relative = PurePosixPath(binding.reference)
        private_prefix = (
            (*candidate.private_root.parts, verification_id)
            if candidate.private_root is not None
            else ("private", verification_id)
        )
        if is_private and relative.parts[: len(private_prefix)] != private_prefix:
            raise ArtifactQuarantineError(
                "private verification artifacts must remain inside their per-verification namespace"
            )
        path, payload = _read_relative_file(
            trial_root,
            relative,
            description=f"artifact role {binding.role}",
            frozen_inputs=frozen_inputs,
        )
        observed_digest = sha256_bytes(payload)
        if observed_digest != binding.digest:
            raise ArtifactQuarantineError(
                f"digest mismatch for artifact role {binding.role} in verification "
                f"{verification_id}",
                reason_code="digest_mismatch",
                subject_reference=binding.reference,
            )
        copies.append(
            ArtifactCopy(
                layout=candidate.layout,
                role=binding.role,
                reference=binding.reference,
                path=path,
                digest=binding.digest,
                payload=payload,
            )
        )

    return _LoadedManifest(
        manifest=VerificationArtifactSetCopy(
            layout=candidate.layout,
            relative_path=candidate.relative_path.as_posix(),
            path=manifest_path,
            payload=manifest_payload,
            artifact_set=artifact_set,
        ),
        verification_manifest=verification_copy,
        artifacts=tuple(sorted(copies, key=lambda item: item.role)),
    )


def _role_inventory(item: _LoadedManifest) -> dict[ArtifactRole, ArtifactBinding]:
    return {binding.role: binding for binding in item.manifest.artifact_set.artifacts}


def _resolve_verification(
    verification_id: str,
    manifests: list[_LoadedManifest],
    *,
    trial_root: Path,
) -> DiscoveredVerificationArtifacts:
    private_manifests = [
        item for item in manifests if item.manifest.layout == "private_verification"
    ]
    if len(private_manifests) != 1:
        raise ArtifactQuarantineError(
            f"verification {verification_id} requires exactly one complete private claim",
            reason_code="missing_or_ambiguous_private_claim",
        )
    private = private_manifests[0]
    assert private.verification_manifest is not None
    if any(item.manifest.layout != "private_verification" for item in manifests) and (
        private.verification_manifest.verification.reason != "initial"
    ):
        raise ArtifactQuarantineError(
            "root compatibility artifacts require an initial verification_id",
            reason_code="root_claim_for_noninitial_verification",
        )

    expected_inventory = _role_inventory(private)
    expected_roles = set(expected_inventory)
    copies_by_role: dict[ArtifactRole, list[ArtifactCopy]] = {
        role: [] for role in expected_inventory
    }
    root_roles: set[ArtifactRole] | None = None

    for item in manifests:
        inventory = _role_inventory(item)
        roles = set(inventory)
        if item.manifest.layout == "private_verification":
            if roles != expected_roles:
                raise ArtifactQuarantineError(
                    f"private artifact inventory changed for verification {verification_id}",
                    reason_code="artifact_inventory_conflict",
                )
        else:
            if not roles.issubset(expected_roles):
                raise ArtifactQuarantineError(
                    f"root artifact inventory is not a public subset for verification "
                    f"{verification_id}",
                    reason_code="artifact_inventory_conflict",
                )
            if root_roles is None:
                root_roles = roles
            elif roles != root_roles:
                raise ArtifactQuarantineError(
                    f"root artifact role inventories disagree for verification {verification_id}",
                    reason_code="artifact_inventory_conflict",
                )
        for role, binding in inventory.items():
            if binding.digest != expected_inventory[role].digest:
                raise ArtifactQuarantineError(
                    f"artifact copies disagree for role {role} in verification {verification_id}",
                    reason_code="artifact_copy_conflict",
                )
        for copy in item.artifacts:
            copies_by_role[copy.role].append(copy)

    resolved: list[ResolvedArtifact] = []
    for role in sorted(copies_by_role):
        copies = tuple(sorted(copies_by_role[role], key=lambda item: item.reference))
        digests = {copy.digest for copy in copies}
        payloads = {copy.payload for copy in copies}
        if len(digests) != 1 or len(payloads) != 1:
            raise ArtifactQuarantineError(
                f"artifact copies disagree for role {role} in verification {verification_id}",
                reason_code="artifact_copy_conflict",
            )
        representative = copies[0]
        resolved.append(
            ResolvedArtifact(
                role=role,
                digest=representative.digest,
                payload=representative.payload,
                copies=copies,
            )
        )

    manifest_copies = tuple(
        sorted(
            (item.manifest for item in manifests),
            key=lambda item: item.relative_path,
        )
    )
    return DiscoveredVerificationArtifacts(
        verification_id=verification_id,
        trial_root=trial_root,
        manifests=manifest_copies,
        verification_manifests=(private.verification_manifest,),
        artifacts=tuple(resolved),
    )


def _diagnostic_sort_key(item: ArtifactScanDiagnostic) -> tuple[str, str, str, str]:
    return (
        item.subject_reference,
        item.artifact_role or "",
        item.decision,
        item.reason_code,
    )


def _included_scan_diagnostics(
    discovered: DiscoveredVerificationArtifacts,
) -> tuple[ArtifactScanDiagnostic, ...]:
    diagnostics: list[ArtifactScanDiagnostic] = []
    for manifest in discovered.manifests:
        diagnostics.append(
            ArtifactScanDiagnostic(
                stage="discovery",
                subject_reference=manifest.relative_path,
                artifact_kind="verification_artifact_set",
                decision="included",
                reason_code="validated_claim",
                verification_id=discovered.verification_id,
                digest=sha256_bytes(manifest.payload),
            )
        )
    for manifest in discovered.verification_manifests:
        diagnostics.append(
            ArtifactScanDiagnostic(
                stage="discovery",
                subject_reference=manifest.relative_path,
                artifact_kind="verification_manifest",
                decision="included",
                reason_code="validated_claim",
                verification_id=discovered.verification_id,
                digest=sha256_bytes(manifest.payload),
            )
        )
    for artifact in discovered.artifacts:
        for copy in artifact.copies:
            diagnostics.append(
                ArtifactScanDiagnostic(
                    stage="discovery",
                    subject_reference=copy.reference,
                    artifact_kind="verification_output",
                    artifact_role=copy.role,
                    decision="included",
                    reason_code="validated_claim",
                    verification_id=discovered.verification_id,
                    digest=copy.digest,
                )
            )
    return tuple(diagnostics)


def _scan_verification_artifacts(
    trial_root: Path,
    *,
    expected_initial_verification_id: str | None = None,
    frozen_inputs: Mapping[str, bytes] | None = None,
) -> ArtifactScanResult:
    if expected_initial_verification_id is not None:
        _safe_component(
            expected_initial_verification_id,
            field="expected_initial_verification_id",
        )

    diagnostics: list[ArtifactScanDiagnostic] = []
    quarantine_errors: list[ArtifactQuarantineError] = []
    by_verification: dict[str, list[_LoadedManifest]] = {}
    root_verification_ids: set[str] = set()
    try:
        candidates = _manifest_candidates(trial_root, frozen_inputs=frozen_inputs)
    except ArtifactQuarantineError as exc:
        quarantine_errors.append(exc)
        diagnostics.append(
            ArtifactScanDiagnostic(
                stage="discovery",
                subject_reference=exc.subject_reference or "private",
                artifact_kind="verification_namespace",
                decision="quarantined",
                reason_code=exc.reason_code,
            )
        )
        return ArtifactScanResult(
            verifications=(),
            diagnostics=tuple(diagnostics),
            quarantine_errors=tuple(quarantine_errors),
        )

    if not candidates:
        return ArtifactScanResult(
            verifications=(),
            diagnostics=(
                ArtifactScanDiagnostic(
                    stage="discovery",
                    subject_reference="trial",
                    artifact_kind="trial",
                    decision="excluded",
                    reason_code="no_supported_artifact_claim",
                ),
            ),
        )

    for candidate in candidates:
        try:
            loaded = _load_candidate(
                trial_root,
                candidate,
                expected_initial_verification_id=expected_initial_verification_id,
                frozen_inputs=frozen_inputs,
            )
        except ArtifactQuarantineError as exc:
            quarantine_errors.append(exc)
            diagnostics.append(
                ArtifactScanDiagnostic(
                    stage="discovery",
                    subject_reference=(exc.subject_reference or candidate.relative_path.as_posix()),
                    artifact_kind="verification_artifact_claim",
                    decision="quarantined",
                    reason_code=exc.reason_code,
                )
            )
            continue
        by_verification.setdefault(
            loaded.manifest.artifact_set.verification_id,
            [],
        ).append(loaded)
        if candidate.layout != "private_verification":
            root_verification_ids.add(loaded.manifest.artifact_set.verification_id)

    if len(root_verification_ids) > 1:
        quarantine_errors.append(
            ArtifactQuarantineError(
                "Harbor root layouts cannot claim multiple initial verification IDs",
                reason_code="multiple_initial_verification_ids",
            )
        )
        for manifests in by_verification.values():
            for item in manifests:
                if item.manifest.layout != "private_verification":
                    diagnostics.append(
                        ArtifactScanDiagnostic(
                            stage="discovery",
                            subject_reference=item.manifest.relative_path,
                            artifact_kind="verification_artifact_set",
                            decision="quarantined",
                            reason_code="multiple_initial_verification_ids",
                            verification_id=item.manifest.artifact_set.verification_id,
                        )
                    )
        return ArtifactScanResult(
            verifications=(),
            diagnostics=tuple(sorted(diagnostics, key=_diagnostic_sort_key)),
            quarantine_errors=tuple(quarantine_errors),
        )

    resolved_items: list[DiscoveredVerificationArtifacts] = []
    for verification_id in sorted(by_verification):
        manifests = by_verification[verification_id]
        try:
            discovered = _resolve_verification(
                verification_id,
                manifests,
                trial_root=trial_root,
            )
        except ArtifactQuarantineError as exc:
            quarantine_errors.append(exc)
            for item in manifests:
                diagnostics.append(
                    ArtifactScanDiagnostic(
                        stage="discovery",
                        subject_reference=item.manifest.relative_path,
                        artifact_kind="verification_artifact_set",
                        decision="quarantined",
                        reason_code=exc.reason_code,
                        verification_id=verification_id,
                    )
                )
            continue
        resolved_items.append(discovered)
        diagnostics.extend(_included_scan_diagnostics(discovered))

    if quarantine_errors:
        diagnostics = [item for item in diagnostics if item.decision != "included"]
        resolved_items = []
    return ArtifactScanResult(
        verifications=tuple(resolved_items),
        diagnostics=tuple(sorted(diagnostics, key=_diagnostic_sort_key)),
        quarantine_errors=tuple(quarantine_errors),
    )


def scan_verification_artifacts(
    trial_dir: str | Path,
    *,
    expected_initial_verification_id: str | None = None,
) -> ArtifactScanResult:
    """Scan every live v2 claim and retain a terminal structured diagnostic.

    This tolerant API never turns a malformed v2 claim into legacy evidence.
    Use :func:`discover_verification_artifacts` for the fail-closed canonical
    wrapper.
    """

    trial_root = Path(trial_dir)
    if not trial_root.is_dir():
        raise ArtifactDiscoveryError("trial_dir must be an existing directory")
    return _scan_verification_artifacts(
        trial_root,
        expected_initial_verification_id=expected_initial_verification_id,
    )


def scan_verification_artifact_snapshot(
    frozen_inputs: Mapping[str, bytes],
    *,
    trial_root: str | Path,
    expected_initial_verification_id: str | None = None,
) -> ArtifactScanResult:
    """Parse one immutable verification-input snapshot without filesystem reads."""

    snapshot: dict[str, bytes] = {}
    for reference, payload in frozen_inputs.items():
        if not isinstance(reference, str):
            raise TypeError("frozen verification input references must be strings")
        if not isinstance(payload, bytes):
            raise TypeError("frozen verification inputs must contain bytes")
        snapshot[reference] = payload
    return _scan_verification_artifacts(
        Path(trial_root),
        expected_initial_verification_id=expected_initial_verification_id,
        frozen_inputs=snapshot,
    )


def discover_verification_artifacts(
    trial_dir: str | Path,
    *,
    expected_initial_verification_id: str | None = None,
) -> tuple[DiscoveredVerificationArtifacts, ...]:
    """Fail closed unless every discovered v2 claim reaches byte consensus."""

    return scan_verification_artifacts(
        trial_dir,
        expected_initial_verification_id=expected_initial_verification_id,
    ).require_clean()


def _parse_verified_outcome(payload: bytes) -> VerifiedOutcome:
    try:
        _validate_json_syntax(payload)
        outcome = VerifiedOutcome.model_validate_json(payload)
    except (ValidationError, ValueError) as exc:
        raise ArtifactQuarantineError("invalid verified_outcome artifact") from exc
    if payload != canonical_json_bytes(outcome):
        raise ArtifactQuarantineError("verified_outcome artifact is not canonical JSON")
    try:
        return validate_public_verified_outcome(outcome)
    except ValueError as exc:
        raise ArtifactQuarantineError(
            "verified_outcome artifact contains private verifier data",
            reason_code="private_data_in_public_outcome",
        ) from exc


def _parse_verification_audit(payload: bytes) -> VerificationAuditRecord:
    try:
        _validate_json_syntax(payload)
        audit = VerificationAuditRecord.model_validate_json(payload)
    except (ValidationError, ValueError) as exc:
        raise ArtifactQuarantineError("invalid verification_audit artifact") from exc
    if payload != canonical_json_bytes(audit):
        raise ArtifactQuarantineError("verification_audit artifact is not canonical JSON")
    return audit


def _validate_projected_observation_parity(
    *,
    projected: VerifiedOutcome,
    outcome: VerifiedOutcome,
    report_label: str,
) -> None:
    for role, digest in projected.report_digests.items():
        if outcome.report_digests.get(role) != digest:
            raise ValueError(f"{report_label} digest disagrees with VerifiedOutcome")
    if projected.gates != outcome.gates:
        raise ValueError(f"{report_label} gates disagree with VerifiedOutcome")
    if projected.metrics != outcome.metrics:
        raise ValueError(f"{report_label} metrics disagree with VerifiedOutcome")


def validate_verification_artifact_semantics(
    discovered: DiscoveredVerificationArtifacts,
    *,
    execution: ExecutionManifest,
    attempt: AttemptManifest,
    task_edition: TaskEditionManifest,
    verifier_revision: VerifierRevisionManifest,
    expected_legacy_instance_seed: int | None = None,
    predecessor_verification: VerificationManifest | None = None,
    frozen_inputs: Mapping[str, bytes] | None = None,
) -> VerifiedOutcome:
    """Validate identity, evidence, reward, report, and outcome parity for one set."""

    try:
        verification = discovered.verification
        if verification.reason == "rescore":
            if predecessor_verification is None:
                raise ValueError("rescore requires its predecessor VerificationManifest")
            verification.assert_rescore_predecessor(predecessor_verification)
        elif predecessor_verification is not None:
            raise ValueError("initial verification cannot accept a predecessor")

        for evidence in verification.evidence_inputs:
            _, content = _read_relative_file(
                discovered.trial_root,
                PurePosixPath(evidence.reference),
                description=f"verification evidence {evidence.role}",
                frozen_inputs=frozen_inputs,
            )
            if sha256_bytes(content) != evidence.digest:
                raise ValueError(f"verification evidence digest mismatch for {evidence.role!r}")

        outcome_artifact = discovered.artifact("verified_outcome")
        outcome = _parse_verified_outcome(outcome_artifact.payload)
        if outcome.verification_id != discovered.verification_id:
            raise ValueError("VerifiedOutcome verification_id mismatch")
        audit = _parse_verification_audit(discovered.artifact("verification_audit").payload)
        validate_verification_audit_bindings(
            audit,
            verification=verification,
            outcome=outcome,
        )
        validate_verified_outcome_bindings(
            outcome,
            verification=verification,
            execution=execution,
            attempt=attempt,
            task_edition=task_edition,
            verifier_revision=verifier_revision,
        )
        for manifest in discovered.manifests:
            validate_verification_artifact_set_bindings(
                manifest.artifact_set,
                verification=verification,
                outcome=outcome,
            )

        reward = discovered.artifact("reward_projection").payload
        if reward not in {b"0\n", b"1\n"}:
            raise ValueError("reward projection must be exact 0 or 1 bytes")
        if outcome.disposition == "model_failure":
            expected_reward = b"0\n"
        else:
            admission = next(
                (gate for gate in outcome.gates if gate.gate_id == task_edition.admission_gate_id),
                None,
            )
            if admission is None:
                raise ValueError("scored outcome is missing its admission gate")
            expected_reward = b"1\n" if admission.passed else b"0\n"
        if reward != expected_reward:
            raise ValueError("reward projection disagrees with VerifiedOutcome")

        try:
            legacy_artifact = discovered.artifact("legacy_elo_report")
        except KeyError:
            legacy_artifact = None
        if legacy_artifact is not None:
            if type(expected_legacy_instance_seed) is not int:
                raise ValueError("legacy Elo parity requires an expected instance seed")
            from qiqcbench.eval.io.legacy import (
                LegacyOutcomeContext,
                parse_elo_score_report_v1_bytes,
                project_elo_score_report_v1,
            )

            adapters = [
                adapter
                for adapter in verifier_revision.legacy_adapters
                if adapter.adapter_kind == "elo_score_report_v1"
                and adapter.artifact_role == "legacy_elo_report"
            ]
            if len(adapters) != 1:
                raise ValueError("legacy Elo report requires exactly one revision adapter")

            legacy_report = parse_elo_score_report_v1_bytes(legacy_artifact.payload)
            projected = project_elo_score_report_v1(
                legacy_report,
                context=LegacyOutcomeContext(
                    attempt=attempt,
                    execution=execution,
                    verification=verification,
                    disposition=outcome.disposition,
                    expected_instance_seed=expected_legacy_instance_seed,
                    legacy_report_digest=legacy_artifact.digest,
                    public_material_digests={
                        "instruction": task_edition.instruction_digest,
                        **task_edition.public_material_digests,
                    },
                ),
                task_edition=task_edition,
                verifier_revision=verifier_revision,
                adapter_id=adapters[0].adapter_id,
            )
            _validate_projected_observation_parity(
                projected=projected,
                outcome=outcome,
                report_label="legacy Elo report",
            )

        try:
            task_report_artifact = discovered.artifact("task_score_report")
        except KeyError:
            task_report_artifact = None
        rubric_adapters = [
            adapter
            for adapter in verifier_revision.legacy_adapters
            if adapter.adapter_kind == "freeform_rubric_json_v1"
            and adapter.artifact_role == "task_score_report"
        ]
        if task_report_artifact is not None and rubric_adapters:
            if len(rubric_adapters) != 1:
                raise ValueError("task score report requires exactly one revision adapter")
            from qiqcbench.eval.io.legacy import (
                FreeformRubricOutcomeContext,
                project_freeform_rubric_report_bytes,
            )

            projected = project_freeform_rubric_report_bytes(
                task_report_artifact.payload,
                reward,
                context=FreeformRubricOutcomeContext(
                    attempt=attempt,
                    execution=execution,
                    verification=verification,
                    disposition=outcome.disposition,
                    task_report_digest=task_report_artifact.digest,
                    public_material_digests={
                        "instruction": task_edition.instruction_digest,
                        **task_edition.public_material_digests,
                    },
                ),
                task_edition=task_edition,
                verifier_revision=verifier_revision,
                adapter_id=rubric_adapters[0].adapter_id,
            )
            _validate_projected_observation_parity(
                projected=projected,
                outcome=outcome,
                report_label="task score report",
            )
        return outcome
    except ArtifactQuarantineError:
        raise
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise ArtifactQuarantineError(
            f"verification artifact semantic parity failed: {exc}"
        ) from exc


def scan_verification_artifact_semantics(
    discovered: DiscoveredVerificationArtifacts,
    *,
    execution: ExecutionManifest,
    attempt: AttemptManifest,
    task_edition: TaskEditionManifest,
    verifier_revision: VerifierRevisionManifest,
    expected_legacy_instance_seed: int | None = None,
    predecessor_verification: VerificationManifest | None = None,
    frozen_inputs: Mapping[str, bytes] | None = None,
) -> SemanticValidationScanResult:
    """Run strict semantic validation and emit one public-safe decision.

    The strict validator remains the source of truth.  This wrapper deliberately
    omits exception text and observed values from its diagnostic so ingestion
    can retain a stable quarantine decision without publishing verifier detail.
    """

    subject_reference = f"verification/{discovered.verification_id}"
    try:
        outcome = validate_verification_artifact_semantics(
            discovered,
            execution=execution,
            attempt=attempt,
            task_edition=task_edition,
            verifier_revision=verifier_revision,
            expected_legacy_instance_seed=expected_legacy_instance_seed,
            predecessor_verification=predecessor_verification,
            frozen_inputs=frozen_inputs,
        )
    except ArtifactQuarantineError:
        return SemanticValidationScanResult(
            outcome=None,
            diagnostic=ArtifactScanDiagnostic(
                stage="semantic_validation",
                subject_reference=subject_reference,
                artifact_kind="verification",
                decision="quarantined",
                reason_code="semantic_contract_failure",
                verification_id=discovered.verification_id,
            ),
        )
    return SemanticValidationScanResult(
        outcome=outcome,
        diagnostic=ArtifactScanDiagnostic(
            stage="semantic_validation",
            subject_reference=subject_reference,
            artifact_kind="verification",
            decision="included",
            reason_code="semantic_contract_validated",
            verification_id=discovered.verification_id,
        ),
    )


def read_verification_artifacts(
    trial_dir: str | Path,
    verification_id: str,
    *,
    expected_initial_verification_id: str | None = None,
) -> DiscoveredVerificationArtifacts:
    """Read one exact verification lineage from a Harbor trial."""

    requested = _safe_component(verification_id, field="verification_id")
    for discovered in discover_verification_artifacts(
        trial_dir,
        expected_initial_verification_id=expected_initial_verification_id,
    ):
        if discovered.verification_id == requested:
            return discovered
    raise ArtifactSetNotFoundError(
        f"no VerificationArtifactSet found for verification_id {requested!r}"
    )


def read_private_verification_artifacts(
    trial_dir: str | Path,
    verification_id: str,
) -> DiscoveredVerificationArtifacts:
    """Read one immutable private lineage without requiring unrelated branches clean."""

    trial_root = Path(trial_dir)
    if not trial_root.is_dir():
        raise ArtifactDiscoveryError("trial_dir must be an existing directory")
    requested = _safe_component(verification_id, field="verification_id")
    candidates = [
        _ManifestCandidate(
            layout="private_verification",
            relative_path=private_root / requested / _MANIFEST_NAME,
            private_namespace=requested,
            private_root=private_root,
        )
        for private_root in _PRIVATE_LAYOUT_ROOTS
        if _path_claim_exists(
            trial_root.joinpath(*(private_root / requested / _MANIFEST_NAME).parts)
        )
    ]
    if not candidates:
        raise ArtifactSetNotFoundError(
            f"no private VerificationArtifactSet found for verification_id {requested!r}"
        )
    loaded_claims = [
        _load_candidate(
            trial_root,
            candidate,
            expected_initial_verification_id=None,
        )
        for candidate in candidates
    ]
    for candidate, loaded in zip(candidates, loaded_claims, strict=True):
        if loaded.manifest.artifact_set.verification_id != requested:
            raise ArtifactQuarantineError(
                "private VerificationArtifactSet does not match the requested verification ID",
                reason_code="private_verification_id_mismatch",
                subject_reference=candidate.relative_path.as_posix(),
            )
    return _resolve_verification(
        requested,
        loaded_claims,
        trial_root=trial_root,
    )


__all__ = [
    "ArtifactCopy",
    "ArtifactDiscoveryError",
    "ArtifactLayout",
    "ArtifactQuarantineError",
    "ArtifactScanResult",
    "ArtifactSetNotFoundError",
    "DiscoveredVerificationArtifacts",
    "ResolvedArtifact",
    "SemanticValidationScanResult",
    "VerificationArtifactSetCopy",
    "VerificationManifestCopy",
    "discover_verification_artifacts",
    "read_verification_artifacts",
    "read_private_verification_artifacts",
    "scan_verification_artifact_semantics",
    "scan_verification_artifacts",
    "validate_verification_artifact_semantics",
]
