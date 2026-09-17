"""Optional free-form rubric dual-write into canonical verifier artifacts."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from qiqcbench.eval.contracts.outcome import (
    OutcomeDisposition,
    VerificationManifest,
)
from qiqcbench.eval.io.canonical import sha256_bytes
from qiqcbench.eval.io.legacy import (
    FreeformRubricOutcomeContext,
    LegacyProjectionError,
    project_freeform_rubric_report_bytes,
)
from qiqcbench.eval.verifier.bindings import (
    EvidenceIntegrityError,
    bind_verification,
    resolve_predecessor_verification,
    target_trial_reference,
)
from qiqcbench.eval.verifier.context import VerifierRunContext, load_verifier_run_context
from qiqcbench.eval.verifier.writer import (
    ArtifactPayload,
    VerificationWriteResult,
    write_verification_artifacts,
)

_REPORT_ROLE = "task_score_report"
_CONTEXT_UNSET = object()


def _strict_reward_bytes(reward_binary: int) -> bytes:
    if type(reward_binary) is not int or reward_binary not in (0, 1):
        raise TypeError("reward_binary must be the integer 0 or 1")
    return b"1\n" if reward_binary == 1 else b"0\n"


def _write_compatibility_bytes(destination: Path, payload: bytes) -> None:
    """Write an absent legacy output, never clobbering an existing one."""

    if os.path.lexists(destination):
        if destination.is_symlink() or not destination.is_file():
            raise FileExistsError("legacy compatibility output is not a regular file")
        if destination.read_bytes() != payload:
            raise FileExistsError("legacy compatibility output contains different bytes")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)


def emit_legacy_rubric_report(
    report_bytes: bytes,
    *,
    reward_binary: int,
    report_path: str | os.PathLike[str],
    reward_path: str | os.PathLike[str],
    trial_root: str | os.PathLike[str],
    evidence_paths: Mapping[str, str | os.PathLike[str]],
    disposition: OutcomeDisposition,
    environ: Mapping[str, str] | None = None,
    config_root: str | os.PathLike[str] | None = None,
    predecessor_verification: VerificationManifest | None = None,
    verifier_context: VerifierRunContext | None | object = _CONTEXT_UNSET,
) -> VerificationWriteResult | None:
    """Preserve exact legacy bytes and optionally add a fully bound v2 set.

    Absence of verifier context remains the permissive task-development path.
    Presence opts into strict catalog, identity, evidence, adapter, and parity
    validation before any compatibility path is published.
    """

    if not isinstance(report_bytes, bytes):
        raise TypeError("report_bytes must be exact bytes")
    reward_bytes = _strict_reward_bytes(reward_binary)
    context = (
        load_verifier_run_context(environ)
        if verifier_context is _CONTEXT_UNSET
        else verifier_context
    )
    if context is not None and not isinstance(context, VerifierRunContext):
        raise TypeError("verifier_context must be a VerifierRunContext or None")
    report_destination = Path(report_path)
    reward_destination = Path(reward_path)
    root = Path(trial_root)
    if context is None:
        _write_compatibility_bytes(report_destination, report_bytes)
        _write_compatibility_bytes(reward_destination, reward_bytes)
        return None

    if _REPORT_ROLE in evidence_paths:
        raise ValueError(f"evidence_paths must not replace reserved role {_REPORT_ROLE!r}")
    report_reference = target_trial_reference(
        report_destination,
        trial_root=root,
        role="task score report",
    )
    reward_reference = target_trial_reference(
        reward_destination,
        trial_root=root,
        role="reward projection",
    )
    try:
        bound = bind_verification(
            context=context,
            trial_root=root,
            evidence_paths=evidence_paths,
            config_root=config_root,
        )
    except EvidenceIntegrityError:
        raise
    except ValueError as exc:
        raise LegacyProjectionError(str(exc)) from exc
    adapters = [
        adapter
        for adapter in bound.verifier_revision.legacy_adapters
        if adapter.adapter_kind == "freeform_rubric_json_v1"
        and adapter.artifact_role == _REPORT_ROLE
    ]
    if len(adapters) != 1:
        raise LegacyProjectionError(
            "rubric dual-write requires exactly one revision-owned task-score adapter"
        )
    outcome = project_freeform_rubric_report_bytes(
        report_bytes,
        reward_bytes,
        context=FreeformRubricOutcomeContext(
            attempt=context.attempt,
            execution=context.execution,
            verification=bound.verification,
            disposition=disposition,
            task_report_digest=sha256_bytes(report_bytes),
            public_material_digests={
                "instruction": bound.task_edition.instruction_digest,
                **bound.task_edition.public_material_digests,
            },
        ),
        task_edition=bound.task_edition,
        verifier_revision=bound.verifier_revision,
        adapter_id=adapters[0].adapter_id,
    )

    report_parent = Path(report_reference).parent
    compatibility_paths = {
        "verified_outcome": (report_parent / "verified_outcome.json").as_posix(),
        "reward_projection": reward_reference,
        "task_score_report": report_reference,
    }
    predecessor = resolve_predecessor_verification(
        context=context,
        trial_root=root,
        supplied=predecessor_verification,
    )
    return write_verification_artifacts(
        trial_root=root,
        verification=bound.verification,
        outcome=outcome,
        execution=context.execution,
        attempt=context.attempt,
        task_edition=bound.task_edition,
        verifier_revision=bound.verifier_revision,
        predecessor_verification=predecessor,
        task_report=ArtifactPayload(
            reference=Path(report_reference).name,
            content=report_bytes,
        ),
        initial_compatibility_paths=(compatibility_paths if context.reason == "initial" else None),
        initial_artifact_set_reference=(
            (report_parent / "verification_artifact_set.json").as_posix()
            if context.reason == "initial"
            else None
        ),
        private_directory_reference=(
            "verifier/artifacts/private"
            if report_parent == Path("verifier/artifacts")
            else "private"
        ),
    )


__all__ = ["emit_legacy_rubric_report"]
