"""Runtime identities for execution-bound task editions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from qiqcbench.eval.contracts.base import Digest, Identifier, StrictModel

OFFICIAL_RUNTIME_IDENTITY_INPUT_KEY = "official_runtime_identity"

# Historical runtime-manifest contracts remain readable. Current Shadow uses
# one repository-complete apparatus; old editions are not new-launch choices.
CURRENT_REPOSITORY_RUNTIME_EDITIONS = {
    "time_budgeted_shadow_surrogate_60q": (
        "task_edition_d5c95b02c016fc157ca0a9dae1800cdbe3e5ad77bd4fc2f70595ceaf79bade2b"
    ),
}


def assert_current_repository_runtime_edition(task_id: str, task_edition_id: str) -> None:
    """Reject legacy new launches without changing archived identity readers."""
    current = CURRENT_REPOSITORY_RUNTIME_EDITIONS.get(task_id)
    if current is not None and task_edition_id != current:
        raise ValueError(
            "legacy Shadow task editions are retired for new launches; use the current "
            "repository-complete task edition"
        )


# Retained for validation/interpretation of historical private-slot records.
OFFICIAL_STUDY_ANCHOR_VERIFIER_ENVS: dict[str, tuple[str, str]] = {
    "time_budgeted_shadow_surrogate_60q": (
        "QIQCBENCH_SHADOW_OFFICIAL_STUDY_PRECOMMIT_SHA256",
        "QIQCBENCH_SHADOW_OFFICIAL_STUDY_OUTCOME_SHA256",
    ),
}
OFFICIAL_RUNTIME_MANIFEST_REQUIRED_TASKS = frozenset(OFFICIAL_STUDY_ANCHOR_VERIFIER_ENVS)


class OfficialRuntimeIdentityBinding(StrictModel):
    """Registered identity of one emitted candidate or canonical manifest."""

    schema_version: Literal[1] = 1
    release_tier: Literal["candidate", "canonical"]
    task_edition_id: Identifier
    instance_id: Identifier
    instance_source_revision_value: Digest
    edition_manifest_sha256: Digest
    study_precommit_sha256: Digest
    study_outcome_sha256: Digest


def official_runtime_identity_from_runtime_inputs(
    runtime_inputs: Mapping[str, object],
) -> OfficialRuntimeIdentityBinding | None:
    """Return the typed binding, failing closed when a present value is malformed."""

    payload = runtime_inputs.get(OFFICIAL_RUNTIME_IDENTITY_INPUT_KEY)
    if payload is None:
        return None
    try:
        return OfficialRuntimeIdentityBinding.model_validate(payload)
    except ValueError as exc:
        raise ValueError("official runtime identity in the instance binding is malformed") from exc
