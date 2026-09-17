"""Strict loaders for host-side evaluation configuration.

Evaluation policy is deliberately independent of qsim ``TaskSpec``.  These
helpers read only ``configs/evaluation`` (or an explicitly supplied root) and
validate each document through its versioned Pydantic contract.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel
from ruamel.yaml import YAML

if TYPE_CHECKING:
    from qiqcbench.eval.contracts.identity import (
        AnyTaskEditionManifest,
        InstanceRef,
        VerifierRevisionManifest,
    )

ModelT = TypeVar("ModelT", bound=BaseModel)

_CONFIG_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_yaml = YAML(typ="safe")
_yaml.allow_duplicate_keys = False


def evaluation_config_root(root: str | Path | None = None) -> Path:
    """Return the evaluation-config root without importing qsim config code."""
    if root is not None:
        return Path(root)
    if configs := os.environ.get("QIQCBENCH_CONFIGS"):
        return Path(configs) / "evaluation"
    return Path(__file__).resolve().parents[4] / "configs" / "evaluation"


def validate_config_id(value: str, *, field: str = "config_id") -> str:
    """Reject path traversal and ambiguous filesystem identifiers."""
    if not isinstance(value, str) or _CONFIG_ID.fullmatch(value) is None:
        raise ValueError(f"Invalid {field} {value!r}")
    return value


def load_yaml_model(path: str | Path, model_type: type[ModelT]) -> ModelT:
    """Load one YAML mapping and validate it with ``model_type``."""
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        payload = _yaml.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation config must be a mapping: {config_path}")
    return model_type.model_validate(payload)


def load_player(player_name: str, *, root: str | Path | None = None):
    from qiqcbench.eval.contracts.identity import PlayerManifest

    name = validate_config_id(player_name, field="player_name")
    return load_yaml_model(
        evaluation_config_root(root) / "players" / f"{name}.yaml", PlayerManifest
    )


def load_execution_profile(profile_name: str, *, root: str | Path | None = None):
    from qiqcbench.eval.contracts.identity import ExecutionProfile

    name = validate_config_id(profile_name, field="profile_name")
    path = evaluation_config_root(root) / "execution_profiles" / f"{name}.yaml"
    return load_yaml_model(path, ExecutionProfile)


def load_rating_policy(policy_id: str, *, root: str | Path | None = None):
    """Load one rating policy through the explicit ID-to-model resolver."""

    from qiqcbench.eval.aggregation.policy import rating_policy_model_for

    policy = validate_config_id(policy_id, field="rating_policy_id")
    model_type = rating_policy_model_for(policy)
    path = evaluation_config_root(root) / "rating_policies" / f"{policy}.yaml"
    manifest = load_yaml_model(path, model_type)
    if manifest.rating_policy_id != policy:
        raise ValueError(
            f"Rating policy ID {manifest.rating_policy_id!r} does not match catalog path {policy!r}"
        )
    return manifest


def load_evidence_policy(policy_id: str, *, root: str | Path | None = None):
    """Load one evidence policy through the explicit ID-to-model resolver."""

    from qiqcbench.eval.aggregation.policy import evidence_policy_model_for

    policy = validate_config_id(policy_id, field="evidence_policy_id")
    model_type = evidence_policy_model_for(policy)
    path = evaluation_config_root(root) / "evidence_policies" / f"{policy}.yaml"
    manifest = load_yaml_model(path, model_type)
    if manifest.evidence_policy_id != policy:
        raise ValueError(
            "Evidence policy ID "
            f"{manifest.evidence_policy_id!r} does not match catalog path {policy!r}"
        )
    return manifest


def load_instance_ref(
    task_id: str,
    instance_name: str,
    *,
    root: str | Path | None = None,
) -> InstanceRef:
    """Load one public-safe instance reference from the task catalog."""
    from qiqcbench.eval.contracts.identity import InstanceRef

    task = validate_config_id(task_id, field="task_id")
    instance = validate_config_id(instance_name, field="instance_name")
    path = evaluation_config_root(root) / "instance_sets" / task / f"{instance}.yaml"
    return load_yaml_model(path, InstanceRef)


def load_task_edition(
    task_id: str,
    edition_name: str,
    *,
    root: str | Path | None = None,
):
    from qiqcbench.eval.contracts.identity import validate_task_edition_manifest

    task = validate_config_id(task_id, field="task_id")
    edition = validate_config_id(edition_name, field="edition_name")
    path = evaluation_config_root(root) / "task_editions" / task / f"{edition}.yaml"
    with path.open(encoding="utf-8") as stream:
        payload = _yaml.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation config must be a mapping: {path}")
    manifest = validate_task_edition_manifest(payload)
    if manifest.task_id != task:
        raise ValueError(
            f"TaskEdition task_id {manifest.task_id!r} does not match catalog path {task!r}"
        )
    return manifest


def load_verifier_revision(
    task_id: str,
    revision_name: str,
    *,
    root: str | Path | None = None,
    task_edition: AnyTaskEditionManifest | None = None,
) -> VerifierRevisionManifest:
    from qiqcbench.eval.contracts.identity import (
        VerifierRevisionManifest,
        validate_verifier_revision_for_edition,
    )

    task = validate_config_id(task_id, field="task_id")
    revision = validate_config_id(revision_name, field="revision_name")
    path = evaluation_config_root(root) / "verifier_revisions" / task / f"{revision}.yaml"
    manifest = load_yaml_model(path, VerifierRevisionManifest)
    if manifest.task_id != task:
        raise ValueError(
            f"VerifierRevision task_id {manifest.task_id!r} does not match catalog path {task!r}"
        )
    if task_edition is not None:
        validate_verifier_revision_for_edition(task_edition, manifest)
    return manifest


def load_task_edition_by_id(
    task_edition_id: str,
    *,
    root: str | Path | None = None,
):
    """Resolve exactly one TaskEdition by its content-derived identity."""

    expected = validate_config_id(task_edition_id, field="task_edition_id")
    matches = [
        load_task_edition(task_id, edition_name, root=root)
        for task_id, edition_name in list_task_editions(root=root)
    ]
    matches = [manifest for manifest in matches if manifest.task_edition_id == expected]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one TaskEdition for {expected!r}; found {len(matches)}")
    return matches[0]


def load_verifier_revision_by_id(
    verifier_revision_id: str,
    *,
    root: str | Path | None = None,
    task_edition: AnyTaskEditionManifest | None = None,
):
    """Resolve exactly one VerifierRevision by its content-derived identity."""

    from qiqcbench.eval.contracts.identity import validate_verifier_revision_for_edition

    expected = validate_config_id(verifier_revision_id, field="verifier_revision_id")
    matches = [
        load_verifier_revision(task_id, revision_name, root=root)
        for task_id, revision_name in list_verifier_revisions(root=root)
    ]
    matches = [manifest for manifest in matches if manifest.verifier_revision_id == expected]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one VerifierRevision for {expected!r}; found {len(matches)}"
        )
    manifest = matches[0]
    if task_edition is not None:
        validate_verifier_revision_for_edition(task_edition, manifest)
    return manifest


def list_task_editions(*, root: str | Path | None = None) -> list[tuple[str, str]]:
    """List ``(task_id, edition filename stem)`` pairs in stable order."""
    catalog = evaluation_config_root(root) / "task_editions"
    if not catalog.is_dir():
        return []
    editions: list[tuple[str, str]] = []
    for path in catalog.glob("*/*.yaml"):
        task_id = validate_config_id(path.parent.name, field="task_id")
        edition_name = validate_config_id(path.stem, field="edition_name")
        editions.append((task_id, edition_name))
    return sorted(editions)


def list_verifier_revisions(*, root: str | Path | None = None) -> list[tuple[str, str]]:
    """List ``(task_id, revision filename stem)`` pairs in stable order."""

    catalog = evaluation_config_root(root) / "verifier_revisions"
    if not catalog.is_dir():
        return []
    revisions: list[tuple[str, str]] = []
    for path in catalog.glob("*/*.yaml"):
        task_id = validate_config_id(path.parent.name, field="task_id")
        revision_name = validate_config_id(path.stem, field="revision_name")
        revisions.append((task_id, revision_name))
    return sorted(revisions)


def list_rating_policies(*, root: str | Path | None = None) -> list[str]:
    """List checked rating-policy artifact names in stable order."""

    catalog = evaluation_config_root(root) / "rating_policies"
    if not catalog.is_dir():
        return []
    return sorted(
        validate_config_id(path.stem, field="rating_policy_id") for path in catalog.glob("*.yaml")
    )


def list_evidence_policies(*, root: str | Path | None = None) -> list[str]:
    """List checked evidence-policy artifact names in stable order."""

    catalog = evaluation_config_root(root) / "evidence_policies"
    if not catalog.is_dir():
        return []
    return sorted(
        validate_config_id(path.stem, field="evidence_policy_id") for path in catalog.glob("*.yaml")
    )


def list_instance_refs(
    task_id: str | None = None,
    *,
    root: str | Path | None = None,
) -> list[tuple[str, str]]:
    """List public ``(task_id, instance filename stem)`` pairs in stable order."""
    catalog = evaluation_config_root(root) / "instance_sets"
    if task_id is None:
        if not catalog.is_dir():
            return []
        paths = catalog.glob("*/*.yaml")
    else:
        task = validate_config_id(task_id, field="task_id")
        task_catalog = catalog / task
        if not task_catalog.is_dir():
            return []
        paths = task_catalog.glob("*.yaml")

    instances: list[tuple[str, str]] = []
    for path in paths:
        task = validate_config_id(path.parent.name, field="task_id")
        instance_name = validate_config_id(path.stem, field="instance_name")
        instances.append((task, instance_name))
    return sorted(instances)
