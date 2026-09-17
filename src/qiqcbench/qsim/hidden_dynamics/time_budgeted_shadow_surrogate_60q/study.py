"""Prospective three-root release study for the surrogate task.

The study is deliberately task-local.  It freezes one scientific method and
three opaque, structurally admitted roots *before* any scored targets are
materialized.  Every root is then certified once, in order.  One failure (or
an interrupted started slot) retires the whole study; roots are never replaced.

Only commitments and bounded status/digest records are suitable during the
prospective study. After (and only after) a complete 3/3 pass, fixture 0 may be
opened and activated as a public *development* fixture. The master, the other
two roots, every candidate index, and all pair mappings remain owner-private.
That public fixture is never leak-safe evidence for scored runs.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import struct
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from qiqcbench.eval.io.canonical import canonical_digest, canonical_json_bytes, sha256_bytes
from qiqcbench.eval.io.private_state import NamedPrivateState

from . import construction

TASK_ID = construction.TASK_ID
SCIENTIFIC_REVISION = "windowed_parity_network_v9"
PRECOMMIT_ARTIFACT_TYPE = "shadow_surrogate_successor_study_precommit"
OUTCOME_ARTIFACT_TYPE = "shadow_surrogate_successor_study_outcome"
STUDY_SCHEMA_VERSION = 1
STUDY_ROOT_COUNT = 3
FIXTURE_ROOT_INDEX = 0
MAX_CANDIDATE_INDEX = 1_000_000

_MASTER_COMMITMENT_DOMAIN = b"qiqcbench/time_budgeted_shadow_surrogate_60q/v9/study-master/v1\0"
_ROOT_DERIVATION_DOMAIN = b"qiqcbench/time_budgeted_shadow_surrogate_60q/v9/root-candidate/v1\0"
_ROOT_COMMITMENT_DOMAIN = b"qiqcbench/time_budgeted_shadow_surrogate_60q/v9/root-opening/v1\0"
_STUDY_ID_DOMAIN = "qiqcbench/time_budgeted_shadow_surrogate_60q/v9/study-id/v1"
_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40,64}\Z")
_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_STUDY_ID = re.compile(r"shadow-surrogate-v9-[0-9a-f]{24}\Z")

# The study records these exact deterministic pre-manifest bytes for every
# slot.  Unknown or missing roles fail closed; an implementation cannot hide
# an additional release input behind an untyped filename.
PRE_MANIFEST_FILE_ROLES = (
    "hidden_device_config",
    "hidden_truth",
    "public_device_spec",
    "public_observable_index",
    "public_target_challenge",
)

# A tracked precommit may be added after the clean method commit.  These are
# the task-science paths that must still match that commit at execution time.
METHOD_PATHS = (
    "configs/harbor/time_budgeted_shadow_surrogate_60q_codex.yaml",
    "configs/tasks/time_budgeted_shadow_surrogate_60q.yaml",
    "docker/qsim",
    "harbor_tasks/time_budgeted_shadow_surrogate_60q/build_instance.py",
    "harbor_tasks/time_budgeted_shadow_surrogate_60q/environment",
    "harbor_tasks/time_budgeted_shadow_surrogate_60q/instruction.md",
    "harbor_tasks/time_budgeted_shadow_surrogate_60q/task.toml",
    "harbor_tasks/time_budgeted_shadow_surrogate_60q/tests",
    "pyproject.toml",
    # Freeze the entire qsim authority tree, not a hand-maintained subset of
    # imports.  The one excluded file is a data-only activation pointer whose
    # exact permitted post-study edit is validated separately below.
    "src/qiqcbench/qsim",
    "tools/run_shadow_surrogate_successor_study.py",
    "uv.lock",
)
ACTIVATION_POINTER_PATH = (
    "src/qiqcbench/qsim/hidden_dynamics/time_budgeted_shadow_surrogate_60q/active_fixture.py"
)

# These root-specific generated files necessarily differ after fixture-0
# emission.  They are bound by the outcome pre-manifest role map and final
# manifest, not by the earlier clean method commit.
GENERATED_OUTPUT_PATHS = (
    "configs/devices/boundedgate_60q_v0.hidden.example.yaml",
    "configs/devices/boundedgate_60q_v0.public.yaml",
    "configs/task_materials/time_budgeted_shadow_surrogate_60q/hidden/edition_manifest.json",
    "configs/task_materials/time_budgeted_shadow_surrogate_60q/hidden/successor_study_outcome.json",
    "configs/task_materials/time_budgeted_shadow_surrogate_60q/hidden/successor_study_precommit.json",
    "configs/task_materials/time_budgeted_shadow_surrogate_60q/hidden/truth.json",
    "configs/task_materials/time_budgeted_shadow_surrogate_60q/public/observable_index.json",
    "configs/task_materials/time_budgeted_shadow_surrogate_60q/public/target_challenge.json",
)

RUNTIME_RECEIPT_ENV = "QIQCBENCH_STUDY_RUNTIME_RECEIPT"
REQUIRED_CPU_LIMIT_CORES = 4
REQUIRED_MEMORY_LIMIT_BYTES = 4096 * 1024 * 1024


class StudyError(RuntimeError):
    """The prospective study contract or immutable state is invalid."""


class StudyAlreadyTerminalError(StudyError):
    """A caller attempted to rerun a terminal study."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _validate_hex64(value: str, *, label: str) -> str:
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a full lowercase SHA-256 hex digest")
    return value


class MethodBindingV1(_StrictModel):
    git_commit: str
    git_dirty: Literal[False] = False
    runtime_image_digest: str
    cpu_limit_cores: Literal[REQUIRED_CPU_LIMIT_CORES]
    memory_limit_bytes: Literal[REQUIRED_MEMORY_LIMIT_BYTES]
    resource_enforcement: Literal["container_cgroup_exact_v1"]

    @field_validator("git_commit")
    @classmethod
    def _valid_commit(cls, value: str) -> str:
        if _GIT_COMMIT.fullmatch(value) is None:
            raise ValueError("git_commit must be one full lowercase Git object ID")
        return value

    @field_validator("runtime_image_digest")
    @classmethod
    def _valid_image(cls, value: str) -> str:
        if _IMAGE_DIGEST.fullmatch(value) is None:
            raise ValueError("runtime_image_digest must be sha256:<64 lowercase hex>")
        return value


class RootSlotCommitmentV1(_StrictModel):
    slot_index: int
    root_commitment_sha256: str

    @field_validator("root_commitment_sha256")
    @classmethod
    def _valid_digest(cls, value: str) -> str:
        return _validate_hex64(value, label="root_commitment_sha256")


class RootSelectionPolicyV1(_StrictModel):
    derivation_scheme: Literal["hmac_sha256_uint64_candidate_index_v1"]
    selection_rule: Literal["first_three_structural_admission_passes_in_candidate_order"]
    candidate_index_limit_exclusive: Literal[MAX_CANDIDATE_INDEX]
    root_count: Literal[STUDY_ROOT_COUNT]
    fixture_root_index: Literal[FIXTURE_ROOT_INDEX]
    replacement_allowed: Literal[False]
    retry_after_start_allowed: Literal[False]
    ordered_root_slots: tuple[RootSlotCommitmentV1, ...]

    @model_validator(mode="after")
    def _ordered_slots(self) -> RootSelectionPolicyV1:
        if tuple(slot.slot_index for slot in self.ordered_root_slots) != tuple(
            range(STUDY_ROOT_COUNT)
        ):
            raise ValueError("ordered_root_slots must contain slots 0, 1, 2 exactly once")
        commitments = tuple(slot.root_commitment_sha256 for slot in self.ordered_root_slots)
        if len(set(commitments)) != STUDY_ROOT_COUNT:
            raise ValueError("root commitments must be unique")
        return self


class AcceptancePolicyV1(_StrictModel):
    rule: Literal["all_3_slots_pass"]
    required_passes: Literal[STUDY_ROOT_COUNT]
    first_failure_stops_later_slots: Literal[True]
    started_without_terminal_retires_study: Literal[True]
    fixture_selected_only_after_complete_pass: Literal[True]


class SuccessorStudyPrecommitV1(_StrictModel):
    schema_version: Literal[STUDY_SCHEMA_VERSION]
    artifact_type: Literal[PRECOMMIT_ARTIFACT_TYPE]
    task_id: Literal[TASK_ID]
    study_id: str
    scientific_revision: Literal[SCIENTIFIC_REVISION]
    method: MethodBindingV1
    master_key_commitment_sha256: str
    root_selection: RootSelectionPolicyV1
    acceptance: AcceptancePolicyV1
    public_policy: dict[str, Any]
    certificate_policy: dict[str, Any]

    @field_validator("study_id")
    @classmethod
    def _valid_study_id(cls, value: str) -> str:
        if _STUDY_ID.fullmatch(value) is None:
            raise ValueError("study_id has the wrong content-derived form for this edition")
        return value

    @field_validator("master_key_commitment_sha256")
    @classmethod
    def _valid_master_commitment(cls, value: str) -> str:
        return _validate_hex64(value, label="master_key_commitment_sha256")


class RootOpeningV1(_StrictModel):
    slot_index: int
    candidate_index: int
    root_seed_hex: str
    root_commitment_sha256: str

    @field_validator("root_seed_hex", "root_commitment_sha256")
    @classmethod
    def _valid_hex(cls, value: str, info: Any) -> str:
        return _validate_hex64(value, label=info.field_name)


class PrivateStudyPlanV1(_StrictModel):
    schema_version: Literal[STUDY_SCHEMA_VERSION]
    task_id: Literal[TASK_ID]
    study_id: str
    precommit_sha256: str
    master_key_hex: str
    master_key_commitment_sha256: str
    root_openings: tuple[RootOpeningV1, ...]

    @field_validator("precommit_sha256", "master_key_hex", "master_key_commitment_sha256")
    @classmethod
    def _valid_digest(cls, value: str, info: Any) -> str:
        return _validate_hex64(value, label=info.field_name)

    @model_validator(mode="after")
    def _ordered_openings(self) -> PrivateStudyPlanV1:
        if tuple(item.slot_index for item in self.root_openings) != tuple(range(STUDY_ROOT_COUNT)):
            raise ValueError("private root openings must contain slots 0, 1, 2 in order")
        indices = tuple(item.candidate_index for item in self.root_openings)
        if any(index < 0 or index >= MAX_CANDIDATE_INDEX for index in indices):
            raise ValueError("candidate index lies outside the frozen search envelope")
        if tuple(sorted(indices)) != indices or len(set(indices)) != STUDY_ROOT_COUNT:
            raise ValueError("admitted candidate indices must be unique and increasing")
        return self


class ExecutionRuntimeReceiptV1(_StrictModel):
    schema_version: Literal[STUDY_SCHEMA_VERSION]
    artifact_type: Literal["shadow_surrogate_successor_study_runtime_receipt"]
    precommit_sha256: str
    precommit_publication_git_commit: str
    runtime_image_digest: str
    container_id: str
    image_identity_source: Literal["docker_inspect_container_image_id_v1"]
    cpu_limit_cores: Literal[REQUIRED_CPU_LIMIT_CORES]
    cpu_limit_source: Literal["docker_hostconfig_plus_cgroup_v1_or_v2"]
    memory_limit_bytes: Literal[REQUIRED_MEMORY_LIMIT_BYTES]
    memory_limit_source: Literal["docker_hostconfig_plus_cgroup_v1_or_v2"]
    readonly_rootfs: Literal[True]
    repository_mount_read_only: Literal[True]
    network_mode: Literal["none"]

    @field_validator("precommit_sha256")
    @classmethod
    def _valid_precommit_digest(cls, value: str) -> str:
        return _validate_hex64(value, label="precommit_sha256")

    @field_validator("runtime_image_digest")
    @classmethod
    def _valid_image(cls, value: str) -> str:
        if _IMAGE_DIGEST.fullmatch(value) is None:
            raise ValueError("runtime receipt image digest has the wrong form")
        return value

    @field_validator("precommit_publication_git_commit")
    @classmethod
    def _valid_publication_commit(cls, value: str) -> str:
        if _GIT_COMMIT.fullmatch(value) is None:
            raise ValueError("precommit publication commit has the wrong form")
        return value

    @field_validator("container_id")
    @classmethod
    def _valid_container_id(cls, value: str) -> str:
        return _validate_hex64(value, label="container_id")


class PrivateSlotStartedV1(_StrictModel):
    schema_version: Literal[STUDY_SCHEMA_VERSION]
    study_id: str
    slot_index: int
    root_commitment_sha256: str

    @field_validator("root_commitment_sha256")
    @classmethod
    def _valid_digest(cls, value: str) -> str:
        return _validate_hex64(value, label="root_commitment_sha256")


class PrivateSlotTerminalV1(_StrictModel):
    schema_version: Literal[STUDY_SCHEMA_VERSION]
    study_id: str
    slot_index: int
    status: Literal["passed", "certificate_failed", "execution_failed"]
    certificate_sha256: str | None
    pre_manifest_files_sha256: dict[str, str] | None

    @model_validator(mode="after")
    def _artifact_shape(self) -> PrivateSlotTerminalV1:
        if self.certificate_sha256 is not None:
            _validate_hex64(self.certificate_sha256, label="certificate_sha256")
        if self.status == "passed":
            if self.certificate_sha256 is None or self.pre_manifest_files_sha256 is None:
                raise ValueError("passing terminal marker must bind certificate and files")
            if set(self.pre_manifest_files_sha256) != set(PRE_MANIFEST_FILE_ROLES):
                raise ValueError("passing terminal marker has the wrong material roles")
            for digest in self.pre_manifest_files_sha256.values():
                _validate_hex64(digest, label="pre_manifest_files_sha256 value")
        elif self.pre_manifest_files_sha256 is not None:
            raise ValueError("failed terminal marker cannot claim complete material files")
        return self


SlotStatus = Literal[
    "passed",
    "certificate_failed",
    "execution_failed",
    "started_without_terminal",
    "unrun_after_failure",
]


class StudySlotOutcomeV1(_StrictModel):
    slot_index: int
    root_commitment_sha256: str
    started: bool
    terminal: bool
    status: SlotStatus
    certificate_sha256: str | None
    pre_manifest_files_sha256: dict[str, str] | None

    @field_validator("root_commitment_sha256")
    @classmethod
    def _valid_root_commitment(cls, value: str) -> str:
        return _validate_hex64(value, label="root_commitment_sha256")

    @model_validator(mode="after")
    def _status_shape(self) -> StudySlotOutcomeV1:
        if self.certificate_sha256 is not None:
            _validate_hex64(self.certificate_sha256, label="certificate_sha256")
        if self.pre_manifest_files_sha256 is not None:
            if set(self.pre_manifest_files_sha256) != set(PRE_MANIFEST_FILE_ROLES):
                raise ValueError("pre-manifest material roles do not match the frozen inventory")
            for digest in self.pre_manifest_files_sha256.values():
                _validate_hex64(digest, label="pre_manifest_files_sha256 value")

        if self.status == "passed":
            if not self.started or not self.terminal:
                raise ValueError("a passing slot must be started and terminal")
            if self.certificate_sha256 is None or self.pre_manifest_files_sha256 is None:
                raise ValueError("a passing slot must bind its certificate and all material bytes")
        elif self.status in {"certificate_failed", "execution_failed"}:
            if not self.started or not self.terminal:
                raise ValueError("a failed slot must be started and terminal")
            if self.pre_manifest_files_sha256 is not None:
                raise ValueError("a failed slot cannot claim a complete pre-manifest file set")
        elif self.status == "started_without_terminal":
            if not self.started or self.terminal:
                raise ValueError("an interrupted slot is started but nonterminal")
            if self.certificate_sha256 is not None or self.pre_manifest_files_sha256 is not None:
                raise ValueError("an interrupted slot cannot claim terminal artifacts")
        elif self.status == "unrun_after_failure":
            if self.started or self.terminal:
                raise ValueError("an unrun slot cannot be started or terminal")
            if self.certificate_sha256 is not None or self.pre_manifest_files_sha256 is not None:
                raise ValueError("an unrun slot cannot bind artifacts")
        return self


class SuccessorStudyOutcomeV1(_StrictModel):
    schema_version: Literal[STUDY_SCHEMA_VERSION]
    artifact_type: Literal[OUTCOME_ARTIFACT_TYPE]
    task_id: Literal[TASK_ID]
    study_id: str
    precommit_sha256: str
    precommit_publication_git_commit: str
    runtime_receipt_sha256: str
    recovery_runtime_receipt_sha256: str | None
    status: Literal["passed", "retired"]
    acceptance_rule: Literal["all_3_slots_pass"]
    fixture_root_index: Literal[FIXTURE_ROOT_INDEX]
    selected_fixture_slot: Literal[FIXTURE_ROOT_INDEX] | None
    slots: tuple[StudySlotOutcomeV1, ...]

    @field_validator("precommit_sha256", "runtime_receipt_sha256")
    @classmethod
    def _valid_precommit_digest(cls, value: str) -> str:
        return _validate_hex64(value, label="precommit_sha256")

    @field_validator("precommit_publication_git_commit")
    @classmethod
    def _valid_publication_commit(cls, value: str) -> str:
        if _GIT_COMMIT.fullmatch(value) is None:
            raise ValueError("precommit publication commit has the wrong form")
        return value

    @field_validator("recovery_runtime_receipt_sha256")
    @classmethod
    def _valid_optional_recovery_digest(cls, value: str | None) -> str | None:
        if value is not None:
            _validate_hex64(value, label="recovery_runtime_receipt_sha256")
        return value

    @model_validator(mode="after")
    def _all_or_retired(self) -> SuccessorStudyOutcomeV1:
        if tuple(slot.slot_index for slot in self.slots) != tuple(range(STUDY_ROOT_COUNT)):
            raise ValueError("outcome must record all three slots in order")
        statuses = tuple(slot.status for slot in self.slots)
        if self.status == "passed":
            if statuses != ("passed",) * STUDY_ROOT_COUNT or self.selected_fixture_slot != 0:
                raise ValueError(
                    "study passes only when all three slots pass and fixture 0 is selected"
                )
            if self.recovery_runtime_receipt_sha256 is not None:
                raise ValueError("a passing study cannot use an interruption-recovery receipt")
            return self

        if self.selected_fixture_slot is not None:
            raise ValueError("a retired study cannot select a fixture")
        failure_indices = [
            index
            for index, status in enumerate(statuses)
            if status in {"certificate_failed", "execution_failed", "started_without_terminal"}
        ]
        if len(failure_indices) != 1:
            raise ValueError("a retired fail-fast outcome must contain exactly one first failure")
        first_failure = failure_indices[0]
        if (
            self.recovery_runtime_receipt_sha256 is not None
            and statuses[first_failure] != "started_without_terminal"
        ):
            raise ValueError(
                "a recovery runtime receipt is valid only for interrupted-slot retirement"
            )
        if any(status != "passed" for status in statuses[:first_failure]):
            raise ValueError("every slot before the first failure must have passed")
        if any(status != "unrun_after_failure" for status in statuses[first_failure + 1 :]):
            raise ValueError("every slot after the first failure must remain unrun")
        return self


class StudyBindingV1(_StrictModel):
    """Validated proof fields to bind into a final edition manifest."""

    schema_version: Literal[STUDY_SCHEMA_VERSION]
    study_id: str
    precommit_sha256: str
    precommit_publication_git_commit: str
    outcome_sha256: str
    fixture_root_index: Literal[FIXTURE_ROOT_INDEX]

    @field_validator("precommit_sha256", "outcome_sha256")
    @classmethod
    def _valid_digest(cls, value: str, info: Any) -> str:
        return _validate_hex64(value, label=info.field_name)

    @field_validator("precommit_publication_git_commit")
    @classmethod
    def _valid_publication_commit(cls, value: str) -> str:
        if _GIT_COMMIT.fullmatch(value) is None:
            raise ValueError("precommit publication commit has the wrong form")
        return value


# Slot 0 opens publicly as the development fixture after a 3/3 pass; the two
# remaining roots stay owner-private and are the only official-tier instances.
# The assignment is frozen so an opened slot is never relabeled between tiers:
# candidate rounds draw slot 1, canonical releases draw slot 2.
OFFICIAL_SLOT_TIERS: Mapping[int, str] = MappingProxyType({1: "candidate", 2: "canonical"})


class OfficialStudyBindingV1(_StrictModel):
    """Validated proof fields to bind into an official-tier edition manifest."""

    schema_version: Literal[STUDY_SCHEMA_VERSION]
    study_id: str
    precommit_sha256: str
    precommit_publication_git_commit: str
    outcome_sha256: str
    root_slot_index: Literal[1, 2]

    @field_validator("precommit_sha256", "outcome_sha256")
    @classmethod
    def _valid_digest(cls, value: str, info: Any) -> str:
        return _validate_hex64(value, label=info.field_name)

    @field_validator("precommit_publication_git_commit")
    @classmethod
    def _valid_publication_commit(cls, value: str) -> str:
        if _GIT_COMMIT.fullmatch(value) is None:
            raise ValueError("precommit publication commit has the wrong form")
        return value


_PROOF_SEAL = object()


@dataclass(frozen=True)
class SuccessorStudyFixtureProof:
    """Opaque loaded proof; direct construction is rejected by validation."""

    precommit: SuccessorStudyPrecommitV1
    outcome: SuccessorStudyOutcomeV1
    private_plan: PrivateStudyPlanV1
    runtime_receipt: ExecutionRuntimeReceiptV1
    _seal: object = field(repr=False, compare=False)


def _master_commitment(master_key: bytes) -> str:
    if not isinstance(master_key, bytes) or len(master_key) != 32:
        raise ValueError("study master key must be exactly 32 bytes")
    return hashlib.sha256(_MASTER_COMMITMENT_DOMAIN + master_key).hexdigest()


def derive_candidate_root(master_key: bytes, candidate_index: int) -> int:
    """Derive one opaque 256-bit root from a uint64 candidate index."""

    if not isinstance(master_key, bytes) or len(master_key) != 32:
        raise ValueError("study master key must be exactly 32 bytes")
    if type(candidate_index) is not int or not 0 <= candidate_index < 2**64:
        raise ValueError("candidate_index must be a uint64")
    payload = _ROOT_DERIVATION_DOMAIN + struct.pack(">Q", candidate_index)
    return int.from_bytes(hmac.new(master_key, payload, hashlib.sha256).digest(), "big")


def root_commitment_sha256(*, slot_index: int, candidate_index: int, root_seed: int) -> str:
    if type(slot_index) is not int or not 0 <= slot_index < STUDY_ROOT_COUNT:
        raise ValueError("slot_index is outside the frozen study roster")
    if type(candidate_index) is not int or not 0 <= candidate_index < 2**64:
        raise ValueError("candidate_index must be a uint64")
    if type(root_seed) is not int or not 0 <= root_seed < 2**256:
        raise ValueError("root_seed must be a uint256")
    payload = (
        _ROOT_COMMITMENT_DOMAIN
        + struct.pack(">Q", slot_index)
        + struct.pack(">Q", candidate_index)
        + root_seed.to_bytes(32, "big")
    )
    return hashlib.sha256(payload).hexdigest()


def _excluded_seed_values() -> frozenset[int]:
    return frozenset(
        {
            *construction.RETIRED_DEVELOPMENT_INSTANCE_SEEDS,
            *construction.ROUND7_BURNED_CERT_DESIGN_SEEDS,
            *construction.ROUND7_UNCONSUMED_CERT_DESIGN_SEEDS,
            *construction.V8_CERT_DESIGN_SEEDS,
            construction.V8_CERT_ABLATION_DESIGN_SEED,
            *construction.V9_STUDY1_BURNED_CERT_DESIGN_SEEDS,
            construction.V9_STUDY1_BURNED_CERT_ABLATION_DESIGN_SEED,
            *construction.V9_STUDY2_BURNED_CERT_DESIGN_SEEDS,
            construction.V9_STUDY2_BURNED_CERT_ABLATION_DESIGN_SEED,
            *construction.V9_STUDY3_BURNED_CERT_DESIGN_SEEDS,
            construction.V9_STUDY3_BURNED_CERT_ABLATION_DESIGN_SEED,
            *construction.V9_STUDY4_BURNED_CERT_DESIGN_SEEDS,
            construction.V9_STUDY4_BURNED_CERT_ABLATION_DESIGN_SEED,
            *construction.CERT_DESIGN_SEEDS,
            construction.CERT_ABLATION_DESIGN_SEED,
        }
    )


def _validate_fresh_stream_roster() -> None:
    retired_streams = {
        *construction.ROUND7_BURNED_CERT_DESIGN_SEEDS,
        *construction.ROUND7_UNCONSUMED_CERT_DESIGN_SEEDS,
        *construction.V8_CERT_DESIGN_SEEDS,
        construction.V8_CERT_ABLATION_DESIGN_SEED,
        *construction.V9_STUDY1_BURNED_CERT_DESIGN_SEEDS,
        construction.V9_STUDY1_BURNED_CERT_ABLATION_DESIGN_SEED,
        *construction.V9_STUDY2_BURNED_CERT_DESIGN_SEEDS,
        construction.V9_STUDY2_BURNED_CERT_ABLATION_DESIGN_SEED,
        *construction.V9_STUDY3_BURNED_CERT_DESIGN_SEEDS,
        construction.V9_STUDY3_BURNED_CERT_ABLATION_DESIGN_SEED,
        *construction.V9_STUDY4_BURNED_CERT_DESIGN_SEEDS,
        construction.V9_STUDY4_BURNED_CERT_ABLATION_DESIGN_SEED,
    }
    fresh = {*construction.CERT_DESIGN_SEEDS, construction.CERT_ABLATION_DESIGN_SEED}
    if len(construction.CERT_DESIGN_SEEDS) != 3 or len(fresh) != 4:
        raise StudyError(
            "version-9 certificate streams must contain three global plus one ablation"
        )
    if fresh.intersection(retired_streams):
        raise StudyError("version-9 certificate streams reuse a retired earlier-edition stream")
    if {62345, 72345}.intersection(fresh):
        raise StudyError("round-7 streams 62345/72345 are permanently unavailable")


def current_public_policy() -> dict[str, Any]:
    from .scorer import scorer_policy

    observable_index = construction.build_public_observable_index()
    return {
        "observable_index_sha256": canonical_digest(observable_index),
        "response_class": observable_index["response_class"],
        "score_gates": scorer_policy(),
        "runtime_budget": {
            "total_shot_budget": construction.TOTAL_SHOT_BUDGET,
            "max_jobs": construction.MAX_JOBS,
            "max_blocks_per_job": construction.MAX_BLOCKS_PER_JOB,
            "max_settings_per_job": construction.MAX_SETTINGS_PER_JOB,
            "max_shots_per_setting": construction.MAX_SHOTS_PER_SETTING,
        },
    }


def current_certificate_policy() -> dict[str, Any]:
    _validate_fresh_stream_roster()
    ablation_budgets = {
        protocol: {
            "shots_used": construction.certificate_expected_budget(protocol)[0],
            "jobs_used": construction.certificate_expected_budget(protocol)[1],
        }
        for protocol in sorted(construction.CERT_ABLATION_EXPECTED_GATE)
    }
    return {
        "certificate_schema_version": construction.CERTIFICATE_SCHEMA_VERSION,
        "reference_policy_id": construction.REFERENCE_POLICY_ID,
        "global_design_seeds": list(construction.CERT_DESIGN_SEEDS),
        "ablation_design_seed": construction.CERT_ABLATION_DESIGN_SEED,
        "ablation_expected_gates": dict(sorted(construction.CERT_ABLATION_EXPECTED_GATE.items())),
        "minimum_margins": {
            "accuracy": construction.CERT_MIN_ACCURACY_MARGIN,
            "worst_target": construction.CERT_MIN_WORST_TARGET_MARGIN,
            "tail": construction.CERT_MIN_TAIL_MARGIN,
        },
        "budgets": {
            "global": {
                "shots_used": construction.CERT_GLOBAL_BUDGET[0],
                "jobs_used": construction.CERT_GLOBAL_BUDGET[1],
            },
            "even_xyz": {
                "shots_used": construction.CERT_EVEN_XYZ_BUDGET[0],
                "jobs_used": construction.CERT_EVEN_XYZ_BUDGET[1],
            },
            "ablations": ablation_budgets,
        },
    }


def _precommit_without_id(
    precommit: SuccessorStudyPrecommitV1 | Mapping[str, Any],
) -> dict[str, Any]:
    payload = (
        precommit.model_dump(mode="json")
        if isinstance(precommit, SuccessorStudyPrecommitV1)
        else dict(precommit)
    )
    payload.pop("study_id", None)
    return payload


def _study_id(precommit_without_id: Mapping[str, Any]) -> str:
    digest = canonical_digest(
        {"domain": _STUDY_ID_DOMAIN, "precommit_without_study_id": dict(precommit_without_id)}
    )
    return f"shadow-surrogate-v9-{digest[:24]}"


def validate_precommit(precommit: SuccessorStudyPrecommitV1) -> None:
    if precommit.study_id != _study_id(_precommit_without_id(precommit)):
        raise StudyError("precommit study_id does not match its canonical content")
    if precommit.public_policy != current_public_policy():
        raise StudyError("precommit public policy does not match the method in force")
    if precommit.certificate_policy != current_certificate_policy():
        raise StudyError("precommit certificate policy does not match the method in force")


def _is_admitted(candidate: object) -> bool:
    admission = getattr(candidate, "admission", None)
    return getattr(admission, "passed", None) is True


def build_precommit_and_private_plan(
    master_key: bytes,
    *,
    git_commit: str,
    runtime_image_digest: str,
    admission_builder: Callable[[int], object] = construction.build_admission_only,
) -> tuple[SuccessorStudyPrecommitV1, PrivateStudyPlanV1]:
    """Select the first three target-free structural passes and freeze them."""

    master_commitment = _master_commitment(master_key)
    method = MethodBindingV1(
        git_commit=git_commit,
        git_dirty=False,
        runtime_image_digest=runtime_image_digest,
        cpu_limit_cores=REQUIRED_CPU_LIMIT_CORES,
        memory_limit_bytes=REQUIRED_MEMORY_LIMIT_BYTES,
        resource_enforcement="container_cgroup_exact_v1",
    )
    excluded = _excluded_seed_values()
    openings: list[RootOpeningV1] = []
    commitments: list[RootSlotCommitmentV1] = []
    for candidate_index in range(MAX_CANDIDATE_INDEX):
        root_seed = derive_candidate_root(master_key, candidate_index)
        if root_seed in excluded:
            continue
        candidate = admission_builder(root_seed)
        if not _is_admitted(candidate):
            continue
        slot_index = len(openings)
        commitment = root_commitment_sha256(
            slot_index=slot_index,
            candidate_index=candidate_index,
            root_seed=root_seed,
        )
        openings.append(
            RootOpeningV1(
                slot_index=slot_index,
                candidate_index=candidate_index,
                root_seed_hex=root_seed.to_bytes(32, "big").hex(),
                root_commitment_sha256=commitment,
            )
        )
        commitments.append(
            RootSlotCommitmentV1(
                slot_index=slot_index,
                root_commitment_sha256=commitment,
            )
        )
        if len(openings) == STUDY_ROOT_COUNT:
            break
    if len(openings) != STUDY_ROOT_COUNT:
        raise StudyError(
            f"fewer than {STUDY_ROOT_COUNT} structurally admitted roots in the frozen search envelope"
        )

    body: dict[str, Any] = {
        "schema_version": STUDY_SCHEMA_VERSION,
        "artifact_type": PRECOMMIT_ARTIFACT_TYPE,
        "task_id": TASK_ID,
        "scientific_revision": SCIENTIFIC_REVISION,
        "method": method.model_dump(mode="json"),
        "master_key_commitment_sha256": master_commitment,
        "root_selection": {
            "derivation_scheme": "hmac_sha256_uint64_candidate_index_v1",
            "selection_rule": "first_three_structural_admission_passes_in_candidate_order",
            "candidate_index_limit_exclusive": MAX_CANDIDATE_INDEX,
            "root_count": STUDY_ROOT_COUNT,
            "fixture_root_index": FIXTURE_ROOT_INDEX,
            "replacement_allowed": False,
            "retry_after_start_allowed": False,
            "ordered_root_slots": [item.model_dump(mode="json") for item in commitments],
        },
        "acceptance": {
            "rule": "all_3_slots_pass",
            "required_passes": STUDY_ROOT_COUNT,
            "first_failure_stops_later_slots": True,
            "started_without_terminal_retires_study": True,
            "fixture_selected_only_after_complete_pass": True,
        },
        "public_policy": current_public_policy(),
        "certificate_policy": current_certificate_policy(),
    }
    precommit = SuccessorStudyPrecommitV1.model_validate({"study_id": _study_id(body), **body})
    validate_precommit(precommit)
    plan = PrivateStudyPlanV1(
        schema_version=STUDY_SCHEMA_VERSION,
        task_id=TASK_ID,
        study_id=precommit.study_id,
        precommit_sha256=canonical_digest(precommit),
        master_key_hex=master_key.hex(),
        master_key_commitment_sha256=master_commitment,
        root_openings=tuple(openings),
    )
    _validate_plan_against_precommit(plan, precommit)
    _validate_structural_selection(plan, admission_builder=admission_builder)
    return precommit, plan


def _validate_plan_against_precommit(
    plan: PrivateStudyPlanV1,
    precommit: SuccessorStudyPrecommitV1,
) -> None:
    if plan.study_id != precommit.study_id:
        raise StudyError("private plan study_id does not match the public precommit")
    if plan.precommit_sha256 != canonical_digest(precommit):
        raise StudyError("private plan is not bound to the exact public precommit")
    master_key = bytes.fromhex(plan.master_key_hex)
    if len(master_key) != 32:
        raise StudyError("private plan master key is not 32 bytes")
    if _master_commitment(master_key) != plan.master_key_commitment_sha256:
        raise StudyError("private plan master key does not match its commitment")
    if plan.master_key_commitment_sha256 != precommit.master_key_commitment_sha256:
        raise StudyError("private plan has the wrong master-key commitment")
    public_slots = precommit.root_selection.ordered_root_slots
    for opening, public_slot in zip(plan.root_openings, public_slots, strict=True):
        root_seed = int(opening.root_seed_hex, 16)
        if derive_candidate_root(master_key, opening.candidate_index) != root_seed:
            raise StudyError(f"private root opening {opening.slot_index} is not HMAC-derived")
        expected = root_commitment_sha256(
            slot_index=opening.slot_index,
            candidate_index=opening.candidate_index,
            root_seed=root_seed,
        )
        if opening.root_commitment_sha256 != expected:
            raise StudyError(f"private root opening {opening.slot_index} is invalid")
        if public_slot.root_commitment_sha256 != expected:
            raise StudyError(f"private root opening {opening.slot_index} is not precommitted")
        if root_seed in _excluded_seed_values():
            raise StudyError(f"private root opening {opening.slot_index} reuses a retired seed")


def _validate_structural_selection(
    plan: PrivateStudyPlanV1,
    *,
    admission_builder: Callable[[int], object],
) -> None:
    """Reconstruct the first three passes; no earlier admitted root may be skipped."""

    master_key = bytes.fromhex(plan.master_key_hex)
    expected_indices = tuple(item.candidate_index for item in plan.root_openings)
    admitted: list[tuple[int, int]] = []
    for candidate_index in range(expected_indices[-1] + 1):
        root_seed = derive_candidate_root(master_key, candidate_index)
        if root_seed in _excluded_seed_values():
            continue
        if _is_admitted(admission_builder(root_seed)):
            admitted.append((candidate_index, root_seed))
        if len(admitted) == STUDY_ROOT_COUNT:
            break
    observed_indices = tuple(index for index, _seed in admitted)
    if observed_indices != expected_indices:
        raise StudyError(
            "private plan is not the first three structural-admission passes in candidate order"
        )
    for opening, (_index, root_seed) in zip(plan.root_openings, admitted, strict=True):
        if int(opening.root_seed_hex, 16) != root_seed:
            raise StudyError("structural-selection replay derived a different root opening")


def _model_from_canonical_bytes(payload: bytes, model: type[_StrictModel]) -> _StrictModel:
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StudyError("study artifact is not valid JSON") from exc
    parsed = model.model_validate(raw)
    if canonical_json_bytes(parsed) != payload:
        raise StudyError("study artifact is not exact canonical JSON")
    return parsed


def load_precommit(path: str | Path) -> SuccessorStudyPrecommitV1:
    parsed = _model_from_canonical_bytes(
        Path(path).read_bytes(),
        SuccessorStudyPrecommitV1,
    )
    assert isinstance(parsed, SuccessorStudyPrecommitV1)
    validate_precommit(parsed)
    return parsed


def load_outcome(path: str | Path) -> SuccessorStudyOutcomeV1:
    parsed = _model_from_canonical_bytes(Path(path).read_bytes(), SuccessorStudyOutcomeV1)
    assert isinstance(parsed, SuccessorStudyOutcomeV1)
    return parsed


def _git(
    repository_root: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", os.fspath(repository_root), *args],
        check=check,
        capture_output=True,
    )


def clean_method_commit(repository_root: str | Path) -> str:
    """Return HEAD only when the whole repository is clean before precommit."""

    root = Path(repository_root).resolve()
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all").stdout
    if status:
        raise StudyError("prepare requires a completely clean Git working tree")
    commit = _git(root, "rev-parse", "HEAD").stdout.decode().strip()
    if _GIT_COMMIT.fullmatch(commit) is None:
        raise StudyError("Git returned an invalid full commit object ID")
    return commit


def _tracked_exact_payload(path: Path, repository_root: Path) -> bytes:
    root = repository_root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise StudyError("tracked study artifact must live inside the repository") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise StudyError("tracked study artifact must be a regular non-symlink file")
    reference = relative.as_posix()
    tracked = _git(root, "ls-files", "--error-unmatch", "--", reference, check=False)
    if tracked.returncode != 0:
        raise StudyError(f"study artifact is not tracked by Git: {reference}")
    head_payload = _git(root, "show", f"HEAD:{reference}", check=False)
    if head_payload.returncode != 0 or head_payload.stdout != resolved.read_bytes():
        raise StudyError("tracked study artifact bytes do not match the committed HEAD blob")
    return head_payload.stdout


def load_tracked_precommit(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> SuccessorStudyPrecommitV1:
    root = Path(repository_root).resolve()
    parsed = _load_tracked_precommit_artifact(Path(path), root)
    _validate_frozen_method(parsed, root, allowed_activation_seed=None)
    return parsed


def _load_tracked_precommit_artifact(
    path: Path,
    repository_root: Path,
) -> SuccessorStudyPrecommitV1:
    root = repository_root.resolve()
    payload = _tracked_exact_payload(Path(path), root)
    parsed = _model_from_canonical_bytes(payload, SuccessorStudyPrecommitV1)
    assert isinstance(parsed, SuccessorStudyPrecommitV1)
    validate_precommit(parsed)
    return parsed


def _activation_seed(payload: bytes) -> tuple[int, tuple[bytes, bytes]]:
    matches = list(re.finditer(rb"(?m)^INSTANCE_SEED = ([0-9]+)$", payload))
    if len(matches) != 1:
        raise StudyError(
            "activation pointer must contain one exact decimal INSTANCE_SEED assignment"
        )
    match = matches[0]
    return int(match.group(1)), (payload[: match.start(1)], payload[match.end(1) :])


def _validate_frozen_method(
    precommit: SuccessorStudyPrecommitV1,
    repository_root: Path,
    *,
    allowed_activation_seed: int | None,
) -> None:
    root = repository_root.resolve()
    commit_exists = _git(
        root,
        "cat-file",
        "-e",
        f"{precommit.method.git_commit}^{{commit}}",
        check=False,
    )
    if commit_exists.returncode != 0:
        raise StudyError("precommitted method Git object is unavailable")
    method_diff = _git(
        root,
        "diff",
        "--quiet",
        precommit.method.git_commit,
        "--",
        *METHOD_PATHS,
        f":(exclude){ACTIVATION_POINTER_PATH}",
        check=False,
    )
    if method_diff.returncode != 0:
        raise StudyError("task-science method paths differ from the clean precommitted commit")
    frozen_pointer = _git(
        root,
        "show",
        f"{precommit.method.git_commit}:{ACTIVATION_POINTER_PATH}",
        check=False,
    )
    if frozen_pointer.returncode != 0:
        raise StudyError("precommitted activation pointer is unavailable")
    current_pointer = _tracked_exact_payload(root / ACTIVATION_POINTER_PATH, root)
    frozen_seed, frozen_sides = _activation_seed(frozen_pointer.stdout)
    current_seed, current_sides = _activation_seed(current_pointer)
    if frozen_seed not in construction.RETIRED_DEVELOPMENT_INSTANCE_SEEDS:
        raise StudyError("method-freeze commit must retain the explicitly retired fixture pointer")
    if current_sides != frozen_sides:
        raise StudyError("activation pointer changed outside its one decimal seed literal")
    if current_seed == frozen_seed:
        return
    if allowed_activation_seed is None or current_seed != allowed_activation_seed:
        raise StudyError("activation pointer does not select the proof-bound fixture-0 root")


def _write_public_no_clobber(path: Path, payload: bytes) -> None:
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise StudyError("public study artifact parent must be an existing non-symlink directory")
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"refusing to clobber study artifact: {path}")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(path.parent, directory_flags)
    temporary_name = f".{path.name}.{secrets.token_hex(16)}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o644,
            dir_fd=parent_fd,
        )
        try:
            os.fchmod(descriptor, 0o644)
            view = memoryview(payload)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:  # pragma: no cover
                    raise OSError("study artifact write made no progress")
                written += count
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
            descriptor = None
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            raise FileExistsError(f"refusing to clobber study artifact: {path}") from None
        os.fsync(parent_fd)
    finally:
        if descriptor is not None:  # pragma: no cover - defensive cleanup
            os.close(descriptor)
        try:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)


def prepare_successor_study(
    master_key: bytes,
    *,
    git_commit: str,
    runtime_image_digest: str,
    private_state_root: str | Path,
    public_precommit_path: str | Path,
    admission_builder: Callable[[int], object] = construction.build_admission_only,
) -> SuccessorStudyPrecommitV1:
    """Prepare roots without materializing a target or running a certificate."""

    precommit, plan = build_precommit_and_private_plan(
        master_key,
        git_commit=git_commit,
        runtime_image_digest=runtime_image_digest,
        admission_builder=admission_builder,
    )
    persist_prepared_study(
        precommit,
        plan,
        private_state_root=private_state_root,
        public_precommit_path=public_precommit_path,
    )
    return precommit


def persist_prepared_study(
    precommit: SuccessorStudyPrecommitV1,
    plan: PrivateStudyPlanV1,
    *,
    private_state_root: str | Path,
    public_precommit_path: str | Path,
) -> None:
    """Publish an already-built plan once, without repeating admission."""

    validate_precommit(precommit)
    _validate_plan_against_precommit(plan, precommit)
    expected_root = Path(private_state_root)
    if expected_root.name != precommit.study_id:
        raise StudyError("private state root basename must equal the content-derived study_id")
    state = NamedPrivateState(expected_root)
    plan_payload = canonical_json_bytes(plan)
    if (
        not state.write_no_clobber("plan.json", plan_payload)
        and state.read("plan.json") != plan_payload
    ):
        raise FileExistsError("private study plan already exists with different bytes")
    _write_public_no_clobber(Path(public_precommit_path), canonical_json_bytes(precommit))


def _read_private_model(
    state: NamedPrivateState,
    reference: str,
    model: type[_StrictModel],
) -> _StrictModel | None:
    try:
        payload = state.read(reference)
    except FileNotFoundError:
        return None
    return _model_from_canonical_bytes(payload, model)


def _default_instance_materializer(candidate: object) -> object:
    return construction.materialize_admission_candidate(candidate)  # type: ignore[arg-type]


def _default_certifier(instance: object) -> dict[str, Any]:
    from .reference import certify_edition

    return certify_edition(instance)  # type: ignore[arg-type]


def build_pre_manifest_files(instance: object) -> dict[str, bytes]:
    """Render the deterministic edition inputs, excluding the final manifest."""

    from .device_configs import write_device_configs

    with tempfile.TemporaryDirectory(prefix="qiqcbench-shadow-study-") as temporary:
        root = Path(temporary)
        material_paths = construction.write_materials(root / "materials", instance)  # type: ignore[arg-type]
        device_paths = write_device_configs(root / "devices", instance)  # type: ignore[arg-type]
        role_paths = {
            "public_target_challenge": material_paths["target_challenge"],
            "public_observable_index": material_paths["observable_index"],
            "hidden_truth": material_paths["hidden_truth"],
            "public_device_spec": device_paths["public"],
            "hidden_device_config": device_paths["hidden"],
        }
        if set(role_paths) != set(PRE_MANIFEST_FILE_ROLES):  # pragma: no cover - literal guard
            raise StudyError("pre-manifest renderer returned the wrong role inventory")
        return {role: role_paths[role].read_bytes() for role in PRE_MANIFEST_FILE_ROLES}


def _cgroup_cpu_limit() -> tuple[int, str]:
    v2 = Path("/sys/fs/cgroup/cpu.max")
    if v2.is_file():
        parts = v2.read_text(encoding="ascii").strip().split()
        if len(parts) != 2 or parts[0] == "max":
            raise StudyError("successor study requires a finite cgroup-v2 CPU quota")
        quota, period = (int(value) for value in parts)
        if quota <= 0 or period <= 0 or quota % period:
            raise StudyError("cgroup-v2 CPU quota is not an exact integer core limit")
        return quota // period, "cgroup_v2_cpu.max"
    quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota_path.is_file() and period_path.is_file():
        quota = int(quota_path.read_text(encoding="ascii").strip())
        period = int(period_path.read_text(encoding="ascii").strip())
        if quota <= 0 or period <= 0 or quota % period:
            raise StudyError("cgroup-v1 CPU quota is not an exact integer core limit")
        return quota // period, "cgroup_v1_cpu.cfs_quota_us"
    raise StudyError("cannot prove the successor-study CPU cgroup limit")


def _cgroup_memory_limit() -> tuple[int, str]:
    v2 = Path("/sys/fs/cgroup/memory.max")
    if v2.is_file():
        raw = v2.read_text(encoding="ascii").strip()
        if raw == "max":
            raise StudyError("successor study requires a finite cgroup-v2 memory limit")
        return int(raw), "cgroup_v2_memory.max"
    v1 = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if v1.is_file():
        return int(v1.read_text(encoding="ascii").strip()), "cgroup_v1_memory.limit_in_bytes"
    raise StudyError("cannot prove the successor-study memory cgroup limit")


def validate_runtime_environment(
    precommit: SuccessorStudyPrecommitV1,
) -> ExecutionRuntimeReceiptV1:
    """Validate the host-inspected receipt and the in-container cgroup."""

    receipt_path = os.environ.get(RUNTIME_RECEIPT_ENV)
    if receipt_path is None:
        raise StudyError(
            f"{RUNTIME_RECEIPT_ENV} must name the host-inspected read-only runtime receipt"
        )
    parsed = _model_from_canonical_bytes(
        Path(receipt_path).read_bytes(),
        ExecutionRuntimeReceiptV1,
    )
    assert isinstance(parsed, ExecutionRuntimeReceiptV1)
    if parsed.precommit_sha256 != canonical_digest(precommit):
        raise StudyError("runtime receipt does not bind the exact tracked precommit")
    if parsed.runtime_image_digest != precommit.method.runtime_image_digest:
        raise StudyError("host-inspected container Image ID differs from the precommit")
    cpu_limit, cpu_source = _cgroup_cpu_limit()
    memory_limit, memory_source = _cgroup_memory_limit()
    if cpu_limit != REQUIRED_CPU_LIMIT_CORES:
        raise StudyError(
            f"successor study requires exactly {REQUIRED_CPU_LIMIT_CORES} cgroup CPU cores"
        )
    if memory_limit != REQUIRED_MEMORY_LIMIT_BYTES:
        raise StudyError(
            f"successor study requires exactly {REQUIRED_MEMORY_LIMIT_BYTES} cgroup memory bytes"
        )
    # Source strings in the host receipt state that Docker HostConfig was
    # inspected; these local reads independently prove the effective cgroup.
    if not cpu_source.startswith("cgroup_") or not memory_source.startswith("cgroup_"):
        raise StudyError("runtime cgroup source is unsupported")  # pragma: no cover
    return parsed


def validate_execution_repository_state(
    repository_root: str | Path,
    precommit: SuccessorStudyPrecommitV1,
    runtime_receipt: ExecutionRuntimeReceiptV1,
) -> None:
    """Require the exact clean precommit-publication HEAD before slot 0 starts."""

    root = Path(repository_root).resolve()
    head = _git(root, "rev-parse", "HEAD").stdout.decode().strip()
    if head != runtime_receipt.precommit_publication_git_commit:
        raise StudyError("repository HEAD differs from the host-inspected publication commit")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all").stdout
    if status:
        raise StudyError("successor study execution requires a completely clean worktree")
    if runtime_receipt.precommit_sha256 != canonical_digest(precommit):
        raise StudyError("runtime receipt publication does not bind the exact precommit")


def _outcome_from_slots(
    precommit: SuccessorStudyPrecommitV1,
    slots: list[StudySlotOutcomeV1],
    *,
    runtime_receipt_sha256: str,
    recovery_runtime_receipt_sha256: str | None,
    precommit_publication_git_commit: str,
) -> SuccessorStudyOutcomeV1:
    passed = all(slot.status == "passed" for slot in slots)
    return SuccessorStudyOutcomeV1(
        schema_version=STUDY_SCHEMA_VERSION,
        artifact_type=OUTCOME_ARTIFACT_TYPE,
        task_id=TASK_ID,
        study_id=precommit.study_id,
        precommit_sha256=canonical_digest(precommit),
        precommit_publication_git_commit=precommit_publication_git_commit,
        runtime_receipt_sha256=runtime_receipt_sha256,
        recovery_runtime_receipt_sha256=recovery_runtime_receipt_sha256,
        status="passed" if passed else "retired",
        acceptance_rule="all_3_slots_pass",
        fixture_root_index=FIXTURE_ROOT_INDEX,
        selected_fixture_slot=FIXTURE_ROOT_INDEX if passed else None,
        slots=tuple(slots),
    )


def _terminal_to_outcome(
    terminal: PrivateSlotTerminalV1,
    commitment: str,
    *,
    expected_study_id: str,
    expected_slot_index: int,
) -> StudySlotOutcomeV1:
    if terminal.study_id != expected_study_id:
        raise StudyError(f"slot {expected_slot_index} terminal marker has the wrong study")
    if terminal.slot_index != expected_slot_index:
        raise StudyError(f"slot {expected_slot_index} terminal marker has the wrong slot")
    return StudySlotOutcomeV1(
        slot_index=terminal.slot_index,
        root_commitment_sha256=commitment,
        started=True,
        terminal=True,
        status=terminal.status,
        certificate_sha256=terminal.certificate_sha256,
        pre_manifest_files_sha256=terminal.pre_manifest_files_sha256,
    )


def _runtime_receipt_envelope(receipt: ExecutionRuntimeReceiptV1) -> dict[str, Any]:
    """Return every host-attested field except the ephemeral container ID."""

    envelope = receipt.model_dump(mode="json")
    del envelope["container_id"]
    return envelope


def _private_payload(state: NamedPrivateState, reference: str) -> bytes | None:
    try:
        return state.read(reference)
    except FileNotFoundError:
        return None


def _interrupted_recovery_slots(
    state: NamedPrivateState,
    precommit: SuccessorStudyPrecommitV1,
) -> list[StudySlotOutcomeV1]:
    """Reconstruct only an interrupted fail-fast retirement, never execution.

    A different host-inspected container may attest and publish the terminal
    retirement after the original executor disappeared.  It cannot replay
    structural admission, materialize a target, certify a slot, or replace a
    root.  The private marker sequence must prove exactly one started slot
    without a terminal marker and no later activity.
    """

    slots: list[StudySlotOutcomeV1] = []
    interruption_seen = False
    for public_slot in precommit.root_selection.ordered_root_slots:
        slot_index = public_slot.slot_index
        commitment = public_slot.root_commitment_sha256
        started_payload = _private_payload(state, f"slots/{slot_index}/started.json")
        terminal_payload = _private_payload(state, f"slots/{slot_index}/terminal.json")
        if interruption_seen:
            if started_payload is not None or terminal_payload is not None:
                raise StudyError("interrupted recovery found activity after the first failure")
            slots.append(
                StudySlotOutcomeV1(
                    slot_index=slot_index,
                    root_commitment_sha256=commitment,
                    started=False,
                    terminal=False,
                    status="unrun_after_failure",
                    certificate_sha256=None,
                    pre_manifest_files_sha256=None,
                )
            )
            continue
        if started_payload is None:
            if terminal_payload is not None:
                raise StudyError(f"slot {slot_index} has a terminal marker without a start marker")
            raise StudyError(
                "a different-container recovery requires one started-without-terminal slot"
            )
        expected_started = PrivateSlotStartedV1(
            schema_version=STUDY_SCHEMA_VERSION,
            study_id=precommit.study_id,
            slot_index=slot_index,
            root_commitment_sha256=commitment,
        )
        if started_payload != canonical_json_bytes(expected_started):
            raise StudyError(f"slot {slot_index} started marker has invalid bytes")
        if terminal_payload is None:
            slots.append(
                StudySlotOutcomeV1(
                    slot_index=slot_index,
                    root_commitment_sha256=commitment,
                    started=True,
                    terminal=False,
                    status="started_without_terminal",
                    certificate_sha256=None,
                    pre_manifest_files_sha256=None,
                )
            )
            interruption_seen = True
            continue
        terminal = _model_from_canonical_bytes(terminal_payload, PrivateSlotTerminalV1)
        assert isinstance(terminal, PrivateSlotTerminalV1)
        slot_outcome = _terminal_to_outcome(
            terminal,
            commitment,
            expected_study_id=precommit.study_id,
            expected_slot_index=slot_index,
        )
        if slot_outcome.status != "passed":
            raise StudyError(
                "different-container recovery is reserved for started-without-terminal retirement"
            )
        slots.append(slot_outcome)
    if not interruption_seen:  # pragma: no cover - loop exits earlier for the no-marker case
        raise StudyError("different-container recovery found no interrupted slot")
    return slots


def execute_successor_study(
    *,
    public_precommit_path: str | Path,
    public_outcome_path: str | Path,
    private_state_root: str | Path,
    repository_root: str | Path,
    tracked_precommit_loader: Callable[..., SuccessorStudyPrecommitV1] = load_tracked_precommit,
    admission_builder: Callable[[int], object] = construction.build_admission_only,
    instance_materializer: Callable[[object], object] = _default_instance_materializer,
    certifier: Callable[[object], dict[str, Any]] = _default_certifier,
    pre_manifest_builder: Callable[[object], Mapping[str, bytes]] = build_pre_manifest_files,
    runtime_validator: Callable[[SuccessorStudyPrecommitV1], ExecutionRuntimeReceiptV1] = (
        validate_runtime_environment
    ),
    repository_state_validator: Callable[
        [str | Path, SuccessorStudyPrecommitV1, ExecutionRuntimeReceiptV1], None
    ] = validate_execution_repository_state,
) -> SuccessorStudyOutcomeV1:
    """Run each precommitted root at most once, in order, and fail fast."""

    # This check is intentionally first: no target may be materialized until
    # the exact public precommit is present in committed Git history.
    precommit = tracked_precommit_loader(
        public_precommit_path,
        repository_root=repository_root,
    )
    state = NamedPrivateState(private_state_root)
    parsed_plan = _read_private_model(state, "plan.json", PrivateStudyPlanV1)
    if not isinstance(parsed_plan, PrivateStudyPlanV1):
        raise StudyError("private study plan is missing")
    plan = parsed_plan
    _validate_plan_against_precommit(plan, precommit)

    runtime_receipt = runtime_validator(precommit)
    if not isinstance(runtime_receipt, ExecutionRuntimeReceiptV1):
        raise TypeError("runtime validator must return ExecutionRuntimeReceiptV1")
    if runtime_receipt.runtime_image_digest != precommit.method.runtime_image_digest:
        raise StudyError("runtime receipt image digest differs from the precommit")
    repository_state_validator(repository_root, precommit, runtime_receipt)
    runtime_payload = canonical_json_bytes(runtime_receipt)
    current_runtime_receipt_digest = sha256_bytes(runtime_payload)

    existing = _read_private_model(state, "outcome.json", SuccessorStudyOutcomeV1)
    if isinstance(existing, SuccessorStudyOutcomeV1):
        validate_outcome_against_precommit(existing, precommit)
        allowed_receipt_digests = {
            existing.runtime_receipt_sha256,
            existing.recovery_runtime_receipt_sha256,
        }
        if current_runtime_receipt_digest not in allowed_receipt_digests:
            raise StudyError("terminal outcome is bound to a different runtime receipt")
        if (
            existing.precommit_publication_git_commit
            != runtime_receipt.precommit_publication_git_commit
        ):
            raise StudyError("terminal outcome is bound to a different publication commit")
        public_payload = canonical_json_bytes(existing)
        _write_public_no_clobber(Path(public_outcome_path), public_payload)
        return existing

    outcome_destination = Path(public_outcome_path)
    if outcome_destination.exists() or outcome_destination.is_symlink():
        raise FileExistsError(
            "refusing to begin target evaluation with an occupied public outcome path"
        )

    persisted_runtime_payload = _private_payload(state, "runtime_receipt.json")
    if persisted_runtime_payload is None:
        if not state.write_no_clobber("runtime_receipt.json", runtime_payload):  # pragma: no cover
            raise StudyError("initial runtime receipt raced with another executor")
        initial_runtime_receipt_digest = current_runtime_receipt_digest
    else:
        parsed_runtime = _model_from_canonical_bytes(
            persisted_runtime_payload,
            ExecutionRuntimeReceiptV1,
        )
        assert isinstance(parsed_runtime, ExecutionRuntimeReceiptV1)
        initial_runtime_receipt = parsed_runtime
        initial_runtime_receipt_digest = sha256_bytes(persisted_runtime_payload)
        if persisted_runtime_payload != runtime_payload:
            if _runtime_receipt_envelope(initial_runtime_receipt) != _runtime_receipt_envelope(
                runtime_receipt
            ):
                raise StudyError("recovery runtime receipt changed the frozen execution envelope")
            if initial_runtime_receipt.container_id == runtime_receipt.container_id:
                raise StudyError("runtime receipt bytes changed for the same container")

            # A new, independently inspected container is recovery-only.  It
            # may publish a fail-closed interruption outcome, but must not
            # even replay structural admission or call any target/certifier.
            recovery_slots = _interrupted_recovery_slots(state, precommit)
            recovery_reference = "recovery_runtime_receipt.json"
            if not state.write_no_clobber(recovery_reference, runtime_payload):
                if state.read(recovery_reference) != runtime_payload:
                    raise StudyError("study already has a different recovery runtime receipt")
            recovery_digest = current_runtime_receipt_digest
            recovered_outcome = _outcome_from_slots(
                precommit,
                recovery_slots,
                runtime_receipt_sha256=initial_runtime_receipt_digest,
                recovery_runtime_receipt_sha256=recovery_digest,
                precommit_publication_git_commit=(runtime_receipt.precommit_publication_git_commit),
            )
            recovered_payload = canonical_json_bytes(recovered_outcome)
            if not state.write_no_clobber("outcome.json", recovered_payload):
                raise StudyAlreadyTerminalError(
                    "study outcome was published by another recovery executor"
                )
            _write_public_no_clobber(outcome_destination, recovered_payload)
            return recovered_outcome

    # The expensive deterministic structural replay is deliberately below
    # the recovery-only branch so a new container can never resume a root.
    _validate_structural_selection(plan, admission_builder=admission_builder)

    slots: list[StudySlotOutcomeV1] = []
    first_failure = False
    public_slots = precommit.root_selection.ordered_root_slots
    for opening, public_slot in zip(plan.root_openings, public_slots, strict=True):
        slot_index = opening.slot_index
        commitment = public_slot.root_commitment_sha256
        if first_failure:
            slots.append(
                StudySlotOutcomeV1(
                    slot_index=slot_index,
                    root_commitment_sha256=commitment,
                    started=False,
                    terminal=False,
                    status="unrun_after_failure",
                    certificate_sha256=None,
                    pre_manifest_files_sha256=None,
                )
            )
            continue

        started_reference = f"slots/{slot_index}/started.json"
        terminal_reference = f"slots/{slot_index}/terminal.json"
        try:
            started_payload = state.read(started_reference)
        except FileNotFoundError:
            started_payload = None
        try:
            terminal_payload = state.read(terminal_reference)
        except FileNotFoundError:
            terminal_payload = None

        if started_payload is not None:
            expected_started = PrivateSlotStartedV1(
                schema_version=STUDY_SCHEMA_VERSION,
                study_id=precommit.study_id,
                slot_index=slot_index,
                root_commitment_sha256=commitment,
            )
            if started_payload != canonical_json_bytes(expected_started):
                raise StudyError(f"slot {slot_index} started marker has invalid bytes")
            if terminal_payload is None:
                slots.append(
                    StudySlotOutcomeV1(
                        slot_index=slot_index,
                        root_commitment_sha256=commitment,
                        started=True,
                        terminal=False,
                        status="started_without_terminal",
                        certificate_sha256=None,
                        pre_manifest_files_sha256=None,
                    )
                )
                first_failure = True
                continue
            parsed_terminal = _model_from_canonical_bytes(
                terminal_payload,
                PrivateSlotTerminalV1,
            )
            assert isinstance(parsed_terminal, PrivateSlotTerminalV1)
            terminal = parsed_terminal
            slot_outcome = _terminal_to_outcome(
                terminal,
                commitment,
                expected_study_id=precommit.study_id,
                expected_slot_index=slot_index,
            )
            slots.append(slot_outcome)
            first_failure = slot_outcome.status != "passed"
            continue

        if terminal_payload is not None:
            raise StudyError(f"slot {slot_index} has a terminal marker without a start marker")

        started_payload = canonical_json_bytes(
            PrivateSlotStartedV1(
                schema_version=STUDY_SCHEMA_VERSION,
                study_id=precommit.study_id,
                slot_index=slot_index,
                root_commitment_sha256=commitment,
            )
        )
        if not state.write_no_clobber(started_reference, started_payload):  # pragma: no cover
            raise StudyError(f"slot {slot_index} start marker raced with another executor")

        certificate_digest: str | None = None
        try:
            root_seed = int(opening.root_seed_hex, 16)
            candidate = admission_builder(root_seed)
            if not _is_admitted(candidate):
                raise StudyError(
                    f"precommitted root slot {slot_index} no longer passes structural admission"
                )
            instance = instance_materializer(candidate)
            certificate = certifier(instance)
            certificate_payload = canonical_json_bytes(certificate)
            certificate_digest = sha256_bytes(certificate_payload)
            if not state.write_no_clobber(
                f"slots/{slot_index}/certificate.json", certificate_payload
            ):
                raise StudyError(f"slot {slot_index} certificate path already exists")
            rendered = dict(pre_manifest_builder(instance))
            if set(rendered) != set(PRE_MANIFEST_FILE_ROLES):
                raise StudyError(
                    "candidate materializer returned an unknown or incomplete role set"
                )
            file_digests: dict[str, str] = {}
            for role in PRE_MANIFEST_FILE_ROLES:
                payload = rendered[role]
                if not isinstance(payload, bytes):
                    raise TypeError(f"candidate material role {role} is not bytes")
                if not state.write_no_clobber(f"slots/{slot_index}/materials/{role}", payload):
                    raise StudyError(f"slot {slot_index} material role {role} already exists")
                file_digests[role] = sha256_bytes(payload)
            terminal = PrivateSlotTerminalV1(
                schema_version=STUDY_SCHEMA_VERSION,
                study_id=precommit.study_id,
                slot_index=slot_index,
                status="passed",
                certificate_sha256=certificate_digest,
                pre_manifest_files_sha256=file_digests,
            )
        except Exception as exc:
            from .reference import CertificateRejectedError

            failure_payload = canonical_json_bytes(
                {
                    "schema_version": STUDY_SCHEMA_VERSION,
                    "study_id": precommit.study_id,
                    "slot_index": slot_index,
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            state.write_no_clobber(f"slots/{slot_index}/failure.json", failure_payload)
            terminal = PrivateSlotTerminalV1(
                schema_version=STUDY_SCHEMA_VERSION,
                study_id=precommit.study_id,
                slot_index=slot_index,
                status=(
                    "certificate_failed"
                    if isinstance(exc, CertificateRejectedError)
                    else "execution_failed"
                ),
                certificate_sha256=certificate_digest,
                pre_manifest_files_sha256=None,
            )

        terminal_payload = canonical_json_bytes(terminal)
        if not state.write_no_clobber(terminal_reference, terminal_payload):
            raise StudyError(f"slot {slot_index} terminal marker already exists")
        slot_outcome = _terminal_to_outcome(
            terminal,
            commitment,
            expected_study_id=precommit.study_id,
            expected_slot_index=slot_index,
        )
        slots.append(slot_outcome)
        first_failure = slot_outcome.status != "passed"

    outcome = _outcome_from_slots(
        precommit,
        slots,
        runtime_receipt_sha256=initial_runtime_receipt_digest,
        recovery_runtime_receipt_sha256=None,
        precommit_publication_git_commit=(runtime_receipt.precommit_publication_git_commit),
    )
    outcome_payload = canonical_json_bytes(outcome)
    if not state.write_no_clobber("outcome.json", outcome_payload):
        raise StudyAlreadyTerminalError("study outcome was published by another executor")
    _write_public_no_clobber(Path(public_outcome_path), outcome_payload)
    return outcome


def validate_outcome_against_precommit(
    outcome: SuccessorStudyOutcomeV1,
    precommit: SuccessorStudyPrecommitV1,
) -> None:
    if outcome.study_id != precommit.study_id:
        raise StudyError("outcome study_id does not match precommit")
    if outcome.precommit_sha256 != canonical_digest(precommit):
        raise StudyError("outcome does not bind the exact precommit")
    expected_commitments = tuple(
        slot.root_commitment_sha256 for slot in precommit.root_selection.ordered_root_slots
    )
    actual_commitments = tuple(slot.root_commitment_sha256 for slot in outcome.slots)
    if actual_commitments != expected_commitments:
        raise StudyError("outcome root commitments do not match the ordered precommit roster")


def validate_fixture_emission_proof(
    *,
    proof: SuccessorStudyFixtureProof,
    instance_seed: int,
    certificate: Mapping[str, Any],
    pre_manifest_files_sha256: Mapping[str, str],
) -> StudyBindingV1:
    """Validate the complete 3/3 proof for a regenerated fixture-0 edition."""

    if not isinstance(proof, SuccessorStudyFixtureProof) or proof._seal is not _PROOF_SEAL:
        raise StudyError("successor study proof was not produced by the strict tracked loader")
    precommit = proof.precommit
    outcome = proof.outcome
    private_plan = proof.private_plan
    validate_precommit(precommit)
    _validate_plan_against_precommit(private_plan, precommit)
    validate_outcome_against_precommit(outcome, precommit)
    runtime_payload = canonical_json_bytes(proof.runtime_receipt)
    if sha256_bytes(runtime_payload) != outcome.runtime_receipt_sha256:
        raise StudyError("successor study runtime receipt does not match the outcome")
    if (
        proof.runtime_receipt.precommit_publication_git_commit
        != outcome.precommit_publication_git_commit
    ):
        raise StudyError("successor study outcome has the wrong publication commit")
    if proof.runtime_receipt.runtime_image_digest != precommit.method.runtime_image_digest:
        raise StudyError("successor study runtime receipt has the wrong image digest")
    if outcome.status != "passed" or outcome.selected_fixture_slot != FIXTURE_ROOT_INDEX:
        raise StudyError("final emission requires a complete passing 3/3 study outcome")
    opening = private_plan.root_openings[FIXTURE_ROOT_INDEX]
    if instance_seed != int(opening.root_seed_hex, 16):
        raise StudyError("final emission seed is not the precommitted fixture-0 root")
    fixture = outcome.slots[FIXTURE_ROOT_INDEX]
    if certificate.get("instance_seed") != instance_seed:
        raise StudyError("final emission certificate has the wrong fixture instance seed")
    certificate_digest = canonical_digest(dict(certificate))
    if certificate_digest != fixture.certificate_sha256:
        raise StudyError("final emission certificate differs from fixture-0 study evidence")
    normalized_files = dict(pre_manifest_files_sha256)
    if set(normalized_files) != set(PRE_MANIFEST_FILE_ROLES):
        raise StudyError("final emission has an unknown or incomplete pre-manifest role set")
    if normalized_files != fixture.pre_manifest_files_sha256:
        raise StudyError("final emission files differ from fixture-0 study evidence")
    return StudyBindingV1(
        schema_version=STUDY_SCHEMA_VERSION,
        study_id=precommit.study_id,
        precommit_sha256=canonical_digest(precommit),
        precommit_publication_git_commit=outcome.precommit_publication_git_commit,
        outcome_sha256=canonical_digest(outcome),
        fixture_root_index=FIXTURE_ROOT_INDEX,
    )


def preflight_fixture_emission_proof(
    proof: SuccessorStudyFixtureProof,
    *,
    expected_seed: int,
) -> None:
    """Reject missing/forged/wrong-root proof before any target rebuild."""

    if not isinstance(proof, SuccessorStudyFixtureProof) or proof._seal is not _PROOF_SEAL:
        raise StudyError("successor study proof was not produced by the strict tracked loader")
    validate_precommit(proof.precommit)
    _validate_plan_against_precommit(proof.private_plan, proof.precommit)
    validate_outcome_against_precommit(proof.outcome, proof.precommit)
    if proof.outcome.status != "passed" or proof.outcome.selected_fixture_slot != 0:
        raise StudyError("successor study proof is not a complete 3/3 fixture-0 pass")
    fixture_seed = int(proof.private_plan.root_openings[FIXTURE_ROOT_INDEX].root_seed_hex, 16)
    if expected_seed != fixture_seed:
        raise StudyError("requested emission seed is not the proof-bound fixture-0 root")
    receipt_payload = canonical_json_bytes(proof.runtime_receipt)
    if sha256_bytes(receipt_payload) != proof.outcome.runtime_receipt_sha256:
        raise StudyError("successor study runtime receipt does not match the passing outcome")
    if (
        proof.runtime_receipt.precommit_publication_git_commit
        != proof.outcome.precommit_publication_git_commit
    ):
        raise StudyError("successor study proof has inconsistent publication commits")
    if proof.runtime_receipt.runtime_image_digest != proof.precommit.method.runtime_image_digest:
        raise StudyError("successor study runtime image differs from the precommit")


def validate_slot_emission_proof(
    *,
    proof: SuccessorStudyFixtureProof,
    root_slot_index: int,
    instance_seed: int,
    certificate: Mapping[str, Any],
    pre_manifest_files_sha256: Mapping[str, str],
) -> OfficialStudyBindingV1:
    """Validate the complete 3/3 proof for an owner-private edition.

    Mirrors ``validate_fixture_emission_proof`` with the binding drawn from an
    owner-private slot: the emitted seed must open the precommitted slot root,
    and the certificate plus every rendered byte must equal that slot's study
    evidence exactly.  Slot 0 is the public development fixture and is never
    a private edition.
    """

    if root_slot_index not in OFFICIAL_SLOT_TIERS:
        raise StudyError("official emission draws only an owner-private study slot")
    if not isinstance(proof, SuccessorStudyFixtureProof) or proof._seal is not _PROOF_SEAL:
        raise StudyError("successor study proof was not produced by the strict tracked loader")
    precommit = proof.precommit
    outcome = proof.outcome
    private_plan = proof.private_plan
    validate_precommit(precommit)
    _validate_plan_against_precommit(private_plan, precommit)
    validate_outcome_against_precommit(outcome, precommit)
    runtime_payload = canonical_json_bytes(proof.runtime_receipt)
    if sha256_bytes(runtime_payload) != outcome.runtime_receipt_sha256:
        raise StudyError("successor study runtime receipt does not match the outcome")
    if (
        proof.runtime_receipt.precommit_publication_git_commit
        != outcome.precommit_publication_git_commit
    ):
        raise StudyError("successor study outcome has the wrong publication commit")
    if proof.runtime_receipt.runtime_image_digest != precommit.method.runtime_image_digest:
        raise StudyError("successor study runtime receipt has the wrong image digest")
    if outcome.status != "passed" or outcome.selected_fixture_slot != FIXTURE_ROOT_INDEX:
        raise StudyError("official emission requires a complete passing 3/3 study outcome")
    opening = private_plan.root_openings[root_slot_index]
    if instance_seed != int(opening.root_seed_hex, 16):
        raise StudyError("official emission seed is not the precommitted slot root")
    slot = outcome.slots[root_slot_index]
    if certificate.get("instance_seed") != instance_seed:
        raise StudyError("official emission certificate has the wrong slot instance seed")
    certificate_digest = canonical_digest(dict(certificate))
    if certificate_digest != slot.certificate_sha256:
        raise StudyError("official emission certificate differs from the slot's study evidence")
    normalized_files = dict(pre_manifest_files_sha256)
    if set(normalized_files) != set(PRE_MANIFEST_FILE_ROLES):
        raise StudyError("official emission has an unknown or incomplete pre-manifest role set")
    if normalized_files != slot.pre_manifest_files_sha256:
        raise StudyError("official emission files differ from the slot's study evidence")
    return OfficialStudyBindingV1(
        schema_version=STUDY_SCHEMA_VERSION,
        study_id=precommit.study_id,
        precommit_sha256=canonical_digest(precommit),
        precommit_publication_git_commit=outcome.precommit_publication_git_commit,
        outcome_sha256=canonical_digest(outcome),
        root_slot_index=root_slot_index,
    )


def preflight_slot_emission_proof(
    proof: SuccessorStudyFixtureProof,
    *,
    root_slot_index: int,
    expected_seed: int,
) -> None:
    """Reject missing/forged/wrong-slot proof before any official rebuild."""

    if root_slot_index not in OFFICIAL_SLOT_TIERS:
        raise StudyError("official emission draws only an owner-private study slot")
    if not isinstance(proof, SuccessorStudyFixtureProof) or proof._seal is not _PROOF_SEAL:
        raise StudyError("successor study proof was not produced by the strict tracked loader")
    validate_precommit(proof.precommit)
    _validate_plan_against_precommit(proof.private_plan, proof.precommit)
    validate_outcome_against_precommit(proof.outcome, proof.precommit)
    if proof.outcome.status != "passed" or proof.outcome.selected_fixture_slot != 0:
        raise StudyError("successor study proof is not a complete 3/3 fixture-0 pass")
    slot_seed = int(proof.private_plan.root_openings[root_slot_index].root_seed_hex, 16)
    if expected_seed != slot_seed:
        raise StudyError("requested emission seed is not the proof-bound slot root")
    receipt_payload = canonical_json_bytes(proof.runtime_receipt)
    if sha256_bytes(receipt_payload) != proof.outcome.runtime_receipt_sha256:
        raise StudyError("successor study runtime receipt does not match the passing outcome")
    if (
        proof.runtime_receipt.precommit_publication_git_commit
        != proof.outcome.precommit_publication_git_commit
    ):
        raise StudyError("successor study proof has inconsistent publication commits")
    if proof.runtime_receipt.runtime_image_digest != proof.precommit.method.runtime_image_digest:
        raise StudyError("successor study runtime image differs from the precommit")


def load_private_plan(
    private_state_root: str | Path,
    precommit: SuccessorStudyPrecommitV1,
    *,
    admission_builder: Callable[[int], object] | None = None,
) -> PrivateStudyPlanV1:
    state = NamedPrivateState(private_state_root)
    parsed = _read_private_model(state, "plan.json", PrivateStudyPlanV1)
    if not isinstance(parsed, PrivateStudyPlanV1):
        raise StudyError("private study plan is missing")
    _validate_plan_against_precommit(parsed, precommit)
    _validate_structural_selection(
        parsed,
        admission_builder=(
            construction.build_admission_only if admission_builder is None else admission_builder
        ),
    )
    return parsed


def load_tracked_passing_proof(
    *,
    public_precommit_path: str | Path,
    public_outcome_path: str | Path,
    private_state_root: str | Path,
    repository_root: str | Path,
) -> SuccessorStudyFixtureProof:
    """Load exact tracked proof bytes; scientific equality is checked at emit."""

    root = Path(repository_root).resolve()
    precommit = _load_tracked_precommit_artifact(Path(public_precommit_path), root)
    plan = load_private_plan(private_state_root, precommit)
    fixture_seed = int(plan.root_openings[FIXTURE_ROOT_INDEX].root_seed_hex, 16)
    _validate_frozen_method(
        precommit,
        root,
        allowed_activation_seed=fixture_seed,
    )
    outcome_payload = _tracked_exact_payload(Path(public_outcome_path), root)
    parsed = _model_from_canonical_bytes(outcome_payload, SuccessorStudyOutcomeV1)
    assert isinstance(parsed, SuccessorStudyOutcomeV1)
    outcome = parsed
    validate_outcome_against_precommit(outcome, precommit)
    if outcome.status != "passed":
        raise StudyError("tracked successor study outcome is not a complete pass")
    state = NamedPrivateState(private_state_root)
    receipt_model = _read_private_model(
        state,
        "runtime_receipt.json",
        ExecutionRuntimeReceiptV1,
    )
    if not isinstance(receipt_model, ExecutionRuntimeReceiptV1):
        raise StudyError("private execution runtime receipt is missing")
    if sha256_bytes(canonical_json_bytes(receipt_model)) != outcome.runtime_receipt_sha256:
        raise StudyError("private execution runtime receipt does not match the tracked outcome")
    if receipt_model.precommit_publication_git_commit != outcome.precommit_publication_git_commit:
        raise StudyError("private execution receipt has the wrong publication commit")
    if receipt_model.runtime_image_digest != precommit.method.runtime_image_digest:
        raise StudyError("private execution runtime receipt has the wrong image digest")
    return SuccessorStudyFixtureProof(
        precommit=precommit,
        outcome=outcome,
        private_plan=plan,
        runtime_receipt=receipt_model,
        _seal=_PROOF_SEAL,
    )


def validate_owner_private_parent(path: str | Path) -> Path:
    """Create/validate the owner-only ``.eval`` parent used by the CLI."""

    parent = Path(path)
    try:
        parent.mkdir(mode=0o700)
    except FileExistsError:
        pass
    metadata = parent.lstat()
    getuid = getattr(os, "getuid", None)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or parent.is_symlink()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or (getuid is not None and metadata.st_uid != getuid())
    ):
        raise StudyError(
            "private study parent must be an owner-owned non-symlink directory mode 0700"
        )
    return parent


__all__ = [
    "FIXTURE_ROOT_INDEX",
    "GENERATED_OUTPUT_PATHS",
    "MAX_CANDIDATE_INDEX",
    "METHOD_PATHS",
    "OFFICIAL_SLOT_TIERS",
    "OfficialStudyBindingV1",
    "PRE_MANIFEST_FILE_ROLES",
    "PrivateStudyPlanV1",
    "RootOpeningV1",
    "SCIENTIFIC_REVISION",
    "STUDY_ROOT_COUNT",
    "StudyBindingV1",
    "StudyError",
    "SuccessorStudyOutcomeV1",
    "SuccessorStudyFixtureProof",
    "SuccessorStudyPrecommitV1",
    "build_pre_manifest_files",
    "build_precommit_and_private_plan",
    "clean_method_commit",
    "current_certificate_policy",
    "current_public_policy",
    "derive_candidate_root",
    "execute_successor_study",
    "load_outcome",
    "load_precommit",
    "load_private_plan",
    "load_tracked_passing_proof",
    "load_tracked_precommit",
    "persist_prepared_study",
    "prepare_successor_study",
    "preflight_fixture_emission_proof",
    "preflight_slot_emission_proof",
    "root_commitment_sha256",
    "validate_fixture_emission_proof",
    "validate_slot_emission_proof",
    "validate_outcome_against_precommit",
    "validate_owner_private_parent",
    "validate_precommit",
]
