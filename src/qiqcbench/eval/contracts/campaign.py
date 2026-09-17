"""Versioned campaign planning and execution-lineage contracts."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Literal

from pydantic import Field, JsonValue, ValidationInfo, field_validator, model_validator

from .base import (
    Digest,
    Identifier,
    NonEmptyStr,
    StrictModel,
    ensure_finite_json,
    semantic_digest,
)

ComparisonDesign = Literal["complete_randomized_block"]
AnchorRole = Literal["primary", "bridge"]
SeedCommitmentScheme = Literal["sha256_random_256bit"]
_SAFE_RUNTIME_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def commit_verifier_replay_secret(secret: str) -> str:
    """Commit to verifier-only high-entropy replay material without exposing it."""

    if not isinstance(secret, str) or re.fullmatch(r"[0-9a-f]{64}", secret) is None:
        raise ValueError("verifier replay secret must be 32 bytes encoded as lowercase hex")
    material = b"qiqcbench-verifier-replay-secret-v1\0" + bytes.fromhex(secret)
    return hashlib.sha256(material).hexdigest()


def _sorted_unique_ids(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if len(values) != len(set(values)):
        raise ValueError(f"{field} must contain unique IDs")
    return tuple(sorted(values))


def _weights_sum_to_one(weights: dict[str, float], *, field: str) -> None:
    if any(value <= 0.0 for value in weights.values()):
        raise ValueError(f"{field} values must be strictly positive")
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{field} values must sum to 1")


class RetryPolicy(StrictModel):
    """Frozen retry authority; retries replace infrastructure executions only."""

    max_infrastructure_retries: int = Field(ge=0)
    retryable_dispositions: tuple[Literal["infrastructure_failure"], ...] = (
        "infrastructure_failure",
    )
    allow_best_of_n: Literal[False] = False

    @field_validator("retryable_dispositions")
    @classmethod
    def _only_infrastructure_is_retryable(
        cls,
        values: tuple[Literal["infrastructure_failure"], ...],
    ) -> tuple[Literal["infrastructure_failure"], ...]:
        if values != ("infrastructure_failure",):
            raise ValueError("only infrastructure_failure may be execution-retryable")
        return values


class CoverageThresholds(StrictModel):
    """Predeclared minimum coverage ratios for publication decisions."""

    anchor: float = Field(ge=0.0, le=1.0)
    overall: float = Field(ge=0.0, le=1.0)
    family: float = Field(ge=0.0, le=1.0)
    task: float = Field(ge=0.0, le=1.0)
    provisional: float = Field(ge=0.0, le=1.0)


class RandomizationSeedCommitment(StrictModel):
    """Non-enumerable public commitment to private campaign randomization."""

    scheme: SeedCommitmentScheme
    value: Digest


def commit_campaign_randomization_secret(
    secret: str,
) -> RandomizationSeedCommitment:
    """Commit to one high-entropy private planner secret."""

    if not isinstance(secret, str) or re.fullmatch(r"[0-9a-f]{64}", secret) is None:
        raise ValueError("campaign randomization secret must be 32 bytes as lowercase hex")
    value = hashlib.sha256(
        b"qiqcbench-campaign-randomization-secret-v1\0" + bytes.fromhex(secret)
    ).hexdigest()
    return RandomizationSeedCommitment(scheme="sha256_random_256bit", value=value)


class AnchorPoolMember(StrictModel):
    """One rated calibration configuration in a task's utility anchor pool."""

    rated_configuration_id: Identifier
    role: AnchorRole
    utility_weight: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)


class TaskAnchorPool(StrictModel):
    """Predeclared weighted anchor ladder for one TaskEdition panel."""

    task_edition_id: Identifier
    members: tuple[AnchorPoolMember, ...] = Field(min_length=1)

    @field_validator("members")
    @classmethod
    def _normalize_members(
        cls,
        values: tuple[AnchorPoolMember, ...],
    ) -> tuple[AnchorPoolMember, ...]:
        member_ids = [member.rated_configuration_id for member in values]
        if len(member_ids) != len(set(member_ids)):
            raise ValueError("anchor pool rated_configuration_id values must be unique")
        if sum(member.role == "primary" for member in values) != 1:
            raise ValueError("task anchor pool requires exactly one primary member")
        if not any(member.role == "bridge" for member in values):
            raise ValueError("task anchor pool requires at least one bridge member")
        _weights_sum_to_one(
            {member.rated_configuration_id: member.utility_weight for member in values},
            field="anchor pool utility weights",
        )
        return tuple(sorted(values, key=lambda member: member.rated_configuration_id))


class CampaignRatingPolicyBinding(StrictModel):
    """Atomic optional rating layer carried by a campaign execution policy.

    The schema-1 wire keeps these fields at the ``CampaignPolicyManifest`` top
    level for byte compatibility.  This view makes the semantic split explicit
    without changing already-frozen rated campaign policy bytes.
    """

    schema_version: Literal[1] = 1
    family_weights: dict[Identifier, float] = Field(min_length=1)
    task_weights: dict[Identifier, dict[Identifier, float]] = Field(min_length=1)
    calibrated_reference_utilities: dict[Identifier, float] = Field(default_factory=dict)
    estimator_policy_id: Identifier
    rating_policy_digest: Digest

    @model_validator(mode="after")
    def _rating_weights_are_complete(self) -> CampaignRatingPolicyBinding:
        _weights_sum_to_one(self.family_weights, field="family_weights")
        if set(self.task_weights) != set(self.family_weights):
            raise ValueError("task_weights family keys must exactly match family_weights")
        weighted_editions: set[str] = set()
        for family_id, weights in self.task_weights.items():
            _weights_sum_to_one(weights, field=f"task_weights[{family_id}]")
            overlap = weighted_editions.intersection(weights)
            if overlap:
                raise ValueError("one task edition cannot be weighted in multiple families")
            weighted_editions.update(weights)
        if any(
            value < 0.0 or value > 1.0 for value in self.calibrated_reference_utilities.values()
        ):
            raise ValueError("calibrated reference utilities must lie in [0, 1]")
        if not set(self.calibrated_reference_utilities).issubset(weighted_editions):
            raise ValueError(
                "calibrated reference utilities may reference only weighted task editions"
            )
        return self


class CampaignPolicyManifest(StrictModel):
    """Schema-1 campaign policy with its original frozen rating binding."""

    schema_version: Literal[1] = 1
    campaign_policy_digest: Digest | None = None
    target_tier: Literal["development", "candidate", "canonical"]
    attempts_per_task_edition: dict[Identifier, int] = Field(min_length=1)
    stochastic_blocks_per_task_edition: dict[Identifier, int] = Field(default_factory=dict)
    retry_policy: RetryPolicy
    coverage_thresholds: CoverageThresholds
    family_weights: dict[Identifier, float] = Field(min_length=1)
    task_weights: dict[Identifier, dict[Identifier, float]] = Field(min_length=1)
    calibrated_reference_utilities: dict[Identifier, float] = Field(default_factory=dict)
    comparison_design: ComparisonDesign = "complete_randomized_block"
    estimator_policy_id: Identifier
    rating_policy_digest: Digest

    @property
    def rating_policy_binding(self) -> CampaignRatingPolicyBinding:
        """Expose the schema-1 rating fields as one validated atomic binding."""

        return CampaignRatingPolicyBinding(
            family_weights=self.family_weights,
            task_weights=self.task_weights,
            calibrated_reference_utilities=self.calibrated_reference_utilities,
            estimator_policy_id=self.estimator_policy_id,
            rating_policy_digest=self.rating_policy_digest,
        )

    @field_validator("attempts_per_task_edition", "stochastic_blocks_per_task_edition")
    @classmethod
    def _quotas_are_positive(
        cls,
        values: dict[str, int],
    ) -> dict[str, int]:
        if any(isinstance(value, bool) or value <= 0 for value in values.values()):
            raise ValueError("campaign policy quotas must be positive integers")
        return values

    @model_validator(mode="after")
    def _bind_policy_digest(self) -> CampaignPolicyManifest:
        rating_binding = self.rating_policy_binding
        weighted_editions = {
            edition_id for weights in rating_binding.task_weights.values() for edition_id in weights
        }
        if weighted_editions != set(self.attempts_per_task_edition):
            raise ValueError(
                "attempts_per_task_edition keys must exactly match weighted task editions"
            )
        unknown_stochastic_editions = set(self.stochastic_blocks_per_task_edition).difference(
            weighted_editions
        )
        if unknown_stochastic_editions:
            raise ValueError("stochastic block quotas may reference only weighted task editions")
        payload = self.model_dump(mode="json", exclude={"campaign_policy_digest"})
        expected = semantic_digest(payload)
        if self.campaign_policy_digest is None:
            object.__setattr__(self, "campaign_policy_digest", expected)
        elif self.campaign_policy_digest != expected:
            raise ValueError("campaign_policy_digest does not match policy content")
        return self


class CampaignExecutionPolicyManifest(StrictModel):
    """Schema-2 execution/evidence policy for an official record-only campaign.

    It freezes launch tier, assignment quotas, retry behavior, coverage gates,
    and comparison design without selecting any estimator, rating artifact, or
    aggregation weights.  A later rating layer must bind the immutable record
    snapshot independently.
    """

    schema_version: Literal[2] = 2
    campaign_policy_digest: Digest | None = None
    evidence_policy_id: Identifier
    evidence_policy_digest: Digest
    target_tier: Literal["development", "candidate", "canonical"]
    attempts_per_task_edition: dict[Identifier, int] = Field(min_length=1)
    stochastic_blocks_per_task_edition: dict[Identifier, int] = Field(default_factory=dict)
    retry_policy: RetryPolicy
    coverage_thresholds: CoverageThresholds
    calibrated_reference_utilities: dict[Identifier, float] = Field(default_factory=dict)
    comparison_design: ComparisonDesign = "complete_randomized_block"

    @property
    def rating_policy_binding(self) -> None:
        """Record-only execution policies deliberately carry no rating layer."""

        return None

    @field_validator("attempts_per_task_edition", "stochastic_blocks_per_task_edition")
    @classmethod
    def _quotas_are_positive(
        cls,
        values: dict[str, int],
    ) -> dict[str, int]:
        if any(isinstance(value, bool) or value <= 0 for value in values.values()):
            raise ValueError("campaign execution policy quotas must be positive integers")
        return values

    @model_validator(mode="after")
    def _bind_policy_digest(self) -> CampaignExecutionPolicyManifest:
        unknown_stochastic_editions = set(self.stochastic_blocks_per_task_edition).difference(
            self.attempts_per_task_edition
        )
        if unknown_stochastic_editions:
            raise ValueError("stochastic block quotas may reference only planned task editions")
        if any(
            value < 0.0 or value > 1.0 for value in self.calibrated_reference_utilities.values()
        ):
            raise ValueError("calibrated reference utilities must lie in [0, 1]")
        if not set(self.calibrated_reference_utilities).issubset(self.attempts_per_task_edition):
            raise ValueError(
                "calibrated reference utilities may reference only planned task editions"
            )
        payload = self.model_dump(mode="json", exclude={"campaign_policy_digest"})
        expected = semantic_digest(payload)
        if self.campaign_policy_digest is None:
            object.__setattr__(self, "campaign_policy_digest", expected)
        elif self.campaign_policy_digest != expected:
            raise ValueError("campaign_policy_digest does not match execution policy content")
        return self


CampaignPolicy = CampaignPolicyManifest | CampaignExecutionPolicyManifest


class AttemptManifest(StrictModel):
    """One predeclared statistical player/edition/instance attempt."""

    schema_version: Literal[1] = 1
    attempt_id: Identifier
    campaign_id: Identifier
    player_id: Identifier
    rated_configuration_id: Identifier | None = None
    campaign_policy_digest: Digest | None = None
    task_edition_id: Identifier
    task_family_id: Identifier
    instance_id: Identifier
    instance_cluster_id: Identifier
    comparison_block_id: Identifier
    attempt_slot: int = Field(ge=0)
    stochastic_block_slot: int | None = Field(default=None, ge=0)
    execution_profile_id: Identifier
    harbor_assignment_id: Identifier
    annotations: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("annotations")
    @classmethod
    def _annotations_are_finite(cls, values: dict[str, JsonValue]) -> dict[str, JsonValue]:
        ensure_finite_json(values, path="annotations")
        return values


class ExecutionManifest(StrictModel):
    """One infrastructure execution, including retry lineage, of an attempt."""

    schema_version: Literal[1] = 1
    execution_id: Identifier
    attempt_id: Identifier
    retry_ordinal: int = Field(ge=0)
    harbor_config_name: NonEmptyStr
    harbor_job_name: NonEmptyStr
    qsim_evidence_nonce: Digest
    verifier_replay_secret_commitment: Digest
    supersedes_execution_id: Identifier | None = None
    annotations: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("annotations")
    @classmethod
    def _annotations_are_finite(cls, values: dict[str, JsonValue]) -> dict[str, JsonValue]:
        ensure_finite_json(values, path="annotations")
        return values

    @field_validator("harbor_config_name", "harbor_job_name")
    @classmethod
    def _runtime_names_are_safe(cls, value: str) -> str:
        if _SAFE_RUNTIME_NAME.fullmatch(value) is None:
            raise ValueError("Harbor config/job names must be safe single-component names")
        return value

    @model_validator(mode="after")
    def _retry_lineage_is_explicit(self) -> ExecutionManifest:
        if self.retry_ordinal == 0 and self.supersedes_execution_id is not None:
            raise ValueError("initial execution cannot set supersedes_execution_id")
        if self.retry_ordinal > 0 and self.supersedes_execution_id is None:
            raise ValueError("retry execution requires supersedes_execution_id")
        if self.supersedes_execution_id == self.execution_id:
            raise ValueError("execution cannot supersede itself")
        return self

    def assert_retry_predecessor(self, predecessor: ExecutionManifest) -> None:
        """Fail if ``predecessor`` cannot be this retry's direct parent."""

        if self.retry_ordinal == 0:
            raise ValueError("initial execution has no retry predecessor")
        if self.supersedes_execution_id != predecessor.execution_id:
            raise ValueError("retry does not reference the supplied predecessor execution")
        if self.attempt_id != predecessor.attempt_id:
            raise ValueError("retry and predecessor must preserve attempt_id")
        if self.retry_ordinal != predecessor.retry_ordinal + 1:
            raise ValueError("retry_ordinal must increment its predecessor by one")


class CampaignManifest(StrictModel):
    """Frozen assignments and policy inputs for an auditable evaluation campaign."""

    schema_version: Literal[1] = 1
    campaign_id: Identifier
    player_ids: tuple[Identifier, ...] = Field(min_length=1)
    rated_configuration_ids: tuple[Identifier, ...] = ()
    campaign_policy_digest: Digest | None = None
    policy: CampaignPolicy | None = None
    task_edition_ids: tuple[Identifier, ...] = Field(min_length=1)
    task_family_ids: tuple[Identifier, ...] = Field(min_length=1)
    instance_ids: tuple[Identifier, ...] = Field(min_length=1)
    execution_profile_ids: tuple[Identifier, ...] = Field(min_length=1)
    attempts: tuple[AttemptManifest, ...] = ()
    initial_executions: tuple[ExecutionManifest, ...] = ()
    execution_order: tuple[Identifier, ...] = ()
    task_anchor_pools: tuple[TaskAnchorPool, ...] = ()
    randomization_seed_commitment: RandomizationSeedCommitment
    annotations: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator(
        "player_ids",
        "rated_configuration_ids",
        "task_edition_ids",
        "task_family_ids",
        "instance_ids",
        "execution_profile_ids",
    )
    @classmethod
    def _normalize_id_collections(
        cls,
        values: tuple[str, ...],
        info: ValidationInfo,
    ) -> tuple[str, ...]:
        field_name = info.field_name or "IDs"
        return _sorted_unique_ids(values, field=field_name)

    @field_validator("attempts")
    @classmethod
    def _normalize_attempts(
        cls, values: tuple[AttemptManifest, ...]
    ) -> tuple[AttemptManifest, ...]:
        return tuple(sorted(values, key=lambda value: value.attempt_id))

    @field_validator("initial_executions")
    @classmethod
    def _normalize_initial_executions(
        cls, values: tuple[ExecutionManifest, ...]
    ) -> tuple[ExecutionManifest, ...]:
        return tuple(sorted(values, key=lambda value: value.execution_id))

    @field_validator("execution_order")
    @classmethod
    def _execution_order_is_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("execution_order must contain unique execution IDs")
        return values

    @field_validator("task_anchor_pools")
    @classmethod
    def _normalize_anchor_pools(
        cls,
        values: tuple[TaskAnchorPool, ...],
    ) -> tuple[TaskAnchorPool, ...]:
        edition_ids = [pool.task_edition_id for pool in values]
        if len(edition_ids) != len(set(edition_ids)):
            raise ValueError("one task edition cannot declare multiple anchor pools")
        return tuple(sorted(values, key=lambda value: value.task_edition_id))

    @field_validator("annotations")
    @classmethod
    def _annotations_are_finite(cls, values: dict[str, JsonValue]) -> dict[str, JsonValue]:
        ensure_finite_json(values, path="annotations")
        return values

    @model_validator(mode="after")
    def _collections_are_structurally_consistent(self) -> CampaignManifest:
        if (self.policy is None) != (self.campaign_policy_digest is None):
            raise ValueError("campaign policy and campaign_policy_digest must be supplied together")
        if self.policy is not None:
            if self.campaign_policy_digest != self.policy.campaign_policy_digest:
                raise ValueError("campaign_policy_digest does not match embedded policy")
        elif self.rated_configuration_ids or self.task_anchor_pools:
            raise ValueError(
                "policy-less campaign cannot declare rated configurations or anchor pools"
            )

        attempt_ids = [attempt.attempt_id for attempt in self.attempts]
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError("attempt_id values must be unique")
        assignment_ids = [attempt.harbor_assignment_id for attempt in self.attempts]
        if len(assignment_ids) != len(set(assignment_ids)):
            raise ValueError("harbor_assignment_id values must be unique")

        if self.attempts:
            expected_refs = {
                "player_ids": set(self.player_ids),
                "task_edition_ids": set(self.task_edition_ids),
                "task_family_ids": set(self.task_family_ids),
                "instance_ids": set(self.instance_ids),
                "execution_profile_ids": set(self.execution_profile_ids),
            }
            actual_refs = {
                "player_ids": {attempt.player_id for attempt in self.attempts},
                "task_edition_ids": {attempt.task_edition_id for attempt in self.attempts},
                "task_family_ids": {attempt.task_family_id for attempt in self.attempts},
                "instance_ids": {attempt.instance_id for attempt in self.attempts},
                "execution_profile_ids": {
                    attempt.execution_profile_id for attempt in self.attempts
                },
            }
            for field_name, expected in expected_refs.items():
                if actual_refs[field_name] != expected:
                    raise ValueError(f"attempt references must exactly match campaign {field_name}")
            if any(attempt.campaign_id != self.campaign_id for attempt in self.attempts):
                raise ValueError("every attempt campaign_id must match the CampaignManifest")
            observed_rated_ids = {
                attempt.rated_configuration_id
                for attempt in self.attempts
                if attempt.rated_configuration_id is not None
            }
            if self.policy is None:
                if observed_rated_ids or any(
                    attempt.campaign_policy_digest is not None for attempt in self.attempts
                ):
                    raise ValueError(
                        "policy-less attempts cannot declare rating or policy bindings"
                    )
            else:
                if any(
                    attempt.rated_configuration_id is None for attempt in self.attempts
                ) or observed_rated_ids != set(self.rated_configuration_ids):
                    raise ValueError(
                        "policy-bound attempt rated_configuration_id references must exactly "
                        "match campaign"
                    )
                if any(
                    attempt.campaign_policy_digest != self.campaign_policy_digest
                    for attempt in self.attempts
                ):
                    raise ValueError(
                        "policy-bound attempt campaign_policy_digest must exactly match campaign"
                    )

        planned_cells = [
            (
                attempt.rated_configuration_id or attempt.player_id,
                attempt.task_edition_id,
                attempt.instance_id,
                attempt.stochastic_block_slot,
                attempt.attempt_slot,
            )
            for attempt in self.attempts
        ]
        if len(planned_cells) != len(set(planned_cells)):
            raise ValueError("duplicate player/edition/instance/slot planned cell")

        block_signatures: dict[str, tuple[object, ...]] = {}
        for attempt in self.attempts:
            signature = (
                attempt.task_edition_id,
                attempt.task_family_id,
                attempt.instance_id,
                attempt.instance_cluster_id,
                attempt.execution_profile_id,
                attempt.stochastic_block_slot,
            )
            prior = block_signatures.setdefault(attempt.comparison_block_id, signature)
            if prior != signature:
                raise ValueError(
                    "one comparison_block_id cannot mix edition, instance, profile, or budget"
                )

        execution_ids = [execution.execution_id for execution in self.initial_executions]
        if len(execution_ids) != len(set(execution_ids)):
            raise ValueError("initial execution_id values must be unique")
        if any(execution.retry_ordinal != 0 for execution in self.initial_executions):
            raise ValueError("CampaignManifest may contain only an initial execution per attempt")
        for field_name in (
            "harbor_config_name",
            "harbor_job_name",
            "qsim_evidence_nonce",
            "verifier_replay_secret_commitment",
        ):
            values = [getattr(execution, field_name) for execution in self.initial_executions]
            if len(values) != len(set(values)):
                raise ValueError(f"initial execution {field_name} values must be unique")
        execution_attempt_ids = [execution.attempt_id for execution in self.initial_executions]
        if len(execution_attempt_ids) != len(set(execution_attempt_ids)):
            raise ValueError("each attempt must have exactly one initial execution")
        if self.initial_executions:
            if set(execution_attempt_ids) != set(attempt_ids):
                raise ValueError("initial executions must cover every planned attempt exactly once")
            if set(self.execution_order) != set(execution_ids) or len(self.execution_order) != len(
                execution_ids
            ):
                raise ValueError("execution_order must be a full permutation of initial executions")

        anchor_editions = {pool.task_edition_id for pool in self.task_anchor_pools}
        if not anchor_editions.issubset(self.task_edition_ids):
            raise ValueError("anchor pools must belong to campaign task editions")
        anchor_members = {
            member.rated_configuration_id
            for pool in self.task_anchor_pools
            for member in pool.members
        }
        if not anchor_members.issubset(self.rated_configuration_ids):
            raise ValueError("anchor members must belong to campaign rated configurations")

        if self.policy is not None and self.policy.rating_policy_binding is not None:
            if set(self.policy.family_weights) != set(self.task_family_ids):
                raise ValueError("policy family_weights keys must match campaign task families")
            weighted_editions = {
                edition_id
                for weights in self.policy.task_weights.values()
                for edition_id in weights
            }
            if weighted_editions != set(self.task_edition_ids):
                raise ValueError("policy task_weights must match campaign task editions")
            if self.attempts:
                edition_families: dict[str, str] = {}
                for attempt in self.attempts:
                    prior = edition_families.setdefault(
                        attempt.task_edition_id,
                        attempt.task_family_id,
                    )
                    if prior != attempt.task_family_id:
                        raise ValueError("one task edition cannot belong to multiple task families")
                for family_id, weights in self.policy.task_weights.items():
                    expected_editions = {
                        edition_id
                        for edition_id, task_family_id in edition_families.items()
                        if task_family_id == family_id
                    }
                    if set(weights) != expected_editions:
                        raise ValueError(
                            "attempt task families must match policy task_weights panels"
                        )
        return self


__all__ = [
    "AnchorPoolMember",
    "AnchorRole",
    "AttemptManifest",
    "CampaignExecutionPolicyManifest",
    "CampaignPolicy",
    "CampaignPolicyManifest",
    "CampaignRatingPolicyBinding",
    "CampaignManifest",
    "ComparisonDesign",
    "CoverageThresholds",
    "ExecutionManifest",
    "RandomizationSeedCommitment",
    "RetryPolicy",
    "SeedCommitmentScheme",
    "TaskAnchorPool",
    "commit_campaign_randomization_secret",
    "commit_verifier_replay_secret",
]
