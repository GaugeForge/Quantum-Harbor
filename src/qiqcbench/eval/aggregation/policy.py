"""Frozen policy contract and explicit resolvers for ``blocked_bt_v2``.

Development TaskEditions may name policy IDs that this module does not know.
The analysis is intentionally closed: it resolves one
checked rating-policy artifact and one explicitly registered endpoint policy.
Hints on TaskEndpoint are never used for dispatch.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Self, TypeAlias

from pydantic import Field, field_validator, model_validator

from qiqcbench.eval.contracts.base import (
    Digest,
    Identifier,
    NonEmptyStr,
    StrictModel,
    semantic_digest,
)

if TYPE_CHECKING:
    from pathlib import Path

    from qiqcbench.eval.contracts.campaign import CampaignPolicy
    from qiqcbench.eval.contracts.identity import (
        AnyTaskEditionManifest,
        TaskEndpoint,
        TaskRatingUtility,
    )


AnalysisTier: TypeAlias = Literal["development", "candidate", "canonical"]
EndpointAggregationRole: TypeAlias = Literal["racing", "calibrated_absolute"]
EndpointMeasurementSemantics: TypeAlias = Literal[
    "deterministic",
    "verifier_standard_error",
]


class OfficialEvidencePolicyV1(StrictModel):
    """Frozen completeness contract for official rating-neutral evidence."""

    schema_version: Literal[1] = 1
    evidence_policy_id: Literal["official_evidence_v1"] = "official_evidence_v1"
    implementation_id: Literal["official_evidence_v1"] = "official_evidence_v1"
    evidence_policy_digest: Digest | None = None
    comparison_design: Literal["complete_randomized_block"] = "complete_randomized_block"
    canonical_anchor_coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    canonical_overall_coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    canonical_family_coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    canonical_task_coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    canonical_finite_catalog_instance_cell_coverage: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    provisional_overall_coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    minimum_superpopulation_instance_clusters: int = Field(gt=0)

    @model_validator(mode="after")
    def _frozen_relationships_and_digest(self) -> Self:
        if not math.isclose(self.canonical_anchor_coverage, 1.0, abs_tol=0.0):
            raise ValueError("canonical anchor coverage must be exactly 1")
        if not math.isclose(
            self.canonical_finite_catalog_instance_cell_coverage,
            1.0,
            abs_tol=0.0,
        ):
            raise ValueError("canonical finite-catalog instance-cell coverage must be exactly 1")
        payload = self.model_dump(mode="json", exclude={"evidence_policy_digest"})
        expected = semantic_digest(payload)
        if self.evidence_policy_digest is None:
            object.__setattr__(self, "evidence_policy_digest", expected)
        elif self.evidence_policy_digest != expected:
            raise ValueError("evidence_policy_digest does not match policy content")
        return self


class AggregationPolicyRegistration(StrictModel):
    """One endpoint-policy implementation admitted by a rating policy."""

    aggregation_policy_id: Identifier
    implementation_id: Identifier
    aggregation_role: EndpointAggregationRole
    measurement_semantics: EndpointMeasurementSemantics
    candidate_eligible: bool
    canonical_eligible: bool
    weighted_eligible: bool

    @model_validator(mode="after")
    def _canonical_implies_candidate(self) -> Self:
        if self.canonical_eligible and not self.candidate_eligible:
            raise ValueError("canonical endpoint policy must also be candidate eligible")
        return self


_REGISTRATION_PAYLOADS: tuple[dict[str, Any], ...] = (
    {
        "aggregation_policy_id": "racing_deterministic_practical_tie_v1",
        "implementation_id": "racing_deterministic_practical_tie_v1",
        "aggregation_role": "racing",
        "measurement_semantics": "deterministic",
        "candidate_eligible": True,
        "canonical_eligible": True,
        "weighted_eligible": True,
    },
    {
        "aggregation_policy_id": "racing_independent_normal_se_v1",
        "implementation_id": "racing_independent_normal_se_v1",
        "aggregation_role": "racing",
        "measurement_semantics": "verifier_standard_error",
        "candidate_eligible": True,
        "canonical_eligible": True,
        "weighted_eligible": False,
    },
    {
        "aggregation_policy_id": "calibrated_absolute_clipped_linear_v1",
        "implementation_id": "calibrated_absolute_clipped_linear_v1",
        "aggregation_role": "calibrated_absolute",
        "measurement_semantics": "deterministic",
        "candidate_eligible": True,
        "canonical_eligible": True,
        "weighted_eligible": True,
    },
    {
        "aggregation_policy_id": ("calibrated_absolute_clipped_linear_independent_normal_se_v1"),
        "implementation_id": ("calibrated_absolute_clipped_linear_independent_normal_se_v1"),
        "aggregation_role": "calibrated_absolute",
        "measurement_semantics": "verifier_standard_error",
        "candidate_eligible": True,
        "canonical_eligible": True,
        "weighted_eligible": False,
    },
    {
        "aggregation_policy_id": "racing_jeffreys_delta_normal_migration_v1",
        "implementation_id": "racing_jeffreys_delta_normal_migration_v1",
        "aggregation_role": "racing",
        "measurement_semantics": "verifier_standard_error",
        "candidate_eligible": True,
        "canonical_eligible": False,
        "weighted_eligible": False,
    },
)
_REGISTRATION_BY_ID = MappingProxyType(
    {item["aggregation_policy_id"]: item for item in _REGISTRATION_PAYLOADS}
)


class BlockedBtV2Policy(StrictModel):
    """Content-derived official policy for block-aware evaluation v2."""

    schema_version: Literal[1] = 1
    rating_policy_id: Literal["blocked_bt_v2"] = "blocked_bt_v2"
    implementation_id: Literal["blocked_bt_v2_v1"] = "blocked_bt_v2_v1"
    rating_policy_digest: Digest | None = None
    aggregation_policies: list[AggregationPolicyRegistration] = Field(min_length=1)

    comparison_design: Literal["complete_randomized_block"] = "complete_randomized_block"
    pair_weighting: Literal["equal_scheduled_pairs_per_block"]
    hurdle_policy: Literal["scored_pass_scored_fail_model_failure"]
    practical_tie_inclusive: Literal[True]

    primary_ridge_lambda: float = Field(gt=0.0, allow_inf_nan=False)
    optimizer_method: Literal["scipy_lbfgsb"]
    optimizer_initialization: Literal["zeros"]
    optimizer_parameter_order: Literal["utf8_byte_sorted_non_anchor"]
    optimizer_maxiter: int = Field(gt=0)
    optimizer_ftol: float = Field(gt=0.0, allow_inf_nan=False)
    optimizer_gtol: float = Field(gt=0.0, allow_inf_nan=False)
    optimizer_maxls: int = Field(gt=0)
    optimizer_maxcor: int = Field(gt=0)
    maximum_absolute_final_gradient: float = Field(gt=0.0, allow_inf_nan=False)

    elo_anchor: float = Field(allow_inf_nan=False)
    elo_scale: float = Field(gt=0.0, allow_inf_nan=False)
    overall_probability_clipping_epsilon: float = Field(
        gt=0.0,
        lt=0.5,
        allow_inf_nan=False,
    )

    candidate_bootstrap_replicates: int = Field(gt=0)
    canonical_bootstrap_replicates: int = Field(gt=0)
    bootstrap_rng: Literal["PCG64"]
    bootstrap_seed_domain: NonEmptyStr
    bootstrap_purpose_ids: list[Identifier] = Field(min_length=1)
    percentile_method: Literal["linear"]
    percentile_bounds: list[float] = Field(min_length=2, max_length=2)
    minimum_valid_fit_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    minimum_anchor_connectivity_fraction: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    zero_width_relative_tolerance: float = Field(gt=0.0, allow_inf_nan=False)
    prior_sensitivity_multipliers: list[float] = Field(min_length=2, max_length=2)
    maximum_prior_sensitive_movement_elo: float = Field(gt=0.0, allow_inf_nan=False)

    weighted_utility_policy: Literal["deterministic_homogeneous"]
    calibration_method: Literal["clipped_piecewise_linear"]
    calibrated_normal_point_estimate: Literal["analytic_expected_utility"]
    model_failure_utility: Literal[0.0]
    required_gate_failure_utility: Literal[0.0]

    canonical_anchor_coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    canonical_overall_coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    canonical_family_coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    canonical_task_coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    canonical_finite_catalog_instance_cell_coverage: float = Field(
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
    )
    provisional_overall_coverage: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    minimum_superpopulation_instance_clusters: int = Field(gt=0)

    @field_validator("aggregation_policies")
    @classmethod
    def _registrations_are_exact_and_ordered(
        cls,
        values: list[AggregationPolicyRegistration],
    ) -> list[AggregationPolicyRegistration]:
        ids = [item.aggregation_policy_id for item in values]
        if len(ids) != len(set(ids)):
            raise ValueError("aggregation policy IDs must be unique")
        observed = {item.aggregation_policy_id: item.model_dump(mode="json") for item in values}
        expected = {
            str(policy_id): dict(payload) for policy_id, payload in _REGISTRATION_BY_ID.items()
        }
        if observed != expected:
            raise ValueError("blocked_bt_v2 aggregation-policy registry mismatch")
        return sorted(values, key=lambda item: item.aggregation_policy_id)

    @field_validator("bootstrap_purpose_ids")
    @classmethod
    def _bootstrap_purposes_are_frozen(cls, values: list[str]) -> list[str]:
        expected = [
            "outer_cluster_indices",
            "inner_block_indices",
            "attempt_indices",
            "latent_metric",
        ]
        if values != expected:
            raise ValueError("bootstrap purpose IDs do not match blocked_bt_v2")
        return values

    @model_validator(mode="after")
    def _frozen_relationships_and_digest(self) -> Self:
        if self.candidate_bootstrap_replicates >= self.canonical_bootstrap_replicates:
            raise ValueError("candidate bootstrap count must be below canonical count")
        if self.percentile_bounds != [0.025, 0.975]:
            raise ValueError("blocked_bt_v2 percentile bounds must be [0.025, 0.975]")
        if self.prior_sensitivity_multipliers != [0.5, 2.0]:
            raise ValueError("blocked_bt_v2 prior multipliers must be [0.5, 2.0]")
        if not math.isclose(self.canonical_anchor_coverage, 1.0, abs_tol=0.0):
            raise ValueError("canonical anchor coverage must be exactly 1")
        if not math.isclose(
            self.canonical_finite_catalog_instance_cell_coverage,
            1.0,
            abs_tol=0.0,
        ):
            raise ValueError("canonical finite-catalog instance-cell coverage must be exactly 1")
        payload = self.model_dump(mode="json", exclude={"rating_policy_digest"})
        expected = semantic_digest(payload)
        if self.rating_policy_digest is None:
            object.__setattr__(self, "rating_policy_digest", expected)
        elif self.rating_policy_digest != expected:
            raise ValueError("rating_policy_digest does not match policy content")
        return self

    @property
    def aggregation_policy_by_id(self) -> dict[str, AggregationPolicyRegistration]:
        return {item.aggregation_policy_id: item for item in self.aggregation_policies}


_RATING_POLICY_MODELS = MappingProxyType({"blocked_bt_v2": BlockedBtV2Policy})
_RATING_POLICY_GOLDEN_DIGESTS = MappingProxyType(
    {"blocked_bt_v2": "9a16796da00e04ee1b14f7871d3ed4e4382ff23f2c1534fe977d595b00918cc8"}
)
_EVIDENCE_POLICY_MODELS = MappingProxyType({"official_evidence_v1": OfficialEvidencePolicyV1})
_EVIDENCE_POLICY_GOLDEN_DIGESTS = MappingProxyType(
    {"official_evidence_v1": "c72a768a7f3acf0bacdfcabbc3b8eed7d6958f4c7f2d546ee5c2b0af35d8879d"}
)


def evidence_policy_model_for(policy_id: str) -> type[OfficialEvidencePolicyV1]:
    """Resolve one supported evidence-policy ID to its strict artifact model."""

    try:
        return _EVIDENCE_POLICY_MODELS[policy_id]
    except KeyError as exc:
        raise ValueError(f"unsupported evidence policy ID {policy_id!r}") from exc


def resolve_evidence_policy(
    policy_id: str,
    *,
    expected_digest: str,
    root: str | Path | None = None,
) -> OfficialEvidencePolicyV1:
    """Load the frozen evidence artifact and verify its exact expected digest."""

    from qiqcbench.eval.io.config import load_evidence_policy

    model_type = evidence_policy_model_for(policy_id)
    policy = load_evidence_policy(policy_id, root=root)
    if not isinstance(policy, model_type):
        raise ValueError("evidence-policy resolver returned the wrong model")
    if policy.evidence_policy_id != policy_id:
        raise ValueError("evidence-policy artifact ID mismatch")
    if policy.evidence_policy_digest != _EVIDENCE_POLICY_GOLDEN_DIGESTS[policy_id]:
        raise ValueError("evidence-policy artifact differs from its append-only golden body")
    if policy.evidence_policy_digest != expected_digest:
        raise ValueError("evidence-policy digest mismatch")
    return policy


def resolve_campaign_evidence_policy(
    campaign_policy: CampaignPolicy,
    *,
    root: str | Path | None = None,
) -> OfficialEvidencePolicyV1:
    """Resolve the exact evidence policy bound by a schema-2 campaign policy."""

    from qiqcbench.eval.contracts.campaign import CampaignExecutionPolicyManifest

    if not isinstance(campaign_policy, CampaignExecutionPolicyManifest):
        raise ValueError("campaign policy does not carry a schema-2 evidence binding")
    return resolve_evidence_policy(
        campaign_policy.evidence_policy_id,
        expected_digest=campaign_policy.evidence_policy_digest,
        root=root,
    )


def validate_campaign_evidence_policy_binding(
    campaign_policy: CampaignPolicy,
    *,
    root: str | Path | None = None,
) -> OfficialEvidencePolicyV1:
    """Fail closed when schema-2 official gates differ from frozen evidence policy."""

    policy = resolve_campaign_evidence_policy(campaign_policy, root=root)
    expected_thresholds = {
        "anchor": policy.canonical_anchor_coverage,
        "overall": policy.canonical_overall_coverage,
        "family": policy.canonical_family_coverage,
        "task": policy.canonical_task_coverage,
        "provisional": policy.provisional_overall_coverage,
    }
    if campaign_policy.coverage_thresholds.model_dump(mode="json") != expected_thresholds:
        raise ValueError(
            "official campaign coverage thresholds must exactly match the bound evidence policy"
        )
    if campaign_policy.comparison_design != policy.comparison_design:
        raise ValueError(
            "official campaign comparison design must exactly match the bound evidence policy"
        )
    return policy


def rating_policy_model_for(policy_id: str) -> type[BlockedBtV2Policy]:
    """Resolve one supported rating-policy ID to its strict artifact model."""

    try:
        return _RATING_POLICY_MODELS[policy_id]
    except KeyError as exc:
        raise ValueError(f"unsupported rating policy ID {policy_id!r}") from exc


def resolve_rating_policy(
    policy_id: str,
    *,
    expected_digest: str,
    root: str | Path | None = None,
) -> BlockedBtV2Policy:
    """Load the checked artifact and bind an official caller's expected digest."""

    from qiqcbench.eval.io.config import load_rating_policy

    model_type = rating_policy_model_for(policy_id)
    policy = load_rating_policy(policy_id, root=root)
    if not isinstance(policy, model_type):
        raise ValueError("rating-policy resolver returned the wrong model")
    if policy.rating_policy_id != policy_id:
        raise ValueError("rating-policy artifact ID mismatch")
    if policy.rating_policy_digest != _RATING_POLICY_GOLDEN_DIGESTS[policy_id]:
        raise ValueError("rating-policy artifact differs from its append-only golden body")
    if policy.rating_policy_digest != expected_digest:
        raise ValueError("rating-policy digest mismatch")
    return policy


def resolve_campaign_rating_policy(
    campaign_policy: CampaignPolicy,
    *,
    root: str | Path | None = None,
) -> BlockedBtV2Policy:
    """Resolve the exact rating policy bound into a CampaignPolicyManifest."""

    binding = campaign_policy.rating_policy_binding
    if binding is None:
        raise ValueError("record-only campaign policy has no rating policy binding")
    return resolve_rating_policy(
        binding.estimator_policy_id,
        expected_digest=binding.rating_policy_digest,
        root=root,
    )


def effective_superpopulation_cluster_minimum(
    campaign_policy: CampaignPolicy,
    *,
    task_edition_minimum: int,
) -> int:
    """Return the edition floor, tightened by the bound canonical policy."""

    if (
        isinstance(task_edition_minimum, bool)
        or not isinstance(task_edition_minimum, int)
        or task_edition_minimum <= 0
    ):
        raise ValueError("task-edition superpopulation minimum must be a positive integer")
    if campaign_policy.target_tier != "canonical":
        return task_edition_minimum
    if campaign_policy.rating_policy_binding is not None:
        policy_minimum = resolve_campaign_rating_policy(
            campaign_policy
        ).minimum_superpopulation_instance_clusters
    else:
        policy_minimum = resolve_campaign_evidence_policy(
            campaign_policy
        ).minimum_superpopulation_instance_clusters
    return max(
        task_edition_minimum,
        policy_minimum,
    )


def resolve_endpoint_aggregation_policy(
    policy: BlockedBtV2Policy,
    aggregation_policy_id: str | None,
    *,
    target_tier: AnalysisTier,
) -> AggregationPolicyRegistration | None:
    """Resolve one endpoint policy, leaving unknown development IDs open."""

    registration = (
        None
        if aggregation_policy_id is None
        else policy.aggregation_policy_by_id.get(aggregation_policy_id)
    )
    if registration is None:
        if target_tier == "development":
            return None
        raise ValueError("official rating endpoint requires a recognized aggregation_policy_id")
    if target_tier == "candidate" and not registration.candidate_eligible:
        raise ValueError("aggregation policy is not candidate eligible")
    if target_tier == "canonical" and not registration.canonical_eligible:
        raise ValueError("aggregation policy is not canonical eligible")
    return registration


def validate_task_edition_aggregation_policy(
    policy: BlockedBtV2Policy,
    task_edition: AnyTaskEditionManifest,
    *,
    target_tier: AnalysisTier,
    resolved_utility: TaskRatingUtility | None = None,
) -> AnyTaskEditionManifest:
    """Validate one edition's task utility against the selected rating policy."""

    utility = resolved_utility or task_edition.rating_utility
    if utility is None:
        raise ValueError("rating-neutral TaskEdition requires a resolved rating utility")
    endpoint_by_id = {endpoint.endpoint_id: endpoint for endpoint in task_edition.endpoints}
    try:
        endpoints: list[TaskEndpoint] = [
            endpoint_by_id[endpoint_id] for endpoint_id in utility.endpoint_ids
        ]
    except KeyError as exc:
        raise ValueError("resolved rating utility references an unknown endpoint") from exc
    if any(
        endpoint.aggregation_role not in ("racing", "calibrated_absolute") for endpoint in endpoints
    ):
        raise ValueError("resolved rating utility references an ineligible endpoint")
    registrations: list[AggregationPolicyRegistration | None] = []
    for endpoint in endpoints:
        registration = resolve_endpoint_aggregation_policy(
            policy,
            endpoint.aggregation_policy_id,
            target_tier=target_tier,
        )
        registrations.append(registration)
        if registration is None:
            continue
        if registration.aggregation_role != endpoint.aggregation_role:
            raise ValueError("endpoint aggregation role disagrees with aggregation policy")
        if registration.measurement_semantics != endpoint.measurement_semantics:
            raise ValueError("endpoint measurement semantics disagree with aggregation policy")
        if endpoint.aggregation_role != "racing" and endpoint.below_gate_rankable:
            raise ValueError("below_gate_rankable is official-v1 racing-only")

    if target_tier == "development" or utility.kind != "weighted_endpoints":
        return task_edition

    roles = {endpoint.aggregation_role for endpoint in endpoints}
    if len(roles) != 1:
        raise ValueError("official weighted utility cannot mix aggregation roles")
    if any(endpoint.measurement_semantics != "deterministic" for endpoint in endpoints):
        raise ValueError("official weighted utility must be deterministic")
    if any(
        registration is None or not registration.weighted_eligible for registration in registrations
    ):
        raise ValueError("aggregation policy does not support official weighted utility")
    gate_policies = {tuple(endpoint.required_gate_ids) for endpoint in endpoints}
    if len(gate_policies) != 1:
        raise ValueError("official weighted utility must share required-gate conditioning")
    if roles == {"racing"}:
        hurdle_policies = {
            (endpoint.hurdle_gate_id, endpoint.below_gate_rankable) for endpoint in endpoints
        }
        if len(hurdle_policies) != 1:
            raise ValueError("official weighted racing utility must share one hurdle policy")
    return task_edition


__all__ = [
    "AggregationPolicyRegistration",
    "AnalysisTier",
    "BlockedBtV2Policy",
    "effective_superpopulation_cluster_minimum",
    "rating_policy_model_for",
    "resolve_campaign_rating_policy",
    "resolve_endpoint_aggregation_policy",
    "resolve_rating_policy",
    "validate_task_edition_aggregation_policy",
]
