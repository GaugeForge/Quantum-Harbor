"""Identity and scientific-contract models for evaluation framework v2."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Annotated, Any, Literal, Self, TypeAlias
from urllib.parse import urlsplit

from pydantic import Field, JsonValue, TypeAdapter, field_validator, model_validator

from qiqcbench.eval.contracts.base import (
    Digest,
    Identifier,
    NonEmptyStr,
    StrictModel,
    ensure_finite_json,
    semantic_id,
)

Direction: TypeAlias = Literal["higher_is_better", "lower_is_better"]
AggregationRole: TypeAlias = Literal["profile_only", "calibrated_absolute", "racing"]
LegacyAdapterKind: TypeAlias = Literal[
    "elo_score_report_v1",
    "freeform_rubric_json_v1",
]
LegacyArtifactRole: TypeAlias = Literal["legacy_elo_report", "task_score_report"]
BindingMode: TypeAlias = Literal[
    "verifier_only",
    "qsim_and_verifier",
    "fixed_instance_stochastic_block",
]
CommitmentScheme: TypeAlias = Literal[
    "public_content_sha256",
    "hmac_sha256",
    "random_256bit",
]

_SENSITIVE_KEY = re.compile(
    r"(?:^|_)(?:token|access_token|auth_token|api_key|secret|password|credential|"
    r"private_key|authorization)(?:_|$)",
    re.IGNORECASE,
)
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_EXTERNAL_URL = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_AUTHORIZATION_VALUE = re.compile(r"^\s*(?:Bearer|Basic)\s+\S+", re.IGNORECASE)
_RUNTIME_ENV_NAME = re.compile(r"[A-Z_][A-Z0-9_]*\Z")
_RESERVED_RUNTIME_ENV = frozenset(
    {
        "COMPOSE_FILE",
        "COMPOSE_PATH_SEPARATOR",
        "COMPOSE_PROFILES",
        "COMPOSE_PROJECT_NAME",
        "QIQCBENCH_TASK_ID",
        "QSIM_BACKEND_MODE",
        "QIQCBENCH_SURFACES",
        "QIQCBENCH_EXECUTION_CONTEXT_ID",
    }
)


def _assign_or_validate_id(model: StrictModel, field_name: str, expected: str) -> None:
    supplied = getattr(model, field_name)
    if supplied is None:
        object.__setattr__(model, field_name, expected)
    elif supplied != expected:
        raise ValueError(f"{field_name} does not match the manifest's semantic content")


def _validate_public_value(
    value: Any,
    *,
    path: str,
    allow_public_urls: bool = False,
) -> Any:
    ensure_finite_json(value, path=path)
    if isinstance(value, str):
        if value.startswith(("/", "~/")) or _WINDOWS_ABSOLUTE_PATH.match(value):
            raise ValueError(f"{path} must not contain a machine-local path")
        if _EXTERNAL_URL.match(value):
            if not allow_public_urls:
                raise ValueError(f"{path} must not contain an external URL")
            parsed = urlsplit(value)
            if (
                parsed.scheme.lower() not in {"http", "https"}
                or parsed.hostname is None
                or parsed.username is not None
                or parsed.password is not None
            ):
                raise ValueError(f"{path} must be a public HTTP(S) URL without user information")
        if _AUTHORIZATION_VALUE.match(value):
            raise ValueError(f"{path} must not contain an authorization value")
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_public_value(
                item,
                path=f"{path}[{index}]",
                allow_public_urls=allow_public_urls,
            )
        return value
    if isinstance(value, dict):
        for key, item in value.items():
            if _SENSITIVE_KEY.search(key):
                raise ValueError(f"{path} contains a sensitive key {key!r}")
            _validate_public_value(
                item,
                path=f"{path}.{key}",
                allow_public_urls=allow_public_urls,
            )
    return value


def _require_unique(values: list[str], *, field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")


class ResourceBudget(StrictModel):
    """Concrete resource selection for one task execution profile."""

    wall_clock_s: int = Field(gt=0)
    turns: int | None = Field(default=None, ge=0)
    tokens: int | None = Field(default=None, ge=0)
    tool_calls: int | None = Field(default=None, ge=0)
    experiment_calls: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class ResourceEnvelope(StrictModel):
    """Maximum resource conditions allowed by a scientific task edition."""

    max_wall_clock_s: int = Field(gt=0)
    max_turns: int | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, ge=0)
    max_tool_calls: int | None = Field(default=None, ge=0)
    max_experiment_calls: int | None = Field(default=None, ge=0)
    max_cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class PlayerManifest(StrictModel):
    """Immutable identity of the agent system, independent of task conditions."""

    schema_version: Literal[1] = 1
    player_id: Identifier | None = None
    display_name: NonEmptyStr
    public_metadata: dict[Identifier, JsonValue] = Field(default_factory=dict)
    harness_id: Identifier
    harness_version: NonEmptyStr
    model_provider: Identifier
    model_id: NonEmptyStr
    model_revision: NonEmptyStr | None = None
    executable_model_id: NonEmptyStr | None = None
    reasoning_effort: NonEmptyStr | None = None
    inference_parameters: dict[Identifier, JsonValue] = Field(default_factory=dict)
    agent_image_digest: Digest | None = None
    agent_code_digest: Digest | None = None
    base_system_prompt_digest: Digest | None = None
    tool_implementation_digests: dict[Identifier, Digest] = Field(default_factory=dict)

    @field_validator("inference_parameters")
    @classmethod
    def _safe_inference_parameters(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        _validate_public_value(value, path="inference_parameters")
        return value

    @field_validator("public_metadata")
    @classmethod
    def _safe_public_metadata(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        _validate_public_value(
            value,
            path="public_metadata",
            allow_public_urls=True,
        )
        return value

    @model_validator(mode="after")
    def _bind_player_id(self) -> Self:
        payload = {
            "schema_version": self.schema_version,
            "harness_id": self.harness_id,
            "harness_version": self.harness_version,
            "model_provider": self.model_provider,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "executable_model_id": self.executable_model_id,
            "reasoning_effort": self.reasoning_effort,
            "inference_parameters": self.inference_parameters,
            "agent_image_digest": self.agent_image_digest,
            "agent_code_digest": self.agent_code_digest,
            "base_system_prompt_digest": self.base_system_prompt_digest,
            "tool_implementation_digests": self.tool_implementation_digests,
        }
        _assign_or_validate_id(self, "player_id", semantic_id("player", payload))
        return self


class ExecutionProfile(StrictModel):
    """Concrete task-varying runtime conditions selected by a campaign."""

    schema_version: Literal[1] = 1
    execution_profile_id: Identifier | None = None
    backend_track: Identifier
    surfaces: list[Identifier] = Field(min_length=1)
    network_policy: Identifier
    harness_wrapper_digest: Digest | None = None
    runtime_image_digests: dict[Identifier, Digest] = Field(default_factory=dict)
    runtime_configuration_digest: Digest | None = None
    resource_budget: ResourceBudget

    @model_validator(mode="after")
    def _bind_execution_profile_id(self) -> Self:
        _require_unique(self.surfaces, field_name="surfaces")
        reserved_roles = {"agent", "main", "verifier", "rescore"}.intersection(
            self.runtime_image_digests
        )
        if reserved_roles:
            raise ValueError(
                "runtime_image_digests contains roles owned by PlayerManifest or "
                f"VerifierRevisionManifest: {sorted(reserved_roles)!r}"
            )
        object.__setattr__(
            self,
            "surfaces",
            self._freeze_contract_value(sorted(self.surfaces)),
        )
        payload = self.model_dump(
            mode="json",
            exclude={"execution_profile_id"},
        )
        _assign_or_validate_id(
            self,
            "execution_profile_id",
            semantic_id("execution_profile", payload),
        )
        return self


class RatedConfigurationManifest(StrictModel):
    """Release-level claim binding one player to task execution profiles."""

    schema_version: Literal[1] = 1
    rated_configuration_id: Identifier | None = None
    player_id: Identifier
    task_execution_profile_ids: dict[Identifier, Identifier] = Field(min_length=1)
    campaign_policy_digest: Digest

    @model_validator(mode="after")
    def _bind_rated_configuration_id(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"rated_configuration_id"})
        _assign_or_validate_id(
            self,
            "rated_configuration_id",
            semantic_id("rated_configuration", payload),
        )
        return self


class CalibrationPoint(StrictModel):
    """One predeclared native-value to utility calibration point."""

    native_value: float = Field(allow_inf_nan=False)
    utility: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class TaskEndpoint(StrictModel):
    """One authoritative gate, metric, resource, or diagnostic endpoint."""

    endpoint_id: Identifier
    metric_role: Literal["gate", "quality", "resource", "diagnostic"]
    unit: NonEmptyStr
    direction: Direction
    aggregation_role: AggregationRole
    measurement_semantics: Literal["deterministic", "verifier_standard_error"]
    practical_equivalence_margin: float = Field(ge=0.0, allow_inf_nan=False)
    required_gate_ids: list[Identifier]
    hurdle_gate_id: Identifier | None = None
    gate_role: Literal["hurdle", "reporting_only"] | None = None
    gate_rule: Literal["threshold", "boolean", "composite"] | None = None
    gate_threshold: float | None = Field(default=None, allow_inf_nan=False)
    gate_threshold_source: NonEmptyStr | None = None
    below_gate_rankable: bool
    aggregation_policy_id: Identifier | None = None
    transform_hint: NonEmptyStr | None = None
    likelihood_hint: NonEmptyStr | None = None
    source_digests: dict[Identifier, Digest]
    calibration_points: list[CalibrationPoint] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_endpoint_semantics(self) -> Self:
        _require_unique(self.required_gate_ids, field_name="required_gate_ids")
        if not self.source_digests:
            raise ValueError("source_digests must declare at least one public-safe source")

        if self.metric_role == "gate":
            if self.gate_role is None:
                raise ValueError("gate endpoints must declare gate_role")
            if self.gate_rule is None:
                raise ValueError("gate endpoints must declare gate_rule")
            if self.aggregation_role != "profile_only":
                raise ValueError("gate endpoints must use profile_only aggregation")
            if self.hurdle_gate_id is not None or self.required_gate_ids:
                raise ValueError("gate endpoints cannot depend on other gates")
            if self.below_gate_rankable:
                raise ValueError("gate endpoints cannot set below_gate_rankable")
            declared_thresholds = (
                self.gate_threshold is not None,
                self.gate_threshold_source is not None,
            )
            if self.gate_rule == "threshold":
                if sum(declared_thresholds) != 1:
                    raise ValueError(
                        "threshold gate must declare exactly one of gate_threshold or "
                        "gate_threshold_source"
                    )
            elif self.gate_rule == "composite":
                if sum(declared_thresholds) > 1:
                    raise ValueError("composite gate may declare at most one threshold component")
            elif any(declared_thresholds):
                raise ValueError("boolean gates cannot declare a threshold")
        elif any(
            value is not None
            for value in (
                self.gate_role,
                self.gate_rule,
                self.gate_threshold,
                self.gate_threshold_source,
            )
        ):
            raise ValueError("only gate endpoints may declare gate policy fields")

        if self.aggregation_role == "racing":
            if self.metric_role not in ("quality", "resource"):
                raise ValueError("racing endpoints must be quality or resource metrics")
            if self.hurdle_gate_id is None:
                raise ValueError("racing endpoints must declare hurdle_gate_id")
            if self.hurdle_gate_id not in self.required_gate_ids:
                raise ValueError("hurdle_gate_id must also appear in required_gate_ids")
        elif self.hurdle_gate_id is not None:
            raise ValueError("only racing endpoints may declare hurdle_gate_id")

        if self.aggregation_role == "calibrated_absolute":
            if self.metric_role not in ("quality", "resource"):
                raise ValueError(
                    "calibrated_absolute endpoints must be quality or resource metrics"
                )
            if len(self.calibration_points) < 2:
                raise ValueError(
                    "calibrated_absolute endpoints require at least two calibration_points"
                )
            native_values = [point.native_value for point in self.calibration_points]
            if any(
                right <= left for left, right in zip(native_values, native_values[1:], strict=False)
            ):
                raise ValueError("calibration_points native values must strictly increase")
            utilities = [point.utility for point in self.calibration_points]
            if self.direction == "higher_is_better":
                monotone = all(
                    right >= left for left, right in zip(utilities, utilities[1:], strict=False)
                )
            else:
                monotone = all(
                    right <= left for left, right in zip(utilities, utilities[1:], strict=False)
                )
            if not monotone:
                raise ValueError("calibration_points utility must be monotone with direction")
        elif self.calibration_points:
            raise ValueError("calibration_points are only valid for calibrated_absolute endpoints")

        object.__setattr__(
            self,
            "required_gate_ids",
            self._freeze_contract_value(sorted(self.required_gate_ids)),
        )
        return self


class RatingUtilityComponent(StrictModel):
    """One component of a predeclared task-level utility."""

    endpoint_id: Identifier
    weight: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)


class TaskRatingUtility(StrictModel):
    """The edition's single claim to one task-level utility slot."""

    kind: Literal["primary_endpoint", "weighted_endpoints"]
    primary_endpoint_id: Identifier | None = None
    components: list[RatingUtilityComponent] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_utility(self) -> Self:
        if self.kind == "primary_endpoint":
            if self.primary_endpoint_id is None:
                raise ValueError("primary_endpoint utility requires primary_endpoint_id")
            if self.components:
                raise ValueError("primary_endpoint utility cannot declare components")
        else:
            if self.primary_endpoint_id is not None:
                raise ValueError("weighted_endpoints utility cannot declare primary_endpoint_id")
            if len(self.components) < 2:
                raise ValueError("weighted_endpoints utility requires at least two components")
            endpoint_ids = [component.endpoint_id for component in self.components]
            _require_unique(endpoint_ids, field_name="rating utility endpoint IDs")
            if not math.isclose(
                sum(component.weight for component in self.components),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("rating utility component weights must sum to one")
            object.__setattr__(
                self,
                "components",
                self._freeze_contract_value(
                    sorted(self.components, key=lambda component: component.endpoint_id)
                ),
            )
        return self

    @property
    def endpoint_ids(self) -> tuple[str, ...]:
        if self.primary_endpoint_id is not None:
            return (self.primary_endpoint_id,)
        return tuple(component.endpoint_id for component in self.components)


class SourceRevisionCommitment(StrictModel):
    """Public-safe semantic epoch for an instance source or population."""

    scheme: CommitmentScheme
    value: Digest


class InstanceSourcePolicy(StrictModel):
    """Versioned target population and private-binding route for an edition."""

    source_id: Identifier
    source_kind: Identifier
    source_revision: SourceRevisionCommitment
    population_type: Literal["finite_catalog", "superpopulation", "fixed_instance"]
    binding_mode: BindingMode
    declared_instance_ids: list[Identifier] = Field(default_factory=list)
    minimum_distinct_clusters: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_population(self) -> Self:
        _require_unique(self.declared_instance_ids, field_name="declared_instance_ids")
        if (
            self.binding_mode == "fixed_instance_stochastic_block"
            and self.population_type != "fixed_instance"
        ):
            raise ValueError(
                "fixed_instance_stochastic_block binding requires fixed_instance population"
            )
        if self.population_type == "finite_catalog":
            if not self.declared_instance_ids:
                raise ValueError("finite_catalog requires declared_instance_ids")
            if self.minimum_distinct_clusters is not None:
                raise ValueError("finite_catalog cannot declare minimum_distinct_clusters")
        elif self.population_type == "superpopulation":
            if self.declared_instance_ids:
                raise ValueError("superpopulation cannot predeclare a finite instance catalog")
            if self.minimum_distinct_clusters is None:
                raise ValueError("superpopulation requires minimum_distinct_clusters")
        else:
            if len(self.declared_instance_ids) != 1:
                raise ValueError("fixed_instance requires exactly one declared instance ID")
            if self.minimum_distinct_clusters is not None:
                raise ValueError("fixed_instance cannot declare minimum_distinct_clusters")
        object.__setattr__(
            self,
            "declared_instance_ids",
            self._freeze_contract_value(sorted(self.declared_instance_ids)),
        )
        return self


class LegacyEndpointMapping(StrictModel):
    """Declared projection from one legacy artifact field to a v2 endpoint."""

    source_path: NonEmptyStr | None = None
    source_metric_name: NonEmptyStr | None = None
    standard_error_source_path: NonEmptyStr | None = None
    gate_pass_source_path: NonEmptyStr | None = None
    threshold_source_path: NonEmptyStr | None = None
    evaluation_replica_domain: Identifier | None = None
    required_when_scored: bool = True
    endpoint_id: Identifier


class LegacyAdapterSpec(StrictModel):
    """Named verifier-revision adapter for one legacy artifact role."""

    adapter_id: Identifier
    adapter_kind: LegacyAdapterKind
    artifact_role: LegacyArtifactRole
    report_schema_version: int | None = Field(default=None, ge=1)
    task_id_source_path: NonEmptyStr | None = None
    reward_source_path: NonEmptyStr | None = None
    endpoint_mappings: list[LegacyEndpointMapping]

    @model_validator(mode="after")
    def _validate_mappings(self) -> Self:
        if not self.endpoint_mappings:
            raise ValueError("legacy adapter endpoint_mappings must not be empty")
        endpoint_ids = [mapping.endpoint_id for mapping in self.endpoint_mappings]
        _require_unique(endpoint_ids, field_name="legacy adapter target endpoint IDs")
        if self.adapter_kind == "elo_score_report_v1":
            rubric_fields = (
                self.report_schema_version,
                self.task_id_source_path,
                self.reward_source_path,
            )
            if any(value is not None for value in rubric_fields):
                raise ValueError("EloScoreReport v1 adapters cannot declare free-form report paths")
            if self.artifact_role != "legacy_elo_report":
                raise ValueError("EloScoreReport v1 adapter must consume legacy_elo_report")
            if any(mapping.source_path is None for mapping in self.endpoint_mappings):
                raise ValueError("EloScoreReport v1 mappings require source_path")
            if any(
                mapping.gate_pass_source_path is not None
                or mapping.threshold_source_path is not None
                for mapping in self.endpoint_mappings
            ):
                raise ValueError("EloScoreReport v1 mappings cannot declare rubric gate paths")
        else:
            if self.artifact_role != "task_score_report":
                raise ValueError("declared JSON rubric adapters must consume task_score_report")
            if self.task_id_source_path is None:
                raise ValueError("declared JSON rubric adapter requires task_id_source_path")
            if any(mapping.source_metric_name is not None for mapping in self.endpoint_mappings):
                raise ValueError("declared JSON rubric mappings cannot use source_metric_name")
        object.__setattr__(
            self,
            "endpoint_mappings",
            self._freeze_contract_value(
                sorted(
                    self.endpoint_mappings,
                    key=lambda mapping: (
                        mapping.source_path or "",
                        mapping.source_metric_name or "",
                        mapping.endpoint_id,
                    ),
                ),
            ),
        )
        return self


class ScientificBehaviorCommitment(StrictModel):
    """Public-safe semantic epoch for task behavior not otherwise exposed."""

    scheme: CommitmentScheme
    value: Digest


class ScientificContractCommitment(StrictModel):
    """Public-safe semantic epoch for an authoritative scoring contract."""

    scheme: CommitmentScheme
    value: Digest


class TaskEditionManifest(StrictModel):
    """Comparable scientific contract, independent of verifier implementation."""

    schema_version: Literal[1] = 1
    task_edition_id: Identifier | None = None
    task_id: Identifier
    task_family_id: Identifier
    compatibility_group: Identifier
    instruction_digest: Digest
    public_material_digests: dict[Identifier, Digest] = Field(default_factory=dict)
    scientific_behavior_commitment: ScientificBehaviorCommitment
    verifier_contract_commitment: ScientificContractCommitment
    allowed_backend_tracks: list[Identifier]
    allowed_surfaces: list[Identifier]
    allowed_network_policies: list[Identifier]
    resource_envelope: ResourceEnvelope
    instance_source: InstanceSourcePolicy
    endpoints: list[TaskEndpoint]
    admission_gate_id: Identifier
    rating_utility: TaskRatingUtility

    @model_validator(mode="after")
    def _validate_and_bind_edition(self) -> Self:
        if "instruction" in self.public_material_digests:
            raise ValueError(
                "public_material_digests cannot redefine the reserved instruction role"
            )
        for field_name in (
            "allowed_backend_tracks",
            "allowed_surfaces",
            "allowed_network_policies",
        ):
            values = getattr(self, field_name)
            if not values:
                raise ValueError(f"{field_name} must not be empty")
            _require_unique(values, field_name=field_name)

        endpoints = {endpoint.endpoint_id: endpoint for endpoint in self.endpoints}
        if len(endpoints) != len(self.endpoints):
            raise ValueError("endpoint_id values must be unique")
        admission = endpoints.get(self.admission_gate_id)
        if admission is None or admission.metric_role != "gate":
            raise ValueError("admission_gate_id must reference a declared gate endpoint")
        if admission.gate_role != "hurdle":
            raise ValueError("admission_gate_id must reference a hurdle gate")

        public_sources = {
            "instruction": self.instruction_digest,
            **self.public_material_digests,
        }
        for endpoint in self.endpoints:
            for role, digest in endpoint.source_digests.items():
                if role not in public_sources:
                    raise ValueError(
                        f"endpoint {endpoint.endpoint_id!r} source_digests may reference only "
                        "declared public edition materials"
                    )
                if public_sources[role] != digest:
                    raise ValueError(
                        f"endpoint {endpoint.endpoint_id!r} source digest does not match the "
                        f"declared public material role {role!r}"
                    )

        for endpoint in self.endpoints:
            for gate_id in endpoint.required_gate_ids:
                gate = endpoints.get(gate_id)
                if gate is None or gate.metric_role != "gate":
                    raise ValueError(
                        f"endpoint {endpoint.endpoint_id!r} references missing required gate "
                        f"{gate_id!r}"
                    )
            if endpoint.hurdle_gate_id is not None:
                hurdle = endpoints.get(endpoint.hurdle_gate_id)
                if hurdle is None or hurdle.gate_role != "hurdle":
                    raise ValueError(
                        f"endpoint {endpoint.endpoint_id!r} hurdle_gate_id must reference "
                        "a hurdle gate"
                    )

        rating_utility = self.rating_utility
        if rating_utility is not None:
            for endpoint_id in rating_utility.endpoint_ids:
                endpoint = endpoints.get(endpoint_id)
                if endpoint is None or endpoint.aggregation_role not in (
                    "racing",
                    "calibrated_absolute",
                ):
                    raise ValueError(
                        f"rating utility must reference an eligible endpoint; got {endpoint_id!r}"
                    )

        object.__setattr__(
            self,
            "allowed_backend_tracks",
            self._freeze_contract_value(sorted(self.allowed_backend_tracks)),
        )
        object.__setattr__(
            self,
            "allowed_surfaces",
            self._freeze_contract_value(sorted(self.allowed_surfaces)),
        )
        object.__setattr__(
            self,
            "allowed_network_policies",
            self._freeze_contract_value(sorted(self.allowed_network_policies)),
        )
        object.__setattr__(
            self,
            "endpoints",
            self._freeze_contract_value(sorted(self.endpoints, key=lambda item: item.endpoint_id)),
        )
        payload = self.model_dump(mode="json", exclude={"task_edition_id"})
        _assign_or_validate_id(
            self,
            "task_edition_id",
            semantic_id(
                "task_edition",
                payload,
                unordered_fields={
                    "allowed_backend_tracks",
                    "allowed_surfaces",
                    "allowed_network_policies",
                    "endpoints",
                },
            ),
        )
        return self

    def validate_execution_profile(self, profile: ExecutionProfile) -> ExecutionProfile:
        """Fail if a concrete profile falls outside this edition's envelope."""

        if profile.backend_track not in self.allowed_backend_tracks:
            raise ValueError(f"backend_track {profile.backend_track!r} is outside allowed envelope")
        unsupported_surfaces = set(profile.surfaces).difference(self.allowed_surfaces)
        if unsupported_surfaces:
            raise ValueError(
                f"surfaces {sorted(unsupported_surfaces)!r} are outside allowed envelope"
            )
        if profile.network_policy not in self.allowed_network_policies:
            raise ValueError(
                f"network_policy {profile.network_policy!r} is outside allowed envelope"
            )

        limits = {
            "wall_clock_s": self.resource_envelope.max_wall_clock_s,
            "turns": self.resource_envelope.max_turns,
            "tokens": self.resource_envelope.max_tokens,
            "tool_calls": self.resource_envelope.max_tool_calls,
            "experiment_calls": self.resource_envelope.max_experiment_calls,
            "cost_usd": self.resource_envelope.max_cost_usd,
        }
        for field_name, maximum in limits.items():
            selected = getattr(profile.resource_budget, field_name)
            if maximum is not None and selected is None:
                raise ValueError(
                    f"resource {field_name} requires a concrete selection within the envelope"
                )
            if maximum is None:
                continue
            assert selected is not None
            if selected > maximum:
                raise ValueError(
                    f"resource {field_name}={selected!r} exceeds allowed maximum {maximum!r}"
                )
        return profile

    def validate_instance_binding(self, binding: InstanceBinding) -> InstanceBinding:
        """Validate a private binding against this edition's source declaration."""

        instance = binding.instance_ref
        if binding.task_edition_id != self.task_edition_id:
            raise ValueError("instance binding task_edition_id does not match the edition")
        if instance.source_id != self.instance_source.source_id:
            raise ValueError("instance binding source_id does not match the edition")
        if instance.source_kind != self.instance_source.source_kind:
            raise ValueError("instance binding source_kind does not match the edition")
        if instance.source_revision != self.instance_source.source_revision:
            raise ValueError("instance binding source revision does not match the edition")
        if binding.binding_mode != self.instance_source.binding_mode:
            raise ValueError("instance binding mode does not match the edition")
        if (
            self.instance_source.population_type in ("finite_catalog", "fixed_instance")
            and instance.instance_id not in self.instance_source.declared_instance_ids
        ):
            raise ValueError("instance is outside the edition's declared catalog")
        return binding


class TaskEditionManifestV2(TaskEditionManifest):
    """Rating-neutral scientific observation contract for record-first runs.

    V1 remains byte-for-byte compatible and carries its historical rating
    utility.  V2 deliberately commits no utility interpretation; an optional
    RatingEpoch v2 supplies that interpretation later without changing the
    task edition or its evaluation-comparability boundary.
    """

    schema_version: Literal[2] = 2
    rating_utility: None = None


AnyTaskEditionManifest: TypeAlias = Annotated[
    TaskEditionManifest | TaskEditionManifestV2,
    Field(discriminator="schema_version"),
]
_TASK_EDITION_ADAPTER = TypeAdapter(AnyTaskEditionManifest)


def validate_task_edition_manifest(value: Any) -> AnyTaskEditionManifest:
    """Validate either frozen TaskEdition schema without coercing its version."""

    if isinstance(value, TaskEditionManifest | TaskEditionManifestV2):
        value = value.model_dump(mode="python")
    return _TASK_EDITION_ADAPTER.validate_python(value)


def validate_task_edition_manifest_json(value: str | bytes) -> AnyTaskEditionManifest:
    """Validate canonical JSON bytes against the version-discriminated contract."""

    return _TASK_EDITION_ADAPTER.validate_json(value)


class VerifierRevisionManifest(StrictModel):
    """One implementation revision of a TaskEdition verifier contract."""

    schema_version: Literal[1] = 1
    verifier_revision_id: Identifier | None = None
    task_id: Identifier
    task_edition_id: Identifier
    verifier_contract_commitment: ScientificContractCommitment
    implementation_revision: NonEmptyStr
    artifact_digests: dict[Identifier, Digest]
    runtime_image_digests: dict[Identifier, Digest] = Field(default_factory=dict)
    legacy_adapters: list[LegacyAdapterSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _bind_verifier_revision_id(self) -> Self:
        if not self.artifact_digests:
            raise ValueError("artifact_digests must not be empty")
        reserved_roles = {"agent", "main", "qsim"}.intersection(self.runtime_image_digests)
        if reserved_roles:
            raise ValueError(
                "verifier runtime_image_digests contains roles owned by PlayerManifest or "
                f"ExecutionProfile: {sorted(reserved_roles)!r}"
            )
        adapter_ids = [adapter.adapter_id for adapter in self.legacy_adapters]
        _require_unique(adapter_ids, field_name="legacy adapter IDs")
        artifact_roles = [adapter.artifact_role for adapter in self.legacy_adapters]
        _require_unique(artifact_roles, field_name="legacy adapter artifact roles")
        object.__setattr__(
            self,
            "legacy_adapters",
            self._freeze_contract_value(
                sorted(self.legacy_adapters, key=lambda adapter: adapter.adapter_id)
            ),
        )
        payload = self.model_dump(mode="json", exclude={"verifier_revision_id"})
        # Preserve the identity of deterministic v1 adapters created before
        # uncertainty paths existed.  A declared uncertainty path remains part
        # of the verifier revision's semantic identity.
        for adapter in payload["legacy_adapters"]:
            for mapping in adapter["endpoint_mappings"]:
                if mapping["standard_error_source_path"] is None:
                    del mapping["standard_error_source_path"]
        _assign_or_validate_id(
            self,
            "verifier_revision_id",
            semantic_id("verifier_revision", payload),
        )
        return self


def validate_verifier_revision_for_edition(
    edition: TaskEditionManifest,
    revision: VerifierRevisionManifest,
) -> VerifierRevisionManifest:
    """Bind a scorer implementation to exactly the scientific contract it claims."""

    if revision.task_id != edition.task_id:
        raise ValueError("verifier revision task_id does not match TaskEdition")
    if revision.task_edition_id != edition.task_edition_id:
        raise ValueError("verifier revision task_edition_id does not match TaskEdition")
    if revision.verifier_contract_commitment != edition.verifier_contract_commitment:
        raise ValueError("verifier revision contract commitment does not match TaskEdition")
    endpoints = {endpoint.endpoint_id: endpoint for endpoint in edition.endpoints}
    for adapter in revision.legacy_adapters:
        for mapping in adapter.endpoint_mappings:
            if mapping.endpoint_id not in endpoints:
                raise ValueError(
                    f"legacy adapter {adapter.adapter_id!r} references missing TaskEdition "
                    f"endpoint {mapping.endpoint_id!r}"
                )
            endpoint = endpoints[mapping.endpoint_id]
            if mapping.source_path is None and endpoint.metric_role != "gate":
                raise ValueError(
                    f"legacy adapter {adapter.adapter_id!r} metric endpoint "
                    f"{mapping.endpoint_id!r} requires source_path"
                )
            if adapter.adapter_kind == "freeform_rubric_json_v1":
                if endpoint.metric_role == "gate":
                    if endpoint.gate_rule in {"boolean", "composite"} and (
                        mapping.gate_pass_source_path is None
                    ):
                        raise ValueError(
                            f"legacy adapter {adapter.adapter_id!r} {endpoint.gate_rule} gate "
                            f"{mapping.endpoint_id!r} requires gate_pass_source_path"
                        )
                    if endpoint.gate_rule == "threshold" and mapping.source_path is None:
                        raise ValueError(
                            f"legacy adapter {adapter.adapter_id!r} threshold gate "
                            f"{mapping.endpoint_id!r} requires source_path"
                        )
                    if (
                        endpoint.gate_rule == "composite"
                        and (
                            endpoint.gate_threshold is not None
                            or endpoint.gate_threshold_source is not None
                        )
                        and mapping.source_path is None
                    ):
                        raise ValueError(
                            f"legacy adapter {adapter.adapter_id!r} composite gate "
                            f"{mapping.endpoint_id!r} with a threshold component requires "
                            "source_path"
                        )
                    if endpoint.gate_threshold_source is not None and (
                        mapping.threshold_source_path is None
                    ):
                        raise ValueError(
                            f"legacy adapter {adapter.adapter_id!r} dynamic threshold gate "
                            f"{mapping.endpoint_id!r} requires threshold_source_path"
                        )
                    if endpoint.gate_threshold is not None and (
                        mapping.threshold_source_path is not None
                    ):
                        raise ValueError(
                            f"legacy adapter {adapter.adapter_id!r} fixed threshold gate "
                            f"{mapping.endpoint_id!r} cannot map another threshold"
                        )
                elif (
                    mapping.gate_pass_source_path is not None
                    or mapping.threshold_source_path is not None
                ):
                    raise ValueError(
                        f"legacy adapter {adapter.adapter_id!r} metric endpoint "
                        f"{mapping.endpoint_id!r} cannot declare gate paths"
                    )
            if (
                endpoint.measurement_semantics == "verifier_standard_error"
                and mapping.standard_error_source_path is None
            ):
                raise ValueError(
                    f"legacy adapter {adapter.adapter_id!r} must map standard error for "
                    f"stochastic endpoint {mapping.endpoint_id!r}"
                )
            if (
                endpoint.measurement_semantics == "verifier_standard_error"
                and mapping.evaluation_replica_domain is None
            ):
                raise ValueError(
                    f"legacy adapter {adapter.adapter_id!r} must declare an evaluation "
                    f"replica domain for stochastic endpoint {mapping.endpoint_id!r}"
                )
            if (
                endpoint.measurement_semantics == "deterministic"
                and mapping.standard_error_source_path is not None
            ):
                raise ValueError(
                    f"legacy adapter {adapter.adapter_id!r} cannot map standard error for "
                    f"deterministic endpoint {mapping.endpoint_id!r}"
                )
            if (
                endpoint.measurement_semantics == "deterministic"
                and mapping.evaluation_replica_domain is not None
            ):
                raise ValueError(
                    f"legacy adapter {adapter.adapter_id!r} cannot declare an evaluation "
                    f"replica domain for deterministic endpoint {mapping.endpoint_id!r}"
                )
        mapped_endpoint_ids = {mapping.endpoint_id for mapping in adapter.endpoint_mappings}
        required_endpoint_ids = {
            edition.admission_gate_id,
            *(edition.rating_utility.endpoint_ids if edition.rating_utility is not None else ()),
        }
        missing_required = required_endpoint_ids.difference(mapped_endpoint_ids)
        if missing_required:
            raise ValueError(
                f"legacy adapter {adapter.adapter_id!r} must map admission and rating "
                f"endpoints: {sorted(missing_required)!r}"
            )
        optional_required_endpoint_ids = {
            mapping.endpoint_id
            for mapping in adapter.endpoint_mappings
            if not mapping.required_when_scored and mapping.endpoint_id in required_endpoint_ids
        }
        if optional_required_endpoint_ids:
            raise ValueError(
                f"legacy adapter {adapter.adapter_id!r} cannot make admission or rating "
                f"endpoints optional: {sorted(optional_required_endpoint_ids)!r}"
            )
        for mapping in adapter.endpoint_mappings:
            missing_gate_dependencies = set(
                endpoints[mapping.endpoint_id].required_gate_ids
            ).difference(mapped_endpoint_ids)
            if missing_gate_dependencies:
                raise ValueError(
                    f"legacy adapter {adapter.adapter_id!r} mapping for "
                    f"{mapping.endpoint_id!r} omits required gate mappings: "
                    f"{sorted(missing_gate_dependencies)!r}"
                )
    return revision


class InstanceRef(StrictModel):
    """Public-safe opaque identity for a scientific instance."""

    schema_version: Literal[1] = 1
    instance_id: Identifier | None = None
    source_id: Identifier
    source_kind: Identifier
    source_revision: SourceRevisionCommitment
    commitment_scheme: CommitmentScheme
    safe_commitment: Digest
    public_metadata: dict[Identifier, JsonValue] = Field(default_factory=dict)

    @field_validator("public_metadata")
    @classmethod
    def _safe_public_metadata(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        _validate_public_value(
            value,
            path="public_metadata",
            allow_public_urls=True,
        )
        return value

    @model_validator(mode="after")
    def _bind_instance_id(self) -> Self:
        # Instance identity is edition-independent so finite/fixed catalogs can
        # declare their opaque IDs before the TaskEdition content hash exists.
        payload = {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "source_revision": self.source_revision.model_dump(mode="json"),
            "commitment_scheme": self.commitment_scheme,
            "safe_commitment": self.safe_commitment,
        }
        _assign_or_validate_id(self, "instance_id", semantic_id("instance", payload))
        return self


class InstanceBinding(StrictModel):
    """Private runtime construction binding; never publish this artifact."""

    schema_version: Literal[1] = 1
    instance_binding_id: Identifier | None = None
    task_edition_id: Identifier
    instance_ref: InstanceRef
    binding_mode: BindingMode
    runtime_inputs: dict[Identifier, JsonValue]
    effective_instance_key: Identifier
    instance_cluster_id: Identifier

    @field_validator("runtime_inputs")
    @classmethod
    def _finite_runtime_inputs(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        ensure_finite_json(value, path="runtime_inputs")
        return value

    @model_validator(mode="after")
    def _bind_private_id(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"instance_binding_id"})
        _assign_or_validate_id(
            self,
            "instance_binding_id",
            semantic_id("instance_binding", payload),
        )
        return self


class RuntimeInputRouteSpec(StrictModel):
    """Owner-reviewed private input route for one task-edition binding set."""

    schema_version: Literal[1] = 1
    source_key: Identifier
    qsim_env_key: NonEmptyStr | None = None
    verifier_env_key: NonEmptyStr | None = None
    audit_only: bool = False

    @field_validator("qsim_env_key", "verifier_env_key")
    @classmethod
    def _validate_runtime_env(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if _RUNTIME_ENV_NAME.fullmatch(value) is None:
            raise ValueError("runtime route environment keys must be uppercase env names")
        if value in _RESERVED_RUNTIME_ENV or value.startswith("QIQCBENCH_EVAL_"):
            raise ValueError("runtime route environment key is reserved")
        return value

    @model_validator(mode="after")
    def _validate_destination(self) -> Self:
        if self.audit_only:
            if self.qsim_env_key is not None or self.verifier_env_key is not None:
                raise ValueError("audit-only route cannot declare runtime environment keys")
        elif self.qsim_env_key is None and self.verifier_env_key is None:
            raise ValueError("runtime route must declare at least one destination")
        return self


class ValidatedInstanceBindingSet(StrictModel):
    """Private owner-reviewed/task-validator-produced binding allowlist.

    Binding IDs may commit hidden construction inputs, so this contract belongs
    only in private evaluation state and must never enter public releases.
    It is an auditable allowlist contract, not a cryptographic proof that an
    untrusted caller actually ran the task-owned validator.
    """

    schema_version: Literal[1] = 1
    validated_binding_set_id: Identifier | None = None
    task_edition_id: Identifier
    source_id: Identifier
    source_revision: SourceRevisionCommitment
    validator_revision_digest: Digest
    instance_binding_ids: tuple[Identifier, ...] = Field(min_length=1)
    runtime_input_routes: tuple[RuntimeInputRouteSpec, ...] = ()

    @field_validator("instance_binding_ids")
    @classmethod
    def _normalize_binding_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("validated instance binding IDs must be unique")
        return tuple(sorted(values))

    @field_validator("runtime_input_routes")
    @classmethod
    def _normalize_runtime_routes(
        cls,
        values: tuple[RuntimeInputRouteSpec, ...],
    ) -> tuple[RuntimeInputRouteSpec, ...]:
        source_keys = [item.source_key for item in values]
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("validated runtime routes must have unique source keys")
        qsim_keys = [item.qsim_env_key for item in values if item.qsim_env_key is not None]
        verifier_keys = [
            item.verifier_env_key for item in values if item.verifier_env_key is not None
        ]
        if len(qsim_keys) != len(set(qsim_keys)) or len(verifier_keys) != len(set(verifier_keys)):
            raise ValueError("validated runtime routes cannot reuse one destination env key")
        return tuple(sorted(values, key=lambda item: item.source_key))

    @model_validator(mode="after")
    def _bind_validation_set_id(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"validated_binding_set_id"})
        _assign_or_validate_id(
            self,
            "validated_binding_set_id",
            semantic_id("validated_instance_binding_set", payload),
        )
        return self


def count_effective_instances(bindings: Iterable[InstanceBinding]) -> int:
    """Count task-owned effective instances, not arbitrary seed labels."""

    clusters: dict[tuple[str, str], str] = {}
    for binding in bindings:
        key = (binding.task_edition_id, binding.effective_instance_key)
        previous = clusters.setdefault(key, binding.instance_cluster_id)
        if previous != binding.instance_cluster_id:
            raise ValueError(
                "one effective_instance_key cannot map to multiple instance_cluster_id values"
            )
    return len(clusters)


__all__ = [
    "AggregationRole",
    "BindingMode",
    "CalibrationPoint",
    "CommitmentScheme",
    "Direction",
    "ExecutionProfile",
    "InstanceBinding",
    "InstanceRef",
    "InstanceSourcePolicy",
    "LegacyAdapterKind",
    "LegacyAdapterSpec",
    "LegacyArtifactRole",
    "LegacyEndpointMapping",
    "PlayerManifest",
    "RatedConfigurationManifest",
    "RatingUtilityComponent",
    "ResourceBudget",
    "ResourceEnvelope",
    "RuntimeInputRouteSpec",
    "ScientificBehaviorCommitment",
    "ScientificContractCommitment",
    "SourceRevisionCommitment",
    "AnyTaskEditionManifest",
    "TaskEditionManifest",
    "TaskEditionManifestV2",
    "TaskEndpoint",
    "TaskRatingUtility",
    "VerifierRevisionManifest",
    "count_effective_instances",
    "validate_verifier_revision_for_edition",
    "validate_task_edition_manifest",
    "validate_task_edition_manifest_json",
]
