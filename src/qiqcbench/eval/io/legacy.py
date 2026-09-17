"""Fail-closed projection of legacy reports into evaluation-v2 outcomes.

This adapter is deliberately a host-side compatibility boundary.  It validates
the old wire artifact strictly, follows mappings owned by a verifier revision,
and requires the caller to supply already-planned campaign/execution/
verification context.  It never reads runtime environment variables or imports
task physics.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import Field, JsonValue, ValidationError, field_validator, model_validator

from qiqcbench.eval.contracts.base import (
    Digest,
    Identifier,
    NonEmptyStr,
    StrictModel,
    ensure_finite_json,
)
from qiqcbench.eval.contracts.campaign import AttemptManifest, ExecutionManifest
from qiqcbench.eval.contracts.identity import (
    Direction,
    InstanceBinding,
    LegacyAdapterSpec,
    LegacyEndpointMapping,
    RuntimeInputRouteSpec,
    TaskEditionManifest,
    TaskEndpoint,
    VerifierRevisionManifest,
    validate_task_edition_manifest,
    validate_verifier_revision_for_edition,
)
from qiqcbench.eval.contracts.outcome import (
    DiagnosticObservation,
    EvidenceDigest,
    GateObservation,
    MetricObservation,
    OutcomeDisposition,
    VerificationManifest,
    VerifiedOutcome,
    evaluation_replica_id,
    validate_verified_outcome_bindings,
)

_ARTIFACT_ROLE = "legacy_elo_report"
_GATE_PATH = "gate.value"
_RACE_PATH = "race_metric.value"
_INSTANCE_SEED_ENV = "QIQCBENCH_INSTANCE_SEED"


class LegacyProjectionError(ValueError):
    """A legacy artifact cannot be projected without weakening v2 contracts."""


def derive_elo_v1_expected_labels(
    *,
    verifier_revision: VerifierRevisionManifest,
    instance_binding: InstanceBinding,
    runtime_input_routes: Sequence[RuntimeInputRouteSpec],
) -> dict[str, JsonValue] | None:
    """Derive the Elo-v1 verifier label from one reviewed instance route.

    ``None`` means the revision does not use the legacy EloScoreReport-v1
    adapter.  The compatibility label is a consequence of the frozen binding,
    never a second caller-selected scientific input.
    """

    elo_adapters = tuple(
        adapter
        for adapter in verifier_revision.legacy_adapters
        if adapter.adapter_kind == "elo_score_report_v1"
    )
    if not elo_adapters:
        return None
    if len(elo_adapters) != 1:
        raise ValueError("EloScoreReport v1 requires exactly one revision adapter")

    seed_routes = tuple(
        route for route in runtime_input_routes if route.verifier_env_key == _INSTANCE_SEED_ENV
    )
    if len(seed_routes) != 1:
        raise ValueError(
            "EloScoreReport v1 requires exactly one reviewed verifier route to "
            f"{_INSTANCE_SEED_ENV}"
        )
    source_key = seed_routes[0].source_key
    if source_key not in instance_binding.runtime_inputs:
        raise ValueError("EloScoreReport v1 seed route references a missing binding input")
    seed = instance_binding.runtime_inputs[source_key]
    if type(seed) is not int:
        raise ValueError("EloScoreReport v1 instance_seed must be a strict integer")
    return {"instance_seed": seed}


class LegacyOutcomeContext(StrictModel):
    """Caller-supplied immutable identities and provenance for one projection."""

    attempt: AttemptManifest
    execution: ExecutionManifest
    verification: VerificationManifest
    disposition: OutcomeDisposition
    expected_instance_seed: int
    legacy_report_digest: Digest
    public_material_digests: dict[Identifier, Digest]


class FreeformRubricOutcomeContext(StrictModel):
    """Immutable identities and safe provenance for a declared rubric adapter."""

    attempt: AttemptManifest
    execution: ExecutionManifest
    verification: VerificationManifest
    disposition: OutcomeDisposition
    task_report_digest: Digest
    public_material_digests: dict[Identifier, Digest]


class LegacyDevelopmentEvidence(StrictModel):
    """Binary legacy evidence that deliberately makes no scientific claim."""

    schema_version: Literal[1] = 1
    evidence_kind: Literal["legacy_binary_reward"] = "legacy_binary_reward"
    eligibility: Literal["development_only"] = "development_only"
    provenance_status: Literal["incomplete"] = "incomplete"
    reason_code: Identifier
    reward_binary: Literal[0, 1]
    reward_digest: Digest
    task_report_state: Literal["missing", "unknown_shape", "malformed"]
    task_report_digest: Digest | None = None

    @field_validator("reward_binary", mode="before")
    @classmethod
    def _reward_is_strict_bit(cls, value: Any) -> Any:
        if type(value) is not int or value not in (0, 1):
            raise ValueError("reward_binary must be the integer 0 or 1")
        return value

    @model_validator(mode="after")
    def _report_state_matches_digest(self) -> LegacyDevelopmentEvidence:
        if self.task_report_state == "missing" and self.task_report_digest is not None:
            raise ValueError("missing task report cannot carry a digest")
        if self.task_report_state != "missing" and self.task_report_digest is None:
            raise ValueError("present task report requires a digest")
        return self


class _LegacyGateV1(StrictModel):
    passed: bool
    metric: NonEmptyStr
    value: float = Field(allow_inf_nan=False)
    threshold: float = Field(allow_inf_nan=False)
    direction: Direction


class _LegacyRaceMetricV1(StrictModel):
    name: NonEmptyStr
    value: float = Field(allow_inf_nan=False)
    direction: Direction


class _LegacyEloReportV1(StrictModel):
    schema_version: Literal[1]
    task_id: Identifier
    rating_mode: Literal["elo"]
    reward_binary: Literal[0, 1]
    instance_seed: int
    gate: _LegacyGateV1
    race_metric: _LegacyRaceMetricV1
    extra: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _schema_is_integer_v1(cls, value: Any) -> Any:
        if type(value) is not int or value != 1:
            raise ValueError("schema_version must be the integer 1")
        return value

    @field_validator("reward_binary", mode="before")
    @classmethod
    def _reward_is_an_integer_bit(cls, value: Any) -> Any:
        if type(value) is not int or value not in (0, 1):
            raise ValueError("reward_binary must be the integer 0 or 1")
        return value

    @field_validator("instance_seed", mode="before")
    @classmethod
    def _seed_is_a_strict_integer(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("instance_seed must be an integer")
        return value

    @field_validator("extra")
    @classmethod
    def _extra_is_finite(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        ensure_finite_json(value, path="extra")
        return value


def _strict_report(report: Mapping[str, Any]) -> _LegacyEloReportV1:
    if not isinstance(report, Mapping):
        raise LegacyProjectionError("legacy Elo v1 report must be a mapping")
    try:
        return _LegacyEloReportV1.model_validate(dict(report))
    except (TypeError, ValueError, ValidationError) as exc:
        raise LegacyProjectionError(
            "legacy Elo v1 report failed strict schema validation; mapped standard_error "
            "values must also be finite"
        ) from exc


def _duplicate_checked_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LegacyProjectionError(f"duplicate JSON key in legacy report: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise LegacyProjectionError(f"non-finite JSON constant in legacy report: {value!r}")


def parse_elo_score_report_v1_bytes(payload: bytes) -> dict[str, Any]:
    """Parse exact v1 bytes without duplicate keys, coercion, or non-finite values."""

    if not isinstance(payload, bytes):
        raise TypeError("legacy Elo report payload must be exact bytes")
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LegacyProjectionError("legacy Elo report is not strict UTF-8 JSON") from exc
    return _strict_report(decoded).model_dump(mode="json")


def _parse_json_object_bytes(payload: bytes, *, artifact_name: str) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise TypeError(f"{artifact_name} payload must be exact bytes")
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LegacyProjectionError(f"{artifact_name} is not strict UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise LegacyProjectionError(f"{artifact_name} must be a JSON object")
    try:
        ensure_finite_json(decoded, path=artifact_name)
    except ValueError as exc:
        raise LegacyProjectionError(f"{artifact_name} contains non-finite JSON") from exc
    return decoded


def parse_legacy_binary_reward_bytes(payload: bytes) -> int:
    """Parse historical reward bytes without accepting numeric lookalikes."""

    if not isinstance(payload, bytes):
        raise TypeError("legacy reward payload must be exact bytes")
    try:
        value = payload.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise LegacyProjectionError("legacy reward is not ASCII") from exc
    if value not in {"0", "1"}:
        raise LegacyProjectionError("legacy reward must contain only the integer 0 or 1")
    return int(value)


def retain_unknown_legacy_evidence(
    reward_bytes: bytes,
    *,
    task_report_bytes: bytes | None = None,
) -> LegacyDevelopmentEvidence:
    """Retain an unknown legacy shape without fabricating a v2 scientific outcome."""

    from qiqcbench.eval.io.canonical import sha256_bytes

    reward_binary = parse_legacy_binary_reward_bytes(reward_bytes)
    if task_report_bytes is None:
        state: Literal["missing", "unknown_shape", "malformed"] = "missing"
        report_digest = None
    else:
        report_digest = sha256_bytes(task_report_bytes)
        try:
            _parse_json_object_bytes(task_report_bytes, artifact_name="legacy task report")
        except (LegacyProjectionError, TypeError):
            state = "malformed"
        else:
            state = "unknown_shape"
    return LegacyDevelopmentEvidence(
        reason_code="no_declared_legacy_adapter",
        reward_binary=reward_binary,
        reward_digest=sha256_bytes(reward_bytes),
        task_report_state=state,
        task_report_digest=report_digest,
    )


def _validated_inputs(
    *,
    context: LegacyOutcomeContext,
    task_edition: TaskEditionManifest,
    verifier_revision: VerifierRevisionManifest,
) -> tuple[LegacyOutcomeContext, TaskEditionManifest, VerifierRevisionManifest]:
    """Revalidate trusted Pydantic copies before crossing the adapter boundary."""

    try:
        bound_context = LegacyOutcomeContext.model_validate(context.model_dump(mode="python"))
        edition = validate_task_edition_manifest(task_edition)
        revision = VerifierRevisionManifest.model_validate(
            verifier_revision.model_dump(mode="python")
        )
        validate_verifier_revision_for_edition(edition, revision)
    except (AttributeError, TypeError, ValueError, ValidationError) as exc:
        raise LegacyProjectionError(
            "legacy projection context, TaskEdition, or VerifierRevision is invalid"
        ) from exc
    return bound_context, edition, revision


def _select_adapter(
    revision: VerifierRevisionManifest,
    *,
    adapter_id: str,
) -> LegacyAdapterSpec:
    matches = [adapter for adapter in revision.legacy_adapters if adapter.adapter_id == adapter_id]
    if len(matches) != 1:
        raise LegacyProjectionError(
            f"VerifierRevision does not declare exactly one adapter_id {adapter_id!r}"
        )
    adapter = matches[0]
    if adapter.adapter_kind != "elo_score_report_v1":
        raise LegacyProjectionError(f"adapter_id {adapter_id!r} is not an Elo v1 adapter")
    if adapter.artifact_role != _ARTIFACT_ROLE:
        raise LegacyProjectionError(
            "EloScoreReport v1 adapter artifact_role must be 'legacy_elo_report'"
        )
    return adapter


def _mapped_endpoints(
    adapter: LegacyAdapterSpec,
    edition: TaskEditionManifest,
    report: _LegacyEloReportV1,
) -> tuple[
    TaskEndpoint,
    TaskEndpoint,
    LegacyEndpointMapping,
    LegacyEndpointMapping,
]:
    by_path = {mapping.source_path: mapping for mapping in adapter.endpoint_mappings}
    expected_paths = {_GATE_PATH, _RACE_PATH}
    if set(by_path) != expected_paths or len(adapter.endpoint_mappings) != len(expected_paths):
        raise LegacyProjectionError(
            "EloScoreReport v1 adapter source paths must be exactly gate.value and "
            "race_metric.value"
        )

    gate_mapping = by_path[_GATE_PATH]
    race_mapping = by_path[_RACE_PATH]
    if gate_mapping.source_metric_name != report.gate.metric:
        raise LegacyProjectionError("legacy gate metric does not match its adapter mapping")
    if race_mapping.source_metric_name != report.race_metric.name:
        raise LegacyProjectionError("legacy race metric does not match its adapter mapping")

    endpoints = {endpoint.endpoint_id: endpoint for endpoint in edition.endpoints}
    gate_endpoint = endpoints.get(gate_mapping.endpoint_id)
    race_endpoint = endpoints.get(race_mapping.endpoint_id)
    if gate_endpoint is None or gate_endpoint.metric_role != "gate":
        raise LegacyProjectionError("legacy gate mapping must reference a gate endpoint")
    if gate_endpoint.endpoint_id != edition.admission_gate_id:
        raise LegacyProjectionError("legacy gate mapping must reference the admission gate")
    if race_endpoint is None or race_endpoint.metric_role == "gate":
        raise LegacyProjectionError("legacy race mapping must reference a metric endpoint")
    if race_endpoint.aggregation_role != "racing":
        raise LegacyProjectionError("legacy race mapping must reference a racing endpoint")
    if gate_endpoint.endpoint_id == race_endpoint.endpoint_id:
        raise LegacyProjectionError("legacy gate and race mappings must target distinct endpoints")
    return gate_endpoint, race_endpoint, gate_mapping, race_mapping


def _validate_report_semantics(
    report: _LegacyEloReportV1,
    *,
    context: LegacyOutcomeContext,
    edition: TaskEditionManifest,
    gate_endpoint: TaskEndpoint,
    race_endpoint: TaskEndpoint,
) -> None:
    if report.task_id != edition.task_id:
        raise LegacyProjectionError("legacy report task_id does not match TaskEdition")
    if report.instance_seed != context.expected_instance_seed:
        raise LegacyProjectionError("legacy report instance_seed does not match bound context")
    if report.gate.direction != gate_endpoint.direction:
        raise LegacyProjectionError("legacy gate direction does not match TaskEdition")
    if report.race_metric.direction != race_endpoint.direction:
        raise LegacyProjectionError("legacy race direction does not match TaskEdition")
    if report.reward_binary != int(report.gate.passed):
        raise LegacyProjectionError("legacy reward_binary does not match admission gate")
    if context.disposition == "model_failure" and report.gate.passed:
        raise LegacyProjectionError("model_failure legacy report cannot pass the admission gate")

    if gate_endpoint.gate_rule in {"threshold", "composite"} and (
        gate_endpoint.gate_threshold is not None or gate_endpoint.gate_threshold_source is not None
    ):
        if (
            gate_endpoint.gate_threshold is not None
            and report.gate.threshold != gate_endpoint.gate_threshold
        ):
            raise LegacyProjectionError("legacy gate threshold does not match TaskEdition")
        expected_pass = (
            report.gate.value >= report.gate.threshold
            if gate_endpoint.direction == "higher_is_better"
            else report.gate.value <= report.gate.threshold
        )
        if gate_endpoint.gate_rule == "threshold" and report.gate.passed != expected_pass:
            raise LegacyProjectionError("legacy gate pass decision contradicts its threshold")
        if gate_endpoint.gate_rule == "composite" and report.gate.passed and not expected_pass:
            raise LegacyProjectionError(
                "legacy composite gate cannot pass when its declared threshold component fails"
            )

    if (
        report.gate.metric == report.race_metric.name
        and report.gate.value != report.race_metric.value
    ):
        raise LegacyProjectionError("legacy gate and race values disagree for the same metric")


def _mapped_standard_error(
    report: _LegacyEloReportV1,
    *,
    mapping: LegacyEndpointMapping,
    endpoint: TaskEndpoint,
) -> float | None:
    source_path = mapping.standard_error_source_path
    if endpoint.measurement_semantics == "deterministic":
        if source_path is not None:
            raise LegacyProjectionError(
                f"deterministic endpoint {endpoint.endpoint_id!r} cannot map standard_error"
            )
        return None
    if source_path is None:
        raise LegacyProjectionError(
            f"stochastic endpoint {endpoint.endpoint_id!r} requires a standard_error mapping"
        )

    parts = source_path.split(".")
    if len(parts) < 2 or parts[0] != "extra" or any(not part for part in parts):
        raise LegacyProjectionError(
            "legacy standard_error source path must select a field below extra"
        )
    value: Any = report.extra
    for part in parts[1:]:
        if not isinstance(value, Mapping) or part not in value:
            raise LegacyProjectionError(
                f"legacy report is missing mapped standard_error field {source_path!r}"
            )
        value = value[part]
    if type(value) not in (int, float):
        raise LegacyProjectionError(
            f"mapped standard_error field {source_path!r} must be a strict number"
        )
    standard_error = float(value)
    if not math.isfinite(standard_error) or standard_error < 0.0:
        raise LegacyProjectionError(
            f"mapped standard_error field {source_path!r} must be finite and non-negative"
        )
    return standard_error


def _validate_bound_provenance(
    *,
    context: LegacyOutcomeContext,
    edition: TaskEditionManifest,
    adapter: LegacyAdapterSpec,
) -> None:
    evidence = {item.role: item.digest for item in context.verification.evidence_inputs}
    if adapter.artifact_role in evidence:
        raise LegacyProjectionError(
            "legacy report is a verification output and cannot be a VerificationManifest input"
        )
    expected_public_materials = {
        "instruction": edition.instruction_digest,
        **edition.public_material_digests,
    }
    if context.public_material_digests != expected_public_materials:
        raise LegacyProjectionError(
            "caller public material digests do not exactly match TaskEdition"
        )


def _provenance_diagnostic(
    *,
    adapter: LegacyAdapterSpec,
    context: LegacyOutcomeContext,
    report: _LegacyEloReportV1,
    mappings: tuple[LegacyEndpointMapping, LegacyEndpointMapping],
) -> DiagnosticObservation:
    return DiagnosticObservation(
        code="legacy_elo_v1_projection",
        severity="warning",
        message=(
            "Outcome projected from a legacy EloScoreReport v1 artifact; it is not a "
            "native evaluation-v2 verifier outcome."
        ),
        details={
            "adapter_id": adapter.adapter_id,
            "artifact_role": adapter.artifact_role,
            "legacy_schema_version": report.schema_version,
            "legacy_report_digest": context.legacy_report_digest,
            "legacy_instance_seed_validated": True,
            "legacy_extra_present": bool(report.extra),
            "standard_error_source_paths": sorted(
                {
                    mapping.standard_error_source_path
                    for mapping in mappings
                    if mapping.standard_error_source_path is not None
                }
            ),
        },
    )


def project_elo_score_report_v1(
    report: Mapping[str, Any],
    *,
    context: LegacyOutcomeContext,
    task_edition: TaskEditionManifest,
    verifier_revision: VerifierRevisionManifest,
    adapter_id: str,
) -> VerifiedOutcome:
    """Project one strict legacy Elo report through an explicit revision adapter.

    The caller must supply real planned manifests, the private expected legacy
    seed label, the exact legacy-report digest, and the complete public-material
    binding.  The function derives no campaign, player, instance, execution, or
    verification identity and reads no environment state.
    """

    parsed = _strict_report(report)
    bound_context, edition, revision = _validated_inputs(
        context=context,
        task_edition=task_edition,
        verifier_revision=verifier_revision,
    )
    adapter = _select_adapter(revision, adapter_id=adapter_id)
    gate_endpoint, race_endpoint, gate_mapping, race_mapping = _mapped_endpoints(
        adapter,
        edition,
        parsed,
    )
    _validate_report_semantics(
        parsed,
        context=bound_context,
        edition=edition,
        gate_endpoint=gate_endpoint,
        race_endpoint=race_endpoint,
    )
    if bound_context.disposition == "model_failure":
        gate_standard_error = None
        race_standard_error = None
    else:
        gate_standard_error = _mapped_standard_error(
            parsed,
            mapping=gate_mapping,
            endpoint=gate_endpoint,
        )
        race_standard_error = _mapped_standard_error(
            parsed,
            mapping=race_mapping,
            endpoint=race_endpoint,
        )
    _validate_bound_provenance(
        context=bound_context,
        edition=edition,
        adapter=adapter,
    )

    if bound_context.disposition == "model_failure":
        gate = GateObservation(
            gate_id=gate_endpoint.endpoint_id,
            passed=False,
        )
    else:
        gate_replica_id = None
        if gate_endpoint.measurement_semantics == "verifier_standard_error":
            if gate_mapping.evaluation_replica_domain is None:
                raise LegacyProjectionError(
                    "stochastic legacy gate requires an evaluation replica domain"
                )
            gate_replica_id = evaluation_replica_id(
                execution_id=bound_context.execution.execution_id,
                replay_domain=gate_mapping.evaluation_replica_domain,
                replay_secret_commitment=(
                    bound_context.execution.verifier_replay_secret_commitment
                ),
            )
        gate = GateObservation(
            gate_id=gate_endpoint.endpoint_id,
            passed=parsed.gate.passed,
            observed_value=parsed.gate.value,
            standard_error=gate_standard_error,
            evaluation_replica_id=gate_replica_id,
            threshold_value=parsed.gate.threshold,
            threshold_source=gate_endpoint.gate_threshold_source,
        )
    metrics: tuple[MetricObservation, ...]
    diagnostics = [
        _provenance_diagnostic(
            adapter=adapter,
            context=bound_context,
            report=parsed,
            mappings=(gate_mapping, race_mapping),
        )
    ]
    if bound_context.disposition == "model_failure":
        metrics = ()
        diagnostics.append(
            DiagnosticObservation(
                code="legacy_metric_omitted",
                severity="info",
                message=(
                    "Legacy race sentinel was not projected because the caller bound this "
                    "attempt as a model failure."
                ),
                details={"metric_id": race_endpoint.endpoint_id},
            )
        )
    else:
        replica_id = None
        if race_endpoint.measurement_semantics == "verifier_standard_error":
            if race_mapping.evaluation_replica_domain is None:
                raise LegacyProjectionError(
                    "stochastic legacy projection requires an evaluation replica domain"
                )
            replica_id = evaluation_replica_id(
                execution_id=bound_context.execution.execution_id,
                replay_domain=race_mapping.evaluation_replica_domain,
                replay_secret_commitment=(
                    bound_context.execution.verifier_replay_secret_commitment
                ),
            )
        metrics = (
            MetricObservation(
                metric_id=race_endpoint.endpoint_id,
                value=parsed.race_metric.value,
                measurement_semantics=race_endpoint.measurement_semantics,
                standard_error=race_standard_error,
                evaluation_replica_id=replica_id,
            ),
        )

    attempt = bound_context.attempt
    execution = bound_context.execution
    verification = bound_context.verification
    outcome = VerifiedOutcome(
        campaign_id=attempt.campaign_id,
        attempt_id=attempt.attempt_id,
        player_id=attempt.player_id,
        task_edition_id=attempt.task_edition_id,
        task_family_id=attempt.task_family_id,
        instance_id=attempt.instance_id,
        instance_cluster_id=attempt.instance_cluster_id,
        comparison_block_id=attempt.comparison_block_id,
        execution_id=execution.execution_id,
        verification_id=verification.verification_id,
        verifier_revision_id=verification.verifier_revision_id,
        disposition=bound_context.disposition,
        gates=(gate,),
        metrics=metrics,
        evidence_digests=tuple(
            EvidenceDigest(role=item.role, digest=item.digest)
            for item in verification.evidence_inputs
        ),
        report_digests={adapter.artifact_role: bound_context.legacy_report_digest},
        public_material_digests=bound_context.public_material_digests,
        verifier_contract_commitment=edition.verifier_contract_commitment,
        diagnostics=tuple(diagnostics),
    )
    try:
        return validate_verified_outcome_bindings(
            outcome,
            verification=verification,
            execution=execution,
            attempt=attempt,
            task_edition=edition,
            verifier_revision=revision,
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise LegacyProjectionError(
            "projected legacy outcome violates bound evaluation-v2 contracts"
        ) from exc


def _validated_rubric_inputs(
    *,
    context: FreeformRubricOutcomeContext,
    task_edition: TaskEditionManifest,
    verifier_revision: VerifierRevisionManifest,
) -> tuple[
    FreeformRubricOutcomeContext,
    TaskEditionManifest,
    VerifierRevisionManifest,
]:
    try:
        bound_context = FreeformRubricOutcomeContext.model_validate(
            context.model_dump(mode="python")
        )
        edition = validate_task_edition_manifest(task_edition)
        revision = VerifierRevisionManifest.model_validate(
            verifier_revision.model_dump(mode="python")
        )
        validate_verifier_revision_for_edition(edition, revision)
    except (AttributeError, TypeError, ValueError, ValidationError) as exc:
        raise LegacyProjectionError(
            "rubric projection context, TaskEdition, or VerifierRevision is invalid"
        ) from exc
    return bound_context, edition, revision


def _select_rubric_adapter(
    revision: VerifierRevisionManifest,
    *,
    adapter_id: str,
) -> LegacyAdapterSpec:
    matches = [adapter for adapter in revision.legacy_adapters if adapter.adapter_id == adapter_id]
    if len(matches) != 1:
        raise LegacyProjectionError(
            f"VerifierRevision does not declare exactly one adapter_id {adapter_id!r}"
        )
    adapter = matches[0]
    if adapter.adapter_kind != "freeform_rubric_json_v1":
        raise LegacyProjectionError("adapter is not a free-form rubric JSON adapter")
    if adapter.artifact_role != "task_score_report":
        raise LegacyProjectionError("rubric adapter must consume task_score_report")
    return adapter


def _path_value(report: Mapping[str, Any], path: str | None, *, field: str) -> Any:
    if path is None:
        raise LegacyProjectionError(f"{field} has no declared source path")
    parts = path.split(".")
    if not parts or any(not part for part in parts):
        raise LegacyProjectionError(f"{field} source path must use dotted object keys")
    value: Any = report
    for part in parts:
        if not isinstance(value, Mapping) or part not in value:
            raise LegacyProjectionError(f"rubric report is missing mapped field {path!r}")
        value = value[part]
    return value


def _path_is_absent(report: Mapping[str, Any], path: str) -> bool:
    """Return true only for a genuinely absent key, not a malformed container."""

    value: Any = report
    for part in path.split("."):
        if not isinstance(value, Mapping):
            return False
        if part not in value:
            return True
        value = value[part]
    return False


def _strict_number(report: Mapping[str, Any], path: str | None, *, field: str) -> float:
    value = _path_value(report, path, field=field)
    if type(value) not in (int, float):
        raise LegacyProjectionError(f"mapped field {path!r} must be a strict number")
    number = float(value)
    if not math.isfinite(number):
        raise LegacyProjectionError(f"mapped field {path!r} must be finite")
    return number


def _strict_decision(report: Mapping[str, Any], path: str | None, *, field: str) -> bool:
    value = _path_value(report, path, field=field)
    if type(value) is bool:
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    raise LegacyProjectionError(f"mapped field {path!r} must be a strict boolean or integer bit")


def _rubric_standard_error(
    report: Mapping[str, Any],
    *,
    mapping: LegacyEndpointMapping,
    endpoint: TaskEndpoint,
) -> float | None:
    if endpoint.measurement_semantics == "deterministic":
        return None
    standard_error = _strict_number(
        report,
        mapping.standard_error_source_path,
        field=f"standard error for {endpoint.endpoint_id}",
    )
    if standard_error < 0.0:
        raise LegacyProjectionError(
            f"mapped standard error for {endpoint.endpoint_id!r} must be non-negative"
        )
    return standard_error


def _threshold_pass(endpoint: TaskEndpoint, observed: float, threshold: float) -> bool:
    if endpoint.direction == "higher_is_better":
        return observed >= threshold
    return observed <= threshold


def _rubric_gate(
    report: Mapping[str, Any],
    *,
    mapping: LegacyEndpointMapping,
    endpoint: TaskEndpoint,
    execution_id: str,
    replay_secret_commitment: str,
) -> GateObservation:
    observed = (
        _strict_number(report, mapping.source_path, field=f"gate {endpoint.endpoint_id}")
        if mapping.source_path is not None
        else None
    )
    standard_error = _rubric_standard_error(
        report,
        mapping=mapping,
        endpoint=endpoint,
    )
    if endpoint.gate_threshold is not None:
        threshold = endpoint.gate_threshold
    elif mapping.threshold_source_path is not None:
        threshold = _strict_number(
            report,
            mapping.threshold_source_path,
            field=f"threshold for {endpoint.endpoint_id}",
        )
    else:
        threshold = None

    declared_pass = (
        _strict_decision(
            report,
            mapping.gate_pass_source_path,
            field=f"gate decision for {endpoint.endpoint_id}",
        )
        if mapping.gate_pass_source_path is not None
        else None
    )
    if endpoint.gate_rule == "threshold":
        if observed is None or threshold is None:
            raise LegacyProjectionError(
                f"threshold gate {endpoint.endpoint_id!r} lacks observed value or threshold"
            )
        derived_pass = _threshold_pass(endpoint, observed, threshold)
        if declared_pass is not None and declared_pass != derived_pass:
            raise LegacyProjectionError(
                f"rubric gate decision contradicts threshold for {endpoint.endpoint_id!r}"
            )
        passed = derived_pass
    else:
        if declared_pass is None:
            raise LegacyProjectionError(
                f"{endpoint.gate_rule} gate {endpoint.endpoint_id!r} lacks a pass decision"
            )
        if (
            endpoint.gate_rule == "composite"
            and observed is not None
            and threshold is not None
            and declared_pass
            and not _threshold_pass(endpoint, observed, threshold)
        ):
            raise LegacyProjectionError(
                f"composite gate {endpoint.endpoint_id!r} cannot pass when its declared "
                "threshold component fails"
            )
        passed = declared_pass
    replica_id = None
    if endpoint.measurement_semantics == "verifier_standard_error":
        if mapping.evaluation_replica_domain is None:
            raise LegacyProjectionError(
                "stochastic rubric gate requires an evaluation replica domain"
            )
        replica_id = evaluation_replica_id(
            execution_id=execution_id,
            replay_domain=mapping.evaluation_replica_domain,
            replay_secret_commitment=replay_secret_commitment,
        )
    return GateObservation(
        gate_id=endpoint.endpoint_id,
        passed=passed,
        observed_value=observed,
        standard_error=standard_error,
        evaluation_replica_id=replica_id,
        threshold_value=threshold,
        threshold_source=endpoint.gate_threshold_source,
    )


def project_freeform_rubric_report_bytes(
    report_bytes: bytes,
    reward_bytes: bytes,
    *,
    context: FreeformRubricOutcomeContext,
    task_edition: TaskEditionManifest,
    verifier_revision: VerifierRevisionManifest,
    adapter_id: str,
) -> VerifiedOutcome:
    """Project one explicitly mapped free-form rubric report without task logic."""

    report = _parse_json_object_bytes(report_bytes, artifact_name="rubric task report")
    reward_binary = parse_legacy_binary_reward_bytes(reward_bytes)
    bound_context, edition, revision = _validated_rubric_inputs(
        context=context,
        task_edition=task_edition,
        verifier_revision=verifier_revision,
    )
    adapter = _select_rubric_adapter(revision, adapter_id=adapter_id)
    if adapter.report_schema_version is None:
        if "schema_version" in report:
            raise LegacyProjectionError(
                "unversioned rubric adapter cannot consume a report claiming schema_version"
            )
    else:
        schema_version = report.get("schema_version")
        if type(schema_version) is not int or schema_version != adapter.report_schema_version:
            raise LegacyProjectionError("rubric report schema_version mismatch")
    task_id = _path_value(
        report,
        adapter.task_id_source_path,
        field="rubric task_id",
    )
    if type(task_id) is not str or task_id != edition.task_id:
        raise LegacyProjectionError("rubric report task_id does not match TaskEdition")
    if adapter.reward_source_path is not None:
        report_reward = _strict_decision(
            report,
            adapter.reward_source_path,
            field="rubric reward",
        )
        if int(report_reward) != reward_binary:
            raise LegacyProjectionError("rubric report reward disagrees with reward projection")

    evidence_roles = {item.role for item in bound_context.verification.evidence_inputs}
    if adapter.artifact_role in evidence_roles:
        raise LegacyProjectionError(
            "rubric report is a verification output and cannot be a VerificationManifest input"
        )
    expected_public_materials = {
        "instruction": edition.instruction_digest,
        **edition.public_material_digests,
    }
    if bound_context.public_material_digests != expected_public_materials:
        raise LegacyProjectionError(
            "caller public material digests do not exactly match TaskEdition"
        )

    endpoint_by_id = {endpoint.endpoint_id: endpoint for endpoint in edition.endpoints}
    mapping_by_id = {mapping.endpoint_id: mapping for mapping in adapter.endpoint_mappings}
    diagnostics = [
        DiagnosticObservation(
            code="legacy_rubric_projection",
            severity="warning",
            message=(
                "Outcome projected through an explicit legacy rubric adapter; only declared "
                "scalar fields were retained."
            ),
            details={
                "adapter_id": adapter.adapter_id,
                "artifact_role": adapter.artifact_role,
                "task_report_digest": bound_context.task_report_digest,
                "mapped_endpoint_ids": sorted(mapping_by_id),
            },
        )
    ]
    if bound_context.disposition == "model_failure":
        if reward_binary != 0:
            raise LegacyProjectionError("model_failure rubric report cannot project reward 1")
        gates = (GateObservation(gate_id=edition.admission_gate_id, passed=False),)
        metrics: tuple[MetricObservation, ...] = ()
        diagnostics.append(
            DiagnosticObservation(
                code="legacy_metric_omitted",
                severity="info",
                message="Legacy rubric metrics were omitted for a model failure.",
                details={"mapped_metric_count": 0},
            )
        )
    else:
        gate_items: list[GateObservation] = []
        metric_items: list[MetricObservation] = []
        for mapping in adapter.endpoint_mappings:
            endpoint = endpoint_by_id[mapping.endpoint_id]
            if endpoint.metric_role == "gate":
                gate_items.append(
                    _rubric_gate(
                        report,
                        mapping=mapping,
                        endpoint=endpoint,
                        execution_id=bound_context.execution.execution_id,
                        replay_secret_commitment=(
                            bound_context.execution.verifier_replay_secret_commitment
                        ),
                    )
                )
                continue
            assert mapping.source_path is not None
            if not mapping.required_when_scored and _path_is_absent(report, mapping.source_path):
                continue
            standard_error = _rubric_standard_error(
                report,
                mapping=mapping,
                endpoint=endpoint,
            )
            replica_id = None
            if endpoint.measurement_semantics == "verifier_standard_error":
                if mapping.evaluation_replica_domain is None:
                    raise LegacyProjectionError(
                        "stochastic rubric projection requires an evaluation replica domain"
                    )
                replica_id = evaluation_replica_id(
                    execution_id=bound_context.execution.execution_id,
                    replay_domain=mapping.evaluation_replica_domain,
                    replay_secret_commitment=(
                        bound_context.execution.verifier_replay_secret_commitment
                    ),
                )
            metric_items.append(
                MetricObservation(
                    metric_id=endpoint.endpoint_id,
                    value=_strict_number(
                        report,
                        mapping.source_path,
                        field=f"metric {endpoint.endpoint_id}",
                    ),
                    measurement_semantics=endpoint.measurement_semantics,
                    standard_error=standard_error,
                    evaluation_replica_id=replica_id,
                )
            )
        gates = tuple(gate_items)
        metrics = tuple(metric_items)
        admission = next(
            (gate for gate in gates if gate.gate_id == edition.admission_gate_id),
            None,
        )
        if admission is None:
            raise LegacyProjectionError("rubric adapter did not project the admission gate")
        if int(admission.passed) != reward_binary:
            raise LegacyProjectionError("rubric admission gate disagrees with reward projection")

    attempt = bound_context.attempt
    execution = bound_context.execution
    verification = bound_context.verification
    outcome = VerifiedOutcome(
        campaign_id=attempt.campaign_id,
        attempt_id=attempt.attempt_id,
        player_id=attempt.player_id,
        task_edition_id=attempt.task_edition_id,
        task_family_id=attempt.task_family_id,
        instance_id=attempt.instance_id,
        instance_cluster_id=attempt.instance_cluster_id,
        comparison_block_id=attempt.comparison_block_id,
        execution_id=execution.execution_id,
        verification_id=verification.verification_id,
        verifier_revision_id=verification.verifier_revision_id,
        disposition=bound_context.disposition,
        gates=gates,
        metrics=metrics,
        evidence_digests=tuple(
            EvidenceDigest(role=item.role, digest=item.digest)
            for item in verification.evidence_inputs
        ),
        report_digests={adapter.artifact_role: bound_context.task_report_digest},
        public_material_digests=bound_context.public_material_digests,
        verifier_contract_commitment=edition.verifier_contract_commitment,
        diagnostics=tuple(diagnostics),
    )
    try:
        return validate_verified_outcome_bindings(
            outcome,
            verification=verification,
            execution=execution,
            attempt=attempt,
            task_edition=edition,
            verifier_revision=revision,
        )
    except (TypeError, ValueError, ValidationError) as exc:
        raise LegacyProjectionError(
            "projected rubric outcome violates bound evaluation-v2 contracts"
        ) from exc


__all__ = [
    "FreeformRubricOutcomeContext",
    "LegacyDevelopmentEvidence",
    "LegacyOutcomeContext",
    "LegacyProjectionError",
    "parse_elo_score_report_v1_bytes",
    "parse_legacy_binary_reward_bytes",
    "project_elo_score_report_v1",
    "project_freeform_rubric_report_bytes",
    "retain_unknown_legacy_evidence",
]
