"""Verifier manifests, artifact bindings, and canonical scientific outcomes."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import (
    Field,
    JsonValue,
    ValidationInfo,
    field_validator,
    model_validator,
)

from .base import (
    Digest,
    Identifier,
    NonEmptyStr,
    StrictModel,
    ensure_finite_json,
    semantic_id,
)
from .campaign import AttemptManifest, ExecutionManifest
from .identity import (
    AnyTaskEditionManifest,
    ScientificContractCommitment,
    TaskRatingUtility,
    VerifierRevisionManifest,
    validate_verifier_revision_for_edition,
)

VerificationReason = Literal["initial", "rescore"]
OutcomeDisposition = Literal["scored", "model_failure"]
MeasurementSemantics = Literal["deterministic", "verifier_standard_error"]
EvidenceProvenance = Literal["execution_evidence"]
ReportArtifactRole = Literal["task_score_report", "legacy_elo_report"]
ArtifactSetScope = Literal["private_canonical", "public_compatibility"]
ArtifactRole = Literal[
    "verified_outcome",
    "reward_projection",
    "task_score_report",
    "legacy_elo_report",
    "verification_audit",
]


class HostModelFailureSignal(StrictModel):
    """Exact Harbor-owned model failure without a fabricated verifier outcome."""

    schema_version: Literal[1] = 1
    campaign_id: Identifier
    attempt_id: Identifier
    execution_id: Identifier
    source_role: Literal["trial_result"] = "trial_result"
    source_artifact_digest: Digest
    reason_code: Literal[
        "agent_timeout",
        "no_final_answer",
        "malformed_final_answer",
        "budget_exhausted",
        "missing_required_evidence",
        "model_failure_other",
    ]


def evaluation_replica_id(
    *,
    execution_id: str,
    replay_domain: str,
    replay_secret_commitment: str,
) -> str:
    """Identify one execution-bound verifier replay stream."""

    return semantic_id(
        "evaluation_replica",
        {
            "schema_version": 1,
            "execution_id": execution_id,
            "replay_domain": replay_domain,
            "replay_secret_commitment": replay_secret_commitment,
        },
    )


class EvidenceInput(StrictModel):
    """One exact agent/qsim execution artifact selected for verification.

    Hidden construction, scorer configuration, and exact private bindings are
    deliberately not representable here: they belong in ``InstanceBinding`` or
    verifier-private audit state and must not influence a public verification
    identity through an enumerable digest.
    """

    role: Identifier
    digest: Digest
    reference: NonEmptyStr
    provenance: EvidenceProvenance = "execution_evidence"

    @field_validator("reference")
    @classmethod
    def _reference_is_safe(cls, value: str) -> str:
        return _validate_artifact_reference(value)


def verification_manifest_id(
    *,
    execution_id: str,
    task_edition_id: str,
    verifier_revision_id: str,
    reason: VerificationReason,
    predecessor_verification_id: str | None,
    evidence_inputs: tuple[EvidenceInput, ...],
) -> str:
    """Derive a verification identity from scoring inputs, never outputs."""

    evidence_identity = [
        {
            "role": item.role,
            "digest": item.digest,
            "provenance": item.provenance,
        }
        for item in sorted(evidence_inputs, key=lambda item: (item.role, item.digest))
    ]
    return semantic_id(
        "verification",
        {
            "schema_version": 1,
            "execution_id": execution_id,
            "task_edition_id": task_edition_id,
            "verifier_revision_id": verifier_revision_id,
            "reason": reason,
            "predecessor_verification_id": predecessor_verification_id,
            "evidence_inputs": evidence_identity,
        },
    )


class VerificationManifest(StrictModel):
    """One immutable scoring pass over one execution's preserved evidence."""

    schema_version: Literal[1] = 1
    verification_id: Identifier
    execution_id: Identifier
    task_edition_id: Identifier
    verifier_revision_id: Identifier
    reason: VerificationReason
    predecessor_verification_id: Identifier | None = None
    evidence_inputs: tuple[EvidenceInput, ...] = Field(min_length=1)
    annotations: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("evidence_inputs")
    @classmethod
    def _normalize_evidence_inputs(
        cls, values: tuple[EvidenceInput, ...]
    ) -> tuple[EvidenceInput, ...]:
        roles = [value.role for value in values]
        if len(roles) != len(set(roles)):
            raise ValueError("duplicate evidence input role")
        return tuple(sorted(values, key=lambda value: (value.role, value.digest)))

    @field_validator("annotations")
    @classmethod
    def _annotations_are_finite(cls, values: dict[str, JsonValue]) -> dict[str, JsonValue]:
        ensure_finite_json(values, path="annotations")
        return values

    @model_validator(mode="after")
    def _lineage_and_identity_are_consistent(self) -> VerificationManifest:
        if self.reason == "initial" and self.predecessor_verification_id is not None:
            raise ValueError("initial verification cannot have a predecessor_verification_id")
        if self.reason == "rescore" and self.predecessor_verification_id is None:
            raise ValueError("rescore requires predecessor_verification_id")
        if self.predecessor_verification_id == self.verification_id:
            raise ValueError("verification cannot be its own predecessor")
        expected = verification_manifest_id(
            execution_id=self.execution_id,
            task_edition_id=self.task_edition_id,
            verifier_revision_id=self.verifier_revision_id,
            reason=self.reason,
            predecessor_verification_id=self.predecessor_verification_id,
            evidence_inputs=self.evidence_inputs,
        )
        if self.verification_id != expected:
            raise ValueError(
                "verification_id must be the content-derived ID of its execution, "
                "edition, verifier revision, lineage, and evidence inputs"
            )
        return self

    def assert_rescore_predecessor(self, predecessor: VerificationManifest) -> None:
        """Validate evidence-preserving rescore lineage against its predecessor."""

        if self.reason != "rescore":
            raise ValueError("only a rescore verification has a predecessor")
        if self.predecessor_verification_id != predecessor.verification_id:
            raise ValueError("rescore predecessor_verification_id mismatch")
        if self.execution_id != predecessor.execution_id:
            raise ValueError("rescore must preserve execution_id")
        if self.task_edition_id != predecessor.task_edition_id:
            raise ValueError("rescore must preserve task_edition_id")
        current_evidence = {
            evidence.role: (evidence.digest, evidence.provenance)
            for evidence in self.evidence_inputs
        }
        predecessor_evidence = {
            evidence.role: (evidence.digest, evidence.provenance)
            for evidence in predecessor.evidence_inputs
        }
        if current_evidence != predecessor_evidence:
            raise ValueError("rescore must preserve evidence roles and digests")


def _validate_artifact_reference(value: str) -> str:
    if value.startswith(("/", "~")):
        raise ValueError("artifact reference must be a safe relative reference")
    if "\\" in value:
        raise ValueError("artifact reference must use a relative POSIX path")
    if "://" in value:
        raise ValueError("artifact reference cannot be an external URL")
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ValueError("artifact reference must be a safe relative reference")
    if ".." in path.parts:
        raise ValueError("artifact reference cannot contain path traversal")
    if path in {PurePosixPath("."), PurePosixPath("..")}:
        raise ValueError("artifact reference must identify a file")
    return value


class ArtifactBinding(StrictModel):
    """Role and digest of one immutable verification output."""

    role: ArtifactRole
    reference: NonEmptyStr
    digest: Digest

    @field_validator("reference")
    @classmethod
    def _reference_is_safe(cls, value: str) -> str:
        return _validate_artifact_reference(value)


class VerificationArtifactSet(StrictModel):
    """One typed, immutable output set keyed by a VerificationManifest."""

    schema_version: Literal[1] = 1
    scope: ArtifactSetScope = "private_canonical"
    verification_id: Identifier
    artifacts: tuple[ArtifactBinding, ...] = Field(min_length=1)

    @field_validator("artifacts")
    @classmethod
    def _roles_are_complete_and_unique(
        cls, values: tuple[ArtifactBinding, ...], info: ValidationInfo
    ) -> tuple[ArtifactBinding, ...]:
        roles = [value.role for value in values]
        if len(roles) != len(set(roles)):
            raise ValueError("duplicate artifact role")
        reference_digests: dict[str, str] = {}
        for artifact in values:
            previous = reference_digests.setdefault(artifact.reference, artifact.digest)
            if previous != artifact.digest:
                raise ValueError("one artifact reference cannot bind multiple digests")
        required = {"verified_outcome", "reward_projection"}
        if info.data.get("scope", "private_canonical") == "private_canonical":
            required.add("verification_audit")
        missing = required.difference(roles)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"missing required artifact roles: {names}")
        if (
            info.data.get("scope", "private_canonical") == "public_compatibility"
            and "verification_audit" in roles
        ):
            raise ValueError("public compatibility artifact set cannot expose private audit")
        return tuple(sorted(values, key=lambda value: value.role))


class GateObservation(StrictModel):
    """One authoritative verifier gate decision."""

    gate_id: Identifier
    passed: bool
    observed_value: float | None = Field(default=None, allow_inf_nan=False)
    standard_error: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    evaluation_replica_id: Identifier | None = None
    threshold_value: float | None = Field(default=None, allow_inf_nan=False)
    threshold_source: NonEmptyStr | None = None
    evidence_digest: Digest | None = None

    @model_validator(mode="after")
    def _uncertainty_requires_observed_value(self) -> GateObservation:
        if self.standard_error is not None and self.observed_value is None:
            raise ValueError("gate standard_error requires observed_value")
        if self.standard_error is not None and self.evaluation_replica_id is None:
            raise ValueError("gate standard_error requires evaluation_replica_id")
        if self.standard_error is None and self.evaluation_replica_id is not None:
            raise ValueError("gate evaluation_replica_id requires standard_error")
        return self


class MetricObservation(StrictModel):
    """One task-edition metric observation with explicit error semantics."""

    metric_id: Identifier
    value: float
    measurement_semantics: MeasurementSemantics
    standard_error: float | None = Field(default=None, ge=0.0)
    evaluation_replica_id: Identifier | None = None

    @model_validator(mode="after")
    def _uncertainty_matches_semantics(self) -> MetricObservation:
        if self.measurement_semantics == "deterministic" and self.standard_error is not None:
            raise ValueError("deterministic metric cannot carry standard_error")
        if self.measurement_semantics == "deterministic" and self.evaluation_replica_id is not None:
            raise ValueError("deterministic metric cannot carry evaluation_replica_id")
        if self.measurement_semantics == "verifier_standard_error" and self.standard_error is None:
            raise ValueError("verifier_standard_error metric requires standard_error")
        if (
            self.measurement_semantics == "verifier_standard_error"
            and self.evaluation_replica_id is None
        ):
            raise ValueError("verifier_standard_error metric requires evaluation_replica_id")
        return self


class PenaltyObservation(StrictModel):
    """Auditable penalty fact; policy interpretation remains edition-owned."""

    penalty_id: Identifier
    applied: bool
    amount: float = Field(ge=0.0)
    reason: NonEmptyStr

    @model_validator(mode="after")
    def _non_applied_penalty_is_zero(self) -> PenaltyObservation:
        if not self.applied and self.amount != 0.0:
            raise ValueError("a non-applied penalty must have amount 0")
        return self


class DiagnosticObservation(StrictModel):
    """Public-safe diagnostic separate from rating metrics."""

    code: Identifier
    severity: Literal["info", "warning", "error"]
    message: NonEmptyStr
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("details")
    @classmethod
    def _details_are_finite(cls, values: dict[str, JsonValue]) -> dict[str, JsonValue]:
        ensure_finite_json(values, path="details")
        return values


class EvidenceDigest(StrictModel):
    """Verifier-only role/digest binding retained in private audit."""

    role: Identifier
    digest: Digest


class GateEvidenceDigest(StrictModel):
    """Private audit binding from one declared gate to cited evidence."""

    gate_id: Identifier
    digest: Digest


class VerificationAuditRecord(StrictModel):
    """Verifier-only details excluded from every Harbor-root projection."""

    schema_version: Literal[1] = 1
    verification_id: Identifier
    evidence_digests: tuple[EvidenceDigest, ...] = ()
    gate_evidence_digests: tuple[GateEvidenceDigest, ...] = ()
    penalties: tuple[PenaltyObservation, ...] = ()
    diagnostics: tuple[DiagnosticObservation, ...] = ()
    annotations: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("evidence_digests")
    @classmethod
    def _normalize_evidence_digests(
        cls, values: tuple[EvidenceDigest, ...]
    ) -> tuple[EvidenceDigest, ...]:
        roles = [value.role for value in values]
        if len(roles) != len(set(roles)):
            raise ValueError("duplicate audit evidence digest role")
        return tuple(sorted(values, key=lambda value: value.role))

    @field_validator("gate_evidence_digests")
    @classmethod
    def _normalize_gate_evidence_digests(
        cls, values: tuple[GateEvidenceDigest, ...]
    ) -> tuple[GateEvidenceDigest, ...]:
        gate_ids = [value.gate_id for value in values]
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError("duplicate audit gate evidence binding")
        return tuple(sorted(values, key=lambda value: value.gate_id))

    @field_validator("penalties")
    @classmethod
    def _normalize_penalties(
        cls, values: tuple[PenaltyObservation, ...]
    ) -> tuple[PenaltyObservation, ...]:
        keys = [value.penalty_id for value in values]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate audit penalty observation")
        return tuple(sorted(values, key=lambda value: value.penalty_id))

    @field_validator("diagnostics")
    @classmethod
    def _normalize_diagnostics(
        cls, values: tuple[DiagnosticObservation, ...]
    ) -> tuple[DiagnosticObservation, ...]:
        return tuple(sorted(values, key=lambda value: (value.code, value.message)))

    @field_validator("annotations")
    @classmethod
    def _annotations_are_finite(cls, values: dict[str, JsonValue]) -> dict[str, JsonValue]:
        ensure_finite_json(values, path="annotations")
        return values


class VerifiedOutcome(StrictModel):
    """Verifier-owned facts before writer-enforced public/audit projection."""

    schema_version: Literal[1] = 1
    campaign_id: Identifier
    attempt_id: Identifier
    player_id: Identifier
    task_edition_id: Identifier
    task_family_id: Identifier
    instance_id: Identifier
    instance_cluster_id: Identifier
    comparison_block_id: Identifier
    execution_id: Identifier
    verification_id: Identifier
    verifier_revision_id: Identifier
    disposition: OutcomeDisposition
    gates: tuple[GateObservation, ...] = ()
    metrics: tuple[MetricObservation, ...] = ()
    evidence_digests: tuple[EvidenceDigest, ...] = ()
    report_digests: dict[ReportArtifactRole, Digest] = Field(default_factory=dict)
    public_material_digests: dict[Identifier, Digest] = Field(default_factory=dict)
    verifier_contract_commitment: ScientificContractCommitment
    penalties: tuple[PenaltyObservation, ...] = ()
    diagnostics: tuple[DiagnosticObservation, ...] = ()
    annotations: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("gates")
    @classmethod
    def _normalize_gates(cls, values: tuple[GateObservation, ...]) -> tuple[GateObservation, ...]:
        keys = [value.gate_id for value in values]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate gate observation")
        return tuple(sorted(values, key=lambda value: value.gate_id))

    @field_validator("metrics")
    @classmethod
    def _normalize_metrics(
        cls, values: tuple[MetricObservation, ...]
    ) -> tuple[MetricObservation, ...]:
        keys = [(value.metric_id, value.evaluation_replica_id or "") for value in values]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate metric observation for one evaluation replica")
        return tuple(
            sorted(
                values,
                key=lambda value: (value.metric_id, value.evaluation_replica_id or ""),
            )
        )

    @field_validator("evidence_digests")
    @classmethod
    def _normalize_evidence_digests(
        cls, values: tuple[EvidenceDigest, ...]
    ) -> tuple[EvidenceDigest, ...]:
        roles = [value.role for value in values]
        if len(roles) != len(set(roles)):
            raise ValueError("duplicate evidence digest role")
        return tuple(sorted(values, key=lambda value: value.role))

    @field_validator("penalties")
    @classmethod
    def _normalize_penalties(
        cls, values: tuple[PenaltyObservation, ...]
    ) -> tuple[PenaltyObservation, ...]:
        keys = [value.penalty_id for value in values]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate penalty observation")
        return tuple(sorted(values, key=lambda value: value.penalty_id))

    @field_validator("diagnostics")
    @classmethod
    def _normalize_diagnostics(
        cls, values: tuple[DiagnosticObservation, ...]
    ) -> tuple[DiagnosticObservation, ...]:
        return tuple(sorted(values, key=lambda value: (value.code, value.message)))

    @field_validator("annotations")
    @classmethod
    def _annotations_are_finite(cls, values: dict[str, JsonValue]) -> dict[str, JsonValue]:
        ensure_finite_json(values, path="annotations")
        return values

    @model_validator(mode="after")
    def _scientific_disposition_is_consistent(self) -> VerifiedOutcome:
        if self.disposition == "model_failure" and self.metrics:
            raise ValueError("model_failure cannot carry rating metrics")
        return self


def validate_official_rating_outcome(
    outcome: VerifiedOutcome,
    *,
    task_edition: AnyTaskEditionManifest,
    resolved_utility: TaskRatingUtility | None = None,
) -> VerifiedOutcome:
    """Require the complete one-observation view used by official analysis.

    The base v1 outcome remains development-friendly and may preserve multiple
    execution-bound replicas for one metric. The analysis has a
    narrower contract: one observation per endpoint and complete rating/gate
    facts for every scored attempt.
    """

    metric_ids = [metric.metric_id for metric in outcome.metrics]
    if len(metric_ids) != len(set(metric_ids)):
        raise ValueError("official outcome allows at most one observation per metric endpoint")

    gate_by_id = {gate.gate_id: gate for gate in outcome.gates}
    admission = gate_by_id.get(task_edition.admission_gate_id)
    if outcome.disposition == "model_failure":
        if admission is not None and admission.passed:
            raise ValueError("official model failure cannot pass the admission gate")
        return outcome

    utility = resolved_utility or task_edition.rating_utility
    if utility is None:
        raise ValueError("rating-neutral TaskEdition requires a resolved rating utility")
    endpoint_by_id = {endpoint.endpoint_id: endpoint for endpoint in task_edition.endpoints}
    rating_endpoint_ids = set(utility.endpoint_ids)
    if not rating_endpoint_ids.issubset(endpoint_by_id):
        raise ValueError("resolved rating utility references an unknown endpoint")
    missing_metrics = rating_endpoint_ids.difference(metric_ids)
    if missing_metrics:
        raise ValueError(
            "official scored outcome is missing rating metric observations: "
            f"{sorted(missing_metrics)!r}"
        )

    required_gate_ids = {task_edition.admission_gate_id}
    for endpoint_id in rating_endpoint_ids:
        endpoint = endpoint_by_id[endpoint_id]
        required_gate_ids.update(endpoint.required_gate_ids)
        if endpoint.hurdle_gate_id is not None:
            required_gate_ids.add(endpoint.hurdle_gate_id)
    missing_gates = required_gate_ids.difference(gate_by_id)
    if missing_gates:
        raise ValueError(
            "official scored outcome is missing required gate observations: "
            f"{sorted(missing_gates)!r}"
        )

    metric_by_id = {metric.metric_id: metric for metric in outcome.metrics}
    for endpoint_id in rating_endpoint_ids:
        endpoint = endpoint_by_id[endpoint_id]
        metric = metric_by_id[endpoint_id]
        if metric.measurement_semantics != endpoint.measurement_semantics:
            raise ValueError(
                f"official rating metric {endpoint_id!r} has incompatible measurement semantics"
            )
    for gate_id in required_gate_ids:
        endpoint = endpoint_by_id[gate_id]
        gate = gate_by_id[gate_id]
        if endpoint.measurement_semantics == "verifier_standard_error" and (
            gate.observed_value is None
            or gate.standard_error is None
            or gate.evaluation_replica_id is None
        ):
            raise ValueError(
                f"official stochastic gate {gate_id!r} requires value, standard error, "
                "and replica provenance"
            )
    return outcome


def project_public_verified_outcome(
    outcome: VerifiedOutcome,
) -> tuple[VerifiedOutcome, VerificationAuditRecord]:
    """Split declared scientific facts from verifier-only unstructured audit data.

    The public projection is deliberately closed: adding another public field
    requires a versioned contract change.  Arbitrary strings, paths, secrets,
    evidence digests, diagnostics, penalties, and annotations never cross this
    boundary merely because a verifier placed them in a rich in-memory outcome.
    """

    rich = VerifiedOutcome.model_validate(outcome.model_dump(mode="python"))
    audit = VerificationAuditRecord(
        verification_id=rich.verification_id,
        evidence_digests=rich.evidence_digests,
        gate_evidence_digests=tuple(
            GateEvidenceDigest(gate_id=gate.gate_id, digest=gate.evidence_digest)
            for gate in rich.gates
            if gate.evidence_digest is not None
        ),
        penalties=rich.penalties,
        diagnostics=rich.diagnostics,
        annotations=rich.annotations,
    )
    payload = rich.model_dump(mode="python")
    payload.update(
        {
            "gates": tuple(
                GateObservation.model_validate(
                    {**gate.model_dump(mode="python"), "evidence_digest": None}
                )
                for gate in rich.gates
            ),
            "evidence_digests": (),
            "penalties": (),
            "diagnostics": (),
            "annotations": {},
        }
    )
    public = VerifiedOutcome.model_validate(payload)
    validate_public_verified_outcome(public)
    return public, audit


def validate_public_verified_outcome(outcome: VerifiedOutcome) -> VerifiedOutcome:
    """Fail closed unless an outcome contains only the public contract surface."""

    if outcome.evidence_digests:
        raise ValueError("public VerifiedOutcome cannot carry evidence digests")
    if any(gate.evidence_digest is not None for gate in outcome.gates):
        raise ValueError("public VerifiedOutcome cannot carry gate evidence digests")
    if outcome.penalties:
        raise ValueError("public VerifiedOutcome cannot carry penalty audit details")
    if outcome.diagnostics:
        raise ValueError("public VerifiedOutcome cannot carry verifier diagnostics")
    if outcome.annotations:
        raise ValueError("public VerifiedOutcome cannot carry task annotations")
    return outcome


def validate_verification_audit_bindings(
    audit: VerificationAuditRecord,
    *,
    verification: VerificationManifest,
    outcome: VerifiedOutcome,
) -> VerificationAuditRecord:
    """Bind private audit facts to one public outcome and its exact inputs."""

    if audit.verification_id != verification.verification_id:
        raise ValueError("VerificationAuditRecord verification_id mismatch")
    if outcome.verification_id != verification.verification_id:
        raise ValueError("VerifiedOutcome verification_id mismatch")
    validate_public_verified_outcome(outcome)

    evidence_inputs = {item.role: item.digest for item in verification.evidence_inputs}
    for item in audit.evidence_digests:
        if evidence_inputs.get(item.role) != item.digest:
            raise ValueError(
                f"VerificationAuditRecord evidence digest mismatch for role {item.role!r}"
            )

    gate_ids = {gate.gate_id for gate in outcome.gates}
    audit_evidence_digests = {item.digest for item in audit.evidence_digests}
    for item in audit.gate_evidence_digests:
        if item.gate_id not in gate_ids:
            raise ValueError(f"VerificationAuditRecord references undeclared gate {item.gate_id!r}")
        if item.digest not in audit_evidence_digests:
            raise ValueError(f"VerificationAuditRecord gate evidence mismatch for {item.gate_id!r}")
    return audit


def validate_verified_outcome_bindings(
    outcome: VerifiedOutcome,
    *,
    verification: VerificationManifest,
    execution: ExecutionManifest,
    attempt: AttemptManifest,
    task_edition: AnyTaskEditionManifest,
    verifier_revision: VerifierRevisionManifest,
) -> VerifiedOutcome:
    """Fail closed when separately immutable evaluation artifacts disagree."""

    validate_verifier_revision_for_edition(task_edition, verifier_revision)
    bindings = {
        "campaign_id": (outcome.campaign_id, attempt.campaign_id),
        "attempt_id": (outcome.attempt_id, attempt.attempt_id),
        "player_id": (outcome.player_id, attempt.player_id),
        "task_edition_id": (outcome.task_edition_id, attempt.task_edition_id),
        "task_family_id": (outcome.task_family_id, attempt.task_family_id),
        "instance_id": (outcome.instance_id, attempt.instance_id),
        "instance_cluster_id": (
            outcome.instance_cluster_id,
            attempt.instance_cluster_id,
        ),
        "comparison_block_id": (
            outcome.comparison_block_id,
            attempt.comparison_block_id,
        ),
        "execution.attempt_id": (execution.attempt_id, attempt.attempt_id),
        "execution_id": (outcome.execution_id, execution.execution_id),
        "verification.execution_id": (
            verification.execution_id,
            execution.execution_id,
        ),
        "verification_id": (outcome.verification_id, verification.verification_id),
        "verification.task_edition_id": (
            verification.task_edition_id,
            task_edition.task_edition_id,
        ),
        "outcome.task_edition_id": (
            outcome.task_edition_id,
            task_edition.task_edition_id,
        ),
        "outcome.task_family_id": (
            outcome.task_family_id,
            task_edition.task_family_id,
        ),
        "verification.verifier_revision_id": (
            verification.verifier_revision_id,
            verifier_revision.verifier_revision_id,
        ),
        "outcome.verifier_revision_id": (
            outcome.verifier_revision_id,
            verifier_revision.verifier_revision_id,
        ),
    }
    for label, (observed, expected) in bindings.items():
        if observed != expected:
            raise ValueError(f"VerifiedOutcome binding mismatch for {label}")

    if outcome.verifier_contract_commitment != task_edition.verifier_contract_commitment:
        raise ValueError("VerifiedOutcome verifier contract commitment mismatch")

    evidence_inputs = {evidence.role: evidence.digest for evidence in verification.evidence_inputs}
    for evidence in outcome.evidence_digests:
        if evidence_inputs.get(evidence.role) != evidence.digest:
            raise ValueError(f"VerifiedOutcome evidence digest mismatch for role {evidence.role!r}")

    public_materials = {
        "instruction": task_edition.instruction_digest,
        **task_edition.public_material_digests,
    }
    for role, digest in outcome.public_material_digests.items():
        if public_materials.get(role) != digest:
            raise ValueError(f"VerifiedOutcome public material digest mismatch for role {role!r}")

    endpoints = {endpoint.endpoint_id: endpoint for endpoint in task_edition.endpoints}
    outcome_evidence_digests = {evidence.digest for evidence in outcome.evidence_digests}
    if outcome.disposition == "scored" and not any(
        gate.gate_id == task_edition.admission_gate_id for gate in outcome.gates
    ):
        raise ValueError("scored VerifiedOutcome is missing the TaskEdition admission gate")
    for gate in outcome.gates:
        endpoint = endpoints.get(gate.gate_id)
        if endpoint is None or endpoint.metric_role != "gate":
            raise ValueError(f"VerifiedOutcome references undeclared gate {gate.gate_id!r}")
        model_failure_admission = (
            outcome.disposition == "model_failure"
            and gate.gate_id == task_edition.admission_gate_id
        )
        if model_failure_admission:
            if gate.passed:
                raise ValueError("model_failure cannot pass the TaskEdition admission gate")
            if any(
                value is not None
                for value in (
                    gate.observed_value,
                    gate.standard_error,
                    gate.threshold_value,
                    gate.threshold_source,
                )
            ):
                raise ValueError(
                    "model_failure admission gate cannot carry scientific threshold facts"
                )
        if endpoint.measurement_semantics == "verifier_standard_error":
            if not model_failure_admission and (
                gate.observed_value is None
                or gate.standard_error is None
                or gate.evaluation_replica_id is None
            ):
                raise ValueError(
                    f"VerifiedOutcome gate {gate.gate_id!r} requires observed_value "
                    "standard_error, and evaluation_replica_id"
                )
        elif gate.standard_error is not None or gate.evaluation_replica_id is not None:
            raise ValueError(f"deterministic gate {gate.gate_id!r} cannot carry replay uncertainty")
        has_declared_threshold = (
            endpoint.gate_threshold is not None or endpoint.gate_threshold_source is not None
        )
        if (
            endpoint.gate_rule in {"threshold", "composite"}
            and has_declared_threshold
            and not model_failure_admission
        ):
            if gate.observed_value is None or gate.threshold_value is None:
                raise ValueError(
                    f"threshold gate {gate.gate_id!r} requires observed and threshold values"
                )
            if (
                endpoint.gate_threshold is not None
                and gate.threshold_value != endpoint.gate_threshold
            ):
                raise ValueError(f"VerifiedOutcome gate threshold mismatch for {gate.gate_id!r}")
            if endpoint.gate_threshold_source is not None and (
                gate.threshold_source != endpoint.gate_threshold_source
            ):
                raise ValueError(
                    f"VerifiedOutcome gate threshold source mismatch for {gate.gate_id!r}"
                )
            threshold_component_passed = (
                gate.observed_value >= gate.threshold_value
                if endpoint.direction == "higher_is_better"
                else gate.observed_value <= gate.threshold_value
            )
            if endpoint.gate_rule == "threshold" and gate.passed != threshold_component_passed:
                raise ValueError(
                    f"VerifiedOutcome gate pass decision mismatch for {gate.gate_id!r}"
                )
            if endpoint.gate_rule == "composite" and gate.passed and not threshold_component_passed:
                raise ValueError(
                    f"VerifiedOutcome composite gate {gate.gate_id!r} cannot pass when its "
                    "declared threshold component fails"
                )
        if (
            gate.evidence_digest is not None
            and gate.evidence_digest not in outcome_evidence_digests
        ):
            raise ValueError(f"VerifiedOutcome gate evidence digest mismatch for {gate.gate_id!r}")
    for metric in outcome.metrics:
        endpoint = endpoints.get(metric.metric_id)
        if endpoint is None or endpoint.metric_role == "gate":
            raise ValueError(f"VerifiedOutcome references undeclared metric {metric.metric_id!r}")
        if metric.measurement_semantics != endpoint.measurement_semantics:
            raise ValueError(
                f"VerifiedOutcome measurement semantics mismatch for {metric.metric_id!r}"
            )
        observed_gate_ids = {gate.gate_id for gate in outcome.gates}
        missing_gate_dependencies = set(endpoint.required_gate_ids).difference(observed_gate_ids)
        if missing_gate_dependencies:
            raise ValueError(
                f"VerifiedOutcome metric {metric.metric_id!r} is missing required gate "
                f"observations: {sorted(missing_gate_dependencies)!r}"
            )
    return outcome


def validate_verification_artifact_set_bindings(
    artifact_set: VerificationArtifactSet,
    *,
    verification: VerificationManifest,
    outcome: VerifiedOutcome,
) -> VerificationArtifactSet:
    """Bind one immutable output-role set to its verification and outcome."""

    if artifact_set.verification_id != verification.verification_id:
        raise ValueError("VerificationArtifactSet verification_id mismatch")
    if outcome.verification_id != verification.verification_id:
        raise ValueError("VerifiedOutcome verification_id mismatch")

    artifacts = {artifact.role: artifact for artifact in artifact_set.artifacts}
    artifact_report_digests = {
        role: artifact.digest
        for role in ("task_score_report", "legacy_elo_report")
        if (artifact := artifacts.get(role)) is not None
    }
    if artifact_report_digests != outcome.report_digests:
        raise ValueError("VerifiedOutcome report_digests mismatch")
    return artifact_set


__all__ = [
    "ArtifactBinding",
    "ArtifactRole",
    "ArtifactSetScope",
    "DiagnosticObservation",
    "EvidenceDigest",
    "EvidenceInput",
    "EvidenceProvenance",
    "GateEvidenceDigest",
    "GateObservation",
    "HostModelFailureSignal",
    "MeasurementSemantics",
    "MetricObservation",
    "OutcomeDisposition",
    "PenaltyObservation",
    "ReportArtifactRole",
    "VerificationArtifactSet",
    "VerificationAuditRecord",
    "VerificationManifest",
    "VerificationReason",
    "VerifiedOutcome",
    "evaluation_replica_id",
    "project_public_verified_outcome",
    "validate_official_rating_outcome",
    "validate_public_verified_outcome",
    "validate_verification_audit_bindings",
    "validate_verified_outcome_bindings",
    "validate_verification_artifact_set_bindings",
    "verification_manifest_id",
]
