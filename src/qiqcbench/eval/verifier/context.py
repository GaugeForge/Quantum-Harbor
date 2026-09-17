"""Strict, verifier-only bindings for execution-bound evaluation runs.

Ordinary task development does not need evaluation identities. A
materializer may opt a verifier invocation into framework v2 by supplying one
complete JSON value in :data:`VERIFIER_CONTEXT_ENV`.  Once that value is
present, parsing is deliberately fail-closed: a partial binding must never be
mistaken for canonical evidence.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from collections.abc import Mapping
from typing import Literal

from pydantic import Field, JsonValue, field_validator, model_validator

from qiqcbench.eval.contracts.base import Digest, Identifier, StrictModel, ensure_finite_json
from qiqcbench.eval.contracts.campaign import (
    AttemptManifest,
    ExecutionManifest,
    commit_verifier_replay_secret,
)
from qiqcbench.eval.contracts.outcome import (
    VerificationReason,
    evaluation_replica_id,
)

VERIFIER_CONTEXT_ENV = "QIQCBENCH_EVAL_VERIFIER_CONTEXT"
EvidenceBindingMode = Literal["legacy_unbound", "qsim_context_v1"]


def _duplicate_checked_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key in verifier context: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant in verifier context: {value!r}")


class TaskEditionRuntimeBinding(StrictModel):
    """Task-edition bytes admitted before one verifier invocation."""

    task_id: Identifier
    edition_tier: Literal["development", "candidate", "canonical"]
    edition_manifest_sha256: Digest


class VerifierRunContext(StrictModel):
    """Planner-owned identities available only inside one verifier invocation.

    ``verification_id`` is intentionally absent.  The verifier derives it only
    after hashing the exact evidence inputs selected for this scoring pass.
    """

    schema_version: Literal[1, 2] = 1
    attempt: AttemptManifest
    execution: ExecutionManifest
    task_id: Identifier
    task_edition_name: Identifier
    verifier_revision_name: Identifier
    verifier_revision_id: Identifier
    verifier_replay_secret: Digest
    evidence_binding_mode: EvidenceBindingMode = "legacy_unbound"
    qsim_log_evidence_role: Identifier = "experiment_log"
    instance_binding_id: Identifier | None = None
    reason: VerificationReason = "initial"
    predecessor_verification_id: Identifier | None = None
    expected_legacy_labels: dict[Identifier, JsonValue] = Field(default_factory=dict)
    task_edition_runtime_binding: TaskEditionRuntimeBinding | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )

    @field_validator("expected_legacy_labels")
    @classmethod
    def _legacy_labels_are_finite(cls, values: dict[str, JsonValue]) -> dict[str, JsonValue]:
        ensure_finite_json(values, path="expected_legacy_labels")
        return values

    @model_validator(mode="after")
    def _bindings_and_lineage_are_consistent(self) -> VerifierRunContext:
        if self.execution.attempt_id != self.attempt.attempt_id:
            raise ValueError("execution attempt_id must match the bound attempt")
        if self.reason == "initial" and self.predecessor_verification_id is not None:
            raise ValueError("initial context cannot have a predecessor_verification_id")
        if self.reason == "rescore" and self.predecessor_verification_id is None:
            raise ValueError("rescore context requires predecessor_verification_id")
        if self.evidence_binding_mode == "qsim_context_v1" and self.instance_binding_id is None:
            raise ValueError("qsim_context_v1 requires a private instance_binding_id")
        if self.schema_version == 1 and self.task_edition_runtime_binding is not None:
            raise ValueError(
                "verifier context schema 1 cannot carry a task edition runtime binding"
            )
        if self.schema_version == 2 and self.task_edition_runtime_binding is None:
            raise ValueError("verifier context schema 2 requires a task edition runtime binding")
        if (
            self.task_edition_runtime_binding is not None
            and self.task_edition_runtime_binding.task_id != self.task_id
        ):
            raise ValueError("task edition runtime binding task_id must match the verifier context")
        if (
            commit_verifier_replay_secret(self.verifier_replay_secret)
            != self.execution.verifier_replay_secret_commitment
        ):
            raise ValueError("verifier replay secret does not match execution seed commitment")
        return self


def load_verifier_run_context(
    environ: Mapping[str, str] | None = None,
) -> VerifierRunContext | None:
    """Load an optional canonical-run context from the verifier environment.

    Absence is the sole legacy-only signal.  Blank, malformed, partial, or
    extra-field payloads raise through the strict versioned model.
    """

    source = os.environ if environ is None else environ
    if VERIFIER_CONTEXT_ENV not in source:
        return None
    raw = source[VERIFIER_CONTEXT_ENV]
    if not raw.strip():
        raise ValueError(f"{VERIFIER_CONTEXT_ENV} must contain a complete JSON object")
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{VERIFIER_CONTEXT_ENV} must contain strict JSON") from exc
    return VerifierRunContext.model_validate(payload, strict=True)


def derive_verifier_replica_seed(
    context: VerifierRunContext,
    *,
    domain: str,
) -> int:
    """Derive a stable independent RNG domain from one bound execution."""

    _validate_replica_domain(domain)
    message = (f"qiqcbench-eval-replica-v1\0{domain}\0{context.execution.execution_id}").encode()
    digest = hmac.new(
        bytes.fromhex(context.verifier_replay_secret),
        message,
        hashlib.sha256,
    ).digest()
    return int.from_bytes(digest[:8], "big")


def derive_evaluation_replica_id(
    context: VerifierRunContext,
    *,
    domain: str,
) -> str:
    """Identify the exact execution/domain pair used by the replay seed."""

    _validate_replica_domain(domain)
    return evaluation_replica_id(
        execution_id=context.execution.execution_id,
        replay_domain=domain,
        replay_secret_commitment=context.execution.verifier_replay_secret_commitment,
    )


def _validate_replica_domain(domain: str) -> None:
    if (
        not isinstance(domain, str)
        or not domain
        or any(character.isspace() for character in domain)
    ):
        raise ValueError("replica domain must be a non-empty identifier")


__all__ = [
    "EvidenceBindingMode",
    "TaskEditionRuntimeBinding",
    "VERIFIER_CONTEXT_ENV",
    "VerifierRunContext",
    "derive_evaluation_replica_id",
    "derive_verifier_replica_seed",
    "load_verifier_run_context",
]
