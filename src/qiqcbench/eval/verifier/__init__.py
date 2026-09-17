"""Task-agnostic verifier output helpers for evaluation framework v2."""

from typing import TYPE_CHECKING

from qiqcbench.eval.verifier.bindings import EvidenceIntegrityError
from qiqcbench.eval.verifier.context import (
    VERIFIER_CONTEXT_ENV,
    EvidenceBindingMode,
    TaskEditionRuntimeBinding,
    VerifierRunContext,
    derive_evaluation_replica_id,
    derive_verifier_replica_seed,
    load_verifier_run_context,
)
from qiqcbench.eval.verifier.legacy_rubric import emit_legacy_rubric_report
from qiqcbench.eval.verifier.writer import (
    ArtifactPayload,
    VerificationWriteResult,
    write_verification_artifacts,
)

if TYPE_CHECKING:
    from qiqcbench.eval.verifier.binary_admission import emit_binary_admission_report


def __getattr__(name: str):
    if name == "emit_binary_admission_report":
        from qiqcbench.eval.verifier.binary_admission import emit_binary_admission_report

        return emit_binary_admission_report
    raise AttributeError(name)


__all__ = [
    "ArtifactPayload",
    "EvidenceBindingMode",
    "EvidenceIntegrityError",
    "TaskEditionRuntimeBinding",
    "VERIFIER_CONTEXT_ENV",
    "VerifierRunContext",
    "VerificationWriteResult",
    "derive_evaluation_replica_id",
    "derive_verifier_replica_seed",
    "emit_binary_admission_report",
    "emit_legacy_rubric_report",
    "load_verifier_run_context",
    "write_verification_artifacts",
]
