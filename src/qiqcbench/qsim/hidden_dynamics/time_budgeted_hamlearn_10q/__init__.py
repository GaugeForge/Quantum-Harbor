"""Hidden dynamics for the ``time_budgeted_hamlearn_10q`` task.

Owns the fixed hidden Hamiltonian instance, the black-box probe oracle, the
public materials generator, and the L∞ scorer that back the §9h
feasibility gate. No ``blackbox_analog_dynamics`` qtype / MCP / Harbor surface
is built here — that materialization is deferred until the gate is green.
"""

from qiqcbench.qsim.hidden_dynamics.time_budgeted_hamlearn_10q.construction import (
    BUDGET_US,
    ORACLE_NOISE_SHOTS,
    HamLearnInstance,
    build_instance,
    build_omega_star,
    build_stale_vector,
    support_terms,
    write_materials,
)
from qiqcbench.qsim.hidden_dynamics.time_budgeted_hamlearn_10q.evidence import (
    EvidenceCertification,
    certify_evidence,
)
from qiqcbench.qsim.hidden_dynamics.time_budgeted_hamlearn_10q.oracle import (
    Oracle,
    ProbeEvidence,
    ProbeResult,
    ProbeRow,
)
from qiqcbench.qsim.hidden_dynamics.time_budgeted_hamlearn_10q.schema import (
    HamLearnActivitySummary,
    HamLearnAnswer,
    HamLearnHiddenScorer,
    HamLearnTrajectoryReview,
    HamLearnVerifiableActivity,
)
from qiqcbench.qsim.hidden_dynamics.time_budgeted_hamlearn_10q.scorer import (
    DigestMismatch,
    ScoreReport,
    linf,
    load_hidden_scorer,
    relative_l2,
    score,
    score_from_materials,
    verify_public_material_digests,
)

__all__ = [
    "BUDGET_US",
    "ORACLE_NOISE_SHOTS",
    "DigestMismatch",
    "EvidenceCertification",
    "HamLearnActivitySummary",
    "HamLearnAnswer",
    "HamLearnHiddenScorer",
    "HamLearnTrajectoryReview",
    "HamLearnVerifiableActivity",
    "HamLearnInstance",
    "Oracle",
    "ProbeEvidence",
    "ProbeResult",
    "ProbeRow",
    "ScoreReport",
    "build_instance",
    "build_omega_star",
    "build_stale_vector",
    "certify_evidence",
    "linf",
    "load_hidden_scorer",
    "relative_l2",
    "score",
    "score_from_materials",
    "support_terms",
    "verify_public_material_digests",
    "write_materials",
]
