"""Public + hidden device config for the ``logical_circuit_optimizer`` qtype.

A **device-less / static-instance** qtype (sibling of ``ftqc_resource_estimation``):
no quantum dynamics, no shots, no async job. The "device" is a fixed opaque
Clifford+T logical circuit the agent must re-express with fewer T/T-dagger gates.

Nothing about the *answer* is secret (the reference is the public input circuit — the
task is to produce a verified low-T equivalent), so the hidden config carries only the
stale lab notebook. The verifier-only task material binds the public file digests;
all scientific limits and admission semantics are public.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Public spec (visible to agent) ----------


class PublicEquivalenceCheck(_Strict):
    strategy: Literal["submission_bound_statevector_v1"] = "submission_bound_statevector_v1"
    challenge_state_count: int = Field(16, ge=1)
    fidelity_tolerance: float = Field(1e-8, gt=0.0, lt=1.0)
    soft_deadline_s: int = Field(1500, ge=1)


class PublicLogicalOptSpec(_Strict):
    """Public device spec for ``logical_circuit_optimizer`` (via ``get_device_spec`` /
    ``get_optimization_instance``)."""

    schema_version: int = 2
    device_id: str
    qtype: Literal["logical_circuit_optimizer"] = "logical_circuit_optimizer"
    task_id: str
    device_class: str = "logical_circuit_optimizer"

    n_qubits: int = Field(..., ge=1)
    gate_set: list[str] = Field(
        default_factory=lambda: ["h", "s", "sdg", "x", "y", "z", "cx", "cz", "t", "tdg"]
    )
    circuit_format: Literal["qiqc_clifford_t_unitary_v2"] = "qiqc_clifford_t_unitary_v2"
    input_circuit_file: str = "input_circuit.json"
    initial_t_count: int = Field(..., ge=0)
    max_total_qubits: int = Field(..., ge=1)
    max_gate_entries: int = Field(..., ge=1)
    submission_semantics: Literal["unitary_with_zero_initialized_ancillas"] = (
        "unitary_with_zero_initialized_ancillas"
    )
    objective: str = "minimize_t_plus_tdagger_count"
    rating_mode: Literal["elo"] = "elo"
    race_metric: str = "t_plus_tdagger_count"
    admission_rule: Literal["valid_equivalent_strict_t_count_improvement"] = (
        "valid_equivalent_strict_t_count_improvement"
    )
    equivalence_check: PublicEquivalenceCheck = PublicEquivalenceCheck()

    dynamics_primitive: str = "get_optimization_instance"
    target_wall_clock_runtime_s: int = Field(7200, ge=1)
    notes: str = ""


# ---------- Hidden truth (only inside qsim) ----------


class StaleLogicalOptNotebook(_Strict):
    """Stale prior + trap advice (§6). Surfaced via ``get_lab_notebook``."""

    method_advice: str = (
        "T-count is basically minimal already; just cancel adjacent T and T-dagger "
        "and merge T*T -> S with a few commutations."
    )
    ancilla_advice: str = "Don't bother with ancillas."
    notes: str = "This circuit is close to optimal; a quick local T-cancellation pass is enough."
    reliability: str = "assessment recorded by the last optimization pass"


class HiddenLogicalOptConfig(_Strict):
    """Hidden config for ``logical_circuit_optimizer``. Inside qsim only.

    Minimal by design: the reference is the public input circuit, so there is no
    hidden answer key here. The authoritative scorer config lives in verifier-only
    task materials (``configs/task_materials/<task>/hidden/hidden_scorer.json``)."""

    schema_version: int = 1
    device_id: str
    qtype: Literal["logical_circuit_optimizer"] = "logical_circuit_optimizer"
    task_id: str
    stale_lab_notebook: StaleLogicalOptNotebook = StaleLogicalOptNotebook()
