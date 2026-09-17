"""Public + hidden device config types for the blackbox analog-dynamics qtype.

Mirrors the structural public/hidden split: the public
spec carries the Hamiltonian *dictionary* (term names + Pauli strings) and the
probe constraints the agent needs to design experiments, but never any
coefficient. The hidden config carries the true coefficient vector and the
fixed private RNG salt, and lives only inside the qsim container. The backend
mixes that salt with fresh private entropy for every benchmark attempt, so it
does not fix or replay the runtime shot noise.

The agent learns an unknown Hamiltonian by submitting black-box probe batches:
prepare a product state, evolve under ``U(t)=exp(-i (t/2) sum_P omega_P P)``,
and measure a Pauli observable. The scarce resource is total accepted evolution
time. The first (and currently only) task on this qtype is
``time_budgeted_hamlearn_10q``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


# ---------- Public spec (visible to agent) ----------


class AnalogResourceBudget(_Strict):
    """Public, finite ingress/submission resource policy for this qtype.

    Consumed by the shared enforcement seams: the MCP server bounds the raw
    HTTP request body via ``max_ingress_request_bytes``, and the generic
    ``submit_final_answer`` action walks strings/depth and checks the compact
    serialized envelope before Pydantic validation, then reserves one slot of
    ``max_final_answer_submissions`` per accepted submission. These are
    DoS-hardening caps, not scientific budgets: every value must sit
    comfortably above legitimate agent behavior.
    """

    max_ingress_request_bytes: int = Field(..., ge=1)
    max_final_answer_serialized_bytes: int = Field(..., ge=1)
    max_final_answer_submissions: int = Field(..., ge=1)
    max_answer_string_characters: int = Field(..., ge=1)
    max_final_answer_nesting_depth: int = Field(..., ge=1)
    max_job_result_polls: int = Field(..., ge=1)
    max_probe_jobs: int = Field(..., ge=1)


class AnalogTermEntry(_Strict):
    """One dictionary term: a name and its length-``n_qubits`` Pauli string."""

    name: str
    pauli: str
    weight: int = Field(..., ge=1)


class PublicAnalogSpec(_Strict):
    """Public device spec for the blackbox analog-dynamics qtype.

    Returned verbatim by ``get_device_spec``. Carries the full term dictionary
    and every probe constraint, but no coefficients.
    """

    schema_version: int = 1
    device_id: str
    qtype: Literal["blackbox_analog_dynamics"] = "blackbox_analog_dynamics"
    task_id: str
    device_class: str = "blackbox_analog_dynamics"
    n_qubits: int = Field(..., ge=1)
    num_terms: int = Field(..., ge=1)
    qubit_labels: list[str]
    dynamics_primitive: str = "run_hamiltonian_probe_batch"
    hamiltonian_convention: str
    term_order_convention: str
    coefficient_unit: str = "rad_per_us"
    locality_promise: str
    sparsity_promise_max_nonzero: int = Field(..., ge=1)
    coefficient_bound_rad_per_us: list[float] = Field(..., min_length=2, max_length=2)
    allowed_initial_states: list[str] = Field(..., min_length=1)
    allowed_observable_weight: list[int] = Field(..., min_length=2, max_length=2)
    evolve_time_range_us: list[float] = Field(..., min_length=2, max_length=2)
    evolve_time_resolution_us: float = Field(..., gt=0)
    off_grid_time_policy: Literal["reject"] = "reject"
    return_type: str = "raw_pauli_outcomes"
    num_internal_repetitions: int = Field(..., ge=1)
    nominal_single_probe_noise_1sigma: float = Field(..., gt=0)
    total_evolution_time_budget_us: float = Field(..., gt=0)
    max_probe_rows: int = Field(..., ge=1)
    max_rows_per_batch: int = Field(..., ge=1)
    target_wall_clock_runtime_s: int = Field(..., ge=1)
    budget: AnalogResourceBudget
    notes: str = ""
    terms: list[AnalogTermEntry] = Field(..., min_length=1)


# ---------- Hidden truth (only inside qsim) ----------


class HiddenAnalogTerm(_Strict):
    """One nonzero Hamiltonian term: a Pauli string and its true coefficient."""

    pauli: str
    coefficient: float


class StaleAnalogNotebook(_Strict):
    """Stale prior surfaced (unchanged) via get_lab_notebook."""

    coefficients_rad_per_us: dict[str, float] = Field(default_factory=dict)
    note: str = ""
    reliability: str = "stale; use only as a weak prior"


class HiddenAnalogConfig(_Strict):
    """Hidden truth for the analog-dynamics qtype. Lives only inside qsim."""

    schema_version: int = 1
    device_id: str
    qtype: Literal["blackbox_analog_dynamics"] = "blackbox_analog_dynamics"
    seed: int
    hamiltonian_terms: list[HiddenAnalogTerm] = Field(..., min_length=1)
    stale_lab_notebook: StaleAnalogNotebook = StaleAnalogNotebook()


def hidden_coefficient_vector(
    hidden: HiddenAnalogConfig, public: PublicAnalogSpec
) -> tuple[float, ...]:
    """Return the executed coefficients in public order after cross-validation.

    The public and hidden models are parsed independently, so their shared
    invariants must be checked before the backend executes or commits an
    instance.  In particular, silently dropping an unknown hidden Pauli or
    overwriting a duplicate while the engine sums it would make the commitment
    describe a different Hamiltonian from the one actually simulated.
    """

    if hidden.device_id != public.device_id:
        raise ValueError("public and hidden analog configs name different devices")
    public_paulis = [entry.pauli for entry in public.terms]
    if len(public_paulis) != public.num_terms or len(set(public_paulis)) != len(public_paulis):
        raise ValueError("public analog term dictionary is not a unique num_terms ordering")
    for entry in public.terms:
        actual_weight = sum(character != "I" for character in entry.pauli)
        if (
            len(entry.pauli) != public.n_qubits
            or any(character not in "IXYZ" for character in entry.pauli)
            or entry.weight != actual_weight
        ):
            raise ValueError("public analog term dictionary contains an invalid Pauli entry")

    public_set = set(public_paulis)
    coefficient_by_pauli: dict[str, float] = {}
    lower, upper = public.coefficient_bound_rad_per_us
    if lower > upper:
        raise ValueError("public analog coefficient bounds are reversed")
    for term in hidden.hamiltonian_terms:
        if term.pauli not in public_set:
            raise ValueError(
                "hidden analog Hamiltonian contains a Pauli outside the public dictionary"
            )
        if term.pauli in coefficient_by_pauli:
            raise ValueError("hidden analog Hamiltonian contains a duplicate Pauli")
        coefficient = float(term.coefficient)
        if not lower <= coefficient <= upper:
            raise ValueError("hidden analog coefficient violates the public coefficient bound")
        coefficient_by_pauli[term.pauli] = coefficient
    if (
        sum(coefficient != 0.0 for coefficient in coefficient_by_pauli.values())
        > public.sparsity_promise_max_nonzero
    ):
        raise ValueError("hidden analog Hamiltonian violates the public sparsity promise")
    return tuple(coefficient_by_pauli.get(pauli, 0.0) for pauli in public_paulis)


def hidden_instance_commitment(hidden: HiddenAnalogConfig, public: PublicAnalogSpec) -> str:
    """Execution-binding digest of the exact instance qsim serves.

    Canonical form: the full coefficient vector in the PUBLIC term order (zeros
    included), the Hamiltonian convention, and the term-order convention, as
    compact JSON. qsim logs this digest into its private experiment log at MCP
    registration; the separate verifier independently recomputes it from its
    private scoring truth and refuses (as infrastructure) to score when the two
    disagree, so "qsim executed Hamiltonian A, verifier scored Hamiltonian B"
    cannot silently produce a scientific verdict. The digest reveals no
    coefficients on its own, and it travels only inside the qsim-private
    evidence path (the log is never mounted into the agent container for
    separate-mode bundles).
    """

    import hashlib
    import json

    vector = hidden_coefficient_vector(hidden, public)
    payload = json.dumps(
        {
            "scheme": "sha256_omega_vector_v1",
            "qtype": "blackbox_analog_dynamics",
            "hamiltonian_convention": public.hamiltonian_convention,
            "term_order_convention": public.term_order_convention,
            "omega": vector,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
