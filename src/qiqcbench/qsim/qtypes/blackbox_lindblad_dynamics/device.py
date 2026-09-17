"""Public + hidden device config types for the blackbox Lindblad-dynamics qtype.

Mirrors the structural public/hidden split: the public
spec carries the two-block *dictionary* (coherent term names + Pauli strings, and
the dissipative Kossakowski coordinate descriptors) and the probe constraints the
agent needs to design experiments, but never any coefficient. The hidden config
carries the true coherent terms and the dissipator as a flat list of GKSL
``(P_a, P_b, c)`` terms, and lives only inside the qsim container. Runtime
sampling entropy is deliberately not part of the fixed hidden device config.

The agent learns an unknown open-system generator by submitting black-box probe
batches: prepare a product state, evolve under the GKSL master equation, and
measure a Pauli observable. The scarce resource is total accepted evolution
time.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Public spec (visible to agent) ----------


class LindbladCoherentEntry(_Strict):
    """One coherent dictionary term: a name and its length-``n_qubits`` Pauli string."""

    name: str
    pauli: str
    weight: int = Field(..., ge=1)


class LindbladDissipativeEntry(_Strict):
    """One dissipative Kossakowski coordinate descriptor.

    ``single_site`` entries name a real coordinate of a single-site ``3×3``
    Hermitian block (``c_XX``, ``Re_c_XY`` ...); ``correlated`` entries name a
    nearest-neighbour correlated diagonal jump rate (``g_ZZ`` on an edge).
    """

    name: str
    kind: Literal["single_site", "correlated"]
    qubit: int | None = None
    entry: str | None = None
    edge: list[int] | None = None
    pauli_pair: str | None = None


class LindbladBudget(_Strict):
    """Hard cumulative/per-call resource bounds enforced by qsim.

    ``total_evolution_time_budget_us`` and ``max_probe_rows`` on the spec bound
    the *science*; these bound the *transport and evidence* envelope so a run
    cannot grow logs, artifacts, or queued jobs without limit. The shared
    action/server layers consume the generically named fields
    (``max_ingress_request_bytes``, ``max_metadata_calls``,
    ``max_final_answer_*``); the probe-job and poll meters are reserved by the
    lindblad submit action before allocation.
    """

    max_probe_jobs: int = Field(..., gt=0)
    max_ingress_request_bytes: int = Field(..., gt=0)
    max_metadata_calls: int = Field(..., gt=0)
    max_job_result_polls: int = Field(..., gt=0)
    max_final_answer_serialized_bytes: int = Field(..., gt=0)
    max_final_answer_submissions: int = Field(..., gt=0)
    max_answer_string_characters: int = Field(..., gt=0)


class PublicLindbladSpec(_Strict):
    """Public device spec for the blackbox Lindblad-dynamics qtype.

    Returned verbatim by ``get_device_spec``. Carries the full two-block
    dictionary and every probe constraint, but no coefficients.
    """

    schema_version: int = 1
    device_id: str
    qtype: Literal["blackbox_lindblad_dynamics"] = "blackbox_lindblad_dynamics"
    task_id: str
    device_class: str = "blackbox_lindblad_dynamics"
    n_qubits: int = Field(..., ge=1)
    num_coherent_terms: int = Field(..., ge=1)
    num_dissipative_coords: int = Field(..., ge=1)
    qubit_labels: list[str]
    dynamics_primitive: str = "run_lindblad_probe_batch"
    generator_convention: str
    coherent_term_order: str
    dissipative_term_order: str
    coherent_unit: str = "rad_per_us"
    dissipative_unit: str = "per_us"
    locality_promise: str
    coherent_bound_rad_per_us: list[float] = Field(..., min_length=2, max_length=2)
    dissipative_diag_bound_per_us: list[float] = Field(..., min_length=2, max_length=2)
    dissipative_offdiag_bound_per_us: float = Field(..., gt=0)
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
    budget: LindbladBudget
    target_wall_clock_runtime_s: int = Field(..., ge=1)
    notes: str = ""
    coherent_terms: list[LindbladCoherentEntry] = Field(..., min_length=1)
    dissipative_coords: list[LindbladDissipativeEntry] = Field(..., min_length=1)


# ---------- Hidden truth (only inside qsim) ----------


class HiddenLindbladCoherentTerm(_Strict):
    """One nonzero coherent term: a Pauli string and its true coefficient ``h_P``."""

    pauli: str
    coefficient: float


class HiddenLindbladDissipatorTerm(_Strict):
    """One GKSL dissipator term ``c · D[P_a, P_b]`` with a (possibly complex) ``c``.

    A Hermitian single-site Kossakowski block contributes its nine ``(a,b)``
    entries (off-diagonal pairs carry conjugate ``c``); a correlated diagonal
    rate is a single ``(P, P)`` term with real ``c``.
    """

    pa: str
    pb: str
    c_re: float
    c_im: float = 0.0


class StaleLindbladNotebook(_Strict):
    """Stale prior surfaced (unchanged) via get_lab_notebook."""

    coherent_rad_per_us: dict[str, float] = Field(default_factory=dict)
    dissipative_per_us: dict[str, float] = Field(default_factory=dict)
    note: str = ""
    reliability: str = "stale; use only as a weak prior"


class HiddenLindbladConfig(_Strict):
    """Hidden truth for the Lindblad-dynamics qtype. Lives only inside qsim."""

    schema_version: int = 2
    device_id: str
    qtype: Literal["blackbox_lindblad_dynamics"] = "blackbox_lindblad_dynamics"
    coherent_terms: list[HiddenLindbladCoherentTerm] = Field(..., min_length=1)
    dissipator_terms: list[HiddenLindbladDissipatorTerm] = Field(..., min_length=1)
    stale_lab_notebook: StaleLindbladNotebook = StaleLindbladNotebook()
