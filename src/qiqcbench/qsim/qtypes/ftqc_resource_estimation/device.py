"""Public + hidden device config for the ``ftqc_resource_estimation`` qtype.

This is a **device-less / static-instance** qtype: there is no quantum dynamics,
no shots, and no async job model. The "device" is a pinned instance plus a pinned
FTQC cost model. The shipped families cover physical CCZ2T resource estimation
and MPS-initialized QPE resource planning.

The public spec carries pinned constants and handles to authoritative public
materials — never a held-out answer. Hidden references and scoring thresholds
remain inside qsim.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Public spec (visible to agent) ----------


class FtqcInstanceFiles(_Strict):
    """Legacy v1 public file handles retained for frozen device-spec overlays."""

    raw_pauli_hamiltonian: str = "h2o_sto3g_jw_pauli.csv"
    public_metadata: str = "h2o_pauli_lcu_bliss_public_metadata.json"


class FtqcAlgorithmSummary(_Strict):
    """Public logical cost of the fixed target algorithm (Qualtran counts).

    Frozen at construction from the pinned bloq; the agent uses these directly
    (it need not rebuild the bloq, though qualtran is available in-container).
    """

    algorithm_name: str
    n_algo_qubits: int = Field(..., ge=1)
    n_t: int = Field(..., ge=0)
    n_ccz: int = Field(..., ge=0)


class FtqcPhysicalEstimationMeta(_Strict):
    """Public spec for the physical CCZ2T spacetime-volume estimation family.

    The cost model is public (Qualtran 0.7.0 Gidney-Fowler CCZ2T). The design is
    ``(distillation_l1_d, distillation_l2_d, n_factories, data_block_d)``. Only the
    hardware parameters + reference optimum are held out; broad priors on the
    hardware are published here so the agent can plan its oracle search (they help
    equally under any hidden hardware — the two-world litmus).
    """

    algorithm: FtqcAlgorithmSummary
    cost_model_name: str = "qualtran_ccz2t_gidney_fowler"
    cost_model_version: str = "qualtran_0_7_0"
    objective: str = "minimize_spacetime_volume_qubit_seconds"

    failure_budget: float = Field(..., gt=0.0, le=1.0)
    physical_qubit_cap: int = Field(..., ge=1)

    # Design domains (odd code distances; l2 > l1 enforced by the search).
    l1_distance_min: int = Field(..., ge=3)
    l1_distance_max: int = Field(..., ge=3)
    l2_distance_max: int = Field(..., ge=5)
    n_factory_choices: list[int] = Field(..., min_length=1)
    data_block_distance_min: int = Field(..., ge=3)
    data_block_distance_max: int = Field(..., ge=3)

    # Broad public priors on the held-out hardware (two-world-safe planning aids).
    phys_err_prior_min: float = Field(..., gt=0.0)
    phys_err_prior_max: float = Field(..., gt=0.0)
    cycle_time_us_prior_min: float = Field(..., gt=0.0)
    cycle_time_us_prior_max: float = Field(..., gt=0.0)

    algorithm_files: str = "algorithm_summary.json"


class FtqcMpsQpePlanningMeta(_Strict):
    """Public static-instance metadata for MPS-initialized QPE planning."""

    capability: Literal["mps_qpe_resource_planning"] = "mps_qpe_resource_planning"
    cost_model_version: Literal["berry_mps_qpe_logical_v2"]
    objective: Literal["minimize_total_logical_toffoli_count"]
    case_file: str = "case.json"
    candidate_descriptors_file: str = "candidates.json"
    block_encoding_descriptors_file: str = "block_encodings.json"
    method_digest_file: str = "paper_method_digest.md"
    submission_contract_file: str = "submission_contract.md"
    case_schema_file: str = "case.schema.json"
    candidate_schema_file: str = "candidates.schema.json"
    block_encoding_schema_file: str = "block_encodings.schema.json"
    overlap_table_schema_file: str = "overlap_table.schema.json"
    toy_mps_schema_file: str = "toy_mps.schema.json"
    toy_circuit_schema_file: str = "toy_circuit.schema.json"
    result_schema_file: str = "result.schema.json"
    toy_mps_example_file: str = "examples/toy_mps.json"
    toy_circuit_example_file: str = "examples/toy_circuit.json"
    window_half_support: int = Field(1024, ge=16)


class PublicFtqcSpec(_Strict):
    """Public device spec returned by ``get_device_spec``."""

    schema_version: int = 1
    device_id: str
    qtype: Literal["ftqc_resource_estimation"] = "ftqc_resource_estimation"
    task_id: str
    device_class: str = "ftqc_resource_estimation"

    # Legacy v1 fields remain parseable as explicit nulls. No active capability
    # consumes them.
    molecule: str | None = None
    basis: str | None = None
    decomposition: Literal["pauli_lcu"] | None = None
    fermion_mapping: Literal["jordan_wigner"] | None = None
    pauli_string_order: str | None = None
    system_qubits: int | None = Field(None, ge=1)
    electrons: int | None = Field(None, ge=0)
    spatial_orbitals: int | None = Field(None, ge=1)
    bliss_gauge: str | None = None
    num_bliss_parameters: int | None = Field(None, ge=1)
    bliss_parameter_range_hartree: list[float] | None = Field(None, min_length=2, max_length=2)
    affine_basis_columns: list[str] | None = Field(None, min_length=1)
    raw_identity_coefficient_hartree: float | None = None
    canonical_zero_threshold_hartree: float | None = Field(None, gt=0)
    total_error_budget_hartree: float | None = Field(None, gt=0)
    peak_logical_qubit_cap: int | None = Field(None, ge=1)
    lambda_tie_tolerance_hartree: float | None = Field(None, gt=0)
    instance_files: FtqcInstanceFiles | None = None

    cost_model_version: str
    dynamics_primitive: str
    evaluator_call_cap: int = Field(..., ge=1)
    objective: str
    target_wall_clock_runtime_s: int = Field(..., ge=1)
    notes: str = ""

    physical_estimation: FtqcPhysicalEstimationMeta | None = None
    mps_qpe_planning: FtqcMpsQpePlanningMeta | None = None

    @model_validator(mode="after")
    def validate_family_exclusivity(self) -> PublicFtqcSpec:
        if self.physical_estimation is not None and self.mps_qpe_planning is not None:
            raise ValueError("physical_estimation and mps_qpe_planning are mutually exclusive")
        if self.mps_qpe_planning is not None:
            legacy_fields = (
                self.molecule,
                self.basis,
                self.decomposition,
                self.fermion_mapping,
                self.system_qubits,
                self.bliss_gauge,
                self.num_bliss_parameters,
                self.instance_files,
            )
            if any(value is not None for value in legacy_fields):
                raise ValueError("mps_qpe_planning cannot carry legacy BLISS-family metadata")
        return self


# ---------- Hidden truth (only inside qsim) ----------


class StaleFtqcNotebook(_Strict):
    """Stale prior + trap recommendations.

    The optional scalar fields are retained for v1 config compatibility; active
    tasks leave them unset.
    """

    mu1_hartree: float | None = None
    mu2_hartree: float | None = None
    scalar_lambda_hartree: float | None = None
    recommendations: list[str] = Field(default_factory=list)
    note: str = ""
    reliability: str = "operator notes carried over from a prior compilation study"


class FtqcPhysicalOptimalDesign(_Strict):
    """Held-out minimum-spacetime-volume CCZ2T design + realized cost."""

    distillation_l1_d: int
    distillation_l2_d: int
    n_factories: int
    data_block_d: int
    spacetime_volume_qubit_seconds: float
    footprint_physical_qubits: int
    failure_prob: float


class FtqcPhysicalThresholds(_Strict):
    """Quality-tier thresholds for the physical-estimation family.

    Volume tiers are multiplicative over the held-out optimum. ``cycle_time_rel_tol``
    binds the agent's inferred cycle time; ``phys_err_bracket_ratio`` gold-gates the
    reported physical-error bracket (max/min ≤ ratio, and it must contain truth).
    """

    volume_pass_ratio: float = 1.60
    volume_bronze_ratio: float = 1.15
    cycle_time_rel_tol: float = 1e-2
    volume_bind_rel_tol: float = 1e-3
    phys_err_bracket_ratio: float = 4.0


class FtqcPhysicalHiddenRef(_Strict):
    """Held-out hardware + reference optimum for the physical-estimation family."""

    phys_err: float = Field(..., gt=0.0)
    cycle_time_us: float = Field(..., gt=0.0)
    optimal_design: FtqcPhysicalOptimalDesign
    thresholds: FtqcPhysicalThresholds = FtqcPhysicalThresholds()


class HiddenFtqcConfig(_Strict):
    """Hidden held-out reference for ``ftqc_resource_estimation``. Inside qsim only."""

    schema_version: int = 1
    device_id: str
    qtype: Literal["ftqc_resource_estimation"] = "ftqc_resource_estimation"
    task_id: str

    public_material_digests: dict[str, str] = Field(default_factory=dict)
    stale_lab_notebook: StaleFtqcNotebook = StaleFtqcNotebook()

    physical_estimation: FtqcPhysicalHiddenRef | None = None
