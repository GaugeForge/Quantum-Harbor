"""Pydantic schemas for ``time_budgeted_hamlearn_10q``.

Two schemas live here, mirroring the VQE split discipline (``extra='forbid'``
everywhere so unknown fields cannot smuggle in):

* :class:`HamLearnHiddenScorer` — the hidden scorer payload (never seen by the
  agent): the true 111-coefficient vector, coefficient-error thresholds,
  support set, budget, and the public-material digests that bind it to the
  public task files.
* :class:`HamLearnAnswer` — the agent-facing final-answer schema,
  with the §8c enum literals and §8d plausibility-band validators.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from qiqcbench.qsim.hidden_dynamics.time_budgeted_hamlearn_10q.dictionary import (
    N_TERMS,
    TERM_ORDER,
)
from qiqcbench.qsim.hidden_dynamics.time_budgeted_hamlearn_10q.standalone_contract import (
    validate_hidden_scorer_payload,
)

__all__ = [
    "HAMILTONIAN_CONVENTION",
    "HamLearnActivitySummary",
    "HamLearnAnswer",
    "HamLearnHiddenScorer",
    "HamLearnTrajectoryReview",
    "HamLearnVerifiableActivity",
]

HAMILTONIAN_CONVENTION = "U(t)=exp(-i t/2 sum omega_P P)"

_PROTOCOLS = (
    "fixed_design",
    "adaptive_design",
    "mixed_design",
    "other",
)
_FIT_MODELS = (
    "linear",
    "nonlinear",
    "probabilistic",
    "hybrid",
    "other",
)


class HamLearnHiddenScorer(BaseModel):
    """Hidden scorer payload for the Hamiltonian-learning task.

    Fields:
      * ``public_material_digests``: filename -> sha256 hex of the exact bytes
        written into ``configs/task_materials/<task>/public/``. Binds the
        hidden scorer to the public dictionary / spec / notebook it derives
        from (reuses the VQE sha256 file-digest convention).
      * ``omega_star``: the true 111-coefficient vector, in ``TERM_ORDER``.
      * ``linf_pass/bronze/silver/gold``: §9a thresholds (descending).
      * ``relative_l2_pass``: §9a aggregate coefficient-error threshold.
      * ``support_tau`` / ``support_terms``: §9b diagnostic threshold and the
        set of true terms with ``|omega| > tau`` (23 terms).
      * ``budget_us`` / ``max_probe_rows``: §9e/§11 evolution-time budget and
        accepted-row cap.
      * ``oracle_noise_shots``: fixed internal repetitions per probe (§5d) —
        recorded so the scorer/materials stay consistent with the oracle.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal[3]
    public_material_digests: dict[str, str]
    omega_star: list[float] = Field(..., min_length=N_TERMS, max_length=N_TERMS)
    linf_pass: float
    linf_bronze: float
    linf_silver: float
    linf_gold: float
    relative_l2_pass: float
    support_tau: float
    support_terms: list[str]
    budget_us: float
    max_probe_rows: int
    oracle_noise_shots: int

    @model_validator(mode="before")
    @classmethod
    def _strict_hidden_contract(cls, value: object) -> dict[str, object]:
        return validate_hidden_scorer_payload(
            value,
            term_order=TERM_ORDER,
            coefficient_bound=0.60,
        )


class HamLearnVerifiableActivity(BaseModel):
    """Agent claims that the verifier recomputes from qsim evidence."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    accepted_probe_rows: int = Field(..., ge=1, le=600)
    total_accepted_evolution_time_us: float = Field(..., ge=0.0)
    min_accepted_evolve_time_us: float = Field(..., ge=0.010, le=2.500)
    max_accepted_evolve_time_us: float = Field(..., ge=0.010, le=2.500)

    @model_validator(mode="after")
    def _time_range_is_ordered(self) -> HamLearnVerifiableActivity:
        if self.min_accepted_evolve_time_us > self.max_accepted_evolve_time_us:
            raise ValueError("minimum accepted evolve time cannot exceed maximum")
        return self


class HamLearnTrajectoryReview(BaseModel):
    """Human-reviewable method and chronological trajectory; never numerically scored."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    protocol: Literal[_PROTOCOLS]  # type: ignore[valid-type]
    fit_model: Literal[_FIT_MODELS]  # type: ignore[valid-type]
    trajectory_details: str = Field(..., min_length=1, max_length=8192)

    @field_validator("trajectory_details")
    @classmethod
    def _trajectory_details_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("trajectory_details must not be blank")
        return value


class HamLearnActivitySummary(BaseModel):
    """Two-layer activity record submitted with the scientific estimate."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal[2]
    verifiable: HamLearnVerifiableActivity
    trajectory: HamLearnTrajectoryReview


class HamLearnAnswer(BaseModel):
    """Agent final-answer schema. ``extra='forbid'`` per the split."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    # --- required (§8a) ---
    task_id: Literal["time_budgeted_hamlearn_10q"]
    hamiltonian_convention: Literal["U(t)=exp(-i t/2 sum omega_P P)"]
    term_order_convention: Literal["one_local_xyz_then_edge_local_xx_xy_xz_yx_yy_yz_zx_zy_zz"]
    num_terms: Literal[111]
    omega_vector_raw_rad_per_us: list[float] = Field(..., min_length=N_TERMS, max_length=N_TERMS)
    omega_vector_raw_rad_per_us_1sigma: list[float] = Field(
        ..., min_length=N_TERMS, max_length=N_TERMS
    )
    activity_summary: HamLearnActivitySummary

    # --- optional (§8b) ---
    omega_vector_refined_rad_per_us: list[float] | None = Field(
        default=None, min_length=N_TERMS, max_length=N_TERMS
    )
    omega_vector_refined_rad_per_us_1sigma: list[float] | None = Field(
        default=None, min_length=N_TERMS, max_length=N_TERMS
    )
    support_terms_detected: list[str] | None = None
    linf_self_estimate_rad_per_us: float | None = None
    notebook_model_rejected: bool | None = None
    notebook_rejection_reason: str | None = None
    heldout_probe_error_summary: str | None = None

    @field_validator("omega_vector_raw_rad_per_us", "omega_vector_refined_rad_per_us")
    @classmethod
    def _coeff_band(cls, v: list[float] | None) -> list[float] | None:
        # §8d: each coefficient in [-0.75, 0.75] (public bound + slack).
        if v is not None and any(abs(x) > 0.75 + 1e-9 for x in v):
            raise ValueError("coefficient outside plausibility band [-0.75, 0.75]")
        return v

    @field_validator(
        "omega_vector_raw_rad_per_us_1sigma",
        "omega_vector_refined_rad_per_us_1sigma",
    )
    @classmethod
    def _sigma_band(cls, v: list[float] | None) -> list[float] | None:
        # §8d: each 1-sigma in (0, 0.30].
        if v is not None and any(not (0.0 < x <= 0.30 + 1e-9) for x in v):
            raise ValueError("1-sigma outside plausibility band (0, 0.30]")
        return v
