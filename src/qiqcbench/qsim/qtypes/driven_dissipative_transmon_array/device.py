"""Public + hidden device config types for the driven-dissipative transmon-array qtype.

A ``driven_dissipative_transmon_array`` device is an open chain of hard-core
transmon sites (``q0-q1-q2-q3``) operated as an *open-system* analog simulator:
the agent picks one adjacent pair, drives it with energy-selective local pump and
loss reservoirs (engineered dissipation), and autonomously stabilizes the
single-excitation Bell singlet ``|-> = (|ge>-|eg>)/sqrt(2)``.

Public / hidden are STRUCTURALLY separated: both classes
use ``extra="forbid"`` so a hidden-truth field can never be absorbed by the public
spec and vice versa. Mirrors the split in ``bose_hubbard_chain/device.py`` and
``gmon_ring_3q/device.py``.

Unlike the (closed-system) Bose-Hubbard chain, the *realized Hamiltonian itself is
hidden* here: the per-bond hopping ``J``, the per-site detunings (ac-Stark shifts),
the reservoir linewidth ``kappa``, the per-site lifetimes / thermal populations,
the collective Purcell rate, and the asymmetric readout all live in the HIDDEN
config. The public spec advertises only control ranges, the selectable pairs, and
nominal/claimed ranges — never which pair is the hidden best pair.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Public spec (visible to agent) ----------


class DdtaControlLimits(_Strict):
    """Advertised reservoir/timing control envelope the agent designs against."""

    pump_detuning_mhz_range: tuple[float, float] = (-12.0, 12.0)
    loss_detuning_mhz_range: tuple[float, float] = (-12.0, 12.0)
    coupling_mhz_range: tuple[float, float] = Field(
        (0.0, 1.2),
        description="Allowed pump/loss reservoir coupling g_s, g_d / 2pi in MHz.",
    )
    max_duration_us: float = Field(12.0, gt=0)
    duration_resolution_us: float = Field(0.05, gt=0)
    supported_bases: list[str] = ["x", "y", "z"]


class DdtaMeasurementModel(_Strict):
    """Public measurement-chain semantics; numerical assignment rates stay hidden."""

    initial_state_preparation: Literal["exact_computational_basis"] = "exact_computational_basis"
    analysis_rotations: Literal["ideal"] = "ideal"
    born_sampling: Literal["ideal_after_analysis_rotation"] = "ideal_after_analysis_rotation"
    readout_assignment_stage: Literal["post_born_sampling"] = "post_born_sampling"
    readout_assignment_channel: Literal[
        "independent_site_dependent_asymmetric_binary_assignment"
    ] = "independent_site_dependent_asymmetric_binary_assignment"
    readout_assignment_parameters: Literal["hidden"] = "hidden"
    returned_bitstrings: Literal["post_readout_raw"] = "post_readout_raw"


class DdtaResourceBudget(_Strict):
    """Public transport/evidence bounds; none of these fields is a score term."""

    max_experiment_jobs: int = Field(
        4_096,
        gt=0,
        description="Trial-cumulative accepted stabilization jobs across both run tools.",
    )
    max_sweep_points_per_call: int = Field(
        4_096,
        gt=0,
        description="Per-call duration-grid bound for run_stabilization_sweep.",
    )
    max_raw_shot_records_per_call: int = Field(
        2**25,
        gt=0,
        description=(
            "Per-call bound on duration points x shots. One duration-point shot is one "
            "returned raw bitstring record, independent of measured-site count."
        ),
    )
    max_raw_shot_records_total: int = Field(
        2**27,
        gt=0,
        description=(
            "Trial-cumulative raw bitstring records across run_stabilization and "
            "run_stabilization_sweep."
        ),
    )
    max_ingress_request_bytes: int = Field(4 * 2**20, gt=0)
    max_metadata_calls: int = Field(16_384, gt=0)
    max_job_result_polls: int = Field(32_768, gt=0)
    max_final_answer_serialized_bytes: int = Field(256 * 2**10, gt=0)
    max_final_answer_submissions: int = Field(8, gt=0)
    max_answer_string_characters: int = Field(16_384, gt=0)

    @model_validator(mode="after")
    def _validate_record_limits(self) -> DdtaResourceBudget:
        if self.max_raw_shot_records_per_call > self.max_raw_shot_records_total:
            raise ValueError(
                "max_raw_shot_records_per_call cannot exceed max_raw_shot_records_total"
            )
        return self


class PublicDdtaSpec(_Strict):
    """Hardware-manual-style public spec for the driven-dissipative array qtype.

    Only control ranges and nominal/claimed ranges are public; the realized
    Hamiltonian and noise model are hidden. ``advertised_*_range`` give the
    band the true value lives in, never the exact per-bond / per-site truth.
    """

    schema_version: int = 3
    device_id: str
    qtype: Literal["driven_dissipative_transmon_array"] = "driven_dissipative_transmon_array"
    qubits: list[str] = Field(..., min_length=2, description="Ordered open chain, e.g. q0..q3.")
    selectable_pairs: list[str] = Field(
        ...,
        min_length=1,
        description='Adjacent pairs the agent may select, e.g. "q0,q1". Site i maps to qubit q{i}.',
    )
    control_limits: DdtaControlLimits = DdtaControlLimits()
    advertised_hopping_mhz_range: tuple[float, float] = Field(
        (5.4, 6.6), description="Per-bond hopping J/2pi band; true per-bond values are hidden."
    )
    advertised_kappa_mhz_range: tuple[float, float] = Field(
        (1.2, 1.8), description="Reservoir linewidth kappa/2pi band; true value is hidden."
    )
    allowed_initial_states: list[str] = ["gg", "ge", "eg", "ee"]
    measurement_return: Literal["bitstring"] = "bitstring"
    measurement_model: DdtaMeasurementModel = DdtaMeasurementModel()
    max_shots: int = Field(100_000, gt=0)
    budget: DdtaResourceBudget = DdtaResourceBudget()
    notes: str = "Calibrations may be stale. Verify before relying on it."

    @model_validator(mode="after")
    def _validate_single_run_record_limit(self) -> PublicDdtaSpec:
        if self.max_shots > self.budget.max_raw_shot_records_per_call:
            raise ValueError("max_shots cannot exceed budget.max_raw_shot_records_per_call")
        return self


# ---------- Hidden truth (only inside qsim) ----------


class HiddenDdtaReadout(_Strict):
    """Asymmetric, site-dependent computational-basis readout confusion."""

    id: str
    p_0_to_1: float = Field(..., ge=0, le=1)
    p_1_to_0: float = Field(..., ge=0, le=1)


class StaleDdtaNotebookEntry(_Strict):
    """Stale/optimistic calibration claims surfaced via get_lab_notebook (the trap)."""

    last_calibrated: str | None = None
    recommended_pair: str | None = Field(
        None, description="Notebook-recommended pair (no longer the hidden best pair)."
    )
    recommended_delta_s_mhz: float | None = None
    recommended_delta_d_mhz: float | None = None
    recommended_g_s_mhz: float | None = None
    recommended_g_d_mhz: float | None = None
    recommended_duration_us: float | None = None
    expected_f_minus: float | None = Field(
        None, description="Optimistic expected |-> fidelity (stale; real raw value is lower)."
    )
    readout_fidelity_claim: float | None = Field(
        None, description="Symmetric optimistic readout-fidelity claim (hides asymmetry)."
    )
    notes: str | None = None


class HiddenDdtaConfig(_Strict):
    """Hidden truth for the driven-dissipative array qtype. Lives only inside qsim.

    The realized Hamiltonian (hopping + detunings), reservoir linewidth, intrinsic
    lifetimes, thermal populations, collective Purcell rate, and readout confusion
    are ALL hidden. The number of sites is read from ``len(site_t1_us)``; the chain
    is open with ``len(site_t1_us) - 1`` bonds. The best pair is NOT stored — it
    emerges from this physical truth (per-bond J / per-site T1 / thermal asymmetry
    plus the edge-vs-middle spectator-leakage count).
    """

    schema_version: int = 1
    device_id: str
    qtype: Literal["driven_dissipative_transmon_array"] = "driven_dissipative_transmon_array"
    seed: int
    hopping_mhz: list[float] = Field(
        ...,
        min_length=1,
        description="Per-bond nearest-neighbour hopping J/2pi (length n_sites-1).",
    )
    site_detuning_mhz: list[float] = Field(
        ...,
        min_length=2,
        description="Per-site static detuning Delta_i/2pi (residual + representative ac-Stark).",
    )
    kappa_mhz: float = Field(..., gt=0, description="Reservoir Lorentzian linewidth kappa/2pi.")
    site_t1_us: list[float] = Field(..., min_length=2, description="Per-site intrinsic T1 (us).")
    site_tphi_us: list[float] = Field(
        ..., min_length=2, description="Per-site intrinsic pure-dephasing time Tphi (us)."
    )
    site_thermal_pop: list[float] = Field(
        ...,
        min_length=2,
        description="Per-site reservoir thermal population n_th (broadband pump leak); pair-dependent.",
    )
    collective_purcell_mhz: float = Field(
        0.0,
        ge=0,
        description=(
            "Collective Purcell decay rate / 2pi on the selected pair via the shared "
            "readout line. Annihilates |-> but makes |+> superradiant (App F: |+> shorter-lived)."
        ),
    )
    reservoir_rate_scale: float = Field(
        1.0,
        gt=0,
        description="Calibration constant mapping reservoir coupling g (MHz) to an effective rate.",
    )
    ac_stark_mhz_per_g2: float = Field(
        0.0,
        ge=0,
        description=(
            "Drive-induced ac-Stark coefficient: each pair site shifts by "
            "ac_stark_mhz_per_g2 * (g_s^2 + g_d^2) MHz, so the optimal detunings drift "
            "with the reservoir coupling (a hidden 2D co-optimization ridge)."
        ),
    )
    leak_scale: float = Field(
        0.0,
        ge=0,
        description=(
            "Off-resonant strong-drive leakage coefficient: rate ~ leak_scale * g^4, "
            "negligible at low g but dominant at high g, giving the fidelity an "
            "interior-optimal coupling (max g is not best)."
        ),
    )
    readout: list[HiddenDdtaReadout] = Field(..., min_length=2)
    stale_lab_notebook: StaleDdtaNotebookEntry = StaleDdtaNotebookEntry()


__all__ = [
    "DdtaControlLimits",
    "DdtaMeasurementModel",
    "DdtaResourceBudget",
    "HiddenDdtaConfig",
    "HiddenDdtaReadout",
    "PublicDdtaSpec",
    "StaleDdtaNotebookEntry",
]
