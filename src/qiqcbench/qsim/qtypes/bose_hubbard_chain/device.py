"""Public + hidden device config types for the Bose-Hubbard chain qtype.

A ``bose_hubbard_chain`` device is a 1D lattice of bosonic sites (transmon-like
nonlinear oscillators) operated as an analog Hamiltonian simulator: prepare a
product superposition, evolve under a fixed Bose-Hubbard Hamiltonian, and read
out ``sigma^X`` / ``sigma^Y`` quadratures per site.

Public / hidden are STRUCTURALLY separated. The PUBLIC
spec carries nominal on-site potentials, hoppings, and interaction values plus
an envelope within which the realized coefficients may differ. The frozen
per-site / per-bond / interaction offsets that turn those nominal numbers into
the realized Hamiltonian live in the HIDDEN config, alongside the coherence
time, readout confusion, RNG seed, and lab notebook.

This inversion is deliberate. Schema v1 published the
realized Hamiltonian, which made the eigen-spectrum a deterministic function of
public bytes: an agent could diagonalize `get_device_spec` and skip the
experiment entirely. Moving the drift behind the boundary is the only change
that makes the answer non-derivable, so more truth — not less — now lives
inside qsim.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Public spec (visible to agent) ----------


class PublicSite(_Strict):
    id: int = Field(..., ge=0)
    on_site_potential_mhz: float = Field(
        ...,
        description="Nominal on-site potential mu_n / 2pi in MHz, as of the last calibration.",
    )


class PublicDriftEnvelope(_Strict):
    """Published bound on how far each nominal coefficient may have drifted.

    Absolute MHz bounds, applied independently per site / per bond. The realized
    coefficient lies in ``nominal +/- bound``; where inside is hidden truth. The
    envelope is published so a drift fit is a tractable inverse problem rather
    than an unbounded guess — it never narrows the search to a single answer.
    """

    on_site_potential_mhz: float = Field(..., ge=0)
    hopping_mhz: float = Field(..., ge=0)
    interaction_u_mhz: float = Field(..., ge=0)


class BoseHubbardResourceBudget(_Strict):
    """Public compute, evidence, and transport limits; none is a score term."""

    max_experiment_jobs: int = Field(256, gt=0)
    max_sweep_points_per_call: int = Field(2_049, gt=0)
    max_raw_shot_records_per_call: int = Field(2**21, gt=0)
    max_raw_shot_records_total: int = Field(2**26, gt=0)
    max_ingress_request_bytes: int = Field(256 * 2**10, gt=0)
    max_metadata_calls: int = Field(256, gt=0)
    max_job_result_polls: int = Field(4_096, gt=0)
    max_final_answer_serialized_bytes: int = Field(64 * 2**10, gt=0)
    max_final_answer_submissions: int = Field(4, gt=0)
    max_answer_string_characters: int = Field(8_192, gt=0)

    @model_validator(mode="after")
    def _consistent_limits(self) -> BoseHubbardResourceBudget:
        if self.max_raw_shot_records_per_call > self.max_raw_shot_records_total:
            raise ValueError(
                "max_raw_shot_records_per_call cannot exceed max_raw_shot_records_total"
            )
        return self


class PublicBoseHubbardSpec(_Strict):
    """Hardware-manual-style public spec for the Bose-Hubbard chain qtype.

    The Hamiltonian *form* is public: ``H = sum_n mu_n a+_n a_n + (U/2) sum_n
    n_n(n_n-1) + J sum_n (a+_{n+1} a_n + h.c.)``, all coefficients in MHz. The
    coefficients published here are **nominal**. The realized device
    coefficients may differ within ``calibration_drift_envelope_mhz``; their
    exact values are hidden truth.
    """

    schema_version: Literal[3] = 3
    device_id: str
    qtype: Literal["bose_hubbard_chain"] = "bose_hubbard_chain"
    sites: list[PublicSite] = Field(..., min_length=1)
    hopping_mhz: list[float] = Field(
        ...,
        description="Nominal nearest-neighbour hopping J/2pi per bond (length len(sites)-1).",
    )
    interaction_u_mhz: float = Field(..., description="Nominal on-site interaction U/2pi in MHz.")
    calibration_drift_envelope_mhz: PublicDriftEnvelope | None = Field(
        None,
        description=(
            "Published bound on the unknown drift of each nominal coefficient. "
            "None means the device claims no drift since calibration."
        ),
    )
    max_excitations: int = Field(
        2, ge=1, description="Maximum total photon number the device supports."
    )
    max_evolution_time_ns: float = Field(2000.0, gt=0)
    time_resolution_ns: float = Field(1.0, gt=0)
    measurement_return: Literal["bitstring"] = "bitstring"
    max_shots: int = Field(100_000, gt=0)
    budget: BoseHubbardResourceBudget = BoseHubbardResourceBudget()
    notes: str = "Calibrations may be stale; verify before relying on them."

    @model_validator(mode="after")
    def _validate_resource_shape(self) -> PublicBoseHubbardSpec:
        site_ids = [site.id for site in self.sites]
        if sorted(site_ids) != list(range(len(self.sites))):
            raise ValueError("site IDs must be unique and contiguous from zero")
        if len(self.hopping_mhz) != len(self.sites) - 1:
            raise ValueError("hopping_mhz must have len(sites)-1 entries")
        if self.max_shots > self.budget.max_raw_shot_records_per_call:
            raise ValueError("max_shots cannot exceed budget.max_raw_shot_records_per_call")
        return self


# ---------- Hidden truth (only inside qsim) ----------


class HiddenBoseHubbardReadout(_Strict):
    id: int = Field(..., ge=0)
    p_0_to_1: float = Field(..., ge=0, le=1)
    p_1_to_0: float = Field(..., ge=0, le=1)


class HiddenHamiltonianDrift(_Strict):
    """Frozen offsets from the published nominal Hamiltonian.

    ``realized = nominal + drift``, coefficient by coefficient, in MHz. These
    are **hand-picked and frozen per device**, never drawn from the RNG seed at
    run time. Each offset must lie inside the device's published
    ``calibration_drift_envelope_mhz``, or the public contract is a lie.
    """

    on_site_potential_mhz: list[float] = Field(
        ..., min_length=1, description="d_mu_n per site, ordered by site id."
    )
    hopping_mhz: list[float] = Field(
        ..., min_length=1, description="d_J_b per bond, ordered by bond index."
    )
    interaction_u_mhz: float = Field(0.0, description="d_U.")


class StaleBoseHubbardNotebookEntry(_Strict):
    """Optional stale/optimistic calibration written into the lab notebook."""

    last_calibrated: str | None = None
    coherence_t2_ns_claim: float | None = None
    spectral_linewidth_mhz_claim: float | None = None
    expected_two_photon_energies_mhz: list[float] | None = None
    notes: str = ""


class HiddenBoseHubbardConfig(_Strict):
    """Hidden truth for the Bose-Hubbard chain qtype. Lives only inside qsim.

    Coherence/noise plus the calibration drift that separates the realized
    Hamiltonian from the nominal one the agent is shown.
    """

    schema_version: Literal[2] = 2
    device_id: str
    qtype: Literal["bose_hubbard_chain"] = "bose_hubbard_chain"
    seed: int
    t2_ns: float = Field(
        ..., gt=0, description="Effective in-chain coherence time; sets the linewidth."
    )
    hamiltonian_drift: HiddenHamiltonianDrift | None = Field(
        None,
        description=(
            "Frozen offsets from the public nominal Hamiltonian. None means the "
            "realized Hamiltonian equals the published one (derivable by "
            "diagonalization — only appropriate where that is not the science)."
        ),
    )
    readout: list[HiddenBoseHubbardReadout] = Field(..., min_length=1)
    stale_lab_notebook: StaleBoseHubbardNotebookEntry = StaleBoseHubbardNotebookEntry()


__all__ = [
    "BoseHubbardResourceBudget",
    "HiddenBoseHubbardConfig",
    "HiddenBoseHubbardReadout",
    "HiddenHamiltonianDrift",
    "PublicBoseHubbardSpec",
    "PublicDriftEnvelope",
    "PublicSite",
    "StaleBoseHubbardNotebookEntry",
]
