"""Public + hidden device config for the ``kitaev_chain`` qtype.

A ``kitaev_chain`` device is a 1D array of gate-defined quantum dots coupled
through superconductor-semiconductor hybrid segments (a semiconductor Kitaev
chain; Bordin et al. arXiv:2402.19382, van Loo et al. arXiv:2507.01606). Each
bond ``b`` carries an elastic-co-tunnelling hopping ``t_b`` and a
crossed-Andreev pairing ``Delta_b = |Delta_b| e^{i phi_b}``; each dot has an
on-site potential ``mu_i``. Operated through programmable gates (the dot/hybrid
voltages) and **charge readout** (global quantum capacitance + a local charge
sensor).

Public / hidden are STRUCTURALLY separated. The public
spec advertises the *design intent* (a uniform sweet spot of ``n_sites`` bonds,
nominal coupling, claimed protection). The HIDDEN config holds the real
per-bond couplings + phases (the protection-breaking defect lives here), the
nuisance disorder scale, the readout SNRs, the poisoning temperature, and the
RNG seed. The defect itself is the answer, so the hidden truth is never
snapshotted to the agent (tasks set ``QSIM_SNAPSHOT_HIDDEN_TRUTH=0``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Public spec (visible to agent) ----------


class PublicKitaevChainSpec(_Strict):
    """Hardware-manual-style public spec. Advertises the design intent only."""

    schema_version: int = 1
    device_id: str
    qtype: Literal["kitaev_chain"] = "kitaev_chain"
    n_sites: int = Field(..., ge=2, description="Number of quantum dots (bonds = n_sites-1).")
    nominal_coupling_ueV: float = Field(
        ...,
        gt=0,
        description="Advertised uniform sweet-spot coupling |t_b|=|Delta_b| in micro-eV.",
    )
    claimed_protection_exponent: float = Field(
        ...,
        description="Advertised protection exponent (design value, typically n_sites).",
    )
    mu_window_ueV: float = Field(
        ...,
        gt=0,
        description="Maximum |dot detuning| (micro-eV) the gate scans may request.",
    )
    claimed_readout_fidelity: float = Field(
        0.97, ge=0, le=1, description="Claimed single-shot parity readout fidelity."
    )
    max_shots: int = Field(100_000, gt=0)
    # --- optional Clifford-RB register fields (majorana_pulse_control capability) ---
    rb_n_logical_qubits: int | None = Field(
        None, description="Logical Majorana qubits the register hosts (RB task)."
    )
    rb_infidelity_pass_threshold: float | None = Field(
        None,
        ge=0,
        le=1,
        description=(
            "Maximum accepted one-sided leakage-inclusive infidelity quality bound "
            "per compiled Clifford under the task's short-sequence estimator."
        ),
    )
    notes: str = (
        "Semiconductor Kitaev chain operated through programmable gates and "
        "charge readout (global quantum capacitance + local charge sensor). "
        "Couplings reflect fabrication design intent; verify per-bond before "
        "trusting protection claims."
    )


# ---------- Hidden truth (only inside qsim) ----------


class StaleKitaevNotebookEntry(_Strict):
    """Optional stale/optimistic calibration written into the lab notebook."""

    last_calibrated: str | None = None
    uniform_sweet_spot_claim: bool | None = None
    coupling_ueV_claim: float | None = None
    protection_exponent_claim: float | None = None
    parity_lifetime_ms_claim: float | None = None
    # Stale pulse-gate calibration (the RB-task trap): a drive rate that no longer
    # matches the device, so gates built on it over/under-rotate unless recalibrated.
    drive_rate_ueV_per_amp_claim: float | None = None
    bulk_gap_ueV_claim: float | None = None
    notes: str = ""


class HiddenKitaevChainConfig(_Strict):
    """Hidden truth for the Kitaev-chain qtype. Lives only inside qsim.

    ``bond_t_ueV`` / ``bond_delta_ueV`` are the real coupling magnitudes per bond;
    ``bond_phase_rad`` is the pairing phase ``phi_b`` per bond. A clean sweet-spot
    bond has ``t==delta`` and ``phi==0``; the protection-breaking defect is a
    single bond with ``t!=delta`` (amplitude) OR ``phi!=0`` (phase).
    """

    schema_version: int = 1
    device_id: str
    qtype: Literal["kitaev_chain"] = "kitaev_chain"
    seed: int
    bond_t_ueV: list[float] = Field(..., min_length=1)
    bond_delta_ueV: list[float] = Field(..., min_length=1)
    bond_phase_rad: list[float] = Field(..., min_length=1)
    mu_disorder_sigma_ueV: float = Field(
        0.0, ge=0, description="Std of fresh per-dot potential disorder (nuisance)."
    )
    global_snr: float = Field(
        ..., gt=0, description="Global quantum-capacitance single-shot SNR at the resonance."
    )
    local_snr: float = Field(
        ..., gt=0, description="Local charge-sensor single-shot SNR at full parity contrast."
    )
    poisoning_temp_ueV: float = Field(
        ..., gt=0, description="Phenomenological poisoning temperature setting parity polarization."
    )
    spectroscopy_noise_ueV: float = Field(
        ..., gt=0, description="Gaussian measurement noise on sub-chain bulk-gap spectroscopy."
    )
    # --- optional hidden pulse-gate parameters (majorana_pulse_control capability) ---
    # Two coupled topological segments encode one qubit inside a fixed-total-parity
    # manifold. These parameters define its effective pulse dynamics. The real drive
    # rate is the hidden calibration target; the bulk gap sets coherent leakage.
    pulse_drive_rate_ueV: float | None = Field(
        None, description="Realized Rabi rate at amp=1 (the hidden calibration target)."
    )
    pulse_e_gap_ueV: float | None = Field(
        None, description="Leakage-level (bulk quasiparticle) energy ~ 2t."
    )
    pulse_leakage_lambda: float | None = Field(
        None, description="|1>-|2> drive coupling relative to |0>-|1| (leakage strength)."
    )
    pulse_charge_noise_sigma_ueV: float | None = Field(
        None, description="Per-shot detuning fluctuation (dephasing)."
    )
    pulse_amp_noise_frac: float | None = Field(
        None, description="Per-shot fractional drive-amplitude fluctuation."
    )
    pulse_poisoning_per_ns: float | None = Field(
        None,
        description=("Poisoning rate per ns out of the fixed-parity computational manifold."),
    )
    stale_lab_notebook: StaleKitaevNotebookEntry = StaleKitaevNotebookEntry()

    @model_validator(mode="after")
    def _check_bond_lengths(self) -> HiddenKitaevChainConfig:
        n_bonds = len(self.bond_t_ueV)
        if not (len(self.bond_delta_ueV) == n_bonds and len(self.bond_phase_rad) == n_bonds):
            raise ValueError("bond_t_ueV, bond_delta_ueV, bond_phase_rad must have equal length")
        return self


__all__ = [
    "HiddenKitaevChainConfig",
    "PublicKitaevChainSpec",
    "StaleKitaevNotebookEntry",
]
