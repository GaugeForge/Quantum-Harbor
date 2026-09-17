"""Public spec + hidden config for the ``transmon_repeater_array`` qtype.

Structural public/hidden split (both ``extra="forbid"``): the public spec carries
geometry, the LOCC control envelope, *claimed* fidelities, and budgets; the hidden
config carries the real noise model (-> physics.RepeaterParams) and the stale
notebook. The saturation fidelity F_sat is emergent from the hidden params, never
stored.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- public ---------------------------------------------------------------


class RepeaterBudgets(_Strict):
    pairs_cap: int = Field(..., ge=1)  # total distributed pairs consumable in a run
    experiment_rows: int = Field(..., ge=1)  # total experiment rows across the run
    rows_per_batch: int = Field(..., ge=1)
    max_shots_per_row: int = Field(..., ge=1)
    max_depth: int = Field(..., ge=0)


class PublicRepeaterSpec(_Strict):
    schema_version: int = 1
    device_id: str
    qtype: Literal["transmon_repeater_array"] = "transmon_repeater_array"
    task_id: str | None = None

    # geometry (public, descriptive)
    n_transmons: int = Field(..., ge=4)
    alice_memory_qubits: int = Field(..., ge=2)
    bob_memory_qubits: int = Field(..., ge=2)
    n_swap_per_arm: int = Field(..., ge=0)  # public transport length; per-SWAP error is hidden
    target_state: str = "|Phi+> = (|00> + |11>)/sqrt(2)"

    # purification control envelope: each end may apply ANY local unitary (from this gate
    # set) to its two held qubits; the device then measures the second pair and keeps the
    # first when the two outcomes agree (post-selection), converting two pairs into one with
    # some success probability. No operation may act across the two ends.
    local_gate_set: list[str]  # gates an end may apply to its two qubits (incl. rx/ry/rz, cnot)
    qubits_per_end: int = 2  # each party holds its half of pair-1 (q0) and pair-2 (q1)
    postselection: str = "measure the second pair; keep the first pair when the outcomes agree"
    measurement_bases: list[str]  # ZZ, XX, YY (final-fidelity estimate via Pauli correlators)
    cross_end_gate_allowed: Literal[False] = False  # nothing acts across the two ends

    # CLAIMED (nominal/spec, not real) calibration to verify
    claimed_distributed_fidelity: float = Field(..., gt=0, lt=1)
    claimed_per_swap_fidelity: float = Field(..., gt=0, le=1)

    budgets: RepeaterBudgets
    notes: str = ""


# --- hidden ---------------------------------------------------------------


class RepeaterStaleNotebook(_Strict):
    # Honest stale calibration only: a nominal/claimed distributed-pair fidelity to
    # VERIFY (no fabricated measurements, no claim of an achievable purified ceiling,
    # no protocol recommendation — those would be misinformation; difficulty comes from
    # the hidden params the agent measures).
    claimed_distributed_fidelity: float
    last_calibrated: str
    note: str = ""
    transport_noise: str = ""  # honest: "not characterized" (never a structure claim)


class RepeaterTransportBias(_Strict):
    """Non-isotropic transport noise: on every SWAP hop each qubit is flipped
    about ONE hidden unit axis with probability ``p_bias`` -- the same axis on both arms (a
    systematic device direction). ``axis_mode = "random_y_dominant"`` draws the axis per
    attempt, uniformly over the unit sphere restricted to ``n_y^2 - n_x^2 >= min_y2_minus_x2``
    (the region where the textbook x-rotation frame is decidedly wrong); ``"fixed"`` pins it
    to (theta_deg, phi_deg) for regression tests. The realized axis is a per-attempt hidden
    instance, recorded by qsim for the separate-mode verifier only."""

    p_bias: float = Field(..., ge=0, le=1)
    axis_mode: Literal["random_y_dominant", "fixed"] = "random_y_dominant"
    min_y2_minus_x2: float = Field(0.5, ge=-1.0, le=1.0)
    theta_deg: float | None = None
    phi_deg: float | None = None

    @model_validator(mode="after")
    def _fixed_axis_needs_angles(self) -> RepeaterTransportBias:
        if self.axis_mode == "fixed" and (self.theta_deg is None or self.phi_deg is None):
            raise ValueError("axis_mode 'fixed' needs theta_deg and phi_deg")
        return self


class HiddenRepeaterConfig(_Strict):
    schema_version: int = 1
    device_id: str
    qtype: Literal["transmon_repeater_array"] = "transmon_repeater_array"
    seed: int

    # physical noise model (-> physics.RepeaterParams); F0 and F_sat are emergent
    p_gen: float = Field(..., ge=0, le=1)
    n_swap_per_arm: int = Field(..., ge=0)
    p_swap: float = Field(..., ge=0, le=1)
    t_swap_ns: float = Field(..., ge=0)
    t1_us: float = Field(..., gt=0)
    t2_us: float = Field(..., gt=0)  # must satisfy t2 <= 2 t1 (checked in construction)
    p_cnot: float = Field(..., ge=0, le=1)
    p_1q: float = Field(..., ge=0, le=1)
    readout_p01: float = Field(..., ge=0, le=1)
    readout_p10: float = Field(..., ge=0, le=1)
    t_mem_per_round_ns: float = Field(..., ge=0)
    dwell_growth: Literal["exp", "linear"] = "exp"
    # None -> isotropic transport only
    transport_bias: RepeaterTransportBias | None = None

    stale_lab_notebook: RepeaterStaleNotebook
