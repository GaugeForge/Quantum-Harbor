"""Public + hidden device config for the ``neutral_atom_logical_processor`` qtype.

Standard QIQCBench split: the public spec advertises true device structure + ACCURATE
op timing + **CLAIMED (stale/optimistic) fidelities**; the true noise rates are hidden
and worse, so the agent must re-calibrate (the stale-notebook lever). The CONTROLLED-H
identity is published correctly (not a trap). Architecture invariant 2: the public/hidden
split is structural; both classes use ``extra="forbid"``.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------- public ----------------


class OpDurationsUs(_Strict):
    """Public + ACCURATE op timing (microseconds). Drives decoherence/loss, NOT a score."""

    rotation: float = 1.0
    cz: float = 0.5
    move_per_um: float = 0.1
    readout: float = 5.0


class NeutralAtomBudgets(_Strict):
    shot_budget: int = Field(2_000_000, ge=1)  # total shots across all run_atom_program calls
    max_experiment_calls: int = Field(10_000, ge=1)
    max_shots_per_call: int = Field(200_000, ge=1)
    max_ops: int = Field(2_000, ge=1)
    # Exact pre-allocation guard for the two raw uint8 result matrices.  It is a
    # public capability limit, so future devices can raise it deliberately.
    max_recorded_bits_per_call: int = Field(50_000_000, ge=1)
    max_recorded_bits_per_run: int = Field(500_000_000, ge=1)
    max_final_answer_serialized_bytes: int = Field(131_072, ge=1)
    max_final_answer_submissions: int = Field(4, ge=1)
    max_answer_string_characters: int = Field(8_000, ge=1)
    max_final_answer_nesting_depth: int = Field(16, ge=1)
    n_atom_sites: int = Field(9, ge=1)  # tweezer capacity (this task uses 7 data + 2 Bell ancilla)


class ErasureRepairServiceSpec(_Strict):
    """Public contract for high-level located-erasure service episodes."""

    ldu_cadence_epochs: int = Field(2, ge=1)
    public_horizons: list[int] = Field(default_factory=lambda: [24, 32, 40])
    reserve_atoms: int = Field(2, ge=0, le=8)
    replacement_epochs_per_atom: int = Field(2, ge=1, le=16)
    max_episodes_per_call: int = Field(8_000, ge=1)
    episode_budget: int = Field(1_000_000, ge=1)
    max_raw_result_uncompressed_bytes: int = Field(268_435_456, ge=1)
    role_loss_per_epoch_range: list[float] = Field(
        default_factory=lambda: [0.0005, 0.006], min_length=2, max_length=2
    )
    burst_probability_range: list[float] = Field(
        default_factory=lambda: [0.0, 0.20], min_length=2, max_length=2
    )
    burst_role_loss_per_epoch_range: list[float] = Field(
        default_factory=lambda: [0.005, 0.05], min_length=2, max_length=2
    )
    ldu_miss_probability_range: list[float] = Field(
        default_factory=lambda: [0.0, 0.15], min_length=2, max_length=2
    )
    transport_failure_probability_range: list[float] = Field(
        default_factory=lambda: [0.0, 0.05], min_length=2, max_length=2
    )
    recovery_role_loss_probability_by_block_range: list[float] = Field(
        default_factory=lambda: [0.001, 0.025], min_length=2, max_length=2
    )
    assessment_episodes_per_horizon: int = Field(10_000, ge=1)
    joint_survival_sla: float = Field(0.80, ge=0.0, le=1.0)
    simultaneous_confidence_level: float = Field(0.99, gt=0.0, lt=1.0)
    confidence_method: Literal["bonferroni_clopper_pearson"] = "bonferroni_clopper_pearson"


class PublicNeutralAtomSpec(_Strict):
    """Agent-visible device specification (no hidden truth, no false claims)."""

    schema_version: int = 1
    device_id: str
    qtype: Literal["neutral_atom_logical_processor"] = "neutral_atom_logical_processor"
    task_id: str | None = None

    n_atoms: int = 9
    single_atom_gates: list[str] = Field(
        default_factory=lambda: ["h", "s", "sdg", "x", "y", "z", "rx", "ry", "rz"]
    )
    entangling_gate: Literal["cz"] = "cz"  # Rydberg-blockade controlled-Z
    blockade_radius_um: float = 5.0
    measurement_bases: list[str] = Field(default_factory=lambda: ["z", "x", "y"])
    # The CORRECT controlled-H identity, published plainly (a textbook fact, not a trap):
    controlled_h_identity: str = "CH(c->t) = ry(+pi/4) . cz(c,t) . ry(-pi/4)"

    durations_us: OpDurationsUs = OpDurationsUs()
    budgets: NeutralAtomBudgets = NeutralAtomBudgets()

    # CLAIMED (stale/optimistic) fidelities — NOT the real values (HiddenNeutralAtomNoise
    # is worse). The agent must re-measure; trusting these is the stale-notebook trap.
    claimed_cz_fidelity: float = 0.997
    claimed_single_atom_fidelity: float = 0.9995
    claimed_readout_fidelity: float = 0.99  # claimed symmetric; real is asymmetric
    claimed_transport_note: str = "AOD moves are nearly lossless; move freely"

    # The validity-floor GATE (= F_valid); a run scoring below this is not Elo-rated.
    logical_fidelity_floor: float = 0.60
    erasure_repair: ErasureRepairServiceSpec | None = None

    notes: str = (
        "Calibrations may be stale; verify before relying on the claimed numbers. "
        "Readout is by fluorescence (bright=|1>, dark=|0>); a lost atom reads dark/0 in the "
        "bit record, and results carry a per-shot diagnostic lost-atom mask (lost_b64). "
        "Scored on the final |H_L> fidelity."
    )


# ---------------- hidden ----------------


class HiddenNeutralAtomNoise(_Strict):
    """True hidden noise rates (the agent must measure these; nothing is advertised)."""

    cz_pauli_error: float = Field(0.005, ge=0.0, lt=1.0)  # stochastic Pauli after each CZ
    cz_atom_loss: float = Field(0.002, ge=0.0, lt=1.0)  # honest loss per CZ
    rot_pauli_error: float = Field(0.0005, ge=0.0, lt=1.0)  # per single-atom gate
    move_atom_loss_per_um: float = Field(1e-4, ge=0.0)  # transport loss vs distance
    move_dephasing_per_us: float = Field(2e-4, ge=0.0)  # heating dephasing vs move time
    idle_t2_us: float = Field(1500.0, gt=0.0)  # transverse coherence time
    readout_p1_to_dark: float = Field(0.03, ge=0.0, lt=1.0)  # |1> reads "0" (dominant)
    readout_p0_to_bright: float = Field(0.01, ge=0.0, lt=1.0)  # |0> reads "1"
    readout_atom_loss: float = Field(0.001, ge=0.0, lt=1.0)


class HiddenErasureRepairNoise(_Strict):
    """Hidden stochastic parameters for generic located-erasure service episodes."""

    role_loss_per_epoch: float = Field(0.001, ge=0.0, lt=1.0)
    burst_probability: float = Field(0.10, ge=0.0, lt=1.0)
    burst_role_loss_per_epoch: float = Field(0.025, ge=0.0, lt=1.0)
    missed_detection_probability: float = Field(0.035, ge=0.0, lt=1.0)
    transport_failure_probability: float = Field(0.0065, ge=0.0, lt=1.0)
    # Per-block scaling of the loss rate, in block order (A, B). Any nonuniformity is part of the
    # persistent apparatus regime and is observable only through block-tagged public raw traces;
    # a task-specific apparatus may use equal multipliers.
    block_loss_multipliers: list[float] = Field(
        default_factory=lambda: [1.0, 1.0], min_length=2, max_length=2
    )
    recovery_role_loss_probability_by_block: list[Annotated[float, Field(ge=0.0, lt=1.0)]] = Field(
        default_factory=lambda: [0.008, 0.008], min_length=2, max_length=2
    )


class HiddenNeutralAtomConfig(_Strict):
    schema_version: int = 1
    device_id: str
    qtype: Literal["neutral_atom_logical_processor"] = "neutral_atom_logical_processor"
    seed: int

    noise: HiddenNeutralAtomNoise = HiddenNeutralAtomNoise()
    erasure_repair_noise: HiddenErasureRepairNoise | None = None


__all__ = [
    "PublicNeutralAtomSpec",
    "HiddenNeutralAtomConfig",
    "HiddenNeutralAtomNoise",
    "OpDurationsUs",
    "NeutralAtomBudgets",
    "ErasureRepairServiceSpec",
    "HiddenErasureRepairNoise",
]
