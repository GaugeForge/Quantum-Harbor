"""Public + hidden device config for the ``neutral_atom_dm_processor`` qtype.

Standard QIQCBench split: the public spec advertises the true device structure +
ACCURATE op timing + CLAIMED (stale/optimistic) fidelities; the true noise rates
are hidden and worse (the stale-notebook lever). The hidden config additionally
carries the task's tunable ``scoring`` anchors so difficulty tuning is
YAML-only; anchors freeze before any rated run and later changes bump the
device version (plan-review commitment discipline). Architecture invariant 2:
both classes use ``extra="forbid"``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------- public ----------------


class OpDurationsUs(_Strict):
    """Public + ACCURATE op timing (microseconds). Multiplied by ``idle_scale``.

    Timing drives decoherence/loss; it is never itself a score.
    """

    rotation_local: float = 2.0
    rotation_global: float = 4.0
    cz: float = 0.4
    move_per_um: float = 0.2
    move_settle: float = 15.0
    readout: float = 30.0
    reset: float = 10.0


class ZonedLayoutSpec(_Strict):
    """Zoned array geometry: one storage row + one Rydberg gate zone.

    Site indexing (public, exact): storage sites are ``0..n_storage_sites-1`` at
    x = site * site_pitch_um, y = 0. Gate-zone slot ``s`` (0-based) provides two
    positions, site IDs ``n_storage_sites + 2s`` and ``n_storage_sites + 2s + 1``,
    at x = (2s or 2s+1) * site_pitch_um, y = gate_zone_offset_um. Move distance
    between sites = |x_a - x_b| + (gate_zone_offset_um if the zones differ else 0).
    CZ requires the pair to occupy the TWO positions of ONE slot.
    """

    n_storage_sites: int = Field(14, ge=1)
    n_gate_slots: int = Field(2, ge=1)
    site_pitch_um: float = Field(6.0, gt=0.0)
    gate_zone_offset_um: float = Field(20.0, gt=0.0)

    @property
    def n_sites(self) -> int:
        return self.n_storage_sites + 2 * self.n_gate_slots

    def site_x_um(self, site: int) -> float:
        if site < self.n_storage_sites:
            return site * self.site_pitch_um
        return (site - self.n_storage_sites) * self.site_pitch_um

    def is_gate_position(self, site: int) -> bool:
        return site >= self.n_storage_sites

    def slot_of(self, site: int) -> int | None:
        if site < self.n_storage_sites:
            return None
        return (site - self.n_storage_sites) // 2

    def move_distance_um(self, site_a: int, site_b: int) -> float:
        dx = abs(self.site_x_um(site_a) - self.site_x_um(site_b))
        if self.is_gate_position(site_a) != self.is_gate_position(site_b):
            dx += self.gate_zone_offset_um
        return dx


class NeutralAtomDmStateMemory(_Strict):
    """Public working-set limit and the rule that prices a program against it.

    The simulator evaluates an exact density matrix, so a program's memory cost is
    set by its ATOM COUNT and its BRANCH STRUCTURE, not by its length --
    ``max_program_cost_units`` bounds work, not bytes, and the two are different
    quantities. This block is public because an agent cannot plan a schedule it
    cannot price. ``validate_schedule`` returns the computed figures and refuses an
    over-budget program with the same reason submission gives.

    The rule, in full. One matrix is ``matrix_bytes_formula`` bytes. Walking the
    ops before the terminal readout block:

      * a program holds at least ``peak_matrices_unitary`` matrices: the state
        plus the largest scratch any gate or noise kernel allocates;
      * a mid-circuit measure op holds
        ``peak_matrices_measure_base + 2**free_bits``: one matrix per outcome
        state it builds plus a scratch, where ``free_bits`` counts the atoms that
        op measures WITHOUT ``expect`` (an ``expect`` readout parks the
        contradicting branch instead of building it, so it adds no free bits);
      * the branches an EARLIER fan-out left pending are held on disk, not in
        memory, while their sibling is evaluated, so they add nothing.

    The peak over that walk, times the matrix size, must not exceed ``budget_mib``.
    """

    budget_mib: int = Field(10_752, ge=1)
    matrix_bytes_formula: str = "16 * 4**n_atoms"
    peak_matrices_unitary: float = Field(1.25, gt=0.0)
    peak_matrices_measure_base: float = Field(0.125, gt=0.0)


class NeutralAtomDmBudgets(_Strict):
    shot_budget: int = Field(5_000_000, ge=1)
    max_experiment_calls: int = Field(10_000, ge=1)
    max_shots_per_call: int = Field(200_000, ge=1)
    max_ops: int = Field(4_000, ge=1)
    max_atoms: int = Field(14, ge=1)  # hard cap = simulator capacity (4^14 rho)
    max_sweep_points: int = Field(16, ge=1)
    # Branch-tree containment: mid-circuit measurements split deterministic
    # branches, and both caps are public so one legal program's branch tree
    # stays bounded (agent-side and verifier replay enforce identically). These
    # caps are per REQUEST and say nothing about what else is running: the
    # engine's separate concurrent working-set budget (see ``memory.py``)
    # is what keeps SEVERAL admitted programs from summing past the container.
    max_branches: int = Field(256, ge=1)
    # Sum over ops of live_branches * 4^n_atoms. A ~250-op 12-atom FT-shaped
    # program with a thin flag tree costs ~2e10; deep 14-atom programs exceed
    # the cap and reject cleanly (13-14 atoms stay legal for shallow programs;
    # the race metric's atom penalty prices them).
    max_program_cost_units: float = Field(1.0e11, gt=0.0)
    max_recorded_bits_per_call: int = Field(50_000_000, ge=1)
    max_recorded_bits_per_run: int = Field(500_000_000, ge=1)
    # Memory, which none of the caps above bounds. Separate from
    # max_program_cost_units on purpose: that one is a WORK cap.
    state_memory: NeutralAtomDmStateMemory = NeutralAtomDmStateMemory()


class PublicNeutralAtomDmSpec(_Strict):
    """Agent-visible device specification (no hidden truth; claims marked as claims)."""

    # 2: budgets.state_memory publishes the working-set limit and its pricing
    # rule used by admission.
    schema_version: int = 2
    device_id: str
    qtype: Literal["neutral_atom_dm_processor"] = "neutral_atom_dm_processor"
    task_id: str | None = None

    layout: ZonedLayoutSpec = ZonedLayoutSpec()
    single_atom_gates: list[str] = Field(default_factory=lambda: ["rx", "ry", "rz"])
    entangling_gate: Literal["cz"] = "cz"
    init_states: list[str] = Field(default_factory=lambda: ["0", "1", "+", "-"])
    measurement: str = "fluorescence, Z basis only, mid-circuit on any subset, optional reset"

    noise_scale_range: list[float] = Field(default_factory=lambda: [0.25, 2.0])
    idle_scale_range: list[float] = Field(default_factory=lambda: [0.25, 4.0])

    durations_us: OpDurationsUs = OpDurationsUs()
    budgets: NeutralAtomDmBudgets = NeutralAtomDmBudgets()

    # CLAIMED (stale/optimistic) fidelities — NOT the real values. Re-measure.
    claimed_cz_fidelity: float = 0.9992
    claimed_rotation_fidelity: float = 0.9998
    claimed_readout_fidelity: float = 0.99  # claimed symmetric; truth is asymmetric
    claimed_transport_note: str = "AOD moves are nearly lossless; move freely"

    notes: str = (
        "Zoned array; CZ only for a pair co-located in one gate-zone slot; moves, gates, and "
        "readout take time and time costs coherence. noise_scale rescales all error rates "
        "together — the standard way to check how your logical error scales with the physical "
        "error rate. Calibrations may be stale; verify before trusting. You are scored on "
        "logical Bell fidelity at noise_scale = 1, demonstrated second-order error scaling, "
        "and atom/move economy; runs below a validity floor are not rated. The exact "
        "simulation also has a working-set limit that program length does not predict: "
        "see budgets.state_memory, and call validate_schedule, which prices your "
        "schedule and refuses an over-budget one for the same reason submission would."
    )


# ---------------- hidden ----------------


class HiddenNeutralAtomDmNoise(_Strict):
    """True hidden noise rates. Every rate is multiplied by ``noise_scale``.

    Amplitude damping (t1_us) and pure dephasing (t2_star_us) are exact Kraus
    channels applied over elapsed schedule time x idle_scale. Atom loss is
    modeled as classically accumulated forced-dark probability per atom applied
    at readout (post-loss gate back-action deliberately unmodeled) — this
    definition IS the hidden model and the verifier replays it identically.
    """

    cz_depolarizing: float = Field(1.5e-3, ge=0.0, lt=1.0)
    rotation_depolarizing: float = Field(2e-4, ge=0.0, lt=1.0)
    prep_error: float = Field(1e-3, ge=0.0, lt=1.0)  # per-atom init depolarizing
    t1_us: float = Field(4.0e6, gt=0.0)
    t2_star_us: float = Field(1.0e5, gt=0.0)  # 100 ms; see the device YAML for why
    move_loss: float = Field(6e-5, ge=0.0, lt=1.0)  # per move, forced-dark accumulation
    move_dephasing: float = Field(2e-4, ge=0.0, lt=1.0)  # phase-flip prob per move
    readout_p1_to_0: float = Field(0.017, ge=0.0, lt=1.0)
    readout_p0_to_1: float = Field(0.006, ge=0.0, lt=1.0)


class HiddenNeutralAtomDmScoring(_Strict):
    """Tunable scoring anchors. Read only by the verifier.

    Part of the standard hidden-model commitment: freeze before any rated
    run; bump the device version to change them afterwards.
    """

    f_valid: float = Field(0.55, ge=0.0, le=1.0)
    slope_threshold: float = Field(1.7, gt=0.0)
    # Fault tolerance of the READOUT: the fidelity the agent's own declared decode
    # yields on the replayed basis programs must also be second order in the noise
    # scale (a transversal product-basis readout is first order on this non-CSS
    # code -- measured 0.88 on this device against 2.04 for a flagged measure-out).
    readout_slope_threshold: float = Field(1.7, gt=0.0)
    # Gate structure: at ``schedule.gate_start`` each block must already hold a
    # logical qubit -- the largest eigenvalue of its ideally decoded logical qubit
    # (encoded blocks measure ~0.999; an unencoded |00000> block decodes to 0.625).
    pre_gate_purity_floor: float = Field(0.9, gt=0.5, le=1.0)
    replay_noise_scales: list[float] = Field(default_factory=lambda: [0.35, 0.5, 0.7, 1.0])
    replay_shots: int = Field(20_000, ge=1)
    alpha_atom_penalty: float = Field(0.02, ge=0.0)
    beta_move_penalty: float = Field(0.002, ge=0.0)
    atoms_ref: int = Field(12, ge=1)
    # Every CZ needs at least two tweezer moves; the literature protocol (verified
    # preparation + pieceable round-robin gate + flagged measure-out) measures 385
    # moves on this device, so the reference sits at 400 rather than the 60.
    m_ref_moves: int = Field(400, ge=0)
    min_accepted_replay_shots: int = Field(500, ge=1)
    min_accept_weight: float = Field(0.05, gt=0.0, le=1.0)
    raw_consistency_sigmas: float = Field(5.0, gt=0.0)
    # A genuine logical Bell pair is locally maximally mixed on each block:
    # every single-block logical expectation is ~0 while the two-block
    # correlators are near their Bell values. A coincidence-counting decode over
    # locally-definite atoms has a single-block marginal near +-1, so this bound
    # separates a real entangled logical pair from a faked one. Generous headroom
    # above the readout-asymmetry bias a genuine pair actually carries.
    local_mixedness_max: float = Field(0.3, ge=0.0, le=1.0)
    # Logical Bell fidelity of the PREPARED state itself (the shared preparation
    # replayed exactly, reduced to the declared [[5,1,3]] data atoms, decoded
    # ideally per block, maximized over the logical Pauli frame). Every separable
    # state -- including any classically-correlated preparation -- is bounded by
    # 1/2 even after decoding (decoding is a local channel), so a floor above 1/2
    # is an entanglement witness no decode can talk its way past. Kept equal to
    # ``f_valid`` so it never rejects a protocol the measured floor admits.
    f_logical_state_floor: float = Field(0.55, gt=0.5, le=1.0)


class HiddenNeutralAtomDmConfig(_Strict):
    schema_version: int = 1
    device_id: str
    qtype: Literal["neutral_atom_dm_processor"] = "neutral_atom_dm_processor"
    seed: int

    noise: HiddenNeutralAtomDmNoise = HiddenNeutralAtomDmNoise()
    scoring: HiddenNeutralAtomDmScoring = HiddenNeutralAtomDmScoring()


__all__ = [
    "HiddenNeutralAtomDmConfig",
    "HiddenNeutralAtomDmNoise",
    "HiddenNeutralAtomDmScoring",
    "NeutralAtomDmBudgets",
    "OpDurationsUs",
    "PublicNeutralAtomDmSpec",
    "ZonedLayoutSpec",
]
