"""Public + hidden device config for the ``neutral_atom_ftqc_compiler`` qtype.

A **device-less / static-instance** qtype (sibling to
``ftqc_resource_estimation``): no quantum dynamics, no shots, no async job model.
The "device" is a zoned reconfigurable neutral-atom surface-code architecture
described entirely by a **public, authoritative deterministic cost model** —
surface-code tiles (``2 d²`` atoms), global Rydberg CZ, transversal vs
lattice-surgery logical CNOTs, magic-state factories, AOD movement constraints,
and the analytic logical-error model ``P(d) = a (p/p*)^⌊(d+1)/2⌋``.

The public spec carries that generic cost model. The per-instance compilation
target (the modular-multiplier Clifford+T netlist + DAG, and the maintainer-pinned
budget reference depth) is delivered separately via ``get_target_circuit``. The
hidden config carries only the scoring anchors (``F_naive``/``F_opt``) and the
stale lab notebook; because the optimum is derivable from the public cost model,
nothing the agent must *discover* lives here (the run keeps
``QSIM_SNAPSHOT_HIDDEN_TRUTH=0``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_QTYPE = "neutral_atom_ftqc_compiler"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Public spec (visible to agent) ----------


class PublicNeutralAtomSpec(_Strict):
    """Public device spec for ``neutral_atom_ftqc_compiler`` (via ``get_device_spec``).

    Every field here is authoritative: the agent optimizes its compilation against
    this cost model, and the verifier recomputes the score from the same model.
    """

    schema_version: int = 1
    device_id: str
    qtype: Literal["neutral_atom_ftqc_compiler"] = "neutral_atom_ftqc_compiler"
    task_id: str
    device_class: str = "neutral_atom_ftqc_compiler"
    architecture: str = "zoned_reconfigurable_neutral_atom"

    # --- surface-code tiles ---
    available_code_distances: list[int] = Field(default_factory=lambda: [3, 5, 7, 9])
    tile_atoms_model: str = "2*d**2"

    # --- logical-error model ---
    error_prefactor_a: float = Field(0.03, gt=0)
    threshold_p_star: float = Field(0.01, gt=0)
    physical_error_p: float = Field(0.003, gt=0)
    logical_error_model: str = "a*(p/p_star)**floor((d+1)/2)"
    logical_error_budget_eps: float = Field(0.12, gt=0)

    # --- timing model ---
    t_round_s: float = Field(1e-3, gt=0)
    transversal_cnot_rounds: int = Field(1, ge=1)
    lattice_surgery_rounds_model: str = "d"  # a lattice-surgery CNOT costs d rounds
    t_teleport_rounds: int = Field(2, ge=1)
    # Structurally pinned to the engine's hardcoded charge (op_round_cost
    # returns 0 for non-cx/t/tdg ops): a spec claiming any other value must
    # fail validation rather than silently diverge from the recompute.
    single_qubit_clifford_rounds: Literal[0] = 0  # transversal in place
    # --- exact score recompute ---
    schedule_makespan_model: str = (
        "rounds(op): cx granted transversal = transversal_cnot_rounds; cx "
        "lattice_surgery (including a transversal claim that is not granted, and a "
        "cx with no impl set) = d; t/tdg = t_teleport_rounds; every other op "
        "(single-qubit Cliffords) = single_qubit_clifford_rounds. gate_makespan = "
        "sum over distinct time_slice values of max(rounds(op) over the ops in that "
        "slice); slices execute sequentially; unused slice indices are free"
    )
    total_rounds_model: str = (
        "max(gate_makespan, factory_rounds) with factory_rounds = "
        "ceil(t_count / n_factories) * factory_throughput_cycles_per_t * d"
    )
    total_atoms_model: str = (
        "(n_data_qubits + n_routing_tiles) * 2*d**2 + n_factories * "
        "factory_footprint_tiles * 2*d_distill**2 with d_distill = "
        "max(factory_min_distill_distance, d - factory_distill_distance_offset)"
    )
    physical_qubit_seconds_model: str = "total_atoms * total_rounds * t_round_s"
    transversal_grant_rule: str = (
        "claims are judged per time_slice, among the cx entries claiming "
        "impl=transversal in that slice: (i) both endpoints must have a tile_layout "
        "entry; (ii) blockade matching: a logical qubit may appear in at most one "
        "claimed pair in the slice (both offending claims are charged); (iii) "
        "pairwise 1-D no-crossing on the layout rows, checked among the claims "
        "surviving (i)-(ii) only: for any two surviving pairs (c_a->t_a), "
        "(c_b->t_b), require (row[c_a] - row[c_b]) * (row[t_a] - row[t_b]) > 0 "
        "strictly (equal rows count as a collision; both members of a violating "
        "pair are charged). A claim failing any of (i)-(iii) is charged as "
        "lattice_surgery (a score penalty, not INVALID); with the slice-max rule "
        "one charged CNOT raises its whole slice's cost to d rounds. row[q] is "
        "the row of qubit q's tile_layout entry; if tile_layout repeats a "
        "logical_qubit, its last entry wins"
    )

    # --- magic-state factory ---
    factory_recipe: str = "fifteen_to_one"
    factory_footprint_tiles: int = Field(12, ge=1)
    factory_throughput_cycles_per_t: int = Field(18, ge=1)
    factory_distill_distance_offset: int = Field(2, ge=0)  # d_distill = d - offset
    factory_min_distill_distance: int = Field(3, ge=1)
    magic_state_output_error_model: str = "35*p**3"

    # --- AOD movement constraints ---
    aod_constraints: str = (
        "same AOD row/column moves together (collective); rows cannot cross rows, "
        "columns cannot cross columns (order-preserving); SLM traps are fixed"
    )

    # --- tool surface ---
    target_primitive: str = "get_target_circuit"
    target_circuit_file: str = "netlist_seed_0.json"
    objective: str = (
        "minimize physical_qubit_seconds = total_physical_atoms * critical_path_rounds "
        "* t_round_s, subject to FT-correctness, AOD-legality, and the logical-error "
        "budget V_err*P(d) <= eps with V_err = n_data_qubits * n_ref_cycles"
    )
    target_wall_clock_runtime_s: int = Field(7200, ge=1)
    notes: str = ""


# ---------- Hidden truth (only inside qsim) ----------


class StaleNeutralAtomNotebook(_Strict):
    """Stale compilation recommendations (§6a) — every actionable item is a trap."""

    cnot_recipe: str = "use lattice surgery for all logical CNOTs (standard, robust)"
    code_distance_advice: int = 9
    factory_advice: str = "one magic-state factory is simpler"
    routing_advice: str = "treat connectivity as a graph; insert SWAP networks"
    movement_advice: str = "atom movement is expensive; minimize moves above all else"
    note: str = (
        "Use the standard surface-code compilation; lattice surgery everywhere; a "
        "single factory and a safe large distance keep things simple."
    )
    reliability: str = "operator notes carried over from a prior compilation study"


class NeutralAtomScoreAnchors(_Strict):
    """Per-instance scoring anchors (§5b). Not in the ELO ranking path; the

    authoritative race metric is the verifier-recomputed physical-qubit-seconds.
    These anchor only the optional normalized view ``(F_naive-score)/(F_naive-F_opt)``.
    """

    instance_seed: int = 0
    reference_code_distance: int = 7
    f_naive_qubit_seconds: float
    f_opt_qubit_seconds: float


class HiddenNeutralAtomConfig(_Strict):
    """Hidden config for ``neutral_atom_ftqc_compiler``. Inside qsim only."""

    schema_version: int = 1
    device_id: str
    qtype: Literal["neutral_atom_ftqc_compiler"] = "neutral_atom_ftqc_compiler"
    task_id: str

    anchors: list[NeutralAtomScoreAnchors] = Field(default_factory=list)
    stale_lab_notebook: StaleNeutralAtomNotebook = StaleNeutralAtomNotebook()


__all__ = [
    "PublicNeutralAtomSpec",
    "HiddenNeutralAtomConfig",
    "StaleNeutralAtomNotebook",
    "NeutralAtomScoreAnchors",
]
