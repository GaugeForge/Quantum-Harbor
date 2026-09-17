"""Public + hidden device config for the ``surface_code_lattice_surgery`` qtype.

The chip layout, code geometries, extraction schedules, the fixed CNOT schedule, the
byproduct frame convention, budgets and floors are **public and authoritative**; the
circuit-level noise model (per-qubit T1/T2, per-CZ error rates, asymmetric measurement +
reset errors, the leakage state machine and its topology) is **hidden** (architecture
invariant 2: the split is structural — both classes use ``extra="forbid"``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------- public ----------------


class AncillaSpec(_Strict):
    qubit: int
    pos: list[int] = Field(..., min_length=2, max_length=2)
    basis: Literal["Z", "X"]
    support: list[int]  # data qubit ids measured by this plaquette


class LayoutSpec(_Strict):
    """One hosted code layout (memory patch or merged configuration)."""

    name: str
    data_qubits: list[int]
    ancillas: list[AncillaSpec]


class LsBudgets(_Strict):
    shot_budget: int = Field(..., ge=1)  # total shots across ALL experiment calls
    shot_rounds_budget: int = Field(..., ge=1)  # total shots x syndrome-rounds
    max_rounds: int = Field(32, ge=1)
    max_shots_per_call: int = Field(100_000, ge=1)


class PublicLatticeSurgerySpec(_Strict):
    """Agent-visible device specification (no hidden truth)."""

    schema_version: int = 1
    device_id: str
    qtype: Literal["surface_code_lattice_surgery"] = "surface_code_lattice_surgery"
    task_id: str | None = None

    code: Literal["rotated_surface_code"] = "rotated_surface_code"
    n_chip_qubits: int
    chip_coords: list[list[int]]  # chip qubit id -> [x, y] (data odd/odd, ancilla even/even)
    layouts: list[LayoutSpec]  # d3_control, d3_intermediate, d3_target, d5, merged_zz, merged_xx
    routing_zz_data: list[int]  # routing row data qubits (active only in zz merges)
    routing_xx_data: list[int]  # routing col data qubits (active only in xx merges)

    cycle_time_us: float = 1.0
    schedule_note: str = ""

    # Fixed scoring layouts. The submitted memory DEM is ALWAYS replayed at
    # memory_scoring_rounds; the CNOT DEM at the fixed public CNOT schedule.
    memory_scoring_rounds: int = 12
    cnot_rounds: int = 9
    merge_rounds: int = 3
    transition_pre_rounds: int = 2
    transition_post_rounds: int = 2

    decoder_types: list[str] = Field(default_factory=lambda: ["mwpm", "belief_matching"])

    # Nominal/stale physical scale; the true rates are hidden and inhomogeneous.
    nominal_cz_error: float = 0.0035
    nominal_t1_us: float = 70.0

    budgets: LsBudgets
    floor_mem: float
    floor_cnot: float
    cnot_per_entry_cap: float
    replay_shots_memory: int
    replay_shots_per_entry: int

    # Byproduct frame convention (public, verifier-owned): per CNOT reporting
    # configuration, which named parity groups fold into each output observable.
    frame_convention: dict[str, dict[str, list[str]]]
    observable_names: list[str]

    notes: str = ""


# ---------------- hidden ----------------


class QubitNoise(_Strict):
    t1_us: float = Field(..., gt=0.0)
    t2_us: float = Field(..., gt=0.0)
    p_reset: float = Field(..., ge=0.0, lt=1.0)
    p_m01: float = Field(..., ge=0.0, lt=1.0)  # readout misclassification 0 -> 1
    p_m10: float = Field(..., ge=0.0, lt=1.0)  # readout misclassification 1 -> 0


class CzError(_Strict):
    q1: int
    q2: int
    p: float = Field(..., ge=0.0, lt=1.0)


class LeakageSite(_Strict):
    """A leakage-prone MEASURE qubit (stuck-ancilla model, McEwen-style): while leaked,
    the ancilla's measurement misreports, flipping its record each lived round with
    probability ``p_flip_round``. Bulk (``stuck``) sites keep a short lifetime,
    presuming a per-cycle leakage-removal mechanism; ``activation`` sites are
    merge-window seam ancillas whose leak may instead persist for the remainder of the
    window (no leakage removal on freshly-activated routing ancillas) — a run-gap
    re-initialization always clears the state. The induced detection events are
    long-range and spacetime-correlated."""

    qubit: int
    p_enter: float = Field(..., ge=0.0, lt=1.0)  # per measured round (or per activation)
    lifetime: int = Field(..., ge=1, le=64)  # rounds (truncated at the run boundary)
    p_flip_round: float = Field(..., ge=0.0, le=1.0)  # record-flip prob per lived round
    # "stuck": may enter at any measured round (bulk sites). "activation": enters only
    # once per contiguous MID-CIRCUIT measurement run (a cold round-0 boot is a global
    # reset, not an activation event), ``delay`` rounds after the run start — the
    # merge-window seam mechanism: the delay/lifetime mix carries the hidden
    # per-site detector->parity attribution.
    mode: Literal["stuck", "activation"] = "stuck"
    delay: int = Field(0, ge=0, le=8)  # activation mode: entry offset within the run


class StaleLsNotebook(_Strict):
    noise_model: str = ""
    routing_region: str = ""
    surgery_advice: str = ""
    decoder_advice: str = ""
    correlation_advice: str = ""
    budget_advice: str = ""
    note: str = ""


class HiddenLatticeSurgeryConfig(_Strict):
    schema_version: int = 1
    device_id: str
    qtype: Literal["surface_code_lattice_surgery"] = "surface_code_lattice_surgery"
    seed: int

    cycle_time_us: float = 1.0
    qubits: dict[int, QubitNoise]
    cz_errors: list[CzError]
    leakage: list[LeakageSite]
    stale_lab_notebook: StaleLsNotebook = StaleLsNotebook()
