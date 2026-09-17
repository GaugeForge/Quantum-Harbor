"""Public + hidden device config for the ``cycle_error_recon`` qtype.

A 5-qubit line of CZ-coupled qubits running a scheduled hard cycle
``G = CZ01 CZ12 CZ23 CZ34`` (``G^2 = I``) under two schedules (``parallel_2``,
``serial_4``). The public spec carries only conventions, dictionaries, candidate
generators, and budgets; the hidden config carries the two physical error channels,
the asymmetric readout, and the stale lab notebook (architecture invariant 2:
public/hidden split is structural).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------- public ----------------


class ScheduleLayer(_Strict):
    """One clock layer of a schedule: the CZ edges that fire simultaneously."""

    layer: int
    cz_edges: list[tuple[int, int]]


class ScheduleSpec(_Strict):
    name: Literal["parallel_2", "serial_4"]
    layers: list[ScheduleLayer]


class CerBudgets(_Strict):
    """Run-long experiment budgets."""

    hard_cycle_exposure: int = Field(..., ge=1)  # sum over rows of x*m*R*S
    raw_shots: int = Field(..., ge=1)  # incl. readout calibration
    experiment_rows: int = Field(..., ge=1)
    rows_per_batch: int = Field(32, ge=1)
    wall_clock_s: int = Field(1200, ge=1)
    max_answer_string_characters: int = Field(4_096, ge=1)
    max_final_answer_serialized_bytes: int = Field(32_768, ge=1)
    max_final_answer_submissions: int = Field(1, ge=1)


class PublicCerSpec(_Strict):
    """Agent-visible device specification (no hidden truth)."""

    schema_version: int = 1
    device_id: str
    qtype: Literal["cycle_error_recon"] = "cycle_error_recon"
    task_id: str | None = None

    n_qubits: int = 5
    connectivity: list[tuple[int, int]] = Field(
        default_factory=lambda: [(0, 1), (1, 2), (2, 3), (3, 4)]
    )
    native_hard_gate: Literal["cz"] = "cz"
    bit_order: Literal["leftmost_is_q0"] = "leftmost_is_q0"
    measurement_return: Literal["raw_five_bit_counts"] = "raw_five_bit_counts"

    schedules: list[ScheduleSpec]
    ideal_cycle: str = "CZ01 CZ12 CZ23 CZ34"
    cycle_is_involution: bool = True

    # The channel order is authoritative: ideal cycle, then coherent rotation, then
    # stochastic Pauli error. U_Q(theta) = exp(-i theta Q / 2).
    channel_order: Literal["ideal_cycle__coherent__stochastic"] = (
        "ideal_cycle__coherent__stochastic"
    )
    coherent_angle_range_rad: tuple[float, float] = (0.0, 0.20)
    stochastic_total_max: float = 0.08
    max_nonzero_stochastic_terms: int = 14

    # The fixed 30-term intrinsic stochastic dictionary, order authoritative.
    stochastic_dictionary: list[str]
    # The 5 coherent-generator candidates, positive phase.
    coherent_generator_candidates: list[str] = Field(
        default_factory=lambda: ["none", "X2", "Z1Z2", "Z2Z3", "Z1Z3"]
    )
    canonical_zero_threshold: float = 1e-12

    # Readout model (asymmetric, qubit-local). Hidden values in [0, 0.08].
    readout_model: Literal["asymmetric_qubit_local"] = "asymmetric_qubit_local"
    readout_error_max: float = 0.08
    state_prep: Literal["ideal"] = "ideal"

    allowed_fold_factors: list[int] = Field(default_factory=lambda: [1, 3, 5, 7])
    allowed_block_repetitions: list[int] = Field(default_factory=lambda: [2, 4, 8, 16, 32])

    budgets: CerBudgets
    notes: str = ""


# ---------------- hidden ----------------


class HiddenScheduleChannel(_Strict):
    """The hidden physical error channel of one schedule."""

    coherent_generator: Literal["none", "X2", "Z1Z2", "Z2Z3", "Z1Z3"]
    coherent_angle_rad: float = Field(..., ge=0.0, le=0.20)
    # intrinsic stochastic probabilities keyed by dictionary Pauli label (sparse)
    stochastic: dict[str, float]


class HiddenReadout(_Strict):
    r01: list[float] = Field(..., min_length=5, max_length=5)  # P(1|0) per qubit
    r10: list[float] = Field(..., min_length=5, max_length=5)  # P(0|1) per qubit


class StaleCerNotebook(_Strict):
    """Stale, partly-misleading lab notebook."""

    model_config = ConfigDict(extra="forbid")

    parallel_cycle_fidelity: float = 0.9854
    serial_cycle_fidelity: float = 0.9858
    preferred_schedule: str = "serial_4"
    coherent_error_detected: bool = False
    correlated_error_probability: float = 0.0
    readout_model: str = "symmetric"
    fold_model: str = "one global linear + quadratic fit"
    note: str = ""


class HiddenCerConfig(_Strict):
    schema_version: int = 2
    device_id: str
    qtype: Literal["cycle_error_recon"] = "cycle_error_recon"

    parallel_2: HiddenScheduleChannel
    serial_4: HiddenScheduleChannel
    readout: HiddenReadout
    stale_lab_notebook: StaleCerNotebook = StaleCerNotebook()
