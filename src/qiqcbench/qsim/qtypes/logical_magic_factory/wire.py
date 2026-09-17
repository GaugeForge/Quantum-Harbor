"""Logical magic-factory control + result wire schemas (qtype-local).

Mirrors how other self-registering qtypes keep their request schemas local and
contribute their result type to the dynamic ``JobData`` union via
``QtypeDescriptor.result_data_models`` -- so no edit to ``core/wire.py`` is
needed. ``SCHEMA_VERSION`` and ``_Strict`` are imported from ``core/wire.py``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from qiqcbench.qsim.core.wire import SCHEMA_VERSION, _Strict, _StrictRequest

# ---------- Control (requests) ----------


class MagicBenchmarkPointRequest(_StrictRequest):
    """Draw magic states from the factory and run a single- or two-copy circuit.

    ``n_copies=1``: draw one copy, optionally twirl it, then measure the
    requested Pauli ``basis``. Consumes ``shots`` magic states.

    ``n_copies=2``: draw two copies, optionally twirl each, then run the fixed
    joint (CNOT + logical-Hadamard + computational-basis) circuit and measure
    both. ``basis`` is not meaningful here (the joint circuit is fixed) and
    must be omitted. Consumes ``2 * shots`` magic states.
    """

    schema_version: int = SCHEMA_VERSION
    n_copies: Literal[1, 2]
    twirl: bool = False
    basis: Literal["x", "y", "z"] | None = None
    shots: int = Field(..., gt=0, le=100_000)

    @model_validator(mode="after")
    def _check_basis_matches_n_copies(self) -> MagicBenchmarkPointRequest:
        if self.n_copies == 1 and self.basis is None:
            raise ValueError("basis is required when n_copies == 1")
        if self.n_copies == 2 and self.basis is not None:
            raise ValueError(
                "basis must be omitted when n_copies == 2 (the joint circuit "
                "is fixed; otherwise semantically-equivalent requests would "
                "produce different wire payloads)"
            )
        return self


class MagicBenchmarkBatchRequest(_StrictRequest):
    """A batch of magic-benchmark points, run in order against the run-long factory."""

    schema_version: int = SCHEMA_VERSION
    points: list[MagicBenchmarkPointRequest] = Field(..., min_length=1, max_length=256)


# ---------- Result (joins the dynamic JobData union via result_data_models) ----------


class JobMagicBenchmarkPoint(_Strict):
    """Raw per-shot outcome for one requested point.

    ``flagged[s] == 1`` iff shot ``s``'s copy (copy 1, for n_copies=2) was
    syndrome-flagged (not clean); ``0`` = accepted/clean. For ``n_copies=2``,
    ``flagged2``/``outcomes2`` carry copy 2's syndrome flag and its measured
    bit from the joint circuit. Post-selection is the agent's choice, made
    from these raw per-shot arrays -- the engine does not pre-filter.
    """

    n_copies: Literal[1, 2]
    twirl: bool
    basis: str | None
    shots: int
    rejected: bool = False
    reject_reason: str | None = None

    flagged: list[int] = Field(default_factory=list)
    outcomes: list[int] = Field(default_factory=list)
    flagged2: list[int] = Field(default_factory=list)
    outcomes2: list[int] = Field(default_factory=list)

    magic_states_consumed_this_point: int = 0
    magic_states_consumed_total: int = 0
    budget_remaining: int = 0


class JobMagicBenchmarkData(_Strict):
    """Points processed by one run_magic_benchmark_batch call, in request order."""

    kind: Literal["magic_benchmark"] = "magic_benchmark"
    points: list[JobMagicBenchmarkPoint] = Field(..., min_length=1)


# ---------- Cultivation line (magic_state_cultivation capability) ----------


class CultivationPointRequest(_StrictRequest):
    """Run ``shots`` injection attempts of one cultivation schedule.

    Every attempt (kept or later discarded by the agent's own post-selection)
    consumes one injection from the run budget. The engine never pre-filters:
    each shot returns the four detector-group flags and the terminal outcome
    bit; post-selection is the agent's choice, applied client-side.
    """

    schema_version: int = SCHEMA_VERSION
    injection_theta_rad: float = Field(..., ge=0.0, le=1.5707963267948966)
    cultivation_rounds: int = Field(..., ge=0, le=4)
    qec_cycles_per_round: int = Field(..., ge=0, le=4)
    escape_cycle_n: int = Field(..., ge=1, le=8)
    measure_axis: Literal["target_axis", "x", "y", "z"]
    shots: int = Field(..., gt=0, le=100_000)


class CultivationBatchRequest(_StrictRequest):
    """A batch of cultivation points, run in order against the run-long line."""

    schema_version: int = SCHEMA_VERSION
    points: list[CultivationPointRequest] = Field(..., min_length=1, max_length=64)


class JobCultivationPoint(_Strict):
    """Raw per-shot evidence for one cultivation point.

    ``flag_<group>[s] == 1`` iff any detector in that group fired on shot
    ``s``; ``outcomes[s]`` is the terminal logical measurement bit. Arrays are
    empty when the point was rejected (budget exhausted; not charged).
    """

    injection_theta_rad: float
    cultivation_rounds: int
    qec_cycles_per_round: int
    escape_cycle_n: int
    measure_axis: str
    shots: int
    rejected: bool = False
    reject_reason: str | None = None

    outcomes: list[int] = Field(default_factory=list)
    flag_injection: list[int] = Field(default_factory=list)
    flag_cultivation: list[int] = Field(default_factory=list)
    flag_qec: list[int] = Field(default_factory=list)
    flag_graft: list[int] = Field(default_factory=list)

    injections_consumed_this_point: int = 0
    injections_consumed_total: int = 0
    budget_remaining: int = 0


class JobCultivationData(_Strict):
    """Points processed by one run_cultivation_batch call, in request order."""

    kind: Literal["cultivation"] = "cultivation"
    points: list[JobCultivationPoint] = Field(..., min_length=1)


__all__ = [
    "CultivationBatchRequest",
    "CultivationPointRequest",
    "JobCultivationData",
    "JobCultivationPoint",
    "JobMagicBenchmarkData",
    "JobMagicBenchmarkPoint",
    "MagicBenchmarkBatchRequest",
    "MagicBenchmarkPointRequest",
]
