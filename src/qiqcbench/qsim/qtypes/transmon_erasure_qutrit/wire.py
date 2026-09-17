"""g-f erasure-qutrit control + result wire schemas (qtype-local).

Mirrors how other self-registering qtypes (e.g. ``dipolar_spin_ensemble``,
``transmon_multilevel_pulse``) keep their request schemas local and contribute
their result type to the dynamic ``JobData`` union via
``QtypeDescriptor.result_data_models`` — so no edit to ``core/wire.py`` is
needed. ``SCHEMA_VERSION`` and ``_Strict`` are imported from ``core/wire.py``.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from qiqcbench.qsim.core.wire import SCHEMA_VERSION, _Strict, _StrictRequest

# ---------- Control (requests) ----------


def _validate_matched_preparation(
    *,
    prep_basis: str,
    prep_state: str,
    measure_basis: str,
) -> None:
    allowed = {"z": {"0L", "1L"}, "x": {"+X", "-X"}}
    if prep_state not in allowed[prep_basis]:
        raise ValueError(f"prep_state does not belong to prep_basis {prep_basis!r}")
    if measure_basis != prep_basis:
        raise ValueError("measure_basis must match prep_basis")


class LogicalMemoryRequest(_StrictRequest):
    """Prepare a logical state, interleave DD + mid-circuit erasure detection for a
    fixed number of rounds at a chosen cycle time, then read out the data qutrit.

    Preparation and measurement use the same basis. Each shot returns the
    ordered raw ancilla bits for every round and the same shot's final
    post-readout qutrit assignment (g/e/f).
    """

    schema_version: int = SCHEMA_VERSION
    shots: int = Field(..., gt=0, le=1_000_000)
    prep_basis: Literal["z", "x"]
    prep_state: Literal["0L", "1L", "+X", "-X"]
    measure_basis: Literal["z", "x"]
    n_rounds: int = Field(..., ge=0)
    cycle_time_us: float = Field(..., gt=0)
    dd: Literal["xy4", "none"] = "xy4"

    @model_validator(mode="after")
    def _validate_matched_basis(self) -> LogicalMemoryRequest:
        _validate_matched_preparation(
            prep_basis=self.prep_basis,
            prep_state=self.prep_state,
            measure_basis=self.measure_basis,
        )
        return self


class LogicalMemorySweepRequest(_StrictRequest):
    """Same as LogicalMemoryRequest but over a grid of round counts (one record per
    point, in order). The sweep axis is the number of erasure-detection rounds;
    total evolution time of a point is ``n_rounds * cycle_time_us``.
    """

    schema_version: int = SCHEMA_VERSION
    shots: int = Field(..., gt=0, le=1_000_000)
    prep_basis: Literal["z", "x"]
    prep_state: Literal["0L", "1L", "+X", "-X"]
    measure_basis: Literal["z", "x"]
    n_rounds_grid: list[Annotated[int, Field(ge=0)]] = Field(..., min_length=1)
    cycle_time_us: float = Field(..., gt=0)
    dd: Literal["xy4", "none"] = "xy4"

    @model_validator(mode="after")
    def _validate_matched_basis(self) -> LogicalMemorySweepRequest:
        _validate_matched_preparation(
            prep_basis=self.prep_basis,
            prep_state=self.prep_state,
            measure_basis=self.measure_basis,
        )
        return self


# ---------- Result (joins the dynamic JobData union via result_data_models) ----------


class JobErasureMemoryPoint(_Strict):
    """Per-shot erasure-detection outcomes for one (n_rounds) sweep point.

    ``erasure_flags[s] == 1`` iff shot ``s`` had at least one mid-circuit erasure
    flagged across all rounds. ``final_outcomes[s]`` is the final data-qutrit
    readout: 0=g (|0_L>), 1=e (leakage), 2=f (|1_L>).
    """

    n_rounds: int = Field(..., ge=0)
    total_evolution_us: float = Field(..., ge=0)
    erasure_flags: list[int] = Field(..., min_length=1)
    final_outcomes: list[int] = Field(..., min_length=1)


class JobErasureMemoryData(_Strict):
    """Compatibility payload with a derived sticky flag per shot."""

    kind: Literal["erasure_memory"] = "erasure_memory"
    points: list[JobErasureMemoryPoint] = Field(..., min_length=1)
    data_qubit: str
    prep_state: str
    prep_basis: Literal["z", "x"]
    measure_basis: Literal["z", "x"]
    cycle_time_us: float
    dd: Literal["xy4", "none"]


class JobErasureMemoryRoundResolvedPoint(_Strict):
    """Round-resolved digital measurements for one logical-memory point.

    The two lists share a shot index. Each syndrome string contains the ordered
    post-readout ancilla bits for all detection rounds; the final assignment is
    the post-readout data-qutrit label ``0=g, 1=e, 2=f``. No post-selection or
    across-round OR is performed in this public payload.
    """

    n_rounds: int = Field(..., ge=0)
    total_evolution_us: float = Field(..., ge=0)
    mid_circuit_ancilla_post_readout_bitstrings: list[str] = Field(..., min_length=1)
    final_qutrit_post_readout_assignments: list[int] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_shot_records(self) -> JobErasureMemoryRoundResolvedPoint:
        syndromes = self.mid_circuit_ancilla_post_readout_bitstrings
        assignments = self.final_qutrit_post_readout_assignments
        if len(syndromes) != len(assignments):
            raise ValueError("syndrome and final-assignment shot counts must match")
        if any(len(bits) != self.n_rounds or set(bits) - {"0", "1"} for bits in syndromes):
            raise ValueError(
                "each ancilla syndrome bitstring must contain exactly n_rounds binary bits"
            )
        if any(value not in {0, 1, 2} for value in assignments):
            raise ValueError("final qutrit assignments must be 0, 1, or 2")
        return self


# This result kind is additive within transport schema 3: v2 emits it, while the
# legacy ``erasure_memory`` kind remains registered to parse historical artifacts.
class JobErasureMemoryRoundResolvedData(_Strict):
    """Raw round-resolved erasure-memory records grouped by sweep point."""

    kind: Literal["erasure_memory_round_resolved"] = "erasure_memory_round_resolved"
    points: list[JobErasureMemoryRoundResolvedPoint] = Field(..., min_length=1)
    data_qubit: str
    prep_state: str
    prep_basis: Literal["z", "x"]
    measure_basis: Literal["z", "x"]
    cycle_time_us: float
    dd: Literal["xy4", "none"]


__all__ = [
    "JobErasureMemoryData",
    "JobErasureMemoryPoint",
    "JobErasureMemoryRoundResolvedData",
    "JobErasureMemoryRoundResolvedPoint",
    "LogicalMemoryRequest",
    "LogicalMemorySweepRequest",
]
