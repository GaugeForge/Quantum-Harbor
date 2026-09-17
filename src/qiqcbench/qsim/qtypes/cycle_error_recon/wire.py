"""cycle_error_recon control + result wire schemas (qtype-local).

Mirrors how ``dipolar_spin_ensemble/wire.py`` defines its own qtype-local request
schemas rather than widening the shared op unions, and registers its own result-data
models (carrying a ``kind`` discriminator) through the qtype DESCRIPTOR so
``core/wire.py`` is never edited.

Two experiment primitives, both batched and run-long budgeted:

- ``run_readout_calibration_batch`` — trusted-basis (ideal-prep) calibration rows; the
  device returns raw 5-bit counts per prepared bitstring. Counts toward the raw-shot
  budget but NOT the hard-cycle-exposure budget.
- ``run_folded_cer_batch`` — canonical folded cycle-error-reconstruction rows. For fold
  factor ``x`` the hard block is ``(noisy cycle)^x`` (x odd, so the ideal action stays
  the cycle ``G``); randomized Pauli dressing is inserted between folded hard blocks. The
  device returns raw 5-bit counts SEPARATELY for each randomization, plus the ideal parity
  sign needed for post-processing. It never returns fitted fidelities, decay constants,
  exact probabilities, or hidden channel parameters.

Budget accounting (hard-cycle exposure, raw shots, accepted rows) accumulates across every
batch call in the run and is echoed back in each result.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from qiqcbench.qsim.core.wire import SCHEMA_VERSION, _Strict, _StrictRequest

# ---------- run_readout_calibration_batch ----------


class ReadoutCalibRow(_StrictRequest):
    """One trusted-basis readout-calibration row (ideal preparation)."""

    prepared_bitstring: str = Field(..., min_length=5, max_length=5, pattern=r"^[01]{5}$")
    shots: int = Field(..., ge=64, le=4096)


class ReadoutCalibBatchRequest(_StrictRequest):
    schema_version: int = SCHEMA_VERSION
    rows: list[ReadoutCalibRow] = Field(..., min_length=1, max_length=32)


# ---------- run_folded_cer_batch ----------


class FoldedCerRow(_StrictRequest):
    """One folded cycle-error-reconstruction row."""

    schedule: Literal["parallel_2", "serial_4"]
    probe_pauli: str = Field(..., min_length=5, max_length=5, pattern=r"^[IXYZ]{5}$")
    fold_factor: Literal[1, 3, 5, 7]
    block_repetitions: Literal[2, 4, 8, 16, 32]
    num_randomizations: int = Field(..., ge=1, le=32)
    shots_per_randomization: int = Field(..., ge=64, le=1024)


class FoldedCerBatchRequest(_StrictRequest):
    schema_version: int = SCHEMA_VERSION
    rows: list[FoldedCerRow] = Field(..., min_length=1, max_length=32)


# ---------- result-data models (registered via the DESCRIPTOR) ----------


class _BudgetView(_Strict):
    """Cumulative run-long budget echoed in every batch result."""

    hard_cycle_exposure_used: int
    hard_cycle_exposure_cap: int
    raw_shots_used: int
    raw_shots_cap: int
    rows_used: int
    rows_cap: int


class ReadoutCalibRowResult(_Strict):
    prepared_bitstring: str
    status: Literal["accepted", "rejected"]
    counts: dict[str, int] | None = None  # raw 5-bit counts (leftmost bit = q0)
    shots: int | None = None
    reject_reason: str | None = None


class JobReadoutCalibData(_Strict):
    kind: Literal["readout_calib_counts"] = "readout_calib_counts"
    rows: list[ReadoutCalibRowResult]
    budget: _BudgetView


class FoldedCerRowResult(_Strict):
    schedule: str
    probe_pauli: str
    fold_factor: int
    block_repetitions: int
    status: Literal["accepted", "rejected"]
    # one raw 5-bit count dict per randomization (leftmost bit = q0)
    randomization_counts: list[dict[str, int]] | None = None
    ideal_parity_sign: int | None = None  # +1 or -1; ideal sign of the probe expectation
    reject_reason: str | None = None


class JobCerCountsData(_Strict):
    kind: Literal["folded_cer_counts"] = "folded_cer_counts"
    rows: list[FoldedCerRowResult]
    budget: _BudgetView


__all__ = [
    "ReadoutCalibRow",
    "ReadoutCalibBatchRequest",
    "FoldedCerRow",
    "FoldedCerBatchRequest",
    "ReadoutCalibRowResult",
    "JobReadoutCalibData",
    "FoldedCerRowResult",
    "JobCerCountsData",
]
