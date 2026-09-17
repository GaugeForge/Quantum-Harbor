"""Control + result wire schemas for ``trapped_ion_state_copy_randomized_measurement``.

Three experiment primitives on the async job model, all metered against a single run-long
state-copy budget (and a randomized-basis budget):

* ``run_readout_calibration`` — prepare a computational calibration state, return raw bitstrings.
* ``run_local_pauli_batch`` — single-copy random local-Pauli measurements: raw bitstrings in ion
  order (the legacy ``orm_pauli_batch`` kind carried ion-reversed strings).
* ``run_copy_block_batch`` — collective copy-block cyclic-shift measurements: raw weighted-cycle
  bounded-real samples for every power ``j <= block_size`` (numerator + denominator when
  ``observable="Z0Z1"``).

The engine never returns the density matrix, exact moments, or hidden targets.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from qiqcbench.qsim.core.wire import SCHEMA_VERSION, _Strict, _StrictRequest


class _BudgetView(_Strict):
    copies_used: int
    copies_budget: int
    bases_used: int
    bases_budget: int
    # Additive: the run-long job meter behind the public ``max_jobs`` budget.
    jobs_used: int = 0
    jobs_budget: int = 0


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #


class ReadoutCalibrationRequest(_StrictRequest):
    schema_version: int = SCHEMA_VERSION
    state_label: str = "all_zero"  # 6-bit string or "all_zero"/"all_one"
    num_shots: int = Field(..., ge=1)


class LocalPauliBatchRequest(_StrictRequest):
    schema_version: int = SCHEMA_VERSION
    basis_family: Literal["random_local_pauli"] = "random_local_pauli"
    observable: Literal["Z0Z1", "I"] = "Z0Z1"
    num_bases: int = Field(..., ge=1)
    shots_per_basis: int = Field(..., ge=1)
    random_seed_label: str | None = None


class CopyBlockBatchRequest(_StrictRequest):
    schema_version: int = SCHEMA_VERSION
    block_size: int = Field(..., ge=2, le=4)
    observable: Literal["Z0Z1", "I"] = "Z0Z1"
    num_blocks: int = Field(..., ge=1)
    # No default: the measurement family is an experimental-design choice the agent makes.
    measurement_family: Literal["simultaneous_weighted_cycle", "separate_cyclic_shift"]


# --------------------------------------------------------------------------- #
# Result data (registered on the qtype descriptor -> JobData/JobResult union)
# --------------------------------------------------------------------------- #


class JobReadoutCalData(_Strict):
    kind: Literal["readout_calibration"] = "readout_calibration"
    state_label: str
    bitstrings: list[str]  # bit i = ion i (little-index-first), after readout
    budget: _BudgetView


class JobOrmData(_Strict):
    """Legacy v2 single-copy payload, retained only for historical artifact reads.

    Its bitstrings were decoded little-index-first, so string position ``i`` held ion ``5 - i``.
    Not emitted by the current schema.
    """

    kind: Literal["orm_pauli_batch"] = "orm_pauli_batch"
    basis_family: str
    observable: str
    bases: list[dict]  # [{"basis": ["X","Z",...], "bitstrings": ["010110", ...]}, ...]
    budget: _BudgetView


class JobLocalPauliData(_Strict):
    """Single-copy random local-Pauli payload produced by the current schema (ion-ordered strings)."""

    kind: Literal["local_pauli_batch"] = "local_pauli_batch"
    record_semantics: Literal["ion_ordered_bitstrings_v1"] = "ion_ordered_bitstrings_v1"
    basis_family: str
    observable: str
    # [{"basis": ["X","Z",...], "bitstrings": ["010110", ...]}, ...]; string position i = ion i
    bases: list[dict]
    budget: _BudgetView


class JobCopyBlockData(_Strict):
    """Legacy v1 marginal-coupling payload, retained only for historical artifact reads."""

    kind: Literal["copy_block_batch"] = "copy_block_batch"
    block_size: int
    observable: str
    measurement_family: str
    records: dict[str, dict[str, list[int]]]
    budget: _BudgetView


# This result kind is additive within transport schema 3: v2 emits it, while
# JobCopyBlockData remains registered solely to parse preserved v1 artifacts.
class JobJointCopyBlockData(_Strict):
    """Physical joint-eigenvalue payload produced by the v2 device."""

    kind: Literal["copy_block_joint_observable_batch"] = "copy_block_joint_observable_batch"
    record_semantics: Literal["commuting_joint_eigenvalues_v1"] = "commuting_joint_eigenvalues_v1"
    block_size: int
    observable: str
    measurement_family: str
    # Same list index across every j/num/den is one joint outcome when the family is simultaneous.
    records: dict[str, dict[str, list[float]]]
    budget: _BudgetView


__all__ = [
    "ReadoutCalibrationRequest",
    "LocalPauliBatchRequest",
    "CopyBlockBatchRequest",
    "JobReadoutCalData",
    "JobOrmData",
    "JobLocalPauliData",
    "JobCopyBlockData",
    "JobJointCopyBlockData",
    "_BudgetView",
]
