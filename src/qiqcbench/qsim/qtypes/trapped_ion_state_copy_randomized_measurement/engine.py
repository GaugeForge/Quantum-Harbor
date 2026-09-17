"""Run-long stateful engine for ``trapped_ion_state_copy_randomized_measurement``.

Builds the hidden 6-ion density matrix ``rho`` once (a single cached 64x64 matrix exponential),
then serves measurement requests by sampling the weighted-permutation observables' small joint
spectral measure — no ``2^{k*n}`` joint copy-block state is ever assembled. A single **state-copy** meter
and a **randomized-basis** meter accumulate across every call in the run; requests that would
exceed either budget fail closed. The engine never returns the density matrix, exact moments, or
hidden targets — only raw bitstrings / raw bounded weighted-cycle samples.
"""

from __future__ import annotations

import threading

import numpy as np

from qiqcbench.qsim.qtypes.trapped_ion_state_copy_randomized_measurement import physics as P
from qiqcbench.qsim.qtypes.trapped_ion_state_copy_randomized_measurement.device import (
    HiddenStateCopyConfig,
)
from qiqcbench.qsim.qtypes.trapped_ion_state_copy_randomized_measurement.wire import (
    CopyBlockBatchRequest,
    JobJointCopyBlockData,
    JobLocalPauliData,
    JobReadoutCalData,
    LocalPauliBatchRequest,
    ReadoutCalibrationRequest,
    _BudgetView,
)


class StateCopyEngine:
    def __init__(
        self,
        hidden: HiddenStateCopyConfig,
        rng: np.random.Generator,
        *,
        max_jobs: int | None = None,
    ):
        self._rng = rng
        self._lock = threading.Lock()
        s = hidden.state
        self._rho = P.build_rho(
            s.J0, s.alpha, s.B_over_J0, s.delta_B_over_J0, s.beta_J0, s.depolarizing_lambda
        )
        self._p01 = list(hidden.readout.p01)
        self._p10 = list(hidden.readout.p10)
        self._copies_budget = hidden.max_state_copies
        self._bases_budget = hidden.max_randomized_bases
        self._max_block = hidden.max_copy_block_size
        self._copies_used = 0
        self._bases_used = 0
        # Run-long job meter (public budget ``max_jobs``): every charged experiment counts.
        self._jobs_budget = max_jobs
        self._jobs_used = 0

    def _budget_view(self) -> _BudgetView:
        return _BudgetView(
            copies_used=self._copies_used,
            copies_budget=self._copies_budget,
            bases_used=self._bases_used,
            bases_budget=self._bases_budget,
            jobs_used=self._jobs_used,
            jobs_budget=self._jobs_budget if self._jobs_budget is not None else 0,
        )

    def _charge(self, copies: int, bases: int) -> str | None:
        if self._jobs_budget is not None and self._jobs_used + 1 > self._jobs_budget:
            return f"job budget exhausted: {self._jobs_used} jobs already run (max {self._jobs_budget})"
        if self._copies_used + copies > self._copies_budget:
            return (
                f"state-copy budget exhausted: {self._copies_used}+{copies} "
                f"> budget {self._copies_budget}"
            )
        if self._bases_used + bases > self._bases_budget:
            return (
                f"randomized-basis budget exhausted: {self._bases_used}+{bases} "
                f"> budget {self._bases_budget}"
            )
        self._copies_used += copies
        self._bases_used += bases
        self._jobs_used += 1
        return None

    def run_readout_calibration(
        self, request: ReadoutCalibrationRequest
    ) -> JobReadoutCalData | str:
        with self._lock:
            copies = request.num_shots
            err = self._charge(copies, bases=1)
            if err:
                return err
            try:
                bitstrings = P.sample_readout_calibration(
                    request.state_label, request.num_shots, self._p01, self._p10, self._rng
                )
            except P.MeasurementError as exc:
                # refund on a bad request
                self._copies_used -= copies
                self._bases_used -= 1
                self._jobs_used -= 1
                return str(exc)
            return JobReadoutCalData(
                state_label=request.state_label,
                bitstrings=bitstrings,
                budget=self._budget_view(),
            )

    def run_local_pauli_batch(self, request: LocalPauliBatchRequest) -> JobLocalPauliData | str:
        with self._lock:
            copies = request.num_bases * request.shots_per_basis
            bases = request.num_bases
            try:
                P.validate_local_pauli(
                    request.basis_family,
                    request.observable,
                    request.num_bases,
                    request.shots_per_basis,
                )
            except P.MeasurementError as exc:
                return str(exc)
            err = self._charge(copies, bases)
            if err:
                return err
            batch = P.sample_local_pauli(
                self._rho,
                request.basis_family,
                request.num_bases,
                request.shots_per_basis,
                self._p01,
                self._p10,
                self._rng,
            )
            return JobLocalPauliData(
                basis_family=request.basis_family,
                observable=request.observable,
                bases=batch,
                budget=self._budget_view(),
            )

    def run_copy_block_batch(self, request: CopyBlockBatchRequest) -> JobJointCopyBlockData | str:
        with self._lock:
            if request.block_size > self._max_block:
                return f"block_size {request.block_size} exceeds max {self._max_block}"
            try:
                P.validate_copy_block(
                    request.block_size,
                    request.num_blocks,
                    request.observable,
                    request.measurement_family,
                )
            except P.MeasurementError as exc:
                return str(exc)
            block_families = (
                2
                if request.observable == "Z0Z1"
                and request.measurement_family != "simultaneous_weighted_cycle"
                else 1
            )
            copies = request.block_size * request.num_blocks * block_families
            err = self._charge(copies, bases=0)
            if err:
                return err
            records = P.sample_copy_block(
                self._rho,
                request.block_size,
                request.num_blocks,
                request.observable,
                request.measurement_family,
                self._rng,
            )
            return JobJointCopyBlockData(
                block_size=request.block_size,
                observable=request.observable,
                measurement_family=request.measurement_family,
                records=records,
                budget=self._budget_view(),
            )
