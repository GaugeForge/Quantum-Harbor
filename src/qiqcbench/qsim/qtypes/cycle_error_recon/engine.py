"""Run-long stateful engine for the ``cycle_error_recon`` qtype.

Generates faithful folded-CER evidence (raw 5-bit counts per randomization + the ideal
parity sign) and trusted-basis readout calibration, accumulating three budget meters
(hard-cycle exposure, raw shots, accepted rows) across every batch call in the run.

Per randomization, the folded-CER circuit is simulated in the Pauli-transfer-matrix
picture (``forward.py``): start from the probe's +1 product eigenstate; for each of the
``m`` (even) block repetitions apply a uniformly random Pauli dressing (an elementwise
sign flip on the Pauli vector) then the folded noisy block ``M^x``; reconstruct the
density matrix, rotate to the probe measurement basis, apply the asymmetric readout
confusion, and multinomially sample ``S`` shots. Averaging the probe parity over
randomizations twirls the folded-block error into a Pauli channel. Because the ideal
Clifford carries a probe around its Pauli orbit, the survival is the product of the
folded-block fidelities along that orbit. The backend never returns fidelities, decay
constants, exact probabilities, or hidden parameters.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from qiqcbench.qsim.qtypes.cycle_error_recon import forward as F
from qiqcbench.qsim.qtypes.cycle_error_recon.device import HiddenCerConfig, PublicCerSpec
from qiqcbench.qsim.qtypes.cycle_error_recon.wire import (
    FoldedCerBatchRequest,
    FoldedCerRowResult,
    JobCerCountsData,
    JobReadoutCalibData,
    ReadoutCalibBatchRequest,
    ReadoutCalibRowResult,
    _BudgetView,
)

_TOL = 0  # integer budgets, no tolerance needed


@dataclass
class _Evidence:
    hard_used: int = 0
    raw_used: int = 0
    rows_used: int = 0
    accepted_cer: int = 0
    accepted_calib: int = 0


def _counts_to_dict(samples: np.ndarray) -> dict[str, int]:
    """Map sampled basis indices (q0 = leftmost bit) to a sparse 5-bit count dict."""
    out: dict[str, int] = {}
    for idx in samples:
        bits = "".join(str((int(idx) >> (F.N - 1 - q)) & 1) for q in range(F.N))
        out[bits] = out.get(bits, 0) + 1
    return out


class CerEngine:
    def __init__(self, hidden: HiddenCerConfig, public: PublicCerSpec, rng: np.random.Generator):
        self._rng = rng
        self._lock = threading.Lock()
        self._public = public
        b = public.budgets
        self._hard_cap = b.hard_cycle_exposure
        self._raw_cap = b.raw_shots
        self._rows_cap = b.experiment_rows
        self._rows_per_batch = b.rows_per_batch
        self._ev = _Evidence()
        self._r01 = list(hidden.readout.r01)
        self._r10 = list(hidden.readout.r10)
        self._conf = F.readout_confusion(self._r01, self._r10)
        # per-schedule noisy-cycle PTM (sparse), built once
        self._M = {}
        for name, ch in (("parallel_2", hidden.parallel_2), ("serial_4", hidden.serial_4)):
            self._M[name] = F.noisy_cycle(
                ch.stochastic, ch.coherent_generator, ch.coherent_angle_rad
            )
        self._dict = set(public.stochastic_dictionary)

    # ---- budget view ----
    def _budget_view(self) -> _BudgetView:
        return _BudgetView(
            hard_cycle_exposure_used=self._ev.hard_used,
            hard_cycle_exposure_cap=self._hard_cap,
            raw_shots_used=self._ev.raw_used,
            raw_shots_cap=self._raw_cap,
            rows_used=self._ev.rows_used,
            rows_cap=self._rows_cap,
        )

    # ---- readout calibration ----
    def run_readout_calibration(self, request: ReadoutCalibBatchRequest) -> JobReadoutCalibData:
        with self._lock:
            rows = [self._calib_row(r) for r in request.rows]
            return JobReadoutCalibData(rows=rows, budget=self._budget_view())

    def _calib_row(self, row) -> ReadoutCalibRowResult:
        prep = row.prepared_bitstring
        if self._ev.raw_used + row.shots > self._raw_cap:
            return ReadoutCalibRowResult(
                prepared_bitstring=prep, status="rejected", reject_reason="raw_shots_exhausted"
            )
        if self._ev.rows_used >= self._rows_cap:
            return ReadoutCalibRowResult(
                prepared_bitstring=prep, status="rejected", reject_reason="row_cap_exhausted"
            )
        # ideal prep -> the basis index, then readout confusion, then sample
        idx = int(prep, 2)  # leftmost char = q0 = most significant in our index scheme
        probs = self._conf[:, idx]
        probs = np.clip(probs, 0, None)
        probs = probs / probs.sum()
        samples = self._rng.choice(F.DIM, size=row.shots, p=probs)
        self._ev.raw_used += row.shots
        self._ev.rows_used += 1
        self._ev.accepted_calib += 1
        return ReadoutCalibRowResult(
            prepared_bitstring=prep,
            status="accepted",
            counts=_counts_to_dict(samples),
            shots=row.shots,
        )

    # ---- folded CER ----
    def run_folded_cer(self, request: FoldedCerBatchRequest) -> JobCerCountsData:
        with self._lock:
            rows = [self._cer_row(r) for r in request.rows]
            return JobCerCountsData(rows=rows, budget=self._budget_view())

    def _reject(self, row, reason: str) -> FoldedCerRowResult:
        return FoldedCerRowResult(
            schedule=row.schedule,
            probe_pauli=row.probe_pauli,
            fold_factor=row.fold_factor,
            block_repetitions=row.block_repetitions,
            status="rejected",
            reject_reason=reason,
        )

    def _cer_row(self, row) -> FoldedCerRowResult:
        weight = sum(ch != "I" for ch in row.probe_pauli)
        if not 1 <= weight <= 3:
            return self._reject(row, "probe_weight_out_of_range")
        x, m, R, S = (
            row.fold_factor,
            row.block_repetitions,
            row.num_randomizations,
            row.shots_per_randomization,
        )
        hard_cost = x * m * R * S
        if self._ev.rows_used >= self._rows_cap:
            return self._reject(row, "row_cap_exhausted")
        if self._ev.hard_used + hard_cost > self._hard_cap:
            return self._reject(row, "hard_cycle_exposure_exhausted")
        if self._ev.raw_used + R * S > self._raw_cap:
            return self._reject(row, "raw_shots_exhausted")

        M = self._M[row.schedule]
        probe = row.probe_pauli
        basis = "".join(ch if ch in "XYZ" else "Z" for ch in probe)
        v0 = F.product_state_vector(F.probe_eigenstate(probe))

        # Randomized compiling: each randomization draws m random Pauli dressings; the noisy
        # state accumulates them, and the per-randomization recovery (conjugation by the net
        # dressing Pauli D, with the ideal block = G and m even => G^m = I) undoes the frame so
        # every randomization's measured probe parity is in the same frame (ideal sign +1).
        rand_counts = []
        for _ in range(R):
            v = v0.copy()
            cs = []
            for _ in range(m):
                c = int(self._rng.integers(F.NP))
                cs.append(c)
                v = F.commute_signs_from_mask(int(F.XMASK[c]), int(F.ZMASK[c])) * v  # dressing
                for _ in range(x):
                    v = M @ v
            # recovery = inverse of the ideal circuit (R = U_ideal^dagger): for k=m..1 apply the
            # ideal block G (x odd) then the dressing c_k. Exact, noiseless, no analytic net-Pauli.
            for c in reversed(cs):
                v = F._G_SPARSE @ v
                v = F.commute_signs_from_mask(int(F.XMASK[c]), int(F.ZMASK[c])) * v
            dist = F.basis_measure_distribution(v, basis, self._conf)
            dist = np.clip(dist, 0, None)
            dist = dist / dist.sum()
            samples = self._rng.choice(F.DIM, size=S, p=dist)
            rand_counts.append(_counts_to_dict(samples))
        ideal_sign = 1

        self._ev.hard_used += hard_cost
        self._ev.raw_used += R * S
        self._ev.rows_used += 1
        self._ev.accepted_cer += 1
        return FoldedCerRowResult(
            schedule=row.schedule,
            probe_pauli=probe,
            fold_factor=x,
            block_repetitions=m,
            status="accepted",
            randomization_counts=rand_counts,
            ideal_parity_sign=ideal_sign,
        )
