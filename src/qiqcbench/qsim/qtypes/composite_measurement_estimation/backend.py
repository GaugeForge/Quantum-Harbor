"""Simulator backend for ``composite_measurement_estimation``.

Holds one run-long :class:`CompositeMeasurementEngine` (pilot budget + evaluator
call cap + one-shot production guard accumulate across calls). Exposes the two
synchronous calculators (observable spec, scheme evaluation) and the two async
experiment entrypoints. Each backend run obtains fresh entropy, then creates
independent pilot, production-basis, and production-outcome random streams.
"""

from __future__ import annotations

import secrets

import numpy as np

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.composite_measurement_estimation.device import (
    HiddenCmeConfig,
    PublicCmeSpec,
)
from qiqcbench.qsim.qtypes.composite_measurement_estimation.engine import (
    CompositeMeasurementEngine,
    CompositeMeasurementError,
)
from qiqcbench.qsim.qtypes.composite_measurement_estimation.wire import (
    JobPilotCountsData,
    JobProductionCountsData,
    LockedProductionRequest,
    PilotBatchRequest,
    PilotRowResult,
    ProductionRowResult,
    _PilotBudgetView,
)

__all__ = ["CmeSimulatorBackend", "build_composite_measurement_simulator_backend"]


class CmeSimulatorBackend:
    def __init__(
        self,
        hidden: HiddenCmeConfig,
        public: PublicCmeSpec,
        *,
        run_entropy: int | None = None,
    ):
        self._device_id = public.device_id
        self._public = public
        if run_entropy is None:
            run_entropy = secrets.randbits(128)
        if (
            not isinstance(run_entropy, int)
            or isinstance(run_entropy, bool)
            or not 0 <= run_entropy < 2**128
        ):
            raise ValueError("run_entropy must be a 128-bit non-negative integer")
        pilot_seed, basis_seed, outcome_seed = np.random.SeedSequence(run_entropy).spawn(3)
        self._engine = CompositeMeasurementEngine(
            hidden,
            public,
            pilot_rng=np.random.default_rng(pilot_seed),
            production_basis_rng=np.random.default_rng(basis_seed),
            production_outcome_rng=np.random.default_rng(outcome_seed),
        )

    @property
    def engine(self) -> CompositeMeasurementEngine:
        return self._engine

    def _failed(self, job_id: str, error: str) -> JobResult:
        return JobResult(job_id=job_id, device_id=self._device_id, status="failed", error=error)

    # --- synchronous calculators ---
    def observable_spec(self) -> dict:
        p = self._public
        return {
            "task_id": p.task_id,
            "device_id": p.device_id,
            "molecule": p.molecule,
            "basis": p.basis,
            "fermion_mapping": p.fermion_mapping,
            "pauli_string_order": p.pauli_string_order,
            "n_qubits": p.n_qubits,
            "num_nonidentity_terms": p.num_nonidentity_terms,
            "hamiltonian_file": p.instance_files.hamiltonian,
            "stale_notebook_file": p.instance_files.stale_notebook,
            "num_components": p.num_components,
            "coverage_floor": p.coverage_floor,
            "pilot_max_settings": p.pilot_max_settings,
            "pilot_max_shots": p.pilot_max_shots,
            "pilot_shots_per_setting": [
                p.pilot_shots_per_setting_min,
                p.pilot_shots_per_setting_max,
            ],
            "pilot_shots_multiple_of": p.pilot_shots_multiple_of,
            "production_settings": p.production_settings,
            "production_shots_per_setting": p.production_shots_per_setting,
            "max_control_variates": p.max_control_variates,
            "evaluator_call_cap": p.evaluator_call_cap,
            "estimator_protocol": p.estimator_protocol,
            "estimator_formula": (
                "E_hat = c_I + (1/(B*R)) * sum_b sum_r [ sum_j c_j m_j + "
                "sum_{j: P_j <= Q_b} (c_j/h_j)(mu(P_j,x_br) - m_j) ]; "
                "mu(P_j,x) = prod_{i: P_j[i]!=I} (-1)^x[i]"
            ),
        }

    def evaluate_scheme(
        self, *, mixture_weights: list, local_basis_probabilities_xyz: list
    ) -> dict:
        return self._engine.evaluate_scheme(mixture_weights, local_basis_probabilities_xyz)

    # --- async experiments ---
    def run_pilot_measurements(
        self, request: PilotBatchRequest, job_id: str, salt: int
    ) -> JobResult:
        rows = [(r.basis, r.shots) for r in request.rows]
        result = self._engine.run_pilot_batch(rows)
        data = JobPilotCountsData(
            rows=[PilotRowResult(**r) for r in result["rows"]],
            budget=_PilotBudgetView(**result["budget"]),
        )
        return JobResult(job_id=job_id, device_id=self._device_id, status="complete", data=data)

    def execute_locked_composite_scheme(
        self, request: LockedProductionRequest, job_id: str, salt: int
    ) -> tuple[JobResult, dict]:
        """Return ``(job_result, verifier_evidence)``. Evidence is logged by the action."""
        try:
            full = self._engine.run_locked_production(
                request.mixture_weights,
                request.local_basis_probabilities_xyz,
                [e.model_dump() for e in request.control_variate_entries],
            )
        except CompositeMeasurementError as exc:
            return self._failed(job_id, str(exc)), {}
        data = JobProductionCountsData(
            rows=[ProductionRowResult(basis=r["basis"], counts=r["counts"]) for r in full["rows"]],
            request_digest=full["request_digest"],
            production_settings=full["production_settings"],
            production_shots_per_setting=full["production_shots_per_setting"],
        )
        result = JobResult(job_id=job_id, device_id=self._device_id, status="complete", data=data)
        return result, full


def build_composite_measurement_simulator_backend(
    hidden: HiddenCmeConfig,
    public: PublicCmeSpec,
    *,
    run_entropy: int | None = None,
) -> CmeSimulatorBackend:
    return CmeSimulatorBackend(hidden, public, run_entropy=run_entropy)
