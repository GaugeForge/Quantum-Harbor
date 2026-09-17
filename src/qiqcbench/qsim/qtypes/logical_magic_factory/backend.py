"""Simulator backend for the logical_magic_factory qtype.

Unlike the digital/transmon backends (which build a fresh stateless engine per
job), this backend holds **one** :class:`MagicFactoryEngine` for the whole
run, so the magic-state budget accumulates across ``run_magic_benchmark_batch``
calls. The engine's RNG is injected once here from ``hidden.seed``
(engine-contract invariant: the factory injects, the engine never reseeds
itself).
"""

from __future__ import annotations

import numpy as np

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.logical_magic_factory.cultivation import (
    CultivationEngine,
    CultivationParams,
    Schedule,
)
from qiqcbench.qsim.qtypes.logical_magic_factory.device import (
    HiddenMagicFactoryConfig,
    PublicMagicFactorySpec,
)
from qiqcbench.qsim.qtypes.logical_magic_factory.engine import MagicFactoryEngine
from qiqcbench.qsim.qtypes.logical_magic_factory.wire import (
    CultivationBatchRequest,
    JobCultivationData,
    JobCultivationPoint,
    JobMagicBenchmarkData,
    MagicBenchmarkBatchRequest,
)

__all__ = ["MagicFactorySimulatorBackend", "build_magic_factory_simulator_backend"]


class MagicFactorySimulatorBackend:
    """Holds the single run-long magic-factory engine and answers benchmark batches."""

    def __init__(self, hidden: HiddenMagicFactoryConfig, public: PublicMagicFactorySpec):
        self._device_id = hidden.device_id
        self._max_shots = public.max_shots
        rng = np.random.default_rng(hidden.seed)
        self._engine = MagicFactoryEngine(hidden, rng)
        # The cultivation line (if present) owns a DISTINCT deterministic RNG
        # stream so the bell path stays byte-identical and neither stream
        # depends on cross-capability call interleaving.
        self._cultivation: CultivationEngine | None = None
        if hidden.cultivation is not None:
            c = hidden.cultivation
            params = CultivationParams(
                **{k: getattr(c, k) for k in CultivationParams.__dataclass_fields__}
            )
            self._cultivation = CultivationEngine(
                params, c.injection_budget, np.random.default_rng([hidden.seed, 1])
            )

    @property
    def engine(self) -> MagicFactoryEngine:
        return self._engine

    @property
    def cultivation_engine(self) -> CultivationEngine | None:
        return self._cultivation

    def _failed(self, job_id: str, error: str) -> JobResult:
        return JobResult(job_id=job_id, device_id=self._device_id, status="failed", error=error)

    def run_magic_benchmark_batch(
        self, request: MagicBenchmarkBatchRequest, job_id: str, salt: int
    ) -> JobResult:
        """Execute one batch of points. ``salt`` is unused: the run-long engine
        owns a single RNG stream so the accumulated budget and noise are
        reproducible per device seed."""
        for point in request.points:
            if point.shots > self._max_shots:
                return self._failed(job_id, f"shots exceeds max_shots {self._max_shots}")
        out_points = self._engine.run_batch(request.points)
        data = JobMagicBenchmarkData(points=out_points)
        return JobResult(
            job_id=job_id,
            device_id=self._device_id,
            status="complete",
            shots=out_points[0].shots if out_points else None,
            data=data,
        )

    def run_cultivation_batch(
        self, request: CultivationBatchRequest, job_id: str, salt: int
    ) -> JobResult:
        """Execute one batch of cultivation points. ``salt`` is unused (single
        run-long RNG stream, as for the bell engine)."""
        if self._cultivation is None:
            return self._failed(job_id, "this device has no cultivation line")
        for point in request.points:
            if point.shots > self._max_shots:
                return self._failed(job_id, f"shots exceeds max_shots {self._max_shots}")
        out_points: list[JobCultivationPoint] = []
        for point in request.points:
            sch = Schedule(
                injection_theta_rad=point.injection_theta_rad,
                cultivation_rounds=point.cultivation_rounds,
                qec_cycles_per_round=point.qec_cycles_per_round,
                escape_cycle_n=point.escape_cycle_n,
            )
            before = self._cultivation.injections_consumed
            raw = self._cultivation.run_point(sch, point.measure_axis, point.shots)
            base = dict(
                injection_theta_rad=point.injection_theta_rad,
                cultivation_rounds=point.cultivation_rounds,
                qec_cycles_per_round=point.qec_cycles_per_round,
                escape_cycle_n=point.escape_cycle_n,
                measure_axis=point.measure_axis,
                shots=point.shots,
                injections_consumed_total=self._cultivation.injections_consumed,
                budget_remaining=self._cultivation.budget_remaining,
            )
            if raw is None:
                out_points.append(
                    JobCultivationPoint(**base, rejected=True, reject_reason="budget_exhausted")
                )
                continue
            out_points.append(
                JobCultivationPoint(
                    **base,
                    outcomes=raw["outcomes"],
                    flag_injection=raw["flag_injection"],
                    flag_cultivation=raw["flag_cultivation"],
                    flag_qec=raw["flag_qec"],
                    flag_graft=raw["flag_graft"],
                    injections_consumed_this_point=self._cultivation.injections_consumed - before,
                )
            )
        data = JobCultivationData(points=out_points)
        return JobResult(
            job_id=job_id,
            device_id=self._device_id,
            status="complete",
            shots=out_points[0].shots if out_points else None,
            data=data,
        )


def build_magic_factory_simulator_backend(
    hidden: HiddenMagicFactoryConfig, public: PublicMagicFactorySpec
) -> MagicFactorySimulatorBackend:
    return MagicFactorySimulatorBackend(hidden, public)
