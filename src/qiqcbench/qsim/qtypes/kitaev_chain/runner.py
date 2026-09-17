"""Request -> JobResult runners for the Kitaev-chain qtype.

Bridges the qtype-local wire requests through ``KitaevChainEngine`` into a
``JobResult``. One fresh engine per job; its RNG combines the committed hidden
seed, private per-attempt entropy, and the per-job salt.
"""

from __future__ import annotations

import secrets
import time

import numpy as np

from qiqcbench.qsim.core.wire import JobResult, JobResultMetadata
from qiqcbench.qsim.qtypes.kitaev_chain.device import HiddenKitaevChainConfig
from qiqcbench.qsim.qtypes.kitaev_chain.engine import KitaevChainEngine
from qiqcbench.qsim.qtypes.kitaev_chain.wire import (
    ChargeStabilityRequest,
    MajoranaCliffordRBRequest,
    MajoranaPulseBatchRequest,
    ProtectionSweepRequest,
    SubchainSpectroscopyRequest,
)


def fresh_run_entropy() -> int:
    """Draw private per-attempt entropy from the operating system."""
    return secrets.randbits(128)


def _make_rng(
    hidden: HiddenKitaevChainConfig,
    salt: int,
    run_entropy: int | None = None,
) -> np.random.Generator:
    if run_entropy is None:
        run_entropy = fresh_run_entropy()
    return np.random.default_rng([hidden.seed, run_entropy, salt])


def build_engine(hidden: HiddenKitaevChainConfig, rng: np.random.Generator) -> KitaevChainEngine:
    return KitaevChainEngine(hidden, rng)


def _result(
    job_id: str, hidden: HiddenKitaevChainConfig, shots: int, data, t_start: float
) -> JobResult:
    return JobResult(
        job_id=job_id,
        device_id=hidden.device_id,
        status="complete",
        shots=shots,
        data=data,
        metadata=JobResultMetadata(wallclock_ms=int((time.perf_counter() - t_start) * 1000)),
    )


# No blanket except around the engine calls below. Every agent-visible bound
# (bond, |mu| window, site ordering, shots, segment count) is checked by
# ``KitaevChainSimulatorBackend`` before the runner is reached, so anything
# raised here is a qsim-owned execution fault. Letting it propagate reaches
# ``JobManager``, which types it as the sanctioned ``JOB_EXECUTION_ERROR`` that
# ``is_internal_job_failure`` recognizes and ``get_job_result`` stamps
# ``failure_kind="qsim_internal"``. Returning ``error=str(exc)`` instead
# replaced that marker with free-form text -- empty for ``MemoryError`` -- so a
# qsim fault was attributed to the model.


def run_charge_stability(
    request: ChargeStabilityRequest,
    hidden: HiddenKitaevChainConfig,
    job_id: str,
    salt: int,
    *,
    run_entropy: int | None = None,
) -> JobResult:
    t_start = time.perf_counter()
    engine = build_engine(hidden, _make_rng(hidden, salt, run_entropy))
    data = engine.run_charge_stability(
        request.bond, list(request.mu_ld_values), list(request.mu_rd_values), request.shots
    )
    return _result(job_id, hidden, request.shots, data, t_start)


def run_protection_sweep(
    request: ProtectionSweepRequest,
    hidden: HiddenKitaevChainConfig,
    job_id: str,
    salt: int,
    *,
    run_entropy: int | None = None,
) -> JobResult:
    t_start = time.perf_counter()
    engine = build_engine(hidden, _make_rng(hidden, salt, run_entropy))
    data = engine.run_protection_sweep(list(request.mu_common_values), request.shots)
    return _result(job_id, hidden, request.shots, data, t_start)


def run_subchain_spectroscopy(
    request: SubchainSpectroscopyRequest,
    hidden: HiddenKitaevChainConfig,
    job_id: str,
    salt: int,
    *,
    run_entropy: int | None = None,
) -> JobResult:
    t_start = time.perf_counter()
    engine = build_engine(hidden, _make_rng(hidden, salt, run_entropy))
    data = engine.run_subchain_spectroscopy(request.site_a, request.site_b, request.shots)
    return _result(job_id, hidden, request.shots, data, t_start)


def run_pulse_batch(
    request: MajoranaPulseBatchRequest,
    hidden: HiddenKitaevChainConfig,
    job_id: str,
    salt: int,
    *,
    run_entropy: int | None = None,
) -> JobResult:
    t_start = time.perf_counter()
    engine = build_engine(hidden, _make_rng(hidden, salt, run_entropy))
    data = engine.run_pulse_batch(request)
    return _result(job_id, hidden, request.shots, data, t_start)


def run_clifford_rb(
    request: MajoranaCliffordRBRequest,
    hidden: HiddenKitaevChainConfig,
    job_id: str,
    salt: int,
    *,
    run_entropy: int | None = None,
) -> JobResult:
    t_start = time.perf_counter()
    engine = build_engine(hidden, _make_rng(hidden, salt, run_entropy))
    data = engine.run_clifford_rb(request)
    return _result(job_id, hidden, request.shots_per_sequence, data, t_start)


__all__ = [
    "build_engine",
    "fresh_run_entropy",
    "run_charge_stability",
    "run_protection_sweep",
    "run_subchain_spectroscopy",
    "run_pulse_batch",
    "run_clifford_rb",
]
