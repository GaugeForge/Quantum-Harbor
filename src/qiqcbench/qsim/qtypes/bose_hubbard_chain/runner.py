"""Evolve-and-measure runners for the Bose-Hubbard chain qtype.

Bridges wire schemas through ``BoseHubbardEngine`` into a ``JobResult``. The
analog primitive is routed via the qtype backend + actions layer (not the
qtype-agnostic ``runner`` façade), so these functions take both the hidden
config (coherence/noise + calibration drift) and the public spec (the nominal
Hamiltonian). ``realized_hamiltonian`` combines the two.
"""

from __future__ import annotations

import secrets
import time

import numpy as np

from qiqcbench.qsim.core.wire import (
    BoseHubbardEvolveRequest,
    BoseHubbardEvolveSweepRequest,
    JobBitstringData,
    JobResult,
    JobResultMetadata,
)
from qiqcbench.qsim.qtypes.bose_hubbard_chain.device import (
    HiddenBoseHubbardConfig,
    PublicBoseHubbardSpec,
)
from qiqcbench.qsim.qtypes.bose_hubbard_chain.engine import (
    BoseHubbardEngine,
    HamiltonianSpec,
)


def fresh_run_entropy() -> int:
    """Draw one private 128-bit entropy root for a benchmark attempt."""

    return secrets.randbits(128)


def _make_rng(
    hidden: HiddenBoseHubbardConfig,
    salt: int,
    run_entropy: int | None = None,
) -> np.random.Generator:
    """Derive an independent per-job stream from fixed truth and fresh entropy."""

    if run_entropy is None:
        run_entropy = fresh_run_entropy()
    if (
        isinstance(run_entropy, bool)
        or not isinstance(run_entropy, int)
        or not 0 <= run_entropy < 2**128
    ):
        raise ValueError("run_entropy must be a 128-bit non-negative integer")
    seed = np.random.SeedSequence(
        [
            int(hidden.seed) & ((1 << 64) - 1),
            run_entropy & ((1 << 64) - 1),
            run_entropy >> 64,
            int(salt),
        ]
    )
    return np.random.default_rng(seed)


def realized_hamiltonian(
    hidden: HiddenBoseHubbardConfig, public: PublicBoseHubbardSpec
) -> HamiltonianSpec:
    """Public *nominal* coefficients plus the hidden frozen calibration drift.

    The single place where nominal becomes realized. Everything the device does
    — every shot, and the hidden reference spectrum itself — is generated from
    the result, so an agent cannot reproduce it from the public spec alone.
    """
    ordered = sorted(public.sites, key=lambda s: s.id)
    mu = [s.on_site_potential_mhz for s in ordered]
    hop = list(public.hopping_mhz)
    u = public.interaction_u_mhz

    drift = hidden.hamiltonian_drift
    if drift is not None:
        if len(drift.on_site_potential_mhz) != len(mu):
            raise ValueError(
                f"hamiltonian_drift.on_site_potential_mhz has "
                f"{len(drift.on_site_potential_mhz)} entries, expected {len(mu)}"
            )
        if len(drift.hopping_mhz) != len(hop):
            raise ValueError(
                f"hamiltonian_drift.hopping_mhz has {len(drift.hopping_mhz)} "
                f"entries, expected {len(hop)}"
            )
        mu = [a + b for a, b in zip(mu, drift.on_site_potential_mhz, strict=True)]
        hop = [a + b for a, b in zip(hop, drift.hopping_mhz, strict=True)]
        u += drift.interaction_u_mhz

    return HamiltonianSpec(
        on_site_mhz=tuple(mu),
        hopping_mhz=tuple(hop),
        interaction_mhz=u,
        max_excitations=public.max_excitations,
    )


def build_engine(
    hidden: HiddenBoseHubbardConfig,
    public: PublicBoseHubbardSpec,
    rng: np.random.Generator,
) -> BoseHubbardEngine:
    spec = realized_hamiltonian(hidden, public)
    readout = {r.id: (r.p_0_to_1, r.p_1_to_0) for r in hidden.readout}
    return BoseHubbardEngine(spec=spec, t2_ns=hidden.t2_ns, readout=readout, rng=rng)


def _run(
    *,
    hidden: HiddenBoseHubbardConfig,
    public: PublicBoseHubbardSpec,
    init_sites: list[int],
    time_grid_ns: list[float],
    measure: list[tuple[int, str]],
    shots: int,
    job_id: str,
    salt: int,
    run_entropy: int | None,
) -> JobResult:
    t_start = time.perf_counter()
    rng = _make_rng(hidden, salt, run_entropy)
    engine = build_engine(hidden, public, rng)
    # No blanket except here. Every agent-supplied control is bound by the wire
    # schema, the action preflight, and the backend validation, all of which run
    # first, so anything raised below is a qsim-owned execution fault. Letting it
    # propagate reaches JobManager, which types it as the sanctioned
    # JOB_EXECUTION_ERROR that ``is_internal_job_failure`` recognizes and
    # ``get_job_result`` stamps ``failure_kind="qsim_internal"``. Catching it here
    # replaced that marker with ``str(exc)`` -- empty for MemoryError, which is
    # falsy, so the poll event carried no error and no stamp at all -- and the
    # harper verifier then charged a qsim fault to the model.
    per_time = engine.sample_time_series(
        init_sites=init_sites,
        time_grid_ns=time_grid_ns,
        measure=measure,
        shots=shots,
    )
    return JobResult(
        job_id=job_id,
        device_id=hidden.device_id,
        status="complete",
        shots=shots,
        data=JobBitstringData(
            bitstrings=per_time,
            measured_qubits=[s for s, _ in measure],
        ),
        metadata=JobResultMetadata(
            wallclock_ms=int((time.perf_counter() - t_start) * 1000),
            sweep_coords={"time_ns": [float(t) for t in time_grid_ns]},
        ),
    )


def run_evolution(
    request: BoseHubbardEvolveRequest,
    hidden: HiddenBoseHubbardConfig,
    public: PublicBoseHubbardSpec,
    job_id: str,
    salt: int,
    *,
    run_entropy: int | None = None,
) -> JobResult:
    return _run(
        hidden=hidden,
        public=public,
        init_sites=list(request.init_excited_sites),
        time_grid_ns=[request.evolution_time_ns],
        measure=[(m.site, m.basis) for m in request.measure],
        shots=request.shots,
        job_id=job_id,
        salt=salt,
        run_entropy=run_entropy,
    )


def run_evolution_sweep(
    request: BoseHubbardEvolveSweepRequest,
    hidden: HiddenBoseHubbardConfig,
    public: PublicBoseHubbardSpec,
    job_id: str,
    salt: int,
    *,
    run_entropy: int | None = None,
) -> JobResult:
    return _run(
        hidden=hidden,
        public=public,
        init_sites=list(request.init_excited_sites),
        time_grid_ns=[float(t) for t in request.time_grid_ns],
        measure=[(m.site, m.basis) for m in request.measure],
        shots=request.shots,
        job_id=job_id,
        salt=salt,
        run_entropy=run_entropy,
    )


__all__ = [
    "build_engine",
    "fresh_run_entropy",
    "realized_hamiltonian",
    "run_evolution",
    "run_evolution_sweep",
]
