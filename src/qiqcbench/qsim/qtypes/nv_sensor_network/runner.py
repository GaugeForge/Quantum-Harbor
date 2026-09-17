"""Sense-and-measure runners for the NV sensor-network qtype.

Bridges the wire schemas through ``NvSensorNetworkEngine`` into a ``JobResult``.
Like the other analog qtypes, the primitive is routed via the qtype backend +
actions layer (not the qtype-agnostic ``runner`` facade).

No blanket except around ``sample_point``: the wire model checks the probe
protocol (support size, echo-sign length) and ``NvSensorNetworkSimulatorBackend``
checks node membership, the interrogation-time grid, and shots before the runner
is reached, so anything raised below is a qsim-owned execution fault and must
reach ``JobManager`` to be typed as ``JOB_EXECUTION_ERROR``. Catching it and
returning ``error=str(exc)`` destroyed that marker -- empty for ``MemoryError``
-- so a qsim fault was attributed to the model.
"""

from __future__ import annotations

import math
import time

import numpy as np

from qiqcbench.qsim.core.wire import (
    JobNvSensingData,
    JobNvSensingPoint,
    JobResult,
    JobResultMetadata,
    NvSensingProbeRequest,
    NvSensingSweepRequest,
)
from qiqcbench.qsim.qtypes.nv_sensor_network.device import HiddenNvSensorNetworkConfig
from qiqcbench.qsim.qtypes.nv_sensor_network.engine import (
    NvNodeParams,
    NvSensorNetworkEngine,
)

_TWO_PI = 2.0 * math.pi


def _make_rng(hidden: HiddenNvSensorNetworkConfig, salt: int) -> np.random.Generator:
    return np.random.default_rng((hidden.seed * 1_000_003) ^ salt)


def node_params_from_hidden(hidden: HiddenNvSensorNetworkConfig) -> NvNodeParams:
    """Convert the hidden config to engine node arrays (node id = list order)."""
    ordered = sorted(hidden.nodes, key=lambda nd: nd.id)
    return NvNodeParams(
        theta_rad_per_s=tuple(_TWO_PI * nd.field_hz for nd in ordered),
        t2star_s=tuple(nd.t2star_us * 1e-6 for nd in ordered),
        link_fidelity=tuple(nd.link_fidelity for nd in ordered),
        p_0_to_1=tuple(nd.p_0_to_1 for nd in ordered),
        p_1_to_0=tuple(nd.p_1_to_0 for nd in ordered),
    )


def _bell_pairs(probe_type: str, support: list[int]) -> int:
    if probe_type == "link_probe":
        return 1
    if probe_type == "ghz":
        return len(support) - 1
    return 0


def _run(
    *,
    hidden: HiddenNvSensorNetworkConfig,
    probe_type: str,
    support: list[int],
    per_point_tau_s: list[list[float]],
    sign_list: list[int],
    analysis_phase_rad: float,
    shots: int,
    job_id: str,
    salt: int,
) -> JobResult:
    t_start = time.perf_counter()
    rng = _make_rng(hidden, salt)
    engine = NvSensorNetworkEngine(node_params_from_hidden(hidden), rng)
    points: list[JobNvSensingPoint] = []
    for tau_list in per_point_tau_s:
        strings = engine.sample_point(
            probe_type=probe_type,
            support=support,
            tau_list_s=tau_list,
            sign_list=sign_list,
            analysis_phase_rad=analysis_phase_rad,
            shots=shots,
        )
        points.append(JobNvSensingPoint(interrogation_time_s=list(tau_list), bitstrings=strings))
    sweep_coords = (
        {"interrogation_time_s": [p.interrogation_time_s[0] for p in points]}
        if len(points) > 1
        else None
    )
    return JobResult(
        job_id=job_id,
        device_id=hidden.device_id,
        status="complete",
        shots=shots,
        data=JobNvSensingData(
            probe_type=probe_type,
            support=list(support),
            echo_sign=list(sign_list),
            analysis_phase_rad=analysis_phase_rad,
            bell_pairs_consumed=_bell_pairs(probe_type, support),
            points=points,
        ),
        metadata=JobResultMetadata(
            wallclock_ms=int((time.perf_counter() - t_start) * 1000),
            sweep_coords=sweep_coords,
        ),
    )


def run_sensing_probe(
    request: NvSensingProbeRequest,
    hidden: HiddenNvSensorNetworkConfig,
    job_id: str,
    salt: int,
) -> JobResult:
    return _run(
        hidden=hidden,
        probe_type=request.probe_type,
        support=list(request.support),
        per_point_tau_s=[list(request.interrogation_time_s)],
        sign_list=list(request.echo_sign),
        analysis_phase_rad=request.analysis_phase_rad,
        shots=request.shots,
        job_id=job_id,
        salt=salt,
    )


def run_sensing_sweep(
    request: NvSensingSweepRequest,
    hidden: HiddenNvSensorNetworkConfig,
    job_id: str,
    salt: int,
) -> JobResult:
    k = len(request.support)
    per_point = [[float(g)] * k for g in request.interrogation_time_grid_s]
    return _run(
        hidden=hidden,
        probe_type=request.probe_type,
        support=list(request.support),
        per_point_tau_s=per_point,
        sign_list=list(request.echo_sign),
        analysis_phase_rad=request.analysis_phase_rad,
        shots=request.shots,
        job_id=job_id,
        salt=salt,
    )


__all__ = ["node_params_from_hidden", "run_sensing_probe", "run_sensing_sweep"]
