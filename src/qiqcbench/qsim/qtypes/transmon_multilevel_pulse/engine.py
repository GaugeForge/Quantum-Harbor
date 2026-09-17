"""Multilevel-transmon engine: render pulse segments, propagate, sample shots.

Wraps the pure-numpy :mod:`physics` propagator into the qsim ``JobResult``
contract. Returns raw per-shot level outcomes (0..n_levels-1) after the hidden
level-resolved readout confusion — the agent aggregates these into P0..P3 and
chooses any readout mitigation (raw shots, not counts).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from qiqcbench.qsim.core.wire import (
    JobLevelOutcomeData,
    JobResult,
    JobResultMetadata,
)
from qiqcbench.qsim.qtypes.transmon_multilevel_pulse.device import (
    HiddenTransmonMultilevelConfig,
)
from qiqcbench.qsim.qtypes.transmon_multilevel_pulse.physics import (
    DuffingParams,
    final_populations,
    open_superoperator,
    propagate_density,
)


@dataclass(frozen=True)
class RenderedSegment:
    """A drive segment rendered onto the device sample grid (DAC + Hz arrays)."""

    omega_x_dac: np.ndarray
    omega_y_dac: np.ndarray
    detuning_hz: np.ndarray


def render_analytic(
    *,
    shape: str,
    amp_dac: float,
    duration_ns: float,
    sigma_ns: float | None,
    phase_rad: float,
    carrier_detuning_hz: float,
    sample_dt_ns: float,
) -> RenderedSegment:
    """Render an analytic gaussian/square segment onto the sample grid.

    The carrier phase rotates the single drive between the x and y quadratures
    (a plain analytic pulse has no DRAG derivative term). Endpoints of a gaussian
    are baseline-subtracted so the envelope starts and ends near zero.
    """
    n = max(1, int(round(duration_ns / sample_dt_ns)))
    ts = (np.arange(n) + 0.5) * sample_dt_ns
    if shape == "gaussian":
        sig = sigma_ns if sigma_ns and sigma_ns > 0 else duration_ns / 4.0
        env = np.exp(-((ts - duration_ns / 2.0) ** 2) / (2.0 * sig**2))
        env = env - env.min()
    elif shape == "square":
        env = np.ones(n)
    else:  # pragma: no cover - validated upstream
        raise ValueError(f"Unsupported shape {shape!r}")
    amp = amp_dac * env
    return RenderedSegment(
        omega_x_dac=amp * math.cos(phase_rad),
        omega_y_dac=amp * math.sin(phase_rad),
        detuning_hz=np.full(n, float(carrier_detuning_hz)),
    )


def concat_segments(segments: list[RenderedSegment]) -> RenderedSegment:
    return RenderedSegment(
        omega_x_dac=np.concatenate([s.omega_x_dac for s in segments]),
        omega_y_dac=np.concatenate([s.omega_y_dac for s in segments]),
        detuning_hz=np.concatenate([s.detuning_hz for s in segments]),
    )


class TransmonMultilevelEngine:
    """Stateless-per-call 4-level Duffing engine; one instance per job."""

    def __init__(self, hidden: HiddenTransmonMultilevelConfig, rng: np.random.Generator):
        self.hidden = hidden
        self.rng = rng
        self.n = hidden.readout.assignment.__len__()
        self.params = DuffingParams(
            anharmonicity_hz=hidden.anharmonicity_hz,
            rabi_mhz_per_dac=hidden.rabi_mhz_per_dac,
            amp_compression=hidden.amp_compression,
            t1_s=hidden.t1_s,
            t2_s=hidden.t2_s,
            n_levels=self.n,
        )
        self.assignment = np.array(hidden.readout.assignment, dtype=float)

    def _true_populations(self, seg: RenderedSegment, dt_ns: float) -> np.ndarray:
        u = open_superoperator(
            seg.omega_x_dac, seg.omega_y_dac, seg.detuning_hz, dt_ns, self.params
        )
        rho0 = np.zeros((self.n, self.n), dtype=complex)
        rho0[0, 0] = 1.0
        return final_populations(propagate_density(rho0, u))

    def _sample_outcomes(self, p_true: np.ndarray, shots: int) -> list[str]:
        # Sample the true level, then the reported level via the assignment row.
        true_levels = self.rng.choice(self.n, size=shots, p=p_true)
        out = np.empty(shots, dtype=int)
        for lvl in range(self.n):
            mask = true_levels == lvl
            cnt = int(mask.sum())
            if cnt:
                out[mask] = self.rng.choice(self.n, size=cnt, p=self.assignment[lvl])
        return [str(int(x)) for x in out]

    def run_points(
        self,
        *,
        point_segments: list[RenderedSegment],
        dt_ns: float,
        shots: int,
        job_id: str,
        device_id: str,
        sweep_coords: dict[str, list[float]] | None = None,
    ) -> JobResult:
        t0 = time.perf_counter()
        # No blanket except here. Segments are rendered and placeholder-resolved
        # before this call, and ``MultilevelSimulatorBackend`` checks every
        # agent-visible bound (shots, segment count, sample count, DAC grid,
        # amplitude, duration, detuning), so anything raised below is a qsim-owned
        # execution fault. Letting it propagate reaches ``JobManager``, which types
        # it as the sanctioned ``JOB_EXECUTION_ERROR`` that
        # ``is_internal_job_failure`` recognizes and ``get_job_result`` stamps
        # ``failure_kind="qsim_internal"``. Returning ``error=str(exc)`` instead
        # replaced that marker with free-form text -- empty for ``MemoryError`` --
        # so a qsim fault was attributed to the model.
        outcomes: list[list[str]] = []
        total_dur = 0.0
        for seg in point_segments:
            p_true = self._true_populations(seg, dt_ns)
            outcomes.append(self._sample_outcomes(p_true, shots))
            total_dur = max(total_dur, len(seg.omega_x_dac) * dt_ns)
        return JobResult(
            job_id=job_id,
            device_id=device_id,
            status="complete",
            shots=shots,
            data=JobLevelOutcomeData(outcomes=outcomes, n_levels=self.n),
            metadata=JobResultMetadata(
                sequence_duration_ns=total_dur,
                wallclock_ms=int((time.perf_counter() - t0) * 1000),
                sweep_coords=sweep_coords,
            ),
        )


__all__ = [
    "RenderedSegment",
    "TransmonMultilevelEngine",
    "concat_segments",
    "render_analytic",
]
