"""Render multilevel-pulse wire requests into JobResults via the engine."""

from __future__ import annotations

import itertools

import numpy as np

from qiqcbench.qsim.core.wire import (
    AnalyticDriveSegment,
    DriveSegment,
    JobResult,
    MultilevelPulseRequest,
    MultilevelPulseSweepRequest,
    SampledDriveSegment,
)
from qiqcbench.qsim.qtypes.transmon_multilevel_pulse.device import (
    HiddenTransmonMultilevelConfig,
    PublicTransmonMultilevelSpec,
)
from qiqcbench.qsim.qtypes.transmon_multilevel_pulse.engine import (
    RenderedSegment,
    TransmonMultilevelEngine,
    concat_segments,
    render_analytic,
)


def _make_rng(hidden: HiddenTransmonMultilevelConfig, salt: int) -> np.random.Generator:
    return np.random.default_rng((hidden.seed * 1_000_003) ^ salt)


def _resolve_scalar(value: float | str | None, binding: dict[str, float]) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.startswith("$"):
            raise ValueError(f"Scalar string {value!r} must be a '$name' sweep placeholder")
        name = value[1:]
        if name not in binding:
            raise ValueError(f"Unbound sweep placeholder {value!r}")
        return float(binding[name])
    return float(value)


def _render_segment(seg: DriveSegment, binding: dict[str, float], dt_ns: float) -> RenderedSegment:
    if isinstance(seg, SampledDriveSegment):
        return RenderedSegment(
            omega_x_dac=np.asarray(seg.omega_x_dac, dtype=float),
            omega_y_dac=np.asarray(seg.omega_y_dac, dtype=float),
            detuning_hz=np.asarray(seg.detuning_hz, dtype=float),
        )
    assert isinstance(seg, AnalyticDriveSegment)
    return render_analytic(
        shape=seg.shape,
        amp_dac=_resolve_scalar(seg.amp_dac, binding),
        duration_ns=_resolve_scalar(seg.duration_ns, binding),
        sigma_ns=_resolve_scalar(seg.sigma_ns, binding),
        phase_rad=_resolve_scalar(seg.phase_rad, binding) or 0.0,
        carrier_detuning_hz=_resolve_scalar(seg.carrier_detuning_hz, binding) or 0.0,
        sample_dt_ns=dt_ns,
    )


def _point_segment(
    segments: list[DriveSegment], binding: dict[str, float], dt_ns: float
) -> RenderedSegment:
    return concat_segments([_render_segment(s, binding, dt_ns) for s in segments])


def run_pulse_sequence(
    request: MultilevelPulseRequest,
    hidden: HiddenTransmonMultilevelConfig,
    public: PublicTransmonMultilevelSpec,
    job_id: str,
    salt: int,
) -> JobResult:
    rng = _make_rng(hidden, salt)
    engine = TransmonMultilevelEngine(hidden, rng)
    point = _point_segment(list(request.segments), {}, public.sample_dt_ns)
    return engine.run_points(
        point_segments=[point],
        dt_ns=public.sample_dt_ns,
        shots=request.shots,
        job_id=job_id,
        device_id=hidden.device_id,
    )


def _expand_sweep(
    sweep: dict[str, list[float]], mode: str
) -> tuple[list[str], list[tuple[float, ...]]]:
    names = list(sweep.keys())
    if mode == "zip":
        lengths = {len(v) for v in sweep.values()}
        if len(lengths) != 1:
            raise ValueError("zip sweep requires all value lists to share one length")
        combos = list(zip(*[sweep[n] for n in names], strict=True))
    else:
        combos = list(itertools.product(*[sweep[n] for n in names]))
    return names, combos


def run_pulse_sweep(
    request: MultilevelPulseSweepRequest,
    hidden: HiddenTransmonMultilevelConfig,
    public: PublicTransmonMultilevelSpec,
    job_id: str,
    salt: int,
) -> JobResult:
    rng = _make_rng(hidden, salt)
    engine = TransmonMultilevelEngine(hidden, rng)
    names, combos = _expand_sweep(dict(request.sweep), request.mode)
    points: list[RenderedSegment] = []
    for combo in combos:
        binding = dict(zip(names, combo, strict=True))
        points.append(_point_segment(list(request.template_segments), binding, public.sample_dt_ns))
    coords = {name: [float(c[i]) for c in combos] for i, name in enumerate(names)}
    return engine.run_points(
        point_segments=points,
        dt_ns=public.sample_dt_ns,
        shots=request.shots,
        job_id=job_id,
        device_id=hidden.device_id,
        sweep_coords=coords,
    )


__all__ = ["run_pulse_sequence", "run_pulse_sweep"]
