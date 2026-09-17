"""Simulator backend for the ``transmon_repeater_array`` qtype.

Holds **one** :class:`RepeaterEngine` for the whole run so the pair / row budgets
accumulate across batch calls. The RNG is injected once here (engine-contract invariant:
the factory injects, the engine never reseeds itself). It is seeded from the hidden
construction seed **mixed with fresh private per-attempt entropy**, so repeated benchmark
attempts are independent experiments: the hidden seed fixes the device, never the
realised shots (same runtime principle as ``bose_hubbard_chain``).

The same entropy root also realizes this attempt's **hidden instance**: the
transport-bias axis is drawn from a separate child stream when the hidden config declares
``transport_bias`` with ``axis_mode = "random_y_dominant"``. ``realized_instance()`` exposes
it for the action seam to record for the separate-mode verifier; it is never returned to
the agent. Pass ``run_entropy`` explicitly only for deterministic regression tests.
"""

from __future__ import annotations

import math
import secrets

import numpy as np

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.transmon_repeater_array.device import (
    HiddenRepeaterConfig,
    PublicRepeaterSpec,
    RepeaterTransportBias,
)
from qiqcbench.qsim.qtypes.transmon_repeater_array.engine import RepeaterEngine
from qiqcbench.qsim.qtypes.transmon_repeater_array.wire import PurificationBatchRequest

_MASK64 = (1 << 64) - 1
_AXIS_STREAM_TAG = 0xB1A5  # distinguishes the axis-draw stream from the shot stream
INSTANCE_KIND = "transport_bias_axis_v1"


def fresh_run_entropy() -> int:
    """Draw one private 128-bit entropy root for a benchmark attempt."""

    return secrets.randbits(128)


def draw_bias_axis(
    cfg: RepeaterTransportBias | None, rng: np.random.Generator
) -> tuple[float, float, float] | None:
    """Realize the transport-bias axis of one attempt: None without a bias; the fixed
    (theta, phi) direction; or a point uniform on the unit sphere restricted to
    ``n_y^2 - n_x^2 >= min_y2_minus_x2`` (rejection sampling of Gaussian directions)."""
    if cfg is None:
        return None
    if cfg.axis_mode == "fixed":
        t, f = math.radians(float(cfg.theta_deg)), math.radians(float(cfg.phi_deg))
        return (math.sin(t) * math.cos(f), math.sin(t) * math.sin(f), math.cos(t))
    for _ in range(100_000):
        v = rng.normal(size=3)
        norm = float(np.linalg.norm(v))
        if norm <= 0.0:
            continue
        v = v / norm
        if v[1] * v[1] - v[0] * v[0] >= cfg.min_y2_minus_x2:
            return (float(v[0]), float(v[1]), float(v[2]))
    raise RuntimeError("transport_bias axis region is empty or too small to sample")


class RepeaterSimulatorBackend:
    def __init__(
        self,
        hidden: HiddenRepeaterConfig,
        public: PublicRepeaterSpec,
        *,
        run_entropy: int | None = None,
    ) -> None:
        if run_entropy is None:
            run_entropy = fresh_run_entropy()
        if (
            isinstance(run_entropy, bool)
            or not isinstance(run_entropy, int)
            or not 0 <= run_entropy < 2**128
        ):
            raise ValueError("run_entropy must be a 128-bit non-negative integer")
        self._device_id = hidden.device_id
        self._rows_per_batch = public.budgets.rows_per_batch
        self._run_entropy = run_entropy
        self._transport_bias = hidden.transport_bias
        root = [int(hidden.seed) & _MASK64, run_entropy & _MASK64, run_entropy >> 64]
        shot_rng = np.random.default_rng(np.random.SeedSequence(root))
        axis_rng = np.random.default_rng(np.random.SeedSequence([*root, _AXIS_STREAM_TAG]))
        self._bias_axis = draw_bias_axis(hidden.transport_bias, axis_rng)
        self._engine = RepeaterEngine(hidden, public, shot_rng, bias_axis=self._bias_axis)

    @property
    def engine(self) -> RepeaterEngine:
        return self._engine

    @property
    def run_entropy(self) -> int:
        """Private per-attempt entropy; qsim-internal and never serialized."""

        return self._run_entropy

    @property
    def bias_axis(self) -> tuple[float, float, float] | None:
        """This attempt's realized transport-bias axis (hidden; verifier-only)."""

        return self._bias_axis

    def realized_instance(self) -> dict | None:
        """The per-attempt hidden instance the verifier must re-run against, or None when
        the hidden config carries no transport bias. Recorded by the action seam under
        ``<log_dir>/purification_instance/`` -- never published to the agent."""
        if self._bias_axis is None or self._transport_bias is None:
            return None
        nx, ny, nz = self._bias_axis
        theta = math.degrees(math.acos(max(-1.0, min(1.0, nz))))
        phi = math.degrees(math.atan2(ny, nx))
        return {
            "schema_version": 1,
            "instance_kind": INSTANCE_KIND,
            "device_id": self._device_id,
            "bias_axis": [nx, ny, nz],
            "bias_theta_deg": theta,
            "bias_phi_deg": phi,
            "p_bias": float(self._transport_bias.p_bias),
            "axis_mode": self._transport_bias.axis_mode,
        }

    def _failed(self, job_id: str, error: str) -> JobResult:
        return JobResult(job_id=job_id, device_id=self._device_id, status="failed", error=error)

    def run_purification_batch(
        self, request: PurificationBatchRequest, job_id: str, salt: int
    ) -> JobResult:
        if len(request.rows) > self._rows_per_batch:
            return self._failed(
                job_id, f"batch has {len(request.rows)} rows; max is {self._rows_per_batch}"
            )
        data = self._engine.run_purification_batch(request)
        return JobResult(job_id=job_id, device_id=self._device_id, status="complete", data=data)


def build_transmon_repeater_array_simulator_backend(
    hidden: HiddenRepeaterConfig,
    public: PublicRepeaterSpec,
    *,
    run_entropy: int | None = None,
) -> RepeaterSimulatorBackend:
    return RepeaterSimulatorBackend(hidden, public, run_entropy=run_entropy)
