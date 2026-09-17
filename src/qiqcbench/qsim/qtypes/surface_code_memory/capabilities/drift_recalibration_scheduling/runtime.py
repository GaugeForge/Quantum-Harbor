"""Device-side drift-adaptive recalibration instrument.

A two-tile rotated surface-code memory whose per-tile detector-fire-rate (DFR)
drifts under a hidden non-stationary process. The agent streams raw per-cycle
detector bits plus fixed-firmware-decoder logical-failure bits in windows,
may trigger recalibrations (bounded) and relocations, and finally submits a
declarative :class:`DriftPolicy`. The verifier replays that policy on fresh
hidden drift trajectories through :func:`execute_policy`.

Two load-bearing seams:

* **Sensor vs truth dual path.** The post-recalibration warm-up transient
  multiplies only the OBSERVED detector statistics (what the agent and the
  replayed policy see); sustained-breach accounting always uses the TRUE
  logical error probability, so deployment reliability is assessed against
  the physical memory rather than sensor noise.
* **Emergency override is public firmware semantics.** During a policy's
  warm-up hold, a reactive trigger still fires when the predicted LER exceeds
  ``emergency_override_factor * trigger_ler_per_cycle`` — otherwise every hold
  is a blind window in which no legal policy could react to a catastrophe.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from qiqcbench.qsim.qtypes.surface_code_memory.device import (
    DriftReloqationSpec,
    HiddenDriftReloqation,
    HiddenSurfaceCodeConfig,
    PublicSurfaceCodeSpec,
)
from qiqcbench.qsim.qtypes.surface_code_memory.wire import (
    DriftControlRequest,
    DriftPolicy,
    DriftRiskModel,
    DriftStreamRequest,
    JobDriftControlData,
    JobDriftStreamData,
    _DriftBudgetView,
)


def _pack_bits(value: np.ndarray) -> str:
    packed = np.packbits(np.ascontiguousarray(value, dtype=np.uint8).reshape(-1))
    return base64.b64encode(packed.tobytes()).decode("ascii")


def canonical_risk_model(model: DriftRiskModel) -> dict[str, Any]:
    """Return the immutable risk-model representation used by firmware.

    Firmware predicts from the intercept and exponent only; the reported
    prediction 1-sigma is carried because it is part of the submission the
    verifier binds, not because the policy uses it.
    """
    for name in (
        "log10_ler_per_cycle_intercept_estimate",
        "dfr_exponent_estimate",
        "log10_ler_prediction_1sigma_estimate",
    ):
        if not math.isfinite(float(getattr(model, name))):
            raise ValueError(f"{name} must be finite")
    return {
        "risk_model_schema_version": 2,
        "model_type": model.model_type,
        "log10_ler_per_cycle_intercept_estimate": float(
            model.log10_ler_per_cycle_intercept_estimate
        ),
        "dfr_exponent_estimate": float(model.dfr_exponent_estimate),
        "log10_ler_prediction_1sigma_estimate": float(model.log10_ler_prediction_1sigma_estimate),
    }


def canonical_policy(policy: DriftPolicy, public: PublicSurfaceCodeSpec) -> dict[str, Any]:
    """Fail closed and return the immutable grammar-normalized policy."""
    spec = public.drift_reloqation
    if spec is None:
        raise ValueError("drift-reloqation instrument is not configured for this device")
    if not isinstance(spec, DriftReloqationSpec):
        raise ValueError("policy schema 2 requires drift-reloqation contract version 3")
    if not math.isfinite(float(policy.trigger_ler_per_cycle)):
        raise ValueError("trigger_ler_per_cycle must be finite")
    if policy.trigger_mode in ("scheduled", "hybrid") and policy.schedule_period_windows is None:
        raise ValueError("scheduled/hybrid trigger_mode requires schedule_period_windows")
    if policy.trigger_mode == "reactive_dfr" and policy.schedule_period_windows is not None:
        raise ValueError("reactive_dfr trigger_mode requires schedule_period_windows=null")
    return {
        "policy_schema_version": 2,
        "trigger_mode": policy.trigger_mode,
        "dfr_buffer_windows": int(policy.dfr_buffer_windows),
        "trigger_ler_per_cycle": float(policy.trigger_ler_per_cycle),
        "schedule_period_windows": (
            None if policy.schedule_period_windows is None else int(policy.schedule_period_windows)
        ),
        "cooldown_windows": int(policy.cooldown_windows),
        "post_recalibration_hold_windows": int(policy.post_recalibration_hold_windows),
        "trigger_action": policy.trigger_action,
    }


def canonical_policy_digest(
    risk_model: DriftRiskModel, policy: DriftPolicy, public: PublicSurfaceCodeSpec
) -> str:
    payload = json.dumps(
        {
            "risk_model": canonical_risk_model(risk_model),
            "maintenance_policy": canonical_policy(policy, public),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# Version 2 binds the raw-artifact and composite-relocation contract.
DRIFT_EVIDENCE_SCHEMA_VERSION = 2
DRIFT_INSTANCE_COMMITMENT_NAME = "qiqcbench_drift_reloqation_instance_hmac_v2"
_DRIFT_INSTANCE_KEY_NAME = "qiqcbench_drift_reloqation_instance_key_v2"


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def instance_evidence_commitment(
    hidden: HiddenSurfaceCodeConfig, *, task_id: str | None, instance_seed: int
) -> dict[str, Any]:
    """Opaque equality proof binding evidence to the graded run inputs.

    This task's runtime injects no execution-scoped commitment secret, so the
    HMAC key is derived from the full canonical hidden device dump instead: the
    preimage space is the entire hidden model, never the brute-forceable
    (hidden.seed, instance_seed) pair, so the agent-readable digest reveals
    neither seed.  Deterministic for one env+config so honest rescoring agrees.
    """
    key_envelope = {
        "name": _DRIFT_INSTANCE_KEY_NAME,
        "schema_version": DRIFT_EVIDENCE_SCHEMA_VERSION,
        "task_id": task_id,
        "hidden_device": hidden.model_dump(mode="json"),
    }
    key = hashlib.sha256(_canonical_json_bytes(key_envelope)).digest()
    envelope = {
        "name": DRIFT_INSTANCE_COMMITMENT_NAME,
        "schema_version": DRIFT_EVIDENCE_SCHEMA_VERSION,
        "task_id": task_id,
        "instance_seed": int(instance_seed),
    }
    digest = hmac.new(key, _canonical_json_bytes(envelope), hashlib.sha256).hexdigest()
    return {
        "name": DRIFT_INSTANCE_COMMITMENT_NAME,
        "schema_version": DRIFT_EVIDENCE_SCHEMA_VERSION,
        "hmac_sha256": digest,
    }


@dataclass
class _TileState:
    """Stateful drift process of one tile. All times in device windows."""

    base_dfr: float
    slow_p_windows: float  # slow constant in windows (10x DFR after this many)
    slow_elapsed: int = 0
    jumps: list[tuple[float, int]] = field(default_factory=list)  # (mult, age_windows)
    burst_remaining: int = 0
    burst_mult_active: float = 1.0
    warm_remaining: int = 0
    # Operating windows since this tile's last recalibration completed; the
    # initial tiles were never recalibrated (no warm-up, no hold at start).
    since_recal: int = 10**9


@dataclass(frozen=True)
class _WindowStep:
    window: int
    operating_tile: int
    down: bool
    true_dfr: float | None
    sensor_multiplier: float | None


@dataclass(frozen=True)
class _ControlTransition:
    action: str
    start_window: int
    operating_tile_before_action: int
    operating_tile_after_action: int
    operating_downtime_starts_at_window: int | None
    operating_downtime_completes_at_window: int | None
    relocation_completes_at_window: int | None
    recalibrated_tile: int
    recalibration_starts_at_window: int
    recalibration_completes_at_window: int


class DriftEngine:
    """Run-long two-tile hidden drift process with qsim-owned accounting.

    The same class drives both the agent-facing instrument (via
    :meth:`run_stream` / :meth:`run_control`) and the verifier replay (via
    :func:`execute_policy`, which instantiates a private engine per fresh
    trajectory). One engine = one realization of the hidden process.
    """

    def __init__(
        self,
        hidden: HiddenSurfaceCodeConfig,
        public: PublicSurfaceCodeSpec,
        rng: np.random.Generator,
        *,
        recalibration_budget: int | None = None,
    ):
        if public.drift_reloqation is None or hidden.drift_reloqation is None:
            raise ValueError("drift engine requires public and hidden drift-reloqation material")
        self.public, self.rng = public, rng
        if not isinstance(public.drift_reloqation, DriftReloqationSpec):
            raise ValueError("drift engine requires contract version 3")
        self.spec: DriftReloqationSpec = public.drift_reloqation
        self.drift: HiddenDriftReloqation = hidden.drift_reloqation
        self._n_checks = int(public.n_checks)
        self._w = int(self.spec.window_cycles)
        self._jump_rate_win = self.drift.jump_rate_per_cycle * self._w
        self._burst_rate_win = self.drift.burst_rate_per_cycle * self._w
        self.tiles = [self._fresh_tile(), self._fresh_tile()]
        self.op_tile = 0
        self.now = 0  # device-time window index; advances only via streamed windows
        self.op_down_until = 0  # operating-tile unavailable before this window
        self.tile_ready_at = [0, 0]  # tile busy recalibrating before this window
        # Recalibration takes effect when it COMPLETES: the fresh drift state and
        # the warm-up transient start at the ready time, never at trigger time
        # (otherwise the warm-up would be silently consumed during the downtime
        # and the sensor transient could never be observed).
        self._recal_due: dict[int, int] = {}
        self._windows_used = 0
        self._recals_used = 0
        self._recalibration_budget = (
            self.spec.characterization_recalibration_budget
            if recalibration_budget is None
            else int(recalibration_budget)
        )
        self._lock = threading.Lock()

    # ---- hidden process -------------------------------------------------

    def _fresh_tile(self) -> _TileState:
        slow_p_cycles = self.drift.slow_p_median_cycles * math.exp(
            self.rng.normal(0.0, self.drift.slow_p_sigma)
        )
        base = self.drift.dfr_base * (1.0 + self.rng.normal(0.0, self.drift.dfr_base_spread))
        return _TileState(base_dfr=max(base, 1e-4), slow_p_windows=slow_p_cycles / self._w)

    def _recalibrate_tile(self, tile: int) -> None:
        """Reset calibration drift (slow + jumps) on a tile; bursts are
        environmental and survive recalibration (relocation escapes them)."""
        old = self.tiles[tile]
        fresh = self._fresh_tile()
        fresh.burst_remaining = old.burst_remaining
        fresh.burst_mult_active = old.burst_mult_active
        fresh.warm_remaining = int(self.spec.warmup_observation_windows)
        fresh.since_recal = 0
        self.tiles[tile] = fresh

    def _schedule_recalibration(self, tile: int, completes_at: int) -> None:
        self.tile_ready_at[tile] = completes_at
        self._recal_due[tile] = completes_at

    def _apply_due_recalibrations(self) -> None:
        for tile, due in list(self._recal_due.items()):
            if self.now >= due:
                self._recalibrate_tile(tile)
                del self._recal_due[tile]

    def advance_one_window(self) -> _WindowStep:
        """Advance exactly ``[now, now + 1)`` under the shared state machine."""
        self._apply_due_recalibrations()
        operating = self.op_tile
        down = self.now < self.op_down_until or self.now < self.tile_ready_at[operating]
        observed: tuple[float, float] | None = None
        for t in range(2):
            if self.now < self.tile_ready_at[t]:
                continue
            stepped = self._step_tile(self.tiles[t], observed=t == operating and not down)
            if t == operating and not down:
                observed = stepped
        window = self.now
        self.now += 1
        return _WindowStep(
            window=window,
            operating_tile=operating,
            down=down,
            true_dfr=None if observed is None else observed[0],
            sensor_multiplier=None if observed is None else observed[1],
        )

    def _step_tile(self, tile: _TileState, *, observed: bool) -> tuple[float, float]:
        """Advance one window; return (true_dfr, sensor_multiplier)."""
        # jump arrivals
        if self._jump_rate_win > 0 and self.rng.random() < self._jump_rate_win:
            mult = self.drift.jump_mult_min + self.rng.exponential(
                max(self.drift.jump_mult_exp_mean - self.drift.jump_mult_min, 1e-9)
            )
            tile.jumps.append((mult, 0))
        # burst arrivals
        if self._burst_rate_win > 0 and self.rng.random() < self._burst_rate_win:
            dur = (
                int(
                    self.drift.burst_duration_median_windows
                    * math.exp(self.rng.normal(0.0, self.drift.burst_duration_sigma))
                )
                + 1
            )
            tile.burst_remaining = max(tile.burst_remaining, dur)
            tile.burst_mult_active = self.drift.burst_mult
        slow = 10.0 ** (tile.slow_elapsed / tile.slow_p_windows)
        jump_factor = 1.0
        new_jumps: list[tuple[float, int]] = []
        for mult, age in tile.jumps:
            decayed = 1.0 + (mult - 1.0) * math.exp(-age / self.drift.jump_relax_windows)
            jump_factor *= decayed
            if decayed > 1.005:
                new_jumps.append((mult, age + 1))
        tile.jumps = new_jumps
        burst_factor = tile.burst_mult_active if tile.burst_remaining > 0 else 1.0
        if tile.burst_remaining > 0:
            tile.burst_remaining -= 1
        sensor_mult = 1.0
        if observed and tile.warm_remaining > 0:
            frac = tile.warm_remaining / max(self.spec.warmup_observation_windows, 1)
            sensor_mult = 1.0 + (self.drift.warmup_sensor_mult - 1.0) * frac
            tile.warm_remaining -= 1
        tile.slow_elapsed += 1
        if observed:
            tile.since_recal = min(tile.since_recal + 1, 10**9)
        true_dfr = min(tile.base_dfr * slow * jump_factor * burst_factor, 0.45)
        return true_dfr, sensor_mult

    def _p_l_cycle(self, true_dfr: float) -> float:
        return min(10.0**self.drift.ler_log10_a * true_dfr**self.drift.ler_exponent_b, 0.5)

    def begin_control(self, action: str) -> _ControlTransition | str:
        """Start one action at the current boundary using exact half-open timing."""
        # Boundary completions precede new admissions. Otherwise a second
        # action on the same tile could overwrite a due reset at this boundary.
        self._apply_due_recalibrations()
        if self._recals_used >= self._recalibration_budget:
            return "recalibration budget exhausted"
        start = self.now
        before = self.op_tile
        idle = 1 - before
        if start < self.op_down_until or start < self.tile_ready_at[before]:
            return "operating memory is unavailable (maintenance in progress)"
        if action == "recalibrate_operating_tile":
            tile = before
            recal_start = start
            recal_done = start + self.spec.recalibration_downtime_windows
            down_start, down_done = start, recal_done
            after = before
        elif action == "recalibrate_idle_tile":
            tile = idle
            if start < self.tile_ready_at[tile]:
                return "idle tile is already recalibrating"
            recal_start = start
            recal_done = start + self.spec.recalibration_downtime_windows
            down_start = down_done = None
            after = before
        elif action == "relocate":
            tile = before
            if start < self.tile_ready_at[idle]:
                return "idle tile is not ready (still recalibrating)"
            move_done = start + self.spec.relocation_downtime_windows
            recal_start = move_done
            recal_done = move_done + self.spec.recalibration_downtime_windows
            down_start, down_done = start, move_done
            relocation_done = move_done
            after = idle
            self.op_tile = idle
        else:
            return f"unknown action {action!r}"

        if action != "relocate":
            relocation_done = None

        self._schedule_recalibration(tile, recal_done)
        if down_done is not None:
            self.op_down_until = max(self.op_down_until, down_done)
        self._recals_used += 1
        return _ControlTransition(
            action=action,
            start_window=start,
            operating_tile_before_action=before,
            operating_tile_after_action=after,
            operating_downtime_starts_at_window=down_start,
            operating_downtime_completes_at_window=down_done,
            relocation_completes_at_window=relocation_done,
            recalibrated_tile=tile,
            recalibration_starts_at_window=recal_start,
            recalibration_completes_at_window=recal_done,
        )

    # ---- agent-facing instrument ----------------------------------------

    def budget_view(self) -> _DriftBudgetView:
        return _DriftBudgetView(
            characterization_stream_windows_used_raw=self._windows_used,
            characterization_stream_windows_cap=self.spec.characterization_stream_window_budget,
            characterization_recalibrations_used_raw=self._recals_used,
            characterization_recalibrations_cap=self._recalibration_budget,
        )

    def run_stream(self, request: DriftStreamRequest) -> JobDriftStreamData | str:
        with self._lock:
            n = int(request.windows)
            if n > self.spec.max_windows_per_call:
                return "windows exceeds max_windows_per_call"
            if self._windows_used + n > self.spec.characterization_stream_window_budget:
                return "run-long stream window budget exhausted"
            cycles, checks = self._w, self._n_checks
            events = np.zeros((n, cycles, checks), dtype=np.uint8)
            down = np.zeros(n, dtype=np.uint8)
            fails = np.zeros(n, dtype=np.uint8)
            start = self.now
            operating_tile = self.op_tile
            for i in range(n):
                stepped = self.advance_one_window()
                if stepped.down:
                    down[i] = 1
                else:
                    assert stepped.true_dfr is not None and stepped.sensor_multiplier is not None
                    true_dfr, sensor_mult = stepped.true_dfr, stepped.sensor_multiplier
                    p_obs = min(true_dfr * sensor_mult, 0.5)
                    events[i] = (self.rng.random((cycles, checks)) < p_obs).astype(np.uint8)
                    p_fail = 1.0 - (1.0 - self._p_l_cycle(true_dfr)) ** cycles
                    fails[i] = 1 if self.rng.random() < p_fail else 0
            self._windows_used += n
            return JobDriftStreamData(
                window_cycles=cycles,
                n_checks=checks,
                windows=n,
                start_window=start,
                operating_tile=operating_tile,
                detector_events_b64=_pack_bits(events),
                detector_shape=[n, cycles, checks],
                down_mask_b64=_pack_bits(down),
                logical_failure_bits_b64=_pack_bits(fails),
                budget=self.budget_view(),
            )

    def run_control(self, request: DriftControlRequest) -> JobDriftControlData | str:
        with self._lock:
            transition = self.begin_control(request.action)
            if isinstance(transition, str):
                return transition
            return JobDriftControlData(
                action=request.action,
                start_window=transition.start_window,
                operating_tile_before_action=transition.operating_tile_before_action,
                operating_tile_after_action=transition.operating_tile_after_action,
                operating_downtime_starts_at_window=(
                    transition.operating_downtime_starts_at_window
                ),
                operating_downtime_completes_at_window=(
                    transition.operating_downtime_completes_at_window
                ),
                relocation_completes_at_window=transition.relocation_completes_at_window,
                recalibrated_tile=transition.recalibrated_tile,
                recalibration_starts_at_window=transition.recalibration_starts_at_window,
                recalibration_completes_at_window=transition.recalibration_completes_at_window,
                budget=self.budget_view(),
            )


# ---- verifier-side policy replay -----------------------------------------


@dataclass(frozen=True)
class PolicyRollout:
    breached: bool
    downtime_windows: int
    relocations: int
    recalibrations: int
    maintenance_budget_exceeded: bool
    max_true_ler: float


def execute_policy(
    risk_model: dict[str, Any],
    policy: dict[str, Any],
    hidden: HiddenSurfaceCodeConfig,
    public: PublicSurfaceCodeSpec,
    rng: np.random.Generator,
) -> PolicyRollout:
    """Execute a canonical policy on one fresh hidden trajectory.

    The policy observes only the SENSOR window-DFR (binomial-sampled with the
    warm-up transient applied); sustained-breach accounting uses the TRUE
    logical error probability. Deterministic given ``rng``.
    """
    spec = public.drift_reloqation
    assert spec is not None
    if not isinstance(spec, DriftReloqationSpec):
        raise ValueError("policy replay requires drift-reloqation contract version 3")
    eng = DriftEngine(
        hidden,
        public,
        rng,
        recalibration_budget=spec.deployment_recalibration_budget_per_trajectory,
    )
    n_det_win = spec.window_cycles * public.n_checks
    horizon = spec.deployment_trajectory_windows
    target = spec.target_logical_error_per_cycle
    sustain = spec.sustained_breach_operating_windows
    emergency = spec.emergency_override_factor

    mode = policy["trigger_mode"]
    buf_n = policy["dfr_buffer_windows"]
    la = risk_model["log10_ler_per_cycle_intercept_estimate"]
    eb = risk_model["dfr_exponent_estimate"]
    thresh = policy["trigger_ler_per_cycle"]
    period = policy["schedule_period_windows"]
    cool = policy["cooldown_windows"]
    hold = policy["post_recalibration_hold_windows"]
    action = policy["trigger_action"]

    buf: list[float] = []
    consec = 0
    breached = False
    downtime = 0
    relocs = 0
    last_trigger = -(10**9)
    next_schedule = period if mode in ("scheduled", "hybrid") else None
    pending_relocation = False
    maintenance_budget_exceeded = False
    max_ler = 0.0

    def _start_relocation_if_ready() -> bool:
        nonlocal pending_relocation, last_trigger, relocs, maintenance_budget_exceeded, buf
        nonlocal next_schedule
        idle = 1 - eng.op_tile
        if eng.now < eng.tile_ready_at[idle]:
            return False
        transition = eng.begin_control("relocate")
        if isinstance(transition, str):
            maintenance_budget_exceeded = "budget exhausted" in transition
            pending_relocation = False
            return False
        relocs += 1
        pending_relocation = False
        last_trigger = transition.start_window
        if period is not None:
            next_schedule = transition.start_window + period
        buf = []
        return True

    for _ in range(horizon):
        if pending_relocation:
            _start_relocation_if_ready()
        stepped = eng.advance_one_window()
        t = stepped.window
        if stepped.down:
            downtime += 1
            continue
        assert stepped.true_dfr is not None and stepped.sensor_multiplier is not None
        true_dfr, sensor_mult = stepped.true_dfr, stepped.sensor_multiplier
        pl = eng._p_l_cycle(true_dfr)
        max_ler = max(max_ler, pl)
        if pl > target:
            consec += 1
            if consec >= sustain:
                breached = True
        else:
            consec = 0
        obs = rng.binomial(n_det_win, min(true_dfr * sensor_mult, 0.5)) / n_det_win
        buf.append(obs)
        del buf[:-buf_n]
        pred = 10.0**la * float(np.mean(buf)) ** eb
        trigger = False
        if next_schedule is not None and t + 1 >= next_schedule:
            trigger = True
        if mode in ("reactive_dfr", "hybrid") and (t - last_trigger) >= cool:
            # ``since_recal`` is incremented while the just-finished operating
            # observation is taken, so <= holds exactly the first ``hold``
            # observations after recalibration.
            in_hold = eng.tiles[eng.op_tile].since_recal <= hold
            if not in_hold and pred > thresh:
                trigger = True
            elif in_hold and pred > emergency * thresh:
                trigger = True
        if trigger and not pending_relocation and eng.now < horizon:
            if action == "relocate_then_recalibrate_vacated_tile":
                pending_relocation = True
                _start_relocation_if_ready()
            else:
                transition = eng.begin_control("recalibrate_operating_tile")
                if isinstance(transition, str):
                    maintenance_budget_exceeded = "budget exhausted" in transition
                else:
                    last_trigger = transition.start_window
                    if period is not None:
                        next_schedule = transition.start_window + period
                buf = []
    return PolicyRollout(
        breached=breached,
        downtime_windows=downtime,
        relocations=relocs,
        recalibrations=eng._recals_used,
        maintenance_budget_exceeded=maintenance_budget_exceeded,
        max_true_ler=max_ler,
    )


__all__ = [
    "DRIFT_EVIDENCE_SCHEMA_VERSION",
    "DRIFT_INSTANCE_COMMITMENT_NAME",
    "DriftEngine",
    "PolicyRollout",
    "canonical_policy",
    "canonical_policy_digest",
    "canonical_risk_model",
    "execute_policy",
    "instance_evidence_commitment",
]
