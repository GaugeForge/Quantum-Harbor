"""Causal syndrome-feedback instrument and held-out deployment runtime.

The hidden environment evolves exogenously around persistent, randomly
switching regimes.  An applied trim changes only the instantaneous residual
seen by the physical detector-error mechanisms; it never writes into the
latent environment.  The same sampled mechanisms produce the raw detector
record and logical frame, and a firmware-owned fixed decoder consumes that
record.

Agent controllers are source bundles, never imported by qsim or the verifier.
One isolated streaming process is started for each continuous trajectory.  It
receives only public initialization metadata, the applied trim, and the raw
detector bits from the epoch that has just completed.  Firmware owns integer
quantization, clipping, slew limiting, and the one-epoch actuation delay.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qiqcbench.qsim.qtypes.surface_code_memory import surface_code as SC
from qiqcbench.qsim.qtypes.surface_code_memory.capabilities.syndrome_feedback_control.controller_runner import (
    DEFAULT_RUNNER_ROOT,
    ControllerProcess,
    ControllerRunnerInfrastructureError,
    ControllerSubmissionError,
    controller_bundle_manifest,
)
from qiqcbench.qsim.qtypes.surface_code_memory.capabilities.syndrome_feedback_control.fixed_decoder import (
    FixedFirmwareDecoder,
)
from qiqcbench.qsim.qtypes.surface_code_memory.device import (
    HiddenSurfaceCodeConfig,
    PublicSurfaceCodeSpec,
)
from qiqcbench.qsim.qtypes.surface_code_memory.wire import (
    JobSyndromeControllerProgramData,
    JobSyndromeControlProbeDataV3,
    SyndromeControllerProgramRequest,
    SyndromeControlProbeRequest,
    _ControllerBudgetView,
)

CONTROLLER_SOURCE_DIR = Path("/submission_artifacts/controller")


def _pack_bits(value: np.ndarray) -> str:
    packed = np.packbits(np.ascontiguousarray(value, dtype=np.uint8).reshape(-1))
    return base64.b64encode(packed.tobytes()).decode("ascii")


def _as_finite_matrix(value: list[list[float]], *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (2, 2) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 2x2 matrix")
    return array


# Version 3 binds each raw result to its immutable admission-time budget snapshot.
FEEDBACK_EVIDENCE_SCHEMA_VERSION = 3
FEEDBACK_INSTANCE_COMMITMENT_NAME = "qiqcbench_syndrome_feedback_instance_hmac_v2"
_FEEDBACK_INSTANCE_KEY_NAME = "qiqcbench_syndrome_feedback_instance_key_v2"


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
    """Return an opaque equality proof binding evidence to graded run inputs."""

    key_envelope = {
        "name": _FEEDBACK_INSTANCE_KEY_NAME,
        "schema_version": 2,
        "task_id": task_id,
        "hidden_device": hidden.model_dump(mode="json"),
    }
    key = hashlib.sha256(_canonical_json_bytes(key_envelope)).digest()
    envelope = {
        "name": FEEDBACK_INSTANCE_COMMITMENT_NAME,
        "schema_version": 2,
        "task_id": task_id,
        "instance_seed": int(instance_seed),
    }
    digest = hmac.new(key, _canonical_json_bytes(envelope), hashlib.sha256).hexdigest()
    return {
        "name": FEEDBACK_INSTANCE_COMMITMENT_NAME,
        "schema_version": 2,
        "hmac_sha256": digest,
    }


@dataclass(frozen=True)
class SimulationResult:
    """Lossless trajectory outputs used by public jobs and private scoring."""

    decoded_logical_bits: np.ndarray
    applied_trim_history: np.ndarray
    requested_trim_history: np.ndarray | None
    actuator_limited: np.ndarray
    detector_events: np.ndarray | None
    raw_logical_observables: np.ndarray
    decoder_predictions: np.ndarray


def terminal_failure_to_per_cycle(terminal_failure_probability: float, cycles: int) -> float:
    """Invert independent parity accumulation over ``cycles`` QEC cycles."""

    if cycles <= 0:
        raise ValueError("cycles must be positive")
    probability = min(max(float(terminal_failure_probability), 0.0), 0.5 - 1.0e-12)
    return 0.5 * (1.0 - (1.0 - 2.0 * probability) ** (1.0 / cycles))


class FeedbackEngine:
    """Run-long public instrument plus fresh hidden deployment rollouts."""

    def __init__(
        self,
        hidden: HiddenSurfaceCodeConfig,
        public: PublicSurfaceCodeSpec,
        rng: np.random.Generator,
        *,
        controller_source_dir: str | Path = CONTROLLER_SOURCE_DIR,
        controller_runner_root: str | Path = DEFAULT_RUNNER_ROOT,
        controller_process_type: type[ControllerProcess] = ControllerProcess,
    ) -> None:
        if public.syndrome_feedback is None or hidden.syndrome_feedback_drift is None:
            raise ValueError("feedback engine requires public and hidden feedback material")
        self.hidden, self.public, self.rng = hidden, public, rng
        self.spec = public.syndrome_feedback
        self.drift = hidden.syndrome_feedback_drift
        self.controller_source_dir = Path(controller_source_dir)
        self.controller_runner_root = Path(controller_runner_root)
        self.controller_process_type = controller_process_type
        self._transition = _as_finite_matrix(self.drift.transition, name="hidden transition")
        self._sensor = _as_finite_matrix(
            self.drift.detector_response, name="hidden detector response"
        )
        self._actuator = _as_finite_matrix(
            self.drift.actuator_response, name="hidden actuator response"
        )
        self._baseline = np.asarray(self.drift.detector_baseline, dtype=float)
        self._noise = np.asarray(self.drift.process_noise_std, dtype=float)
        self._shock = np.asarray(self.drift.shock_std, dtype=float)
        self._centers = np.asarray(self.drift.regime_centers, dtype=float)
        if any(
            array.shape != (2,) or not np.isfinite(array).all()
            for array in (self._baseline, self._noise, self._shock)
        ):
            raise ValueError("hidden feedback vectors must contain two finite values")
        if (
            self._centers.ndim != 2
            or self._centers.shape[1] != 2
            or self._centers.shape[0] < 2
            or not np.isfinite(self._centers).all()
        ):
            raise ValueError("hidden feedback regime_centers must be finite [regime, 2]")
        if int(hidden.distance) != int(public.distance):
            raise ValueError("hidden and public surface-code distances disagree")
        if np.max(np.abs(np.linalg.eigvals(self._transition))) >= 1.0:
            raise ValueError("hidden transition must be stable around each regime center")

        self._geo = SC.geometry_from_public_spec(public)
        noise = hidden.noise
        self._dem_params = SC.HiddenDemParams(
            p_data=noise.p_data,
            p_meas=noise.p_meas,
            hook_base=noise.hook_base,
            hook_multipliers=list(noise.hook_multipliers),
            p_leak=noise.p_leak,
            leak_pairs=[
                (pair.s1, pair.s2, pair.dt, int(pair.flips_observable)) for pair in noise.leak_pairs
            ],
        )
        self._n_checks = self._geo.n_checks
        if self._n_checks < 2 or self._n_checks % 2:
            raise ValueError("feedback instrument requires an even number of checks")
        if self._n_checks != int(public.n_checks):
            raise ValueError("executable geometry disagrees with public n_checks")
        rounds = self.spec.control_epoch_cycles
        self._epoch_dem = SC.build_hidden_dem(self._geo, self._dem_params, rounds)
        self._n_epoch_detectors = self._n_checks * rounds
        self._fixed_decoder = FixedFirmwareDecoder(
            self._epoch_dem,
            n_detectors=self._n_epoch_detectors,
        )
        self._dem_a = np.asarray([item[0] for item in self._epoch_dem], dtype=np.int64)
        self._dem_b = np.asarray([item[1] for item in self._epoch_dem], dtype=np.int64)
        self._dem_flips = np.asarray([item[3] for item in self._epoch_dem], dtype=np.uint8)
        base_probabilities = np.asarray([item[2] for item in self._epoch_dem], dtype=float)
        if np.any((base_probabilities <= 0.0) | (base_probabilities >= 0.5)):
            raise ValueError("feedback DEM probabilities must lie in (0, 0.5)")
        self._dem_base_logits = np.log(base_probabilities / (1.0 - base_probabilities))
        self._dem_mode_loadings = self._build_mechanism_mode_loadings()
        nominal_rates = self._nominal_family_rates()
        if not np.allclose(self._baseline, nominal_rates, rtol=0.0, atol=5.0e-6):
            raise ValueError(
                "hidden detector_baseline must equal nominal DEM family rates; "
                f"got {self._baseline.tolist()}, expected {nominal_rates.tolist()}"
            )
        self._lock = threading.Lock()

    def _nominal_family_rates(self) -> np.ndarray:
        parity_products = np.ones(self._n_epoch_detectors, dtype=float)
        for detector_a, detector_b, probability, _flips in self._epoch_dem:
            parity_products[detector_a] *= 1.0 - 2.0 * probability
            if detector_b != SC.BOUNDARY:
                parity_products[detector_b] *= 1.0 - 2.0 * probability
        detector_rates = 0.5 * (1.0 - parity_products)
        by_check = detector_rates.reshape(self._n_checks, self.spec.control_epoch_cycles)
        half = self._n_checks // 2
        return np.asarray([by_check[:half].mean(), by_check[half:].mean()], dtype=float)

    def _build_mechanism_mode_loadings(self) -> np.ndarray:
        half = self._n_checks // 2
        loadings = np.zeros((len(self._epoch_dem), 2), dtype=float)
        for index, (detector_a, detector_b, _probability, _flips) in enumerate(self._epoch_dem):
            checks = [detector_a // self.spec.control_epoch_cycles]
            if detector_b != SC.BOUNDARY:
                checks.append(detector_b // self.spec.control_epoch_cycles)
            for check in checks:
                if not 0 <= check < self._n_checks:
                    raise ValueError("feedback DEM detector lies outside public geometry")
                loadings[index, int(check >= half)] += 1.0 / len(checks)
        if not np.allclose(loadings.sum(axis=1), 1.0):
            raise ValueError("feedback DEM family loadings are malformed")
        return loadings

    def _check_request(self, trajectories: int) -> str | None:
        if trajectories > self.spec.max_trajectories_per_call:
            return "trajectories exceeds max_trajectories_per_call"
        return None

    def _validate_admission_budget(
        self,
        trajectories: int,
        budget: _ControllerBudgetView,
    ) -> None:
        epochs = self.spec.experiment_epochs_per_trajectory
        cycles_per_trajectory = epochs * self.spec.control_epoch_cycles
        expected_trajectory_cap = self.spec.experiment_trajectory_budget
        expected_qec_cycle_cap = expected_trajectory_cap * cycles_per_trajectory
        if (
            budget.trajectories_cap != expected_trajectory_cap
            or budget.qec_cycles_cap != expected_qec_cycle_cap
            or budget.trajectories_used < trajectories
            or budget.trajectories_used > budget.trajectories_cap
            or budget.qec_cycles_used != budget.trajectories_used * cycles_per_trajectory
        ):
            raise RuntimeError("syndrome-feedback admission budget snapshot is inconsistent")

    def _initialize_population(self, trajectories: int) -> tuple[np.ndarray, np.ndarray]:
        regimes = self.rng.integers(0, len(self._centers), size=trajectories)
        deviations = self.rng.normal(
            0.0,
            self.drift.initial_state_std,
            size=(trajectories, 2),
        )
        return self._centers[regimes] + deviations, regimes

    def _advance_population(
        self, state: np.ndarray, regimes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        old_centers = self._centers[regimes]
        switch = self.rng.random(len(regimes)) < self.drift.regime_switch_probability
        if switch.any():
            offsets = self.rng.integers(1, len(self._centers), size=int(switch.sum()))
            regimes = regimes.copy()
            regimes[switch] = (regimes[switch] + offsets) % len(self._centers)
        shocks = self.rng.normal(0.0, self._shock, size=state.shape)
        shocks *= (self.rng.random(len(state)) < self.drift.shock_probability)[:, None]
        deviations = (state - old_centers) @ self._transition.T
        next_state = (
            self._centers[regimes]
            + deviations
            + self.rng.normal(0.0, self._noise, size=state.shape)
            + shocks
        )
        return next_state, regimes

    def _sample_epoch(self, modes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample raw detector/logical bits and decode that exact detector record."""

        trajectories = modes.shape[0]
        if modes.shape != (trajectories, 2) or not np.isfinite(modes).all():
            raise ValueError("feedback detector modes must be finite [trajectory, 2]")
        syndrome = np.zeros((trajectories, self._n_epoch_detectors), dtype=np.uint8)
        observable = np.zeros(trajectories, dtype=np.uint8)
        has_second_detector = self._dem_b != SC.BOUNDARY
        for start in range(0, trajectories, 2_000):
            end = min(start + 2_000, trajectories)
            shifts = modes[start:end] @ self._dem_mode_loadings.T
            if self.drift.detector_overdispersion:
                shifts += self.rng.normal(
                    0.0,
                    self.drift.detector_overdispersion,
                    size=(end - start, 1),
                )
            logits = np.clip(self._dem_base_logits[None, :] + shifts, -18.0, -0.62)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            fired = self.rng.random(probabilities.shape) < probabilities
            syndrome_block = syndrome[start:end]
            observable_block = observable[start:end]
            for mechanism_index in range(len(self._epoch_dem)):
                fault = fired[:, mechanism_index]
                if not fault.any():
                    continue
                syndrome_block[fault, self._dem_a[mechanism_index]] ^= 1
                if has_second_detector[mechanism_index]:
                    syndrome_block[fault, self._dem_b[mechanism_index]] ^= 1
                if self._dem_flips[mechanism_index]:
                    observable_block[fault] ^= 1
        predictions = self._fixed_decoder.decode(syndrome)
        events = syndrome.reshape(
            trajectories,
            self._n_checks,
            self.spec.control_epoch_cycles,
        ).transpose(0, 2, 1)
        return events, observable, predictions

    def _residual_modes(self, state: np.ndarray, applied_trim: np.ndarray) -> np.ndarray:
        # This is deliberately instantaneous.  The controller never modifies
        # ``state`` and therefore cannot freeze or steer the hidden environment.
        residual = state - applied_trim @ self._actuator.T
        # Calibration error in either direction raises physical fault odds;
        # otherwise an extreme trim could improve the device without bound and
        # collapse the task to a static boundary point.  ``detector_response``
        # defines two hidden residual projections and the physical penalty is
        # their squared magnitude.
        projected = residual @ self._sensor.T
        return projected * projected

    def _apply_firmware(
        self, requested_trim: np.ndarray, current_trim: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if requested_trim.shape != current_trim.shape or requested_trim.shape[-1] != 2:
            raise ValueError("requested and current trim arrays must be [trajectory, 2]")
        rounded = np.rint(requested_trim).astype(np.int64)
        clipped = np.clip(rounded, self.spec.trim_min, self.spec.trim_max)
        delta = np.clip(
            clipped - current_trim,
            -self.spec.trim_slew_per_epoch,
            self.spec.trim_slew_per_epoch,
        )
        applied_next = current_trim + delta
        limited = (clipped != rounded) | (applied_next != clipped)
        return applied_next.astype(np.int64), limited

    def simulate_constant(
        self,
        constant_trim: np.ndarray,
        *,
        trajectories: int,
        epochs: int,
        retain_raw: bool,
    ) -> SimulationResult:
        state, regimes = self._initialize_population(trajectories)
        applied = np.broadcast_to(constant_trim, (trajectories, 2)).astype(np.int64).copy()
        applied_history = np.empty((trajectories, epochs, 2), dtype=np.int64)
        decoded = np.empty((trajectories, epochs), dtype=np.uint8)
        raw_logical = np.empty_like(decoded)
        predictions = np.empty_like(decoded)
        raw_parts: list[np.ndarray] | None = [] if retain_raw else None
        for epoch in range(epochs):
            applied_history[:, epoch] = applied
            events, observable, prediction = self._sample_epoch(
                self._residual_modes(state, applied)
            )
            raw_logical[:, epoch] = observable
            predictions[:, epoch] = prediction
            decoded[:, epoch] = observable ^ prediction
            if raw_parts is not None:
                raw_parts.append(events)
            if epoch + 1 < epochs:
                state, regimes = self._advance_population(state, regimes)
        return SimulationResult(
            decoded_logical_bits=decoded,
            applied_trim_history=applied_history,
            requested_trim_history=None,
            actuator_limited=np.zeros((trajectories, epochs, 2), dtype=np.uint8),
            detector_events=np.stack(raw_parts, axis=1) if raw_parts is not None else None,
            raw_logical_observables=raw_logical,
            decoder_predictions=predictions,
        )

    def _controller_init_payload(self) -> dict[str, Any]:
        return {
            "type": "initialize",
            "abi_version": self.spec.controller_abi_version,
            "control_epoch_cycles": self.spec.control_epoch_cycles,
            "detector_shape": [self.spec.control_epoch_cycles, self._n_checks],
            "detector_families": list(self.spec.detector_families),
            "trim_min": self.spec.trim_min,
            "trim_max": self.spec.trim_max,
            "trim_slew_per_epoch": self.spec.trim_slew_per_epoch,
            "controller_latency_epochs": self.spec.controller_latency_epochs,
            "initial_applied_trim": list(self.spec.initial_applied_trim),
        }

    def _controller_step_payload(
        self, epoch: int, applied: np.ndarray, events: np.ndarray
    ) -> dict[str, Any]:
        return {
            "type": "step",
            "trajectory_epoch": epoch,
            "applied_trim": [int(value) for value in applied],
            "detector_bits_b64": _pack_bits(events),
            "detector_shape": [self.spec.control_epoch_cycles, self._n_checks],
        }

    def _process_kwargs(self, expected_manifest_sha256: str) -> dict[str, Any]:
        return {
            "expected_manifest_sha256": expected_manifest_sha256,
            "runner_root": self.controller_runner_root,
            "maximum_files": self.spec.max_controller_bundle_files,
            "maximum_bytes": self.spec.max_controller_bundle_bytes,
            "maximum_response_bytes": self.spec.max_controller_response_bytes,
            "step_timeout_s": self.spec.controller_step_timeout_s,
            "trajectory_timeout_s": self.spec.controller_trajectory_timeout_s,
            "max_requested_trim_abs": self.spec.max_requested_trim_abs,
        }

    def simulate_program(
        self,
        *,
        controller_manifest_sha256: str,
        trajectories: int,
        epochs: int,
        retain_raw: bool,
    ) -> SimulationResult:
        decoded = np.empty((trajectories, epochs), dtype=np.uint8)
        raw_logical = np.empty_like(decoded)
        predictions = np.empty_like(decoded)
        applied_history = np.empty((trajectories, epochs, 2), dtype=np.int64)
        requested_history = np.empty((trajectories, epochs, 2), dtype=float)
        limited_history = np.empty((trajectories, epochs, 2), dtype=np.uint8)
        detector_history = (
            np.empty(
                (
                    trajectories,
                    epochs,
                    self.spec.control_epoch_cycles,
                    self._n_checks,
                ),
                dtype=np.uint8,
            )
            if retain_raw
            else None
        )
        init_payload = self._controller_init_payload()
        for trajectory in range(trajectories):
            state, regimes = self._initialize_population(1)
            applied = np.asarray(self.spec.initial_applied_trim, dtype=np.int64)[None, :]
            with self.controller_process_type(
                self.controller_source_dir,
                **self._process_kwargs(controller_manifest_sha256),
            ) as process:
                process.initialize(init_payload)
                for epoch in range(epochs):
                    applied_history[trajectory, epoch] = applied[0]
                    events, observable, prediction = self._sample_epoch(
                        self._residual_modes(state, applied)
                    )
                    raw_logical[trajectory, epoch] = observable[0]
                    predictions[trajectory, epoch] = prediction[0]
                    decoded[trajectory, epoch] = observable[0] ^ prediction[0]
                    if detector_history is not None:
                        detector_history[trajectory, epoch] = events[0]
                    requested = np.asarray(
                        process.step(self._controller_step_payload(epoch, applied[0], events[0])),
                        dtype=float,
                    )[None, :]
                    requested_history[trajectory, epoch] = requested[0]
                    applied_next, limited = self._apply_firmware(requested, applied)
                    limited_history[trajectory, epoch] = limited[0]
                    if epoch + 1 < epochs:
                        state, regimes = self._advance_population(state, regimes)
                        applied = applied_next
        return SimulationResult(
            decoded_logical_bits=decoded,
            applied_trim_history=applied_history,
            requested_trim_history=requested_history,
            actuator_limited=limited_history,
            detector_events=detector_history,
            raw_logical_observables=raw_logical,
            decoder_predictions=predictions,
        )

    def simulate_policy(
        self,
        policy_factory: Callable[[], Callable[[dict[str, Any]], list[float]]],
        *,
        trajectories: int,
        epochs: int,
        retain_raw: bool = False,
    ) -> SimulationResult:
        """Construction-only in-process seam for independent causal baselines.

        Production scoring always uses :meth:`simulate_program`; this helper
        exists so construction tests can cheaply establish margins for many
        qualitatively different policies without weakening isolation.
        """

        decoded = np.empty((trajectories, epochs), dtype=np.uint8)
        raw_logical = np.empty_like(decoded)
        predictions = np.empty_like(decoded)
        applied_history = np.empty((trajectories, epochs, 2), dtype=np.int64)
        requested_history = np.empty((trajectories, epochs, 2), dtype=float)
        limited_history = np.empty((trajectories, epochs, 2), dtype=np.uint8)
        detector_history = (
            np.empty(
                (trajectories, epochs, self.spec.control_epoch_cycles, self._n_checks),
                dtype=np.uint8,
            )
            if retain_raw
            else None
        )
        for trajectory in range(trajectories):
            policy = policy_factory()
            state, regimes = self._initialize_population(1)
            applied = np.asarray(self.spec.initial_applied_trim, dtype=np.int64)[None, :]
            for epoch in range(epochs):
                applied_history[trajectory, epoch] = applied[0]
                events, observable, prediction = self._sample_epoch(
                    self._residual_modes(state, applied)
                )
                raw_logical[trajectory, epoch] = observable[0]
                predictions[trajectory, epoch] = prediction[0]
                decoded[trajectory, epoch] = observable[0] ^ prediction[0]
                if detector_history is not None:
                    detector_history[trajectory, epoch] = events[0]
                requested = np.asarray(
                    [policy(self._controller_step_payload(epoch, applied[0], events[0]))],
                    dtype=float,
                )
                if requested.shape != (1, 2) or not np.isfinite(requested).all():
                    raise ValueError("construction policy returned an invalid trim")
                requested_history[trajectory, epoch] = requested[0]
                applied_next, limited = self._apply_firmware(requested, applied)
                limited_history[trajectory, epoch] = limited[0]
                if epoch + 1 < epochs:
                    state, regimes = self._advance_population(state, regimes)
                    applied = applied_next
        return SimulationResult(
            decoded_logical_bits=decoded,
            applied_trim_history=applied_history,
            requested_trim_history=requested_history,
            actuator_limited=limited_history,
            detector_events=detector_history,
            raw_logical_observables=raw_logical,
            decoder_predictions=predictions,
        )

    def validate_program(self) -> dict[str, Any]:
        manifest = controller_bundle_manifest(
            self.controller_source_dir,
            maximum_files=self.spec.max_controller_bundle_files,
            maximum_bytes=self.spec.max_controller_bundle_bytes,
        )
        return {
            "valid": True,
            "controller_abi_version": self.spec.controller_abi_version,
            "controller_entrypoint": self.spec.controller_entrypoint,
            "controller_manifest_sha256": manifest.sha256,
            "bundle_files": list(manifest.files),
            "bundle_size_bytes": manifest.size_bytes,
            "performance": "not evaluated",
        }

    def run_probe(
        self,
        request: SyndromeControlProbeRequest,
        *,
        budget: _ControllerBudgetView,
    ) -> JobSyndromeControlProbeDataV3 | str:
        with self._lock:
            if error := self._check_request(request.trajectories):
                return error
            self._validate_admission_budget(request.trajectories, budget)
            trim = np.asarray(request.constant_trim, dtype=np.int64)
            if (
                trim.shape != (2,)
                or np.any(trim < self.spec.trim_min)
                or np.any(trim > self.spec.trim_max)
            ):
                return "constant_trim lies outside the public trim range"
            result = self.simulate_constant(
                trim,
                trajectories=request.trajectories,
                epochs=self.spec.experiment_epochs_per_trajectory,
                retain_raw=True,
            )
            assert result.detector_events is not None
            return JobSyndromeControlProbeDataV3(
                constant_trim=[int(value) for value in trim],
                trajectories=request.trajectories,
                epochs_per_trajectory=self.spec.experiment_epochs_per_trajectory,
                control_epoch_cycles=self.spec.control_epoch_cycles,
                detector_events_b64=_pack_bits(result.detector_events),
                detector_shape=list(result.detector_events.shape),
                decoded_logical_bits_b64=_pack_bits(result.decoded_logical_bits),
                decoded_logical_shape=list(result.decoded_logical_bits.shape),
                applied_trim_history=result.applied_trim_history.astype(int).tolist(),
                budget=budget,
            )

    def run_program(
        self,
        request: SyndromeControllerProgramRequest,
        *,
        budget: _ControllerBudgetView,
    ) -> JobSyndromeControllerProgramData | str:
        with self._lock:
            if error := self._check_request(request.trajectories):
                return error
            self._validate_admission_budget(request.trajectories, budget)
            try:
                result = self.simulate_program(
                    controller_manifest_sha256=request.controller_manifest_sha256,
                    trajectories=request.trajectories,
                    epochs=self.spec.experiment_epochs_per_trajectory,
                    retain_raw=True,
                )
            except ControllerSubmissionError as exc:
                return f"controller program rejected: {exc}"
            assert result.detector_events is not None
            assert result.requested_trim_history is not None
            return JobSyndromeControllerProgramData(
                controller_manifest_sha256=request.controller_manifest_sha256,
                controller_abi_version=self.spec.controller_abi_version,
                trajectories=request.trajectories,
                epochs_per_trajectory=self.spec.experiment_epochs_per_trajectory,
                control_epoch_cycles=self.spec.control_epoch_cycles,
                detector_events_b64=_pack_bits(result.detector_events),
                detector_shape=list(result.detector_events.shape),
                decoded_logical_bits_b64=_pack_bits(result.decoded_logical_bits),
                decoded_logical_shape=list(result.decoded_logical_bits.shape),
                requested_trim_history=result.requested_trim_history.tolist(),
                applied_trim_history=result.applied_trim_history.astype(int).tolist(),
                actuator_limited_bits_b64=_pack_bits(result.actuator_limited),
                actuator_limited_shape=list(result.actuator_limited.shape),
                budget=budget,
            )

    def deployment_rollout(self, controller_manifest_sha256: str) -> SimulationResult:
        """Run the exact staged bundle on fresh long-horizon hidden trajectories."""

        return self.simulate_program(
            controller_manifest_sha256=controller_manifest_sha256,
            trajectories=self.spec.deployment_trajectories,
            epochs=self.spec.deployment_epochs_per_trajectory,
            retain_raw=False,
        )


__all__ = [
    "CONTROLLER_SOURCE_DIR",
    "FEEDBACK_EVIDENCE_SCHEMA_VERSION",
    "FEEDBACK_INSTANCE_COMMITMENT_NAME",
    "ControllerRunnerInfrastructureError",
    "ControllerSubmissionError",
    "FeedbackEngine",
    "SimulationResult",
    "instance_evidence_commitment",
    "terminal_failure_to_per_cycle",
]
