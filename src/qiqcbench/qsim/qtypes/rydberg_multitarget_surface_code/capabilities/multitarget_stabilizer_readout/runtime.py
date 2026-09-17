"""Raw Rydberg characterization and circuit-level logical-memory runtime."""

from __future__ import annotations

import hashlib
import json
import threading

import numpy as np

from qiqcbench.qsim.qtypes.rydberg_multitarget_surface_code.device import (
    HiddenRydbergMultitargetSurfaceCodeConfig,
    PublicRydbergMultitargetSurfaceCodeSpec,
)
from qiqcbench.qsim.qtypes.rydberg_multitarget_surface_code.physics import (
    build_memory_circuit,
    canonical_cycle,
    canonical_cycle_digest,
    correlated_pair_probability,
    cycle_duration_us,
    decode_memory_measurements,
    pack_bits,
    sample_memory_measurements,
)
from qiqcbench.qsim.qtypes.rydberg_multitarget_surface_code.wire import (
    Cz2CharacterizationRequest,
    JobCz2CharacterizationData,
    JobMultitargetMemoryData,
    MultitargetMemoryRequest,
    _ShotBudgetView,
)


def domain_separated_rng(
    *,
    construction_id: str,
    attempt_seed: int,
    job_kind: str,
    salt: int,
) -> np.random.Generator:
    """Derive deterministic, domain-separated streams for replayable simulator jobs."""
    if (
        not isinstance(construction_id, str)
        or not construction_id
        or type(attempt_seed) is not int
        or type(salt) is not int
        or not isinstance(job_kind, str)
        or not job_kind
    ):
        raise ValueError("Rydberg RNG inputs have invalid types or empty identities")
    payload = json.dumps(
        [construction_id, attempt_seed, job_kind, salt],
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    digest = hashlib.sha256(
        b"qiqcbench/rydberg_multitarget_surface_code/v3/job-local-rng\x00" + payload
    ).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


class MultitargetReadoutEngine:
    """Run-wide Rydberg instrument with atomic raw-shot accounting."""

    def __init__(
        self,
        hidden: HiddenRydbergMultitargetSurfaceCodeConfig,
        public: PublicRydbergMultitargetSurfaceCodeSpec,
        rng: np.random.Generator | None = None,
    ):
        if hidden.device_id != public.device_id:
            raise ValueError("public and hidden Rydberg device IDs do not match")
        self.hidden = hidden
        self.public = public
        self.noise = hidden.noise
        self._default_rng = rng
        self._shots_used = 0
        self._lock = threading.Lock()

    def _resolve_rng(self, rng: np.random.Generator | None) -> np.random.Generator:
        selected = self._default_rng if rng is None else rng
        if selected is None:
            raise ValueError("a job-local RNG is required")
        return selected

    def _budget(self, accepted_shots_used: int | None = None) -> _ShotBudgetView:
        return _ShotBudgetView(
            shots_used=self._shots_used if accepted_shots_used is None else accepted_shots_used,
            shots_cap=self.public.budgets.shot_budget,
        )

    def _check_budget(self, shots: int) -> str | None:
        if shots > self.public.budgets.max_shots_per_job:
            return "shots exceeds max_shots_per_job"
        if self._shots_used + shots > self.public.budgets.shot_budget:
            return "run-wide multitarget shot budget exhausted"
        return None

    def _characterization_pair_probability(self, request: Cz2CharacterizationRequest) -> float:
        if request.echo_protocol == "duration_echo":
            probability = (
                self.noise.correlated_pair_error_at_optimum
                + self.noise.correlated_pair_duration_curvature
                * (request.duration_scale - self.noise.duration_scale_optimum) ** 2
            )
        else:
            probability = (
                self.noise.correlated_pair_error_at_optimum
                + self.noise.correlated_pair_phase_curvature
                * (
                    request.target_phase_compensation_rad
                    - self.noise.target_phase_compensation_rad_optimum
                )
                ** 2
            )
        return float(np.clip(request.repetitions * probability, 1.0e-9, 0.45))

    def run_characterization(
        self,
        request: Cz2CharacterizationRequest,
        *,
        rng: np.random.Generator | None = None,
        accepted_shots_used: int | None = None,
    ) -> JobCz2CharacterizationData | str:
        with self._lock:
            if error := self._check_budget(request.shots):
                return error
            active_rng = self._resolve_rng(rng)
            pair_probability = self._characterization_pair_probability(request)
            arm_probability = min(
                0.25,
                request.repetitions * self.noise.independent_clifford_error,
            )
            readout = np.zeros((request.shots, 3), dtype=np.uint8)
            readout ^= (active_rng.random((request.shots, 3)) < arm_probability).astype(np.uint8)
            pair_events = active_rng.random(request.shots) < pair_probability
            readout[:, 1:] ^= pair_events[:, None].astype(np.uint8)
            readout ^= (
                active_rng.random((request.shots, 3))
                < self.noise.characterization_readout_flip_probability
            ).astype(np.uint8)
            self._shots_used += request.shots
            return JobCz2CharacterizationData(
                echo_protocol=request.echo_protocol,
                repetitions=request.repetitions,
                duration_scale=request.duration_scale,
                target_phase_compensation_rad=request.target_phase_compensation_rad,
                shots=request.shots,
                readout_bits_b64=pack_bits(readout),
                readout_shape=[request.shots, 3],
                budget=self._budget(accepted_shots_used),
            )

    def run_memory(
        self,
        request: MultitargetMemoryRequest,
        *,
        rng: np.random.Generator | None = None,
        accepted_shots_used: int | None = None,
    ) -> JobMultitargetMemoryData | str:
        with self._lock:
            if error := self._check_budget(request.shots):
                return error
            try:
                canonical_cycle(request.cycle, self.public)
                digest = canonical_cycle_digest(request.cycle, self.public)
            except ValueError as exc:
                return f"invalid depth-reduced CZ2 cycle: {exc}"
            circuit = build_memory_circuit(
                self.public,
                self.noise,
                cycle=request.cycle,
                duration_scale=request.duration_scale,
                target_phase_compensation_rad=request.target_phase_compensation_rad,
                basis=request.decoded_basis,
                rounds=request.rounds,
            )
            sample = sample_memory_measurements(
                circuit,
                shots=request.shots,
                rng=self._resolve_rng(rng),
                rounds=request.rounds,
            )
            self._shots_used += request.shots
            return JobMultitargetMemoryData(
                decoded_basis=request.decoded_basis,
                rounds=request.rounds,
                shots=request.shots,
                cycle_digest=digest,
                cycle_duration_us=cycle_duration_us(self.public, request.duration_scale),
                duration_scale=request.duration_scale,
                target_phase_compensation_rad=request.target_phase_compensation_rad,
                syndrome_measurement_bits_b64=pack_bits(sample.syndrome_measurements),
                syndrome_measurement_shape=list(sample.syndrome_measurements.shape),
                final_data_measurement_bits_b64=pack_bits(sample.final_data_measurements),
                final_data_measurement_shape=list(sample.final_data_measurements.shape),
                budget=self._budget(accepted_shots_used),
            )

    def replay_design(
        self,
        request: MultitargetMemoryRequest,
        *,
        shots_per_basis: int,
        rng: np.random.Generator | None = None,
    ) -> dict[str, dict[str, int]]:
        """Independently sample and decode the final design in both logical bases."""
        canonical_cycle(request.cycle, self.public)
        active_rng = self._resolve_rng(rng)
        outcomes: dict[str, dict[str, int]] = {}
        for basis in ("Z", "X"):
            circuit = build_memory_circuit(
                self.public,
                self.noise,
                cycle=request.cycle,
                duration_scale=request.duration_scale,
                target_phase_compensation_rad=request.target_phase_compensation_rad,
                basis=basis,
                rounds=request.rounds,
            )
            sample = sample_memory_measurements(
                circuit,
                shots=shots_per_basis,
                rng=active_rng,
                rounds=request.rounds,
            )
            failures = decode_memory_measurements(
                self.public,
                sample,
                basis=basis,
                rounds=request.rounds,
            )
            outcomes[basis] = {
                "failures": int(failures.sum()),
                "shots": shots_per_basis,
            }
        return outcomes

    def submitted_pair_probability(self, request: MultitargetMemoryRequest) -> float:
        """Private diagnostic used by construction reports, never by public result payloads."""
        return correlated_pair_probability(
            self.noise,
            duration_scale=request.duration_scale,
            target_phase_compensation_rad=request.target_phase_compensation_rad,
        )


__all__ = [
    "MultitargetReadoutEngine",
    "canonical_cycle",
    "canonical_cycle_digest",
    "domain_separated_rng",
]
