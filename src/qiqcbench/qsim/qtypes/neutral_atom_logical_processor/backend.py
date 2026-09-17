"""Simulator backend for the ``neutral_atom_logical_processor`` qtype.

Holds **one** :class:`NeutralAtomEngine` for the whole run so the shot budget accumulates
across calls. The RNG is injected once here from ``hidden.seed`` (the engine never reseeds).
"""

from __future__ import annotations

import threading

import numpy as np

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.neutral_atom_logical_processor.capabilities.located_erasure_repair.runtime import (
    LocatedErasureRepairEngine,
    episode_request_digest,
)
from qiqcbench.qsim.qtypes.neutral_atom_logical_processor.capabilities.located_erasure_repair.wire import (
    ErasureRepairEpisodeRequest,
    JobErasureRepairEpisodeData,
)
from qiqcbench.qsim.qtypes.neutral_atom_logical_processor.device import (
    HiddenNeutralAtomConfig,
    PublicNeutralAtomSpec,
)
from qiqcbench.qsim.qtypes.neutral_atom_logical_processor.engine import NeutralAtomEngine
from qiqcbench.qsim.qtypes.neutral_atom_logical_processor.wire import (
    AtomProgramRequest,
    JobAtomShotData,
)

__all__ = ["NeutralAtomSimulatorBackend", "build_neutral_atom_simulator_backend"]


class NeutralAtomSimulatorBackend:
    def __init__(self, hidden: HiddenNeutralAtomConfig, public: PublicNeutralAtomSpec):
        self._device_id = hidden.device_id
        rng = np.random.default_rng(hidden.seed)
        self._engine = NeutralAtomEngine(hidden, public, rng)
        self._reservation_lock = threading.Lock()
        self._reservations: dict[int, AtomProgramRequest] = {}
        self._repair_engine = (
            LocatedErasureRepairEngine(hidden, public)
            if public.erasure_repair is not None
            else None
        )

    @property
    def engine(self) -> NeutralAtomEngine:
        return self._engine

    def reserve_atom_program(self, request: AtomProgramRequest) -> int:
        """Validate and atomically charge a request before JobManager enqueue."""

        token = id(request)
        with self._reservation_lock:
            if token in self._reservations:
                raise RuntimeError("duplicate neutral-atom request reservation")
            error = self._engine.reserve_atom_program(request)
            if error is not None:
                raise ValueError(error)
            self._reservations[token] = request
        return token

    def discard_atom_program_reservation(self, token: int) -> None:
        """Drop an enqueue-failed token; charged capacity remains consumed."""

        with self._reservation_lock:
            self._reservations.pop(token, None)

    def _take_reservation(self, request: AtomProgramRequest, token: int | None) -> str | None:
        if token is None:
            try:
                token = self.reserve_atom_program(request)
            except (RuntimeError, ValueError) as exc:
                return str(exc)
        with self._reservation_lock:
            reserved_request = self._reservations.pop(token, None)
        if reserved_request is not request:
            return "neutral-atom request reservation is missing or does not match"
        return None

    def run_atom_program(
        self,
        request: AtomProgramRequest,
        job_id: str,
        salt: int,
        *,
        reservation_token: int | None = None,
    ) -> JobResult:
        error = self._take_reservation(request, reservation_token)
        if error is not None:
            return JobResult(
                job_id=job_id,
                device_id=self._device_id,
                status="failed",
                error=error,
            )
        result = self._engine.run_atom_program(request, preadmitted=True)
        if isinstance(result, JobAtomShotData):
            return JobResult(
                job_id=job_id, device_id=self._device_id, status="complete", data=result
            )
        return JobResult(job_id=job_id, device_id=self._device_id, status="failed", error=result)

    def reserve_erasure_repair_episodes(self, request: ErasureRepairEpisodeRequest) -> int:
        if self._repair_engine is None:
            raise ValueError("located-erasure control is not configured for this device")
        return self._repair_engine.reserve_episodes(request.episodes)

    def discard_erasure_repair_reservation(
        self, token: int, request: ErasureRepairEpisodeRequest
    ) -> None:
        if self._repair_engine is not None:
            self._repair_engine.discard_reservation(token, episodes=request.episodes)

    def run_located_erasure_control_episodes(
        self,
        request: ErasureRepairEpisodeRequest,
        job_id: str,
        salt: int,
        *,
        reservation_token: int | None = None,
    ) -> JobResult:
        del salt
        if self._repair_engine is None:
            return JobResult(
                job_id=job_id,
                device_id=self._device_id,
                status="failed",
                error="located-erasure repair service is not configured for this device",
            )
        result = self._repair_engine.run(
            request,
            reservation_token=reservation_token,
            entropy_namespace=(
                f"public-characterization-v3:job:{job_id}:request:{episode_request_digest(request)}"
            ),
        )
        if isinstance(result, JobErasureRepairEpisodeData):
            return JobResult(
                job_id=job_id, device_id=self._device_id, status="complete", data=result
            )
        return JobResult(job_id=job_id, device_id=self._device_id, status="failed", error=result)


def build_neutral_atom_simulator_backend(
    hidden: HiddenNeutralAtomConfig, public: PublicNeutralAtomSpec
) -> NeutralAtomSimulatorBackend:
    return NeutralAtomSimulatorBackend(hidden, public)
