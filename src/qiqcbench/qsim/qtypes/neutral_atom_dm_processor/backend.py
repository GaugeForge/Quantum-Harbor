"""Simulator backend for the ``neutral_atom_dm_processor`` qtype.

Holds **one** :class:`NeutralAtomDmEngine` for the whole run so the shot budget
accumulates across calls. The RNG is injected once here from ``hidden.seed``
(the engine never reseeds). Sweep requests charge len(sweep_values) x shots in
a single reservation (bosonic_cavity_qec pattern); execution uses the
preadmitted path so nothing double-charges.
"""

from __future__ import annotations

import threading

import numpy as np

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.neutral_atom_dm_processor.device import (
    HiddenNeutralAtomDmConfig,
    PublicNeutralAtomDmSpec,
)
from qiqcbench.qsim.qtypes.neutral_atom_dm_processor.engine import NeutralAtomDmEngine
from qiqcbench.qsim.qtypes.neutral_atom_dm_processor.wire import (
    AtomProgramRequest,
    AtomProgramSweepRequest,
    JobAtomDmShotData,
    JobAtomDmSweepData,
)

__all__ = ["NeutralAtomDmSimulatorBackend", "build_neutral_atom_dm_simulator_backend"]


class NeutralAtomDmSimulatorBackend:
    def __init__(self, hidden: HiddenNeutralAtomDmConfig, public: PublicNeutralAtomDmSpec):
        self._device_id = hidden.device_id
        rng = np.random.default_rng(hidden.seed)
        self._engine = NeutralAtomDmEngine(hidden, public, rng)
        self._reservation_lock = threading.Lock()
        self._reservations: dict[int, AtomProgramRequest | AtomProgramSweepRequest] = {}

    @property
    def engine(self) -> NeutralAtomDmEngine:
        return self._engine

    def reserve_program(self, request: AtomProgramRequest | AtomProgramSweepRequest) -> int:
        """Validate and atomically charge a request before JobManager enqueue."""

        token = id(request)
        with self._reservation_lock:
            if token in self._reservations:
                raise RuntimeError("duplicate neutral-atom-dm request reservation")
            error = self._engine.reserve_program(request)
            if error is not None:
                raise ValueError(error)
            self._reservations[token] = request
        return token

    def discard_program_reservation(self, token: int) -> None:
        """Drop an enqueue-failed token; charged capacity remains consumed."""

        with self._reservation_lock:
            self._reservations.pop(token, None)

    def _take_reservation(
        self,
        request: AtomProgramRequest | AtomProgramSweepRequest,
        token: int | None,
    ) -> str | None:
        if token is None:
            try:
                token = self.reserve_program(request)
            except (RuntimeError, ValueError) as exc:
                return str(exc)
        with self._reservation_lock:
            reserved_request = self._reservations.pop(token, None)
        if reserved_request is not request:
            return "neutral-atom-dm request reservation is missing or does not match"
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
            return JobResult(job_id=job_id, device_id=self._device_id, status="failed", error=error)
        result = self._engine.run_atom_program(request, preadmitted=True)
        if isinstance(result, JobAtomDmShotData):
            return JobResult(
                job_id=job_id, device_id=self._device_id, status="complete", data=result
            )
        return JobResult(job_id=job_id, device_id=self._device_id, status="failed", error=result)

    def run_atom_program_sweep(
        self,
        request: AtomProgramSweepRequest,
        job_id: str,
        salt: int,
        *,
        reservation_token: int | None = None,
    ) -> JobResult:
        error = self._take_reservation(request, reservation_token)
        if error is not None:
            return JobResult(job_id=job_id, device_id=self._device_id, status="failed", error=error)
        result = self._engine.run_atom_program_sweep(request, preadmitted=True)
        if isinstance(result, JobAtomDmSweepData):
            return JobResult(
                job_id=job_id, device_id=self._device_id, status="complete", data=result
            )
        return JobResult(job_id=job_id, device_id=self._device_id, status="failed", error=result)


def build_neutral_atom_dm_simulator_backend(
    hidden: HiddenNeutralAtomDmConfig, public: PublicNeutralAtomDmSpec
) -> NeutralAtomDmSimulatorBackend:
    return NeutralAtomDmSimulatorBackend(hidden, public)
