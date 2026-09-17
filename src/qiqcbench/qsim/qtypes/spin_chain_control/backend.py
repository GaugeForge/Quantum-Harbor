"""Run-long backend with atomic admission and a common drift timeline."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
from dataclasses import dataclass

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.spin_chain_control.device import (
    HiddenSpinChainControlConfig,
    PublicSpinChainControlSpec,
)
from qiqcbench.qsim.qtypes.spin_chain_control.engine import SpinChainControlEngine
from qiqcbench.qsim.qtypes.spin_chain_control.wire import (
    ControlReadoutReferenceRequest,
    ControlTransferRequest,
)


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ControlReservation:
    token: int
    submission_index: int
    request_digest: str
    kind: str


class SpinChainControlSimulatorBackend:
    """Own one private sampling-entropy root for the complete qsim attempt.

    ``run_entropy`` is an internal deterministic construction/test seam. Normal
    production construction draws it once from the OS CSPRNG and never logs or
    returns it.
    """

    def __init__(
        self,
        hidden: HiddenSpinChainControlConfig,
        public: PublicSpinChainControlSpec,
        *,
        run_entropy: int | None = None,
    ) -> None:
        if run_entropy is None:
            run_entropy = secrets.randbits(128)
        if (
            isinstance(run_entropy, bool)
            or not isinstance(run_entropy, int)
            or not 0 <= run_entropy < 2**128
        ):
            raise ValueError("run_entropy must be a 128-bit non-negative integer")
        self.public = public
        self.engine = SpinChainControlEngine(hidden, public, run_entropy=run_entropy)
        self._lock = threading.Lock()
        self._next_token = 0
        self._next_submission_index = 0
        self._shots_used = 0
        self._jobs_used = 0
        self._bits_used = 0
        self._reservations: dict[
            int, tuple[str, ControlTransferRequest | ControlReadoutReferenceRequest]
        ] = {}

    def _reserve(
        self,
        kind: str,
        request: ControlTransferRequest | ControlReadoutReferenceRequest,
    ) -> ControlReservation:
        budgets = self.public.budgets
        if request.shots > budgets.max_shots_per_job:
            raise ValueError("shots exceeds max_shots_per_job")
        bits = request.shots * len(self.public.measured_qubits)
        if bits > budgets.max_recorded_bits_per_job:
            raise ValueError("request exceeds max_recorded_bits_per_job")
        if isinstance(request, ControlTransferRequest):
            value = request.x_drive_scale_fraction
            if not (
                self.public.x_drive_scale_fraction_min
                <= value
                <= self.public.x_drive_scale_fraction_max
            ):
                raise ValueError("x_drive_scale_fraction is outside the public domain")
            steps = value / self.public.x_drive_scale_fraction_resolution
            if abs(steps - round(steps)) > 1e-7:
                raise ValueError("x_drive_scale_fraction is off the public command grid")
            if request.controller_id not in self.public.controller_ids:
                raise ValueError("unknown controller_id")
        elif request.prepared_bitstring not in self.public.trusted_reference_preparations:
            raise ValueError("prepared_bitstring is not a trusted reference")
        with self._lock:
            if self._shots_used + request.shots > budgets.shot_budget:
                raise ValueError("shot budget exhausted")
            if self._jobs_used + 1 > budgets.experiment_job_budget:
                raise ValueError("experiment-job budget exhausted")
            if self._bits_used + bits > budgets.max_recorded_bits_per_run:
                raise ValueError("run-wide raw-result budget exhausted")
            self._next_token += 1
            self._next_submission_index += 1
            token = self._next_token
            self._shots_used += request.shots
            self._jobs_used += 1
            self._bits_used += bits
            self._reservations[token] = (kind, request)
            return ControlReservation(
                token=token,
                submission_index=self._next_submission_index,
                request_digest=_sha256_json(request.model_dump(mode="json")),
                kind=kind,
            )

    def reserve_transfer(self, request: ControlTransferRequest) -> ControlReservation:
        return self._reserve("transfer", request)

    def reserve_readout_reference(
        self, request: ControlReadoutReferenceRequest
    ) -> ControlReservation:
        return self._reserve("readout_reference", request)

    def discard_reservation(self, reservation: ControlReservation) -> None:
        with self._lock:
            self._reservations.pop(reservation.token, None)

    def _take(
        self,
        reservation: ControlReservation,
        kind: str,
        request: ControlTransferRequest | ControlReadoutReferenceRequest,
    ) -> str | None:
        with self._lock:
            stored = self._reservations.pop(reservation.token, None)
        if stored is None:
            return "structured-control reservation is missing"
        if stored[0] != kind or stored[1] is not request:
            return "structured-control reservation does not match the accepted request"
        return None

    def run_control_transfer(
        self,
        request: ControlTransferRequest,
        job_id: str,
        salt: int,
        *,
        reservation: ControlReservation,
    ) -> JobResult:
        del salt
        if error := self._take(reservation, "transfer", request):
            return JobResult(
                job_id=job_id, device_id=self.public.device_id, status="failed", error=error
            )
        data = self.engine.run_transfer(request, submission_index=reservation.submission_index)
        return JobResult(
            job_id=job_id, device_id=self.public.device_id, status="complete", data=data
        )

    def run_control_readout_reference(
        self,
        request: ControlReadoutReferenceRequest,
        job_id: str,
        salt: int,
        *,
        reservation: ControlReservation,
    ) -> JobResult:
        del salt
        if error := self._take(reservation, "readout_reference", request):
            return JobResult(
                job_id=job_id, device_id=self.public.device_id, status="failed", error=error
            )
        data = self.engine.run_readout_reference(
            request, submission_index=reservation.submission_index
        )
        return JobResult(
            job_id=job_id, device_id=self.public.device_id, status="complete", data=data
        )


def build_spin_chain_control_backend(
    hidden: HiddenSpinChainControlConfig,
    public: PublicSpinChainControlSpec,
    *,
    run_entropy: int | None = None,
) -> SpinChainControlSimulatorBackend:
    return SpinChainControlSimulatorBackend(hidden, public, run_entropy=run_entropy)


__all__ = [
    "ControlReservation",
    "SpinChainControlSimulatorBackend",
    "build_spin_chain_control_backend",
]
