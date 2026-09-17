"""Simulator backend with atomic run-wide admission and drift indexing."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.scheduled_transmon_gate_model.bank import sha256_json
from qiqcbench.qsim.qtypes.scheduled_transmon_gate_model.device import (
    HiddenScheduledTransmonConfig,
    PublicScheduledTransmonSpec,
)
from qiqcbench.qsim.qtypes.scheduled_transmon_gate_model.engine import ScheduledTransmonEngine
from qiqcbench.qsim.qtypes.scheduled_transmon_gate_model.instance import (
    load_public_candidate_bank,
)
from qiqcbench.qsim.qtypes.scheduled_transmon_gate_model.wire import (
    CompilationMirrorRequest,
    CompilationReadoutReferenceRequest,
)


@dataclass(frozen=True)
class CompilationReservation:
    token: int
    submission_index: int
    request_digest: str
    kind: str


class ScheduledTransmonSimulatorBackend:
    def __init__(
        self,
        hidden: HiddenScheduledTransmonConfig,
        public: PublicScheduledTransmonSpec,
        *,
        run_entropy: int | None = None,
    ) -> None:
        self.public = public
        self.bank = load_public_candidate_bank(
            public.task_id,
            public.candidate_bank_file,
            public.candidate_bank_sha256,
        )
        self.engine = ScheduledTransmonEngine(
            hidden,
            public,
            self.bank,
            run_entropy=run_entropy,
        )
        self._lock = threading.Lock()
        self._next_token = 0
        self._next_submission_index = 0
        self._shots_used = 0
        self._jobs_used = 0
        self._recorded_bits_used = 0
        self._reservations: dict[
            int,
            tuple[str, CompilationMirrorRequest | CompilationReadoutReferenceRequest],
        ] = {}

    @property
    def run_entropy(self) -> int:
        """Per-run entropy; qsim-internal and never logged."""

        return self.engine.run_entropy

    def _reserve(
        self,
        kind: str,
        request: CompilationMirrorRequest | CompilationReadoutReferenceRequest,
    ) -> CompilationReservation:
        budgets = self.public.budgets
        width = len(self.public.measured_physical_qubits)
        if request.shots > budgets.max_shots_per_job:
            raise ValueError(
                f"shots {request.shots} exceeds max_shots_per_job {budgets.max_shots_per_job}"
            )
        recorded_bits = request.shots * width
        if recorded_bits > budgets.max_recorded_bits_per_job:
            raise ValueError("request exceeds max_recorded_bits_per_job")
        if isinstance(request, CompilationMirrorRequest):
            if request.candidate_id not in self.public.candidate_ids:
                raise ValueError(f"unknown candidate_id {request.candidate_id!r}")
            if request.mirror_seed not in self.public.mirror_seed_domain:
                raise ValueError(f"mirror_seed {request.mirror_seed} is outside the public domain")
        elif request.prepared_bitstring not in self.public.trusted_reference_preparations:
            raise ValueError("prepared_bitstring is not an allowed trusted reference")

        with self._lock:
            if self._shots_used + request.shots > budgets.shot_budget:
                raise ValueError(
                    f"shot budget exhausted: {self._shots_used}+{request.shots} "
                    f"> {budgets.shot_budget}"
                )
            if self._jobs_used + 1 > budgets.experiment_job_budget:
                raise ValueError("experiment-job budget exhausted")
            if self._recorded_bits_used + recorded_bits > budgets.max_recorded_bits_per_run:
                raise ValueError("run-wide raw-result budget exhausted")
            self._next_token += 1
            self._next_submission_index += 1
            token = self._next_token
            index = self._next_submission_index
            self._shots_used += request.shots
            self._jobs_used += 1
            self._recorded_bits_used += recorded_bits
            self._reservations[token] = (kind, request)
            return CompilationReservation(
                token=token,
                submission_index=index,
                request_digest=sha256_json(request.model_dump(mode="json")),
                kind=kind,
            )

    def reserve_mirror(self, request: CompilationMirrorRequest) -> CompilationReservation:
        return self._reserve("mirror", request)

    def reserve_readout_reference(
        self, request: CompilationReadoutReferenceRequest
    ) -> CompilationReservation:
        return self._reserve("readout_reference", request)

    def discard_reservation(self, reservation: CompilationReservation) -> None:
        """Drop an enqueue-failed pending token; accepted capacity remains charged."""

        with self._lock:
            self._reservations.pop(reservation.token, None)

    def _take(
        self,
        reservation: CompilationReservation,
        expected_kind: str,
        request: CompilationMirrorRequest | CompilationReadoutReferenceRequest,
    ) -> str | None:
        with self._lock:
            stored = self._reservations.pop(reservation.token, None)
        if stored is None:
            return "compilation reservation is missing"
        if stored[0] != expected_kind or stored[1] is not request:
            return "compilation reservation does not match the accepted request"
        return None

    def run_compilation_mirror(
        self,
        request: CompilationMirrorRequest,
        job_id: str,
        salt: int,
        *,
        reservation: CompilationReservation,
    ) -> JobResult:
        if error := self._take(reservation, "mirror", request):
            return JobResult(
                job_id=job_id, device_id=self.public.device_id, status="failed", error=error
            )
        data = self.engine.run_mirror(
            request,
            submission_index=reservation.submission_index,
            shot_salt=salt,
        )
        return JobResult(
            job_id=job_id, device_id=self.public.device_id, status="complete", data=data
        )

    def run_compilation_readout_reference(
        self,
        request: CompilationReadoutReferenceRequest,
        job_id: str,
        salt: int,
        *,
        reservation: CompilationReservation,
    ) -> JobResult:
        if error := self._take(reservation, "readout_reference", request):
            return JobResult(
                job_id=job_id, device_id=self.public.device_id, status="failed", error=error
            )
        data = self.engine.run_readout_reference(
            request,
            submission_index=reservation.submission_index,
            shot_salt=salt,
        )
        return JobResult(
            job_id=job_id, device_id=self.public.device_id, status="complete", data=data
        )


def build_scheduled_transmon_backend(
    hidden: HiddenScheduledTransmonConfig,
    public: PublicScheduledTransmonSpec,
) -> ScheduledTransmonSimulatorBackend:
    return ScheduledTransmonSimulatorBackend(hidden, public)


__all__ = [
    "CompilationReservation",
    "ScheduledTransmonSimulatorBackend",
    "build_scheduled_transmon_backend",
]
