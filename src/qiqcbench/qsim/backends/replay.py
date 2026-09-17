"""Deterministic provider replay substrate for provider request fixtures."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from qiqcbench.qsim.backends.provider_artifacts import (
    ProviderArtifactV1,
    ProviderStatusEvent,
    replay_fixture_path,
)
from qiqcbench.qsim.core.wire import (
    CircuitRequest,
    CircuitSweepRequest,
    JobBitstringData,
    JobResult,
    JobResultMetadata,
)


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _request_shots(request: BaseModel) -> int:
    shots = getattr(request, "shots", None)
    if not isinstance(shots, int):  # pragma: no cover - all wire requests carry shots
        raise TypeError(f"Replay request {type(request).__name__} does not carry integer shots")
    return shots


class ReplayFixtureBackend:
    """Fail-closed provider replay base backed by provider-artifact fixtures."""

    def __init__(
        self,
        task_id: str,
        device_id: str,
        replay_root: str | Path,
        log_dir: str | Path | None = None,
    ) -> None:
        self.task_id = task_id
        self.device_id = device_id
        self.replay_root = Path(replay_root)
        self.log_dir = Path(log_dir) if log_dir is not None else None

    def _replay_request(
        self,
        request: BaseModel,
        job_id: str,
        metadata: JobResultMetadata | None = None,
    ) -> JobResult:
        shots = _request_shots(request)
        try:
            path = replay_fixture_path(
                self.replay_root,
                self.task_id,
                self.device_id,
                request,
                must_exist=True,
            )
        except FileNotFoundError as exc:
            return self._failed(job_id, shots, str(exc))

        try:
            artifact = ProviderArtifactV1.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return self._failed(job_id, shots, f"invalid replay fixture {path}: {exc}")
        validation_error = self._validate_artifact(artifact, request, shots)
        if validation_error is not None:
            return self._failed(job_id, shots, validation_error)

        sidecar = artifact.model_copy(update={"qiqcbench_job_id": job_id})
        self._write_sidecar(job_id, sidecar)

        if artifact.failure is not None:
            error = str(artifact.failure.get("error", artifact.failure))
            return JobResult(
                job_id=job_id,
                device_id=self.device_id,
                status="failed",
                shots=shots,
                error=error,
            )
        if artifact.bitstrings is None:
            return self._failed(
                job_id,
                shots,
                "replay fixture does not contain bitstrings",
            )

        return JobResult(
            job_id=job_id,
            device_id=self.device_id,
            status="complete",
            shots=shots,
            data=JobBitstringData(
                bitstrings=artifact.bitstrings,
                measured_qubits=artifact.measured_qubits,
            ),
            metadata=metadata or JobResultMetadata(),
        )

    def _validate_artifact(
        self,
        artifact: ProviderArtifactV1,
        request: BaseModel,
        shots: int,
    ) -> str | None:
        del request  # generic checks only need identity and shot metadata
        if artifact.task_id != self.task_id:
            return (
                f"replay fixture task_id {artifact.task_id!r} does not match "
                f"requested task {self.task_id!r}"
            )
        if artifact.device_id != self.device_id:
            return (
                f"replay fixture device_id {artifact.device_id!r} does not match "
                f"requested device {self.device_id!r}"
            )
        if artifact.backend_mode != "provider_replay":
            return "replay fixture backend_mode must be 'provider_replay'"
        if artifact.requested_shots != shots:
            return (
                f"replay fixture requested_shots {artifact.requested_shots} "
                f"does not match request shots {shots}"
            )
        if artifact.failure is None and artifact.returned_shots != shots:
            return (
                f"replay fixture returned_shots {artifact.returned_shots} "
                f"does not match request shots {shots}"
            )
        return None

    def _failed(self, job_id: str, shots: int, error: str) -> JobResult:
        artifact = ProviderArtifactV1(
            task_id=self.task_id,
            device_id=self.device_id,
            backend_mode="provider_replay",
            provider_name="replay_fixture",
            backend_name="provider_replay",
            qiqcbench_job_id=job_id,
            requested_shots=shots,
            returned_shots=0,
            measured_qubits=[],
            bitstrings=None,
            status_timeline=[ProviderStatusEvent(state="failed", ts=_utc_now())],
            failure={"error": error},
        )
        self._write_sidecar(job_id, artifact)
        return JobResult(
            job_id=job_id,
            device_id=self.device_id,
            status="failed",
            shots=shots,
            error=error,
        )

    def _write_sidecar(self, job_id: str, artifact: ProviderArtifactV1) -> None:
        if self.log_dir is None:
            return
        path = self.log_dir / "provider_artifacts" / f"{job_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")


class ReplayCircuitBackend(ReplayFixtureBackend):
    """Fail-closed circuit backend backed by provider-artifact fixtures."""

    def run_circuit(
        self,
        request: CircuitRequest,
        job_id: str,
        salt: int,
    ) -> JobResult:
        return self._replay_request(request, job_id)

    def run_circuit_sweep(
        self,
        request: CircuitSweepRequest,
        job_id: str,
        salt: int,
    ) -> JobResult:
        return self._replay_request(request, job_id)
