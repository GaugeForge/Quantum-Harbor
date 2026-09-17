"""Opt-in IBM Quantum live-provider substrate for circuit requests.

The backend is deliberately fail-closed. Construction never touches provider
credentials or the IBM SDK; provider calls happen only after the explicit
``QIQCBENCH_ALLOW_LIVE_PROVIDER=1`` gate is open and local caps pass.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from qiqcbench.qsim.backends._live_constants import (
    IBM_QUANTUM_PLATFORM_CHANNEL,
    LIVE_PROVIDER_DISABLED_ERROR,
)
from qiqcbench.qsim.backends.provider_artifacts import (
    ProviderArtifactV1,
    ProviderStatusEvent,
)
from qiqcbench.qsim.core.wire import (
    CircuitRequest,
    CircuitSweepRequest,
    JobBitstringData,
    JobResult,
    JobResultMetadata,
)

__all__ = [
    "IBM_QUANTUM_PLATFORM_CHANNEL",
    "LIVE_PROVIDER_DISABLED_ERROR",
    "LiveIBMCircuitBackend",
]


@dataclass
class _CompiledLiveRequest:
    circuits: list[QuantumCircuit]
    measured_qubits: list[int]
    metadata: JobResultMetadata
    request_payload: dict[str, Any]


@dataclass
class _LiveProviderHandle:
    provider_job: Any
    provider_job_id: str
    requested_shots: int
    measured_qubits: list[int]
    request_payload: dict[str, Any]
    circuit_count: int
    start_monotonic: float
    status_timeline: list[ProviderStatusEvent]
    metadata: JobResultMetadata
    last_provider_state: str | None = None


@dataclass
class _ExtractedCounts:
    counts: list[dict[str, int]]
    bitstrings: list[list[str]]


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _provider_state(value: Any) -> str:
    if hasattr(value, "name"):
        return str(value.name)
    return str(value)


class LiveIBMCircuitBackend:
    """IBM Quantum live backend substrate for QIQCBench circuit requests."""

    def __init__(
        self,
        *,
        task_id: str,
        device_id: str,
        backend_name: str,
        log_dir: str | Path | None = None,
        token: str | None = None,
        instance: str | None = None,
        channel: str = IBM_QUANTUM_PLATFORM_CHANNEL,
        allow_live_provider: bool = False,
        max_shots: int | None = None,
        max_jobs: int = 1,
        backend_allowlist: Iterable[str] | None = None,
        wall_clock_s: float | None = None,
        monotonic: Callable[[], float] | None = None,
        **_qtype_options: Any,
    ) -> None:
        self.task_id = task_id
        self.device_id = device_id
        self.backend_name = backend_name
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.token = token
        self.instance = instance
        self.channel = channel or IBM_QUANTUM_PLATFORM_CHANNEL
        self.allow_live_provider = allow_live_provider
        self.max_shots = max_shots
        self.max_jobs = max_jobs
        self.backend_allowlist = tuple(
            item.strip() for item in (backend_allowlist or []) if item.strip()
        )
        self.wall_clock_s = wall_clock_s
        self._monotonic = monotonic or time.monotonic
        self._lock = threading.Lock()
        self._handles: dict[str, _LiveProviderHandle] = {}
        self._active_jobs = 0
        self._service: Any | None = None
        self._provider_backend: Any | None = None

    def run_circuit(
        self,
        request: CircuitRequest,
        job_id: str,
        salt: int,
    ) -> JobResult:
        try:
            compiled = self._compile_circuit_request(request)
        except ValueError as exc:
            return self._failed(job_id, request.shots, str(exc))
        return self._submit(compiled, request.shots, job_id)

    def run_circuit_sweep(
        self,
        request: CircuitSweepRequest,
        job_id: str,
        salt: int,
    ) -> JobResult:
        try:
            compiled = self._compile_circuit_sweep_request(request)
        except ValueError as exc:
            return self._failed(job_id, request.shots, str(exc))
        return self._submit(compiled, request.shots, job_id)

    def _compile_circuit_request(self, request: CircuitRequest) -> _CompiledLiveRequest:
        raise NotImplementedError("qtype adapter must compile circuit requests")

    def _compile_circuit_sweep_request(self, request: CircuitSweepRequest) -> _CompiledLiveRequest:
        raise NotImplementedError("qtype adapter must compile circuit sweep requests")

    def poll_job_result(self, job_id: str, current: JobResult) -> JobResult:
        with self._lock:
            handle = self._handles.get(job_id)
        if handle is None:
            return current

        if self.wall_clock_s is not None:
            elapsed = self._monotonic() - handle.start_monotonic
            if elapsed > self.wall_clock_s:
                try:
                    handle.provider_job.cancel()
                except Exception:
                    pass
                return self._finish_failed(
                    job_id,
                    handle,
                    "live provider wall-clock timeout exceeded",
                    state="LOCAL_TIMEOUT",
                )

        try:
            state = _provider_state(handle.provider_job.status())
            final = bool(handle.provider_job.in_final_state())
        except Exception as exc:
            return self._finish_failed(job_id, handle, f"provider status polling failed: {exc}")

        with self._lock:
            if job_id not in self._handles:
                return current
            state_changed = state != handle.last_provider_state
            if state_changed:
                handle.status_timeline.append(ProviderStatusEvent(state=state, ts=_utc_now()))
                handle.last_provider_state = state

        if not final:
            if not state_changed:
                return current
            self._write_artifact(job_id, self._artifact_from_handle(handle))
            return current

        if state == "DONE":
            try:
                extracted = self._extract_counts(
                    handle.provider_job.result(),
                    measured_qubits=handle.measured_qubits,
                )
            except Exception as exc:
                return self._finish_failed(
                    job_id, handle, f"provider result extraction failed: {exc}"
                )
            result = JobResult(
                job_id=job_id,
                device_id=self.device_id,
                status="complete",
                shots=handle.requested_shots,
                data=JobBitstringData(
                    bitstrings=extracted.bitstrings,
                    measured_qubits=handle.measured_qubits,
                ),
                metadata=handle.metadata,
            )
            self._finish_handle(
                job_id,
                handle,
                bitstrings=extracted.bitstrings,
                counts=extracted.counts,
            )
            return result

        if state == "CANCELLED":
            return self._finish_failed(job_id, handle, "provider job cancelled")
        return self._finish_failed(job_id, handle, f"provider job failed with status {state}")

    def _submit(
        self,
        compiled: _CompiledLiveRequest,
        shots: int,
        job_id: str,
    ) -> JobResult:
        if not self.allow_live_provider:
            return self._failed(job_id, shots, LIVE_PROVIDER_DISABLED_ERROR)
        if self.channel != IBM_QUANTUM_PLATFORM_CHANNEL:
            return self._failed_with_artifact(
                job_id,
                shots,
                f"QISKIT_IBM_CHANNEL must be {IBM_QUANTUM_PLATFORM_CHANNEL!r}",
                compiled.measured_qubits,
            )
        if not self.token:
            return self._failed_with_artifact(
                job_id,
                shots,
                "QISKIT_IBM_TOKEN is required for live provider mode",
                compiled.measured_qubits,
            )
        if not self.backend_allowlist:
            return self._failed_with_artifact(
                job_id,
                shots,
                "QIQCBENCH_LIVE_BACKEND_ALLOWLIST is required for live provider mode",
                compiled.measured_qubits,
            )
        if self.backend_name not in self.backend_allowlist:
            return self._failed_with_artifact(
                job_id,
                shots,
                f"backend {self.backend_name!r} is not in live backend allowlist",
                compiled.measured_qubits,
            )
        if self.wall_clock_s is not None and self.wall_clock_s <= 0:
            return self._failed_with_artifact(
                job_id,
                shots,
                "live provider wall-clock timeout exceeded",
                compiled.measured_qubits,
                usage_metadata=self._usage_metadata(compiled),
            )

        if not self._reserve_job_slot():
            return self._failed_with_artifact(
                job_id,
                shots,
                "live provider max_jobs limit reached",
                compiled.measured_qubits,
                usage_metadata=self._usage_metadata(compiled),
            )

        try:
            provider_backend = self._get_provider_backend()
            pass_manager = generate_preset_pass_manager(
                backend=provider_backend,
                optimization_level=1,
            )
            transpiled = [pass_manager.run(circuit) for circuit in compiled.circuits]
            sampler = SamplerV2(mode=provider_backend)
            provider_job = sampler.run(transpiled, shots=shots)
            provider_job_id = str(provider_job.job_id())
        except Exception as exc:
            self._release_job_slot()
            return self._failed_with_artifact(
                job_id,
                shots,
                str(exc),
                compiled.measured_qubits,
                usage_metadata=self._usage_metadata(compiled),
            )

        handle = _LiveProviderHandle(
            provider_job=provider_job,
            provider_job_id=provider_job_id,
            requested_shots=shots,
            measured_qubits=compiled.measured_qubits,
            request_payload=compiled.request_payload,
            circuit_count=len(compiled.circuits),
            start_monotonic=self._monotonic(),
            status_timeline=[],
            metadata=compiled.metadata,
        )
        with self._lock:
            self._handles[job_id] = handle
        self._write_artifact(job_id, self._artifact_from_handle(handle))
        return JobResult(
            job_id=job_id,
            device_id=self.device_id,
            status="running",
            shots=shots,
            metadata=compiled.metadata,
        )

    def _reserve_job_slot(self) -> bool:
        with self._lock:
            if self._active_jobs >= self.max_jobs:
                return False
            self._active_jobs += 1
            return True

    def _release_job_slot(self) -> None:
        with self._lock:
            if self._active_jobs > 0:
                self._active_jobs -= 1

    def _get_provider_backend(self) -> Any:
        with self._lock:
            if self._provider_backend is not None:
                return self._provider_backend
            if self._service is None:
                self._service = QiskitRuntimeService(
                    channel=IBM_QUANTUM_PLATFORM_CHANNEL,
                    token=self.token,
                    instance=self.instance,
                )
            self._provider_backend = self._service.backend(self.backend_name)
            return self._provider_backend

    def _finish_handle(
        self,
        job_id: str,
        handle: _LiveProviderHandle,
        *,
        bitstrings: list[list[str]] | None = None,
        counts: list[dict[str, int]] | None = None,
        failure: dict[str, Any] | None = None,
    ) -> None:
        artifact = self._artifact_from_handle(
            handle,
            bitstrings=bitstrings,
            counts=counts,
            failure=failure,
        )
        self._write_artifact(job_id, artifact)
        with self._lock:
            self._handles.pop(job_id, None)
            if self._active_jobs > 0:
                self._active_jobs -= 1

    def _finish_failed(
        self,
        job_id: str,
        handle: _LiveProviderHandle,
        error: str,
        *,
        state: str | None = None,
    ) -> JobResult:
        if state is not None:
            with self._lock:
                if handle.last_provider_state != state:
                    handle.status_timeline.append(ProviderStatusEvent(state=state, ts=_utc_now()))
                    handle.last_provider_state = state
        safe_error = self._redact_known_values(error)
        self._finish_handle(
            job_id,
            handle,
            failure={"error": safe_error},
        )
        return JobResult(
            job_id=job_id,
            device_id=self.device_id,
            status="failed",
            shots=handle.requested_shots,
            error=safe_error,
            metadata=handle.metadata,
        )

    def _failed(self, job_id: str, shots: int, error: str) -> JobResult:
        return JobResult(
            job_id=job_id,
            device_id=self.device_id,
            status="failed",
            shots=shots,
            error=self._redact_known_values(error),
        )

    def _failed_request_with_artifact(
        self,
        job_id: str,
        shots: int,
        error: str,
        measured_qubits: list[int],
        request_payload: dict[str, Any],
        *,
        circuit_count: int | None = None,
    ) -> JobResult:
        # Gate closed → no artifact: the request never reached the provider.
        if not self.allow_live_provider:
            return self._failed(job_id, shots, error)
        usage_metadata: dict[str, Any] = {"request": request_payload}
        if circuit_count is not None:
            usage_metadata["circuit_count"] = circuit_count
        # Local-validation failures get a distinct provider_name so they're
        # easy to filter out of "real provider" event sets.
        return self._failed_with_artifact(
            job_id,
            shots,
            error,
            measured_qubits,
            usage_metadata=usage_metadata,
            provider_name="local_validation",
        )

    def _failed_with_artifact(
        self,
        job_id: str,
        shots: int,
        error: str,
        measured_qubits: list[int] | None = None,
        usage_metadata: dict[str, Any] | None = None,
        *,
        provider_name: str = IBM_QUANTUM_PLATFORM_CHANNEL,
    ) -> JobResult:
        artifact = ProviderArtifactV1(
            task_id=self.task_id,
            device_id=self.device_id,
            backend_mode="live_provider",
            provider_name=provider_name,
            backend_name=self.backend_name,
            qiqcbench_job_id=job_id,
            provider_job_ids=[],
            requested_shots=shots,
            returned_shots=0,
            measured_qubits=measured_qubits or [],
            status_timeline=[ProviderStatusEvent(state="failed", ts=_utc_now())],
            usage_metadata=usage_metadata,
            failure={"error": self._redact_known_values(error)},
        )
        self._write_artifact(job_id, artifact)
        return self._failed(job_id, shots, error)

    def _artifact_from_handle(
        self,
        handle: _LiveProviderHandle,
        *,
        bitstrings: list[list[str]] | None = None,
        counts: list[dict[str, int]] | None = None,
        failure: dict[str, Any] | None = None,
    ) -> ProviderArtifactV1:
        returned_shots = 0
        if bitstrings:
            returned_shots = len(bitstrings[0])
        return ProviderArtifactV1(
            task_id=self.task_id,
            device_id=self.device_id,
            backend_mode="live_provider",
            provider_name=IBM_QUANTUM_PLATFORM_CHANNEL,
            backend_name=self.backend_name,
            qiqcbench_job_id="",
            provider_job_ids=[handle.provider_job_id],
            requested_shots=handle.requested_shots,
            returned_shots=returned_shots,
            measured_qubits=handle.measured_qubits,
            bitstrings=bitstrings,
            counts=counts,
            status_timeline=handle.status_timeline,
            usage_metadata={
                "circuit_count": handle.circuit_count,
                "request": handle.request_payload,
                "sweep_coords": handle.metadata.sweep_coords,
            },
            failure=failure,
        )

    def _usage_metadata(self, compiled: _CompiledLiveRequest) -> dict[str, Any]:
        return {
            "circuit_count": len(compiled.circuits),
            "request": compiled.request_payload,
            "sweep_coords": compiled.metadata.sweep_coords,
        }

    def _write_artifact(self, job_id: str, artifact: ProviderArtifactV1) -> None:
        if self.log_dir is None:
            return
        path = self.log_dir / "provider_artifacts" / f"{job_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        sidecar = artifact.model_copy(update={"qiqcbench_job_id": job_id})
        path.write_text(sidecar.model_dump_json(indent=2) + "\n", encoding="utf-8")

    def _redact_known_values(self, text: str) -> str:
        redacted = text
        for value in (self.token, self.instance):
            if value:
                redacted = redacted.replace(value, "<redacted>")
        return redacted

    def _extract_counts(
        self,
        provider_result: Any,
        *,
        measured_qubits: list[int],
    ) -> _ExtractedCounts:
        counts: list[dict[str, int]] = []
        bitstrings: list[list[str]] = []
        width = len(measured_qubits)
        for pub_result in provider_result:
            qiskit_counts = self._counts_from_pub_result(pub_result)
            point_counts = self._normalize_qiskit_counts(qiskit_counts, width=width)
            counts.append(point_counts)
            bitstrings.append(self._counts_to_bitstrings(point_counts))
        return _ExtractedCounts(counts=counts, bitstrings=bitstrings)

    def _counts_from_pub_result(self, pub_result: Any) -> dict[str, int]:
        data = pub_result.data
        register = getattr(data, "c", None)
        if register is not None and hasattr(register, "get_counts"):
            return dict(register.get_counts())
        for name in dir(data):
            if name.startswith("_"):
                continue
            try:
                candidate = getattr(data, name)
            except Exception:
                continue
            if hasattr(candidate, "get_counts"):
                return dict(candidate.get_counts())
        raise ValueError("provider result did not expose a classical register with get_counts()")

    def _normalize_qiskit_counts(self, counts: dict[str, int], *, width: int) -> dict[str, int]:
        normalized: dict[str, int] = {}
        for bitstring, count in counts.items():
            clean = str(bitstring).replace(" ", "")
            if len(clean) != width:
                raise ValueError(
                    f"provider count key {bitstring!r} has width {len(clean)}; expected {width}"
                )
            qiqcbench_bitstring = clean[::-1]
            normalized[qiqcbench_bitstring] = normalized.get(qiqcbench_bitstring, 0) + int(count)
        return dict(sorted(normalized.items()))

    def _counts_to_bitstrings(self, counts: dict[str, int]) -> list[str]:
        shots: list[str] = []
        for bitstring, count in sorted(counts.items()):
            shots.extend([bitstring] * int(count))
        return shots
