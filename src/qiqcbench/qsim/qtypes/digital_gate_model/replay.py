"""Digital qtype adapter for provider replay backends."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from qiqcbench.qsim.backends.replay import ReplayCircuitBackend
from qiqcbench.qsim.core.wire import (
    CircuitOp,
    CircuitRequest,
    CircuitSweepRequest,
    JobResult,
    JobResultMetadata,
)
from qiqcbench.qsim.qtypes.digital_gate_model.device import PublicDigitalSpec
from qiqcbench.qsim.qtypes.digital_gate_model.policy import (
    expand_circuit_sweep_points,
    sweep_coords_from_points,
    validate_circuit_non_sweep_parameters,
    validate_digital_circuit,
)


class DigitalReplayBackend(ReplayCircuitBackend):
    """Provider replay backend constructed from digital public device semantics."""

    def __init__(
        self,
        *,
        task_id: str,
        public: PublicDigitalSpec,
        replay_root: str | Path,
        log_dir: str | Path | None = None,
    ) -> None:
        super().__init__(
            task_id=task_id,
            device_id=public.device_id,
            replay_root=replay_root,
            log_dir=log_dir,
        )
        self.connectivity = _bidirectional_connectivity(public.connectivity)
        self.public_qubits = tuple(qubit.id for qubit in public.qubits)
        self.max_shots = public.max_shots

    def run_circuit(
        self,
        request: CircuitRequest,
        job_id: str,
        salt: int,
    ) -> JobResult:
        error = validate_circuit_non_sweep_parameters(request)
        if error is not None:
            return self._failed(job_id, request.shots, error)

        error = validate_digital_circuit(
            request.circuit,
            self.connectivity,
            request.shots,
            max_shots=self.max_shots,
            public_qubits=self.public_qubits,
            require_measurement=True,
            allowed_parameters=[],
        )
        if error is not None:
            return self._failed(job_id, request.shots, error)
        return self._replay_request(request, job_id)

    def run_circuit_sweep(
        self,
        request: CircuitSweepRequest,
        job_id: str,
        salt: int,
    ) -> JobResult:
        error = validate_digital_circuit(
            request.template_circuit,
            self.connectivity,
            request.shots,
            max_shots=self.max_shots,
            public_qubits=self.public_qubits,
            require_measurement=True,
            allowed_parameters=request.parameters,
        )
        if error is not None:
            return self._failed(job_id, request.shots, error)

        try:
            keys, points = expand_circuit_sweep_points(request)
        except ValueError as exc:
            return self._failed(job_id, request.shots, str(exc))
        if not points:
            error = _first_string_parameter_not_bound(request.template_circuit)
            if error is not None:
                return self._failed(job_id, request.shots, error)
        metadata = JobResultMetadata(sweep_coords=sweep_coords_from_points(keys, points))
        return self._replay_request(request, job_id, metadata=metadata)


def _bidirectional_connectivity(
    connectivity: Iterable[tuple[int, int]],
) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for edge in connectivity:
        a, b = int(edge[0]), int(edge[1])
        edges.add((a, b))
        edges.add((b, a))
    return edges


def _first_string_parameter_not_bound(circuit: list[CircuitOp]) -> str | None:
    for op in circuit:
        for param in getattr(op, "params", []):
            if isinstance(param, str):
                return f"unbound circuit parameter {param!r}"
    return None


def build_digital_replay_backend(
    *,
    task_id: str,
    public: PublicDigitalSpec,
    replay_root: str | Path,
    log_dir: str | Path | None = None,
) -> DigitalReplayBackend:
    """Construct a digital provider-replay backend from qtype-local data."""

    return DigitalReplayBackend(
        task_id=task_id,
        public=public,
        replay_root=replay_root,
        log_dir=log_dir,
    )


__all__ = ["DigitalReplayBackend", "build_digital_replay_backend"]
