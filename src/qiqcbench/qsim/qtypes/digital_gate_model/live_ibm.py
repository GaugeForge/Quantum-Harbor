"""Digital qtype adapter for IBM live-provider backends."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from qiskit import QuantumCircuit

from qiqcbench.qsim.backends.live_ibm import (
    LiveIBMCircuitBackend,
    _CompiledLiveRequest,
)
from qiqcbench.qsim.core.wire import (
    CircuitMeasureOp,
    CircuitOp,
    CircuitRequest,
    CircuitSweepRequest,
    GateOp,
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


class DigitalLiveIBMBackend(LiveIBMCircuitBackend):
    """IBM live backend constructed from digital public device semantics."""

    def __init__(
        self,
        *,
        task_id: str,
        public: PublicDigitalSpec,
        backend_name: str,
        log_dir: str | Path | None = None,
        token: str | None = None,
        instance: str | None = None,
        channel: str,
        allow_live_provider: bool,
        max_shots: int | None = None,
        max_jobs: int = 1,
        backend_allowlist: Iterable[str] | None = None,
        wall_clock_s: float | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        super().__init__(
            task_id=task_id,
            device_id=public.device_id,
            backend_name=backend_name,
            log_dir=log_dir,
            token=token,
            instance=instance,
            channel=channel,
            allow_live_provider=allow_live_provider,
            max_shots=max_shots,
            max_jobs=max_jobs,
            backend_allowlist=backend_allowlist,
            wall_clock_s=wall_clock_s,
            monotonic=monotonic,
        )
        self.connectivity = _bidirectional_connectivity(public.connectivity)
        self.public_qubits = tuple(qubit.id for qubit in public.qubits)

    def run_circuit(
        self,
        request: CircuitRequest,
        job_id: str,
        salt: int,
    ) -> JobResult:
        error = validate_circuit_non_sweep_parameters(request)
        if error is not None:
            return self._failed_request_with_artifact(
                job_id,
                request.shots,
                error,
                _safe_measured_qubits(request.circuit),
                request.model_dump(mode="json"),
                circuit_count=1,
            )

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
            return self._failed_request_with_artifact(
                job_id,
                request.shots,
                error,
                _safe_measured_qubits(request.circuit),
                request.model_dump(mode="json"),
                circuit_count=1,
            )

        try:
            compiled = self._compile_circuit_request(request)
        except ValueError as exc:
            return self._failed_request_with_artifact(
                job_id,
                request.shots,
                str(exc),
                _safe_measured_qubits(request.circuit),
                request.model_dump(mode="json"),
                circuit_count=1,
            )
        return self._submit(compiled, request.shots, job_id)

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
            return self._failed_request_with_artifact(
                job_id,
                request.shots,
                error,
                _safe_measured_qubits(request.template_circuit),
                request.model_dump(mode="json"),
            )

        try:
            compiled = self._compile_circuit_sweep_request(request)
        except ValueError as exc:
            return self._failed_request_with_artifact(
                job_id,
                request.shots,
                str(exc),
                _safe_measured_qubits(request.template_circuit),
                request.model_dump(mode="json"),
            )
        return self._submit(compiled, request.shots, job_id)

    def _compile_circuit_request(self, request: CircuitRequest) -> _CompiledLiveRequest:
        return _CompiledLiveRequest(
            circuits=[self._wire_to_qiskit(request.circuit)],
            measured_qubits=self._measured_qubits(request.circuit),
            metadata=JobResultMetadata(),
            request_payload=request.model_dump(mode="json"),
        )

    def _compile_circuit_sweep_request(
        self,
        request: CircuitSweepRequest,
    ) -> _CompiledLiveRequest:
        keys, points = expand_circuit_sweep_points(request)
        return _CompiledLiveRequest(
            circuits=[
                self._wire_to_qiskit(request.template_circuit, bindings=point) for point in points
            ],
            measured_qubits=self._measured_qubits(request.template_circuit),
            metadata=JobResultMetadata(sweep_coords=sweep_coords_from_points(keys, points)),
            request_payload=request.model_dump(mode="json"),
        )

    def _wire_to_qiskit(
        self,
        circuit: list[CircuitOp],
        bindings: dict[str, float] | None = None,
    ) -> QuantumCircuit:
        n_qubits = _num_qubits(circuit)
        n_clbits = _num_clbits(circuit)
        if n_clbits == 0:
            raise ValueError("live provider circuits must include measurement")
        qc = QuantumCircuit(n_qubits, n_clbits)
        for op in circuit:
            if isinstance(op, CircuitMeasureOp):
                qc.measure(op.qubits, op.classical)
                continue
            self._apply_gate(qc, op, bindings)
        return qc

    def _apply_gate(
        self,
        qc: QuantumCircuit,
        op: GateOp,
        bindings: dict[str, float] | None,
    ) -> None:
        name = op.name
        qubits = op.qubits
        params = [_resolve_param(param, bindings) for param in op.params]
        if name == "i":
            qc.id(qubits[0])
        elif name in {"x", "y", "z", "h", "s", "sdg", "t", "tdg"}:
            getattr(qc, name)(qubits[0])
        elif name in {"rx", "ry", "rz"}:
            if len(params) != 1:
                raise ValueError(f"gate {name} requires exactly one parameter")
            getattr(qc, name)(params[0], qubits[0])
        elif name in {"cx", "cz", "swap"}:
            getattr(qc, name)(qubits[0], qubits[1])
        else:  # pragma: no cover - GateOp literal validation should prevent this.
            raise ValueError(f"unsupported gate {name!r}")

    def _measured_qubits(self, circuit: list[CircuitOp]) -> list[int]:
        measured = _safe_measured_qubits(circuit)
        if not measured:
            raise ValueError("live provider circuits must include measurement")
        return measured


def _bidirectional_connectivity(
    connectivity: Iterable[tuple[int, int]],
) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for edge in connectivity:
        a, b = int(edge[0]), int(edge[1])
        edges.add((a, b))
        edges.add((b, a))
    return edges


def _resolve_param(
    value: float | str,
    bindings: dict[str, float] | None,
) -> float:
    if isinstance(value, str):
        if bindings is None or value not in bindings:
            raise ValueError(f"unbound circuit parameter {value!r}")
        return float(bindings[value])
    return float(value)


def _safe_measured_qubits(circuit: list[CircuitOp]) -> list[int]:
    measured: list[int] | None = None
    for op in circuit:
        if isinstance(op, CircuitMeasureOp):
            measured = list(op.qubits)
    return measured or []


def _num_qubits(circuit: list[CircuitOp]) -> int:
    max_qubit = -1
    for op in circuit:
        max_qubit = max(max_qubit, *op.qubits)
    return max_qubit + 1


def _num_clbits(circuit: list[CircuitOp]) -> int:
    max_clbit = -1
    for op in circuit:
        if isinstance(op, CircuitMeasureOp):
            max_clbit = max(max_clbit, *op.classical)
    return max_clbit + 1


def build_digital_live_provider_backend(
    *,
    task_id: str,
    public: PublicDigitalSpec,
    backend_name: str,
    log_dir: str | Path | None = None,
    token: str | None = None,
    instance: str | None = None,
    channel: str,
    allow_live_provider: bool,
    max_shots: int | None = None,
    max_jobs: int = 1,
    backend_allowlist: Iterable[str] | None = None,
    wall_clock_s: float | None = None,
) -> DigitalLiveIBMBackend:
    """Construct a digital IBM live-provider backend from qtype-local data."""

    return DigitalLiveIBMBackend(
        task_id=task_id,
        public=public,
        backend_name=backend_name,
        log_dir=log_dir,
        token=token,
        instance=instance,
        channel=channel,
        allow_live_provider=allow_live_provider,
        max_shots=max_shots,
        max_jobs=max_jobs,
        backend_allowlist=backend_allowlist,
        wall_clock_s=wall_clock_s,
    )


__all__ = ["DigitalLiveIBMBackend", "build_digital_live_provider_backend"]
