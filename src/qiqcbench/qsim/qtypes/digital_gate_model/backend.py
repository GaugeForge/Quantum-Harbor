"""Gate-model digital simulator backend adapter.

Wraps the qtype-local runner and enforces the shared digital-policy gate
before delegating, so policy violations surface as a ``failed``
:class:`JobResult` rather than as engine-level state. The top-level
``qiqcbench.qsim.backends.simulator`` re-exports this class so existing
callers see no API churn.
"""

from __future__ import annotations

from qiqcbench.qsim.core.wire import (
    CircuitOp,
    CircuitRequest,
    CircuitSweepRequest,
    JobResult,
)
from qiqcbench.qsim.qtypes.digital_gate_model.device import (
    HiddenDigitalConfig,
    PublicDigitalSpec,
)
from qiqcbench.qsim.qtypes.digital_gate_model.policy import validate_digital_circuit
from qiqcbench.qsim.qtypes.digital_gate_model.runner import (
    fresh_run_entropy,
    run_circuit_sequence,
    run_circuit_sweep_request,
)


class DigitalSimulatorBackend:
    """Gate-model backend backed by the in-process digital simulator.

    The constructor captures the hidden config plus the *public* connectivity
    edges. Connectivity is normalized into an undirected edge set so that
    ``cx`` on either ``(a, b)`` or ``(b, a)`` of a declared link is accepted
    while any other two-qubit pair is rejected. Identity-order classical
    mapping is enforced because the wire schema does not yet promise faithful
    delivery of arbitrary classical-register permutations to downstream
    consumers; surfacing the mismatch keeps the contract honest.

    ``run_entropy`` is the private per-attempt shot entropy. It is drawn once
    here (once per qsim container) and combined with ``hidden.seed`` and the
    per-call salt, so two attempts against the same committed hidden config
    observe independent shot noise while one attempt stays deterministic for
    a given salt. Pass it explicitly only from regression tests; it must never
    be logged or exposed through any agent- or verifier-visible surface.
    """

    def __init__(
        self,
        hidden: HiddenDigitalConfig,
        connectivity: list[tuple[int, int]],
        max_shots: int | None = None,
        public_qubits: list[int] | None = None,
        *,
        run_entropy: int | None = None,
    ) -> None:
        self.hidden = hidden
        if run_entropy is None:
            run_entropy = fresh_run_entropy()
        if isinstance(run_entropy, bool) or not isinstance(run_entropy, int) or run_entropy < 0:
            raise ValueError("run_entropy must be a non-negative integer")
        self._run_entropy = run_entropy
        self.connectivity: set[tuple[int, int]] = set()
        for edge in connectivity:
            a, b = int(edge[0]), int(edge[1])
            self.connectivity.add((a, b))
            self.connectivity.add((b, a))
        self.max_shots = max_shots
        self.public_qubits = None if public_qubits is None else tuple(public_qubits)

    @property
    def run_entropy(self) -> int:
        """Private per-attempt shot entropy; qsim-internal, never log or expose it."""
        return self._run_entropy

    def _validate_circuit(
        self,
        circuit: list[CircuitOp],
        shots: int,
        *,
        allowed_parameters: list[str],
    ) -> str | None:
        return validate_digital_circuit(
            circuit,
            self.connectivity,
            shots,
            max_shots=self.max_shots,
            public_qubits=self.public_qubits,
            require_measurement=True,
            allowed_parameters=allowed_parameters,
        )

    def _failed(self, job_id: str, shots: int, error: str) -> JobResult:
        return JobResult(
            job_id=job_id,
            device_id=self.hidden.device_id,
            status="failed",
            shots=shots,
            error=error,
        )

    def run_circuit(self, request: CircuitRequest, job_id: str, salt: int) -> JobResult:
        if error := self._validate_circuit(
            request.circuit,
            request.shots,
            allowed_parameters=[],
        ):
            return self._failed(job_id, request.shots, error)
        return run_circuit_sequence(
            request, self.hidden, job_id, salt, run_entropy=self._run_entropy
        )

    def run_circuit_sweep(self, request: CircuitSweepRequest, job_id: str, salt: int) -> JobResult:
        if error := self._validate_circuit(
            request.template_circuit,
            request.shots,
            allowed_parameters=request.parameters,
        ):
            return self._failed(job_id, request.shots, error)
        return run_circuit_sweep_request(
            request, self.hidden, job_id, salt, run_entropy=self._run_entropy
        )


def build_digital_simulator_backend(
    hidden: HiddenDigitalConfig,
    public: PublicDigitalSpec,
) -> DigitalSimulatorBackend:
    """Construct the in-process digital simulator backend from public + hidden config."""
    return DigitalSimulatorBackend(
        hidden,
        public.connectivity,
        public.max_shots,
        public_qubits=[qubit.id for qubit in public.qubits],
    )


__all__ = ["DigitalSimulatorBackend", "build_digital_simulator_backend"]
