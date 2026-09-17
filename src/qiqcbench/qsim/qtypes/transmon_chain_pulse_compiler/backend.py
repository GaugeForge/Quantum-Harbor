"""Chain pulse-compiler simulator backend adapter.

Wraps the qtype-local runner and fails closed on agent-visible limits (shots,
qubit range, linear connectivity, native gate set) before any simulation runs.
Exposes ``run_chain_circuit``; the qtype's MCP action calls it directly on
``state.backend`` (the compilation control surface is distinct from the
pulse/circuit registry slots, mirroring gmon).
"""

from __future__ import annotations

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.transmon_chain_pulse_compiler.device import (
    HiddenChainCompilerConfig,
    PublicChainCompilerSpec,
)
from qiqcbench.qsim.qtypes.transmon_chain_pulse_compiler.runner import run_chain_circuit
from qiqcbench.qsim.qtypes.transmon_chain_pulse_compiler.wire import ChainCircuitRequest


class ChainCompilerSimulatorBackend:
    """Chain pulse-compiler backend backed by the in-process numpy simulator."""

    def __init__(self, hidden: HiddenChainCompilerConfig) -> None:
        self.hidden = hidden
        self.n_qubits: int = hidden.n_qubits
        self.max_shots: int | None = None

    @classmethod
    def from_public(
        cls,
        hidden: HiddenChainCompilerConfig,
        public: PublicChainCompilerSpec,
    ) -> ChainCompilerSimulatorBackend:
        backend = cls(hidden)
        backend.n_qubits = public.n_qubits
        backend.max_shots = public.max_shots
        return backend

    def _failed(self, job_id: str, shots: int, error: str) -> JobResult:
        return JobResult(
            job_id=job_id,
            device_id=self.hidden.device_id,
            status="failed",
            shots=shots,
            error=error,
        )

    def _validate(self, request: ChainCircuitRequest) -> str | None:
        if self.max_shots is not None and request.shots > self.max_shots:
            return f"shots exceeds max_shots {self.max_shots}"
        n = self.n_qubits
        for g in request.circuit:
            for q in g.qubits:
                if q < 0 or q >= n:
                    return f"gate {g.type!r} qubit {q} out of range for n={n}"
            if g.type == "cphase":
                if len(g.qubits) != 2:
                    return "cphase requires exactly 2 qubits"
                if g.qubits[0] == g.qubits[1]:
                    return f"cphase needs two distinct qubits, got {g.qubits}"
                # v1: all-to-all controlled-phase (no linear-routing constraint).
            elif len(g.qubits) != 1:
                return f"{g.type!r} requires exactly 1 qubit"
            if g.virtual and g.type != "rz":
                return f"virtual=True is only valid for rz, not {g.type!r}"
        if request.measure_qubits is not None:
            for q in request.measure_qubits:
                if q < 0 or q >= n:
                    return f"measure qubit {q} out of range for n={n}"
        return None

    def run_chain_circuit(self, request: ChainCircuitRequest, job_id: str, salt: int) -> JobResult:
        if error := self._validate(request):
            return self._failed(job_id, request.shots, error)
        return run_chain_circuit(request, self.hidden, job_id, salt)


def build_chain_compiler_simulator_backend(
    hidden: HiddenChainCompilerConfig,
    public: PublicChainCompilerSpec,
) -> ChainCompilerSimulatorBackend:
    """Construct the in-process chain pulse-compiler simulator backend."""
    return ChainCompilerSimulatorBackend.from_public(hidden, public)


__all__ = [
    "ChainCompilerSimulatorBackend",
    "build_chain_compiler_simulator_backend",
]
