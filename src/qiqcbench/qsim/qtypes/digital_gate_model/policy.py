"""Shared validation policy and sweep helpers for digital circuit backends."""

from __future__ import annotations

import math
from collections.abc import Iterable

from qiqcbench.qsim.core.wire import (
    CircuitMeasureOp,
    CircuitOp,
    CircuitRequest,
    CircuitSweepRequest,
    GateOp,
)
from qiqcbench.qsim.sweep import expand_sweep_points

_ONE_QUBIT_GATES = {"i", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "rx", "ry", "rz"}
_TWO_QUBIT_GATES = {"cx", "cz", "swap"}
_DIGITAL_GATES = _ONE_QUBIT_GATES | _TWO_QUBIT_GATES
_PARAM_COUNTS = {"rx": 1, "ry": 1, "rz": 1}


def _normalized_connectivity(
    connectivity: Iterable[tuple[int, int]],
) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for edge in connectivity:
        a, b = int(edge[0]), int(edge[1])
        edges.add((a, b))
        edges.add((b, a))
    return edges


def validate_digital_circuit(
    circuit: list[CircuitOp],
    connectivity: Iterable[tuple[int, int]],
    shots: int,
    max_shots: int | None = None,
    public_qubits: Iterable[int] | None = None,
    require_measurement: bool = False,
    allowed_parameters: Iterable[str] | None = None,
) -> str | None:
    """Return a shared digital-policy error, or ``None`` when valid."""

    if max_shots is not None and shots > max_shots:
        return f"Requested shots {shots} exceeds max_shots {max_shots}"

    public_qubit_set = None
    if public_qubits is not None:
        public_qubit_set = {int(qubit) for qubit in public_qubits}

    edge_set = _normalized_connectivity(connectivity)
    saw_measurement = False
    allowed_parameter_set = None
    if allowed_parameters is not None:
        allowed_parameter_set = set(allowed_parameters)

    for op in circuit:
        if saw_measurement and isinstance(op, GateOp):
            return "digital circuit measurement must be terminal"
        if public_qubit_set is not None:
            invalid_qubits = [qubit for qubit in op.qubits if qubit not in public_qubit_set]
            if invalid_qubits:
                return (
                    f"qubit(s) {invalid_qubits} not in public device qubits "
                    f"{sorted(public_qubit_set)}"
                )
        if isinstance(op, CircuitMeasureOp):
            saw_measurement = True
            expected = list(range(len(op.qubits)))
            if op.classical != expected:
                return (
                    "CircuitMeasureOp.classical currently supports only "
                    f"identity-order mapping {expected}; got {op.classical}"
                )
        if isinstance(op, GateOp):
            if op.name not in _DIGITAL_GATES:
                return (
                    f"Gate {op.name!r} is not supported by digital_gate_model "
                    f"(allowed: {sorted(_DIGITAL_GATES)})"
                )
            if op.name in _ONE_QUBIT_GATES and len(op.qubits) != 1:
                return f"gate {op.name} requires exactly 1 qubit; got {len(op.qubits)}"
            if op.name in _TWO_QUBIT_GATES and len(op.qubits) != 2:
                return f"gate {op.name} requires exactly 2 qubits; got {len(op.qubits)}"
            expected_params = _PARAM_COUNTS.get(op.name, 0)
            if len(op.params) != expected_params:
                return (
                    f"gate {op.name} requires exactly {expected_params} "
                    f"parameter(s); got {len(op.params)}"
                )
            if any(
                not isinstance(param, str) and not math.isfinite(float(param))
                for param in op.params
            ):
                return f"gate {op.name} parameters must be finite"
            if allowed_parameter_set is not None:
                for param in op.params:
                    if isinstance(param, str) and param not in allowed_parameter_set:
                        return f"unbound circuit parameter {param!r}"
            if len(op.qubits) == 2:
                edge = (op.qubits[0], op.qubits[1])
                if edge not in edge_set:
                    return (
                        f"Two-qubit gate {op.name} on qubits {op.qubits} "
                        "violates public connectivity"
                    )
    if require_measurement and not saw_measurement:
        return "digital circuits must include measurement"
    return None


NON_SWEEP_CIRCUIT_PARAMETERS_ERROR = (
    "CircuitRequest.parameters must be empty for non-sweep calls; "
    "use run_circuit_sweep for parametric circuits."
)


def validate_circuit_non_sweep_parameters(request: CircuitRequest) -> str | None:
    if not request.parameters:
        return None
    return NON_SWEEP_CIRCUIT_PARAMETERS_ERROR


def expand_circuit_sweep_points(
    request: CircuitSweepRequest,
) -> tuple[list[str], list[dict[str, float]]]:
    declared = set(request.parameters)
    sweep_keys = set(request.sweep.keys())
    if declared != sweep_keys:
        raise ValueError(
            f"sweep keys {sorted(sweep_keys)} must equal declared parameters {sorted(declared)}"
        )
    if any(
        not math.isfinite(float(value)) for values in request.sweep.values() for value in values
    ):
        raise ValueError("digital circuit sweep values must be finite")

    keys = request.parameters
    points = expand_sweep_points(keys, request.sweep, request.mode)
    if not points:
        raise ValueError("digital circuit sweeps must include at least one point")
    return keys, points


def sweep_coords_from_points(
    keys: list[str],
    points: list[dict[str, float]],
) -> dict[str, list[float]]:
    coords: dict[str, list[float]] = {k: [] for k in keys}
    for binding in points:
        for k in keys:
            coords[k].append(float(binding[k]))
    return coords
