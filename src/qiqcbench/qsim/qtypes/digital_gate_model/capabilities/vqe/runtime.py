"""Bitstring-backed runtime for digital VQE observable batches.

Builds a brickwork-CZ-line ansatz from the public VQE materials, prepares each
parameter-point state once, reuses it across Pauli measurement settings, and
returns the raw per-shot bitstrings inside a ``JobObservableBitstringData``
envelope. Aggregation, estimation, and scoring are deferred: return
raw shot data, not aggregated counts).
"""

from __future__ import annotations

import time

from qiqcbench.qsim.core.wire import (
    CircuitOp,
    GateOp,
    JobObservableBitstringData,
    JobResult,
    JobResultMetadata,
    ObservableBatchRequest,
    ObservableSettingBitstrings,
)
from qiqcbench.qsim.qtypes.digital_gate_model.capabilities.vqe.materials import (
    VQETaskPublicMaterials,
    validate_observable_batch_request,
)
from qiqcbench.qsim.qtypes.digital_gate_model.capabilities.vqe.observables import (
    measurement_basis_ops,
)
from qiqcbench.qsim.qtypes.digital_gate_model.device import (
    HiddenDigitalConfig,
    HiddenGateErrors,
    HiddenReadoutQubit,
)
from qiqcbench.qsim.qtypes.digital_gate_model.engine import DigitalEngine
from qiqcbench.qsim.qtypes.digital_gate_model.runner import _make_rng

__all__ = ["run_observable_batch_request"]

_N_QUBITS = 10
_N_LAYERS = 4
_PARAMS_PER_LAYER = 2 * _N_QUBITS  # ry/rz pair per qubit.


def _effective_hidden_for_profile(
    hidden: HiddenDigitalConfig,
    materials: VQETaskPublicMaterials,
    profile: str,
) -> HiddenDigitalConfig:
    """Build the hidden config the engine should actually execute under.

    The ``ideal`` profile zeroes one-qubit/two-qubit/readout noise while
    preserving device_id/seed/stale notebook. The ``public_noise`` profile
    swaps in the public-noise model so a contestant can predict the noisy
    expectation without ever seeing hidden truth.
    """
    if profile == "ideal":
        return HiddenDigitalConfig(
            schema_version=hidden.schema_version,
            device_id=hidden.device_id,
            qtype=hidden.qtype,
            seed=hidden.seed,
            gate_errors=HiddenGateErrors(
                one_qubit_depolarizing=0.0,
                two_qubit_cx_depolarizing=0.0,
            ),
            readout=[
                HiddenReadoutQubit(id=r.id, p_0_to_1=0.0, p_1_to_0=0.0) for r in hidden.readout
            ],
            stale_lab_notebook=hidden.stale_lab_notebook,
        )
    if profile == "public_noise":
        public = materials.public_noise
        return HiddenDigitalConfig(
            schema_version=hidden.schema_version,
            device_id=hidden.device_id,
            qtype=hidden.qtype,
            seed=hidden.seed,
            gate_errors=HiddenGateErrors(
                one_qubit_depolarizing=public.one_qubit_depolarizing,
                two_qubit_cx_depolarizing=public.two_qubit_cz_depolarizing,
            ),
            readout=[
                HiddenReadoutQubit(id=r.id, p_0_to_1=r.p_0_to_1, p_1_to_0=r.p_1_to_0)
                for r in public.readout
            ],
            stale_lab_notebook=hidden.stale_lab_notebook,
        )
    raise ValueError(f"Unknown VQE profile: {profile!r}")


def _brickwork_pairs() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Two CZ sublayers of a 10-qubit line in brickwork order."""
    even = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    odd = [(1, 2), (3, 4), (5, 6), (7, 8)]
    return even, odd


def _ansatz_ops(values: list[float]) -> list[CircuitOp]:
    """Build the brickwork-CZ-line ansatz circuit ops for one parameter point.

    Convention ``layer_major_ry_then_rz``: within layer L, qubit-major
    ``ry``/``rz`` parameter pairs drive each qubit in order.
    CZ brickwork (even sublayer then odd sublayer) is inserted after layers
    0, 1, and 2; the final layer has no entangler after it.
    """
    if len(values) != _N_LAYERS * _PARAMS_PER_LAYER:
        raise ValueError(
            f"Expected {_N_LAYERS * _PARAMS_PER_LAYER} ansatz params, got {len(values)}"
        )
    even_pairs, odd_pairs = _brickwork_pairs()
    ops: list[CircuitOp] = []
    cursor = 0
    for layer in range(_N_LAYERS):
        for q in range(_N_QUBITS):
            ops.append(GateOp(name="ry", qubits=[q], params=[values[cursor]]))
            ops.append(GateOp(name="rz", qubits=[q], params=[values[cursor + 1]]))
            cursor += 2
        if layer < _N_LAYERS - 1:
            for a, b in even_pairs:
                ops.append(GateOp(name="cz", qubits=[a, b]))
            for a, b in odd_pairs:
                ops.append(GateOp(name="cz", qubits=[a, b]))
    return ops


def run_observable_batch_request(
    request: ObservableBatchRequest,
    hidden: HiddenDigitalConfig,
    materials: VQETaskPublicMaterials,
    job_id: str,
    salt: int,
    *,
    run_entropy: int | None = None,
) -> JobResult:
    """Execute every (point, Pauli-setting) cell and collect raw bitstrings.

    ``run_entropy`` is the backend's private per-attempt shot entropy (see
    ``DigitalSimulatorBackend``); ``None`` draws a fresh value for this call.
    """
    t_start = time.perf_counter()
    try:
        validate_observable_batch_request(request, materials)
        effective_hidden = _effective_hidden_for_profile(hidden, materials, request.profile)
    except ValueError as exc:
        return JobResult(
            job_id=job_id,
            device_id=hidden.device_id,
            status="failed",
            shots=request.shots_per_setting,
            error=str(exc),
        )

    pauli_terms = materials.hamiltonian.pauli_terms
    results: list[ObservableSettingBitstrings] = []
    measured = list(range(_N_QUBITS))

    rng = _make_rng(effective_hidden, salt, run_entropy)
    engine = DigitalEngine(effective_hidden, rng)

    for point in request.points:
        try:
            prepared = engine.prepare_state(_ansatz_ops(point.values))
        except ValueError as exc:
            return JobResult(
                job_id=job_id,
                device_id=hidden.device_id,
                status="failed",
                shots=request.shots_per_setting,
                error=f"point {point.point_id!r} ansatz failed: {exc}",
            )

        for setting_idx, term in enumerate(pauli_terms):
            try:
                bitstrings = engine.sample_prepared_state(
                    prepared,
                    basis_ops=measurement_basis_ops(term.pauli),
                    measured_qubits=measured,
                    shots=request.shots_per_setting,
                )
            except ValueError as exc:
                return JobResult(
                    job_id=job_id,
                    device_id=hidden.device_id,
                    status="failed",
                    shots=request.shots_per_setting,
                    error=(f"point {point.point_id!r} setting {setting_idx} failed: {exc}"),
                )
            results.append(
                ObservableSettingBitstrings(
                    point_id=point.point_id,
                    setting_id=f"s{setting_idx}",
                    pauli=term.pauli,
                    coefficient=term.coefficient,
                    measured_qubits=measured,
                    bitstrings=bitstrings,
                )
            )

    return JobResult(
        job_id=job_id,
        device_id=hidden.device_id,
        status="complete",
        shots=request.shots_per_setting,
        data=JobObservableBitstringData(results=results),
        metadata=JobResultMetadata(
            wallclock_ms=int((time.perf_counter() - t_start) * 1000),
        ),
    )
