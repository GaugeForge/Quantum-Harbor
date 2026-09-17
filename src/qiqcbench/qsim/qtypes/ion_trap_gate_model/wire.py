"""Qtype-local wire schemas for ion_trap_gate_model.

Reuses ``CircuitOp``/``GateOp`` from ``core/wire.py`` for the native-gate
program (rx/ry/rz/ms/...); defines only the request + result-data payloads
specific to this qtype. The result-data classes are listed in the qtype
``DESCRIPTOR.result_data_models`` so the dynamic ``JobData`` union includes them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from qiqcbench.qsim.core.wire import SCHEMA_VERSION, CircuitOp


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _StrictRequest(_Strict):
    """_Strict with NaN/Infinity refused on every float field."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


# ---------- run_ion_circuit ----------


class IonCircuitRequest(_StrictRequest):
    """Run one native-gate circuit and read out the requested qubits.

    ``circuit`` is a list of gate ops (no measure ops; measurement is on
    ``measured_qubits``). Noise is applied per native gate (1q depolarizing,
    MS depolarizing) and per-qubit asymmetric readout confusion.
    """

    schema_version: int = SCHEMA_VERSION
    shots: int = Field(..., gt=0, le=100_000)
    circuit: list[CircuitOp]
    measured_qubits: list[int] = Field(..., min_length=1)


# ---------- run_randomized_measurement_batch ----------


class RandomizedMeasurementBatchRequest(_StrictRequest):
    """Estimate the second Renyi entropy of a bulk subsystem via randomized
    measurements (Brydges et al. protocol).

    The engine prepares the noisy state from ``prepare_circuit``, then for each
    of ``n_unitaries`` settings draws a fresh CUE-random single-qubit unitary on
    every qubit in ``subsystem_qubits`` (derived deterministically from
    ``randomization_seed``, independent of the hidden noise RNG), and returns
    ``shots_per_unitary`` raw subsystem bitstrings per setting. Noise is drawn
    fresh per shot (no trajectory reuse). One batch corresponds to one cylinder
    circumference; call once per L_y.
    """

    schema_version: int = SCHEMA_VERSION
    prepare_circuit: list[CircuitOp]
    n_unitaries: int = Field(..., ge=1, le=1024)
    shots_per_unitary: int = Field(..., ge=1, le=10_000)
    subsystem_qubits: list[int] = Field(..., min_length=1)
    randomization_seed: int


class ExperimentConditionTag(_StrictRequest):
    """Optional task-owned pre-execution condition label.

    The qtype records and echoes the label without interpreting task semantics.
    A task verifier may require tagged production jobs while leaving untagged
    exploration available.
    """

    schema_version: Literal[1] = 1
    task_id: str = Field(..., min_length=1, max_length=128)
    condition_name: str = Field(..., min_length=1, max_length=64)
    condition_value: float = Field(..., allow_inf_nan=False)


class RandomizedMeasurementBatchRequestV2(_StrictRequest):
    """Run local-Haar randomized measurements with qsim-owned fresh bases."""

    schema_version: int = SCHEMA_VERSION
    protocol_version: Literal[2] = 2
    prepare_circuit: list[CircuitOp]
    n_unitaries: int = Field(..., ge=1, le=1024)
    shots_per_unitary: int = Field(..., ge=1, le=10_000)
    subsystem_qubits: list[int] = Field(..., min_length=1)
    experiment_tag: ExperimentConditionTag | None = None


# ---------- Result data ----------


class JobIonBitstringData(_Strict):
    """Raw per-shot bitstrings from a single native-gate circuit run.

    ``bitstrings[shot]`` has length ``len(measured_qubits)`` with
    ``bitstring[i]`` the value of ``measured_qubits[i]`` (qubit-0-LSB readout
    order = measured order).
    """

    kind: Literal["ion_bitstring"] = "ion_bitstring"
    bitstrings: list[str]
    measured_qubits: list[int]


class RmSettingBitstrings(_Strict):
    """Raw subsystem bitstrings for one randomized-measurement setting."""

    setting_index: int
    bitstrings: list[str]


class JobRandomizedMeasurementData(_Strict):
    """Per-setting raw subsystem bitstrings from a randomized-measurement batch.

    The applied CUE-random bases are a deterministic function of
    ``randomization_seed`` (and the setting index + subsystem) so the verifier
    can reproduce them; the Renyi-2 purity U-statistic itself only needs the
    bitstrings, not the bases.
    """

    kind: Literal["randomized_measurement"] = "randomized_measurement"
    settings: list[RmSettingBitstrings]
    subsystem_qubits: list[int]
    randomization_seed: int
    n_unitaries: int
    shots_per_unitary: int


class JobRandomizedMeasurementDataV2(_Strict):
    """Raw post-readout shots from qsim-private fresh random local bases."""

    kind: Literal["randomized_measurement_v2"] = "randomized_measurement_v2"
    protocol_version: Literal[2] = 2
    settings: list[RmSettingBitstrings]
    subsystem_qubits: list[int]
    experiment_tag: ExperimentConditionTag | None = None
    randomness_policy: Literal["fresh_private_per_job"] = "fresh_private_per_job"
    n_unitaries: int
    shots_per_unitary: int


__all__ = [
    "ExperimentConditionTag",
    "IonCircuitRequest",
    "JobIonBitstringData",
    "JobRandomizedMeasurementData",
    "JobRandomizedMeasurementDataV2",
    "RandomizedMeasurementBatchRequest",
    "RandomizedMeasurementBatchRequestV2",
    "RmSettingBitstrings",
]
