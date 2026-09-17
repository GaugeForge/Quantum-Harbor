"""Public material models + loader for the digital VQE capability family.

Loads the public Hamiltonian, ansatz spec, and public noise model from a
task-material ``public/`` directory. Hidden scorer materials are intentionally
out of scope here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from ruamel.yaml import YAML

from qiqcbench.qsim.core.wire import ObservableBatchRequest

__all__ = [
    "AnsatzExecutionSemantics",
    "AnsatzSpec",
    "EvaluationBudget",
    "HamiltonianSpec",
    "PauliTerm",
    "PublicNoiseModel",
    "PublicNoiseReadout",
    "VQETaskPublicMaterials",
    "load_vqe_public_materials",
    "validate_observable_batch_request",
]

_SUPPORTED_PROFILES: frozenset[str] = frozenset({"ideal", "public_noise"})

_PAULI_LENGTH = 10
_PAULI_ALPHABET = frozenset("IXYZ")
_REQUIRED_FILES = ("hamiltonian.json", "ansatz_spec.yaml", "public_noise_model.json")

_yaml = YAML(typ="safe")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PauliTerm(_Strict):
    pauli: str = Field(..., min_length=_PAULI_LENGTH, max_length=_PAULI_LENGTH)
    coefficient: float

    @field_validator("pauli")
    @classmethod
    def _validate_pauli_characters(cls, value: str) -> str:
        invalid = sorted(set(value) - _PAULI_ALPHABET)
        if invalid:
            raise ValueError(
                f"Pauli string contains unsupported character(s): {', '.join(invalid)}"
            )
        return value


class HamiltonianSpec(_Strict):
    schema_version: Literal[2]
    parameter_convention: Literal["layer_major_ry_then_rz"]
    pauli_terms: list[PauliTerm] = Field(..., min_length=1)


class AnsatzExecutionSemantics(_Strict):
    """Public execution contract for the fixed noisy ansatz circuit."""

    schema_version: Literal[1]
    preserve_all_rotation_slots: Literal[True]
    preserve_all_entangler_slots: Literal[True]
    rotation_gate_count: Literal[80]
    cz_gate_count: Literal[27]


class EvaluationBudget(_Strict):
    """Public, run-long observable-batch admission limits."""

    schema_version: Literal[2]
    ideal_max_submitted_parameter_point_evaluations: int = Field(..., gt=0)
    public_noise_max_submitted_parameter_point_evaluations: int = Field(..., gt=0)
    public_noise_required_shots_per_setting: int = Field(..., gt=1, le=100_000)
    max_parameter_points_per_call: int = Field(..., gt=0)
    max_raw_shot_records_per_call: int = Field(..., gt=0)
    max_raw_shot_records_total: int = Field(..., gt=0)
    max_answer_string_characters: int = Field(..., gt=0, le=16_384)
    max_citation_job_id_characters: int = Field(..., gt=0, le=128)
    max_final_answer_serialized_bytes: int = Field(..., gt=0, le=1_048_576)
    max_final_answer_submissions: int = Field(..., gt=0, le=64)
    max_job_result_polls: int = Field(..., ge=4, le=100_000)
    max_ingress_request_bytes: int = Field(..., gt=0, le=16_777_216)


class AnsatzSpec(_Strict):
    schema_version: Literal[2]
    parameter_convention: Literal["layer_major_ry_then_rz"]
    n_qubits: Literal[10]
    rotation_layers: Literal[4]
    entangler: Literal["brickwork_cz_line"]
    parameter_count: Literal[80]
    initialization_label: Literal["public_graph_state_warm_start"]
    initial_parameters: list[float] = Field(..., min_length=80, max_length=80)
    execution_semantics: AnsatzExecutionSemantics
    evaluation_budget: EvaluationBudget


class PublicNoiseReadout(_Strict):
    id: int = Field(..., ge=0, le=_PAULI_LENGTH - 1)
    p_0_to_1: float = Field(..., ge=0, le=1)
    p_1_to_0: float = Field(..., ge=0, le=1)


class PublicNoiseModel(_Strict):
    schema_version: Literal[2]
    depolarizing_parameter_convention: Literal["total_nonidentity_pauli_probability"]
    one_qubit_depolarizing: float = Field(..., ge=0, le=1)
    two_qubit_cz_depolarizing: float = Field(..., ge=0, le=1)
    readout: list[PublicNoiseReadout] = Field(
        ..., min_length=_PAULI_LENGTH, max_length=_PAULI_LENGTH
    )

    @model_validator(mode="after")
    def _validate_readout_ids(self) -> PublicNoiseModel:
        ids = [entry.id for entry in self.readout]
        expected_ids = set(range(_PAULI_LENGTH))
        if len(set(ids)) != len(ids) or set(ids) != expected_ids:
            raise ValueError("Readout entries must contain each id from 0 through 9 once")
        return self


class VQETaskPublicMaterials(_Strict):
    hamiltonian: HamiltonianSpec
    ansatz: AnsatzSpec
    public_noise: PublicNoiseModel

    @model_validator(mode="after")
    def _validate_evaluation_budget_feasibility(self) -> VQETaskPublicMaterials:
        budget = self.ansatz.evaluation_budget
        if budget.max_raw_shot_records_per_call > budget.max_raw_shot_records_total:
            raise ValueError("per-call raw-shot-record cap cannot exceed the total cap")

        term_count = len(self.hamiltonian.pauli_terms)
        one_noisy_point = term_count * budget.public_noise_required_shots_per_setting
        if budget.max_raw_shot_records_per_call < one_noisy_point:
            raise ValueError(
                "per-call raw-shot-record cap cannot fit one required public_noise point"
            )
        minimum_endpoint_records = one_noisy_point + term_count
        if budget.max_raw_shot_records_total < minimum_endpoint_records:
            raise ValueError(
                "total raw-shot-record cap cannot fit the two required endpoint points"
            )
        return self


def load_vqe_public_materials(public_dir: Path) -> VQETaskPublicMaterials:
    """Load and validate the public VQE materials from ``public_dir``.

    Fails closed with ``FileNotFoundError`` if any required file is absent,
    naming every missing filename in the message.
    """
    missing = [name for name in _REQUIRED_FILES if not (public_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing required VQE public material file(s) in {public_dir}: {', '.join(missing)}"
        )

    hamiltonian_data = json.loads((public_dir / "hamiltonian.json").read_text())
    with (public_dir / "ansatz_spec.yaml").open(encoding="utf-8") as f:
        ansatz_data = _yaml.load(f)
    noise_data = json.loads((public_dir / "public_noise_model.json").read_text())

    return VQETaskPublicMaterials(
        hamiltonian=HamiltonianSpec.model_validate(hamiltonian_data),
        ansatz=AnsatzSpec.model_validate(ansatz_data),
        public_noise=PublicNoiseModel.model_validate(noise_data),
    )


def validate_observable_batch_request(
    request: ObservableBatchRequest,
    materials: VQETaskPublicMaterials,
) -> None:
    """Reject VQE observable-batch requests that disagree with public materials.

    Fail-closed on:
      * unsupported ``profile`` (must be one of ``ideal`` / ``public_noise``);
      * ``parameter_convention`` that does not match the public ansatz spec;
      * any point whose ``values`` length is not ``ansatz.parameter_count``;
      * a request containing more parameter points than either the public
        per-call limit or the profile's entire run-long budget;
      * a request whose expanded ``points * Pauli terms * shots`` raw-record
        count exceeds the public per-call cap;
      * a ``public_noise`` request whose shots per setting are not the exact
        public value.

    Raises ``ValueError`` with a human-readable message; callers in the
    runtime translate this into a failed ``JobResult``.
    """
    if request.profile not in _SUPPORTED_PROFILES:
        supported = ", ".join(sorted(_SUPPORTED_PROFILES))
        raise ValueError(
            f"Unsupported profile {request.profile!r}; supported profiles: {supported}"
        )

    expected_convention = materials.ansatz.parameter_convention
    if request.parameter_convention != expected_convention:
        raise ValueError(
            f"parameter_convention {request.parameter_convention!r} does not match "
            f"public ansatz parameter_convention {expected_convention!r}"
        )

    budget = materials.ansatz.evaluation_budget
    if request.profile == "ideal":
        point_limit = budget.ideal_max_submitted_parameter_point_evaluations
    else:
        point_limit = budget.public_noise_max_submitted_parameter_point_evaluations
        required_shots = budget.public_noise_required_shots_per_setting
        if request.shots_per_setting != required_shots:
            raise ValueError(
                "public_noise shots_per_setting must equal the public required value "
                f"{required_shots}; got {request.shots_per_setting}"
            )
    per_call_point_limit = min(point_limit, budget.max_parameter_points_per_call)
    if len(request.points) > per_call_point_limit:
        raise ValueError(
            f"{request.profile} request contains {len(request.points)} parameter-point "
            f"evaluations; per-call limit is {per_call_point_limit}"
        )
    raw_shot_records = (
        len(request.points) * len(materials.hamiltonian.pauli_terms) * request.shots_per_setting
    )
    if raw_shot_records > budget.max_raw_shot_records_per_call:
        raise ValueError(
            f"observable-batch request expands to {raw_shot_records} raw shot records; "
            f"per-call limit is {budget.max_raw_shot_records_per_call}"
        )

    expected_count = materials.ansatz.parameter_count
    for point in request.points:
        if len(point.values) != expected_count:
            raise ValueError(
                f"point {point.point_id!r} has {len(point.values)} values; "
                f"expected ansatz parameter_count {expected_count}"
            )
