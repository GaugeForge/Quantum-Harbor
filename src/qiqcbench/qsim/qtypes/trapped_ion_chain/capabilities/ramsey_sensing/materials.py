"""Public-material schema + device-level request validation for ramsey_sensing.

Per spec § Capability material: minimal schema with ONLY the fields the
capability runtime needs to validate / execute. Task budgets (N grid, T_tot,
shot caps that map to the answer schema) live in task material
(`configs/task_materials/<task_id>/public/ramsey_sensing_spec.yaml`), not here.

Validation rejects DEVICE-LEVEL violations: shots over cap, off-grid times,
out-of-range ion indices, missing reference_detuning_hz when phase_source ==
'reference'. It does NOT enforce task rules (e.g., N must be in {1,2,4,8,16}
is verifier territory).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from ruamel.yaml import YAML

from qiqcbench.qsim.core.wire import (
    RamseyExperimentRequest,
    RamseySweepRequest,
)

__all__ = [
    "RamseySensingPublicMaterials",
    "load_ramsey_sensing_materials",
    "validate_ramsey_experiment_request",
    "validate_ramsey_sweep_request",
]

_yaml = YAML(typ="safe")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RamseySensingPublicMaterials(_Strict):
    schema_version: Literal[1] = 1
    max_shots_per_run: int = Field(
        ...,
        gt=0,
        description="Maximum shots accepted for one Ramsey experiment or sweep point.",
    )
    free_evolution_resolution_us: float = Field(
        ...,
        gt=0,
        description="Allowed grid spacing for free-evolution times, in microseconds.",
    )


def load_ramsey_sensing_materials(public_dir: Path) -> RamseySensingPublicMaterials:
    """Load ``<public_dir>/ramsey_sensing_capability.yaml``.

    The task may supply additional task-specific YAML files alongside this;
    this loader only reads the device-runtime contract.
    """
    path = Path(public_dir) / "ramsey_sensing_capability.yaml"
    if not path.is_file():
        return RamseySensingPublicMaterials(
            max_shots_per_run=100_000,
            free_evolution_resolution_us=1.0,
        )
    return RamseySensingPublicMaterials.model_validate(_yaml.load(path.read_text(encoding="utf-8")))


def _validate_common(
    *,
    shots: int,
    measured_ions: list[int],
    phase_source: str,
    reference_detuning_hz: float | None,
    materials: RamseySensingPublicMaterials,
    n_ions_in_device: int,
) -> None:
    if shots > materials.max_shots_per_run:
        raise ValueError(f"shots={shots} exceeds max_shots_per_run={materials.max_shots_per_run}")
    if any(q < 0 or q >= n_ions_in_device for q in measured_ions):
        raise ValueError(
            f"measured_ions {measured_ions} out of device range [0, {n_ions_in_device})"
        )
    if phase_source == "reference" and reference_detuning_hz is None:
        raise ValueError("phase_source='reference' requires reference_detuning_hz to be set")


def _validate_time_on_grid(t_us: float, resolution_us: float) -> None:
    if t_us <= 0:
        raise ValueError(f"free_evolution_us must be > 0; got {t_us}")
    quotient = t_us / resolution_us
    if abs(quotient - round(quotient)) > 1e-9:
        raise ValueError(f"free_evolution_us={t_us} not on the {resolution_us} us grid")


def validate_ramsey_experiment_request(
    request: RamseyExperimentRequest,
    materials: RamseySensingPublicMaterials,
    n_ions_in_device: int,
) -> None:
    _validate_common(
        shots=request.shots,
        measured_ions=list(request.measured_ions),
        phase_source=request.phase_source,
        reference_detuning_hz=request.reference_detuning_hz,
        materials=materials,
        n_ions_in_device=n_ions_in_device,
    )
    _validate_time_on_grid(request.free_evolution_us, materials.free_evolution_resolution_us)


def validate_ramsey_sweep_request(
    request: RamseySweepRequest,
    materials: RamseySensingPublicMaterials,
    n_ions_in_device: int,
) -> None:
    _validate_common(
        shots=request.shots,
        measured_ions=list(request.measured_ions),
        phase_source=request.phase_source,
        reference_detuning_hz=request.reference_detuning_hz,
        materials=materials,
        n_ions_in_device=n_ions_in_device,
    )
    for t_us in request.free_evolution_us_values:
        _validate_time_on_grid(t_us, materials.free_evolution_resolution_us)
