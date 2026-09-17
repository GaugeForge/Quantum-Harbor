from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ruamel.yaml import YAML

from qiqcbench.qsim.capabilities import CapabilityFamily
from qiqcbench.qsim.devices import configs_root

BackendMode = Literal["simulator", "provider_replay", "live_provider"]
AgentSurface = Literal["mcp", "qiskit", "qcodes"]
RatingMode = Literal["rubric", "elo"]

_yaml = YAML(typ="safe")


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    device_id: str
    description: str
    wall_clock_budget_s: int
    default_backend_mode: BackendMode | None
    allowed_backend_modes: tuple[BackendMode, ...]
    default_surfaces: tuple[AgentSurface, ...]
    allowed_surfaces: tuple[AgentSurface, ...]
    enabled_capabilities: tuple[CapabilityFamily, ...] = ("basic_measurement",)
    rating_mode: RatingMode = "rubric"


@dataclass(frozen=True)
class TaskContext:
    task_spec: TaskSpec | None
    device_id: str


def task_spec_path(task_id: str, *, root: Path | None = None) -> Path:
    """The config path :func:`load_task_spec` reads for ``task_id``.

    Exposed so a caller that must tell an *absent* declaration from one it
    could not load can check for the entry itself instead of inferring absence
    from which exception the loader raised. Sharing this one definition keeps
    the checked path and the read path the same path.
    """
    cfg_root = configs_root() if root is None else root
    return cfg_root / "tasks" / f"{task_id}.yaml"


def load_task_spec(task_id: str, *, root: Path | None = None) -> TaskSpec:
    path = task_spec_path(task_id, root=root)
    with path.open(encoding="utf-8") as f:
        data = _yaml.load(f)
    schema_version = data.get("schema_version")
    if schema_version != 1:
        raise ValueError(f"Unsupported task schema_version {schema_version!r}")
    rating_mode = data.get("rating_mode", "rubric")
    if rating_mode not in ("rubric", "elo"):
        raise ValueError(f"Unsupported rating_mode {rating_mode!r}")
    return TaskSpec(
        task_id=data["task_id"],
        device_id=data["device_id"],
        description=data["description"],
        wall_clock_budget_s=int(data["wall_clock_budget_s"]),
        default_backend_mode=data.get("default_backend_mode"),
        allowed_backend_modes=tuple(data.get("allowed_backend_modes", [])),
        default_surfaces=tuple(data.get("default_surfaces", ["mcp"])),
        allowed_surfaces=tuple(data.get("allowed_surfaces", ["mcp"])),
        enabled_capabilities=tuple(data.get("enabled_capabilities", ["basic_measurement"])),
        rating_mode=rating_mode,
    )


def resolve_task_context(task_id: str | None, device_id: str | None) -> TaskContext:
    if task_id is not None:
        spec = load_task_spec(task_id)
        return TaskContext(task_spec=spec, device_id=spec.device_id)
    if device_id is None:
        raise ValueError("Either task_id or device_id is required")
    return TaskContext(task_spec=None, device_id=device_id)
