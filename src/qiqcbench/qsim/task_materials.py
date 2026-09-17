"""Task-material directory helpers.

Task materials (Hamiltonians, ansatz specs, public noise models, hidden
scorers, etc.) live under ``configs/task_materials/<task_id>/{public,hidden}``.
The split mirrors the public/hidden device-config split: public
materials are visible to the agent; hidden materials remain inside qsim.
"""

from __future__ import annotations

from pathlib import Path

from qiqcbench.qsim.devices import configs_root

__all__ = [
    "hidden_task_material_dir",
    "public_task_material_dir",
    "task_materials_root",
]


def task_materials_root(root: Path | None = None) -> Path:
    """Resolve the task-materials root directory.

    Returns ``root`` if supplied, otherwise ``<repo>/configs/task_materials``.
    """
    if root is not None:
        return root
    return configs_root() / "task_materials"


def public_task_material_dir(task_id: str, *, root: Path | None = None) -> Path:
    return task_materials_root(root) / task_id / "public"


def hidden_task_material_dir(task_id: str, *, root: Path | None = None) -> Path:
    return task_materials_root(root) / task_id / "hidden"
