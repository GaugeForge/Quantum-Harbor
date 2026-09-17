"""Loader for the public compilation-target netlist (committed task material).

Reads the committed ``netlist_seed_<seed>.json`` (gates + dependency DAG +
pinned budget reference depth) from the task's **public** materials. The JSON is
the single source of truth shared byte-for-byte between the qsim backend and the
Harbor verifier (which re-reads it without importing qiqcbench), so both agree on
the exact instance. The JSON is produced at materialization time.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


def _materials_public_dir(task_id: str) -> Path:
    """Resolve the public task-materials dir for ``task_id``.

    Resolution order (robust across the qsim sidecar, the separate verifier
    image, a standalone container, and a repo checkout):

    1. ``QIQCBENCH_TASK_MATERIALS_DIR`` env override;
    2. ``configs_root()/task_materials`` — honors ``QIQCBENCH_CONFIGS`` (set to
       ``/app/configs`` in the compose sidecar), the canonical config root;
    3. the in-container config locations ``/app/configs`` (bind mount / verifier
       symlink) and ``/app/qiqcbench_configs`` (baked);
    4. walk up from this module for ``configs/task_materials`` (repo checkout).
    """
    override = os.environ.get("QIQCBENCH_TASK_MATERIALS_DIR")
    if override:
        return Path(override) / task_id / "public"

    candidates: list[Path] = []
    try:  # lazy import: avoid any discovery-time cycle
        from qiqcbench.qsim.devices import configs_root

        candidates.append(configs_root() / "task_materials")
    except Exception:  # pragma: no cover - configs_root unavailable
        pass
    candidates.append(Path("/app/configs/task_materials"))
    candidates.append(Path("/app/qiqcbench_configs/task_materials"))
    here = Path(__file__).resolve()
    candidates.extend(parent / "configs" / "task_materials" for parent in here.parents)

    for root in candidates:
        if root.is_dir():
            return root / task_id / "public"
    raise FileNotFoundError(
        "Could not locate configs/task_materials; set QIQCBENCH_TASK_MATERIALS_DIR"
    )


def netlist_filename(seed: int) -> str:
    return f"netlist_seed_{int(seed)}.json"


@lru_cache(maxsize=16)
def load_target_netlist(task_id: str, seed: int = 0) -> dict[str, Any]:
    """Return the committed public netlist dict for ``task_id`` / ``seed``."""
    path = _materials_public_dir(task_id) / netlist_filename(seed)
    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


__all__ = ["load_target_netlist", "netlist_filename"]
