"""Loader for the public task materials served by the qsim backend.

Reads the frozen anonymized input circuit + public spec from
``configs/task_materials/<task_id>/public/`` (materialized offline by
``hidden_dynamics.<task_id>.construction``). No pyzx / hidden truth — the backend
only serves the agent-visible opaque circuit + objective.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from qiqcbench.qsim.task_materials import public_task_material_dir


def _materials_public_dir(task_id: str) -> Path:
    # Env override first (tests), else the canonical resolver (respects
    # $QIQCBENCH_CONFIGS inside the qsim/verifier containers).
    override = os.environ.get("QIQCBENCH_TASK_MATERIALS_DIR")
    if override:
        return Path(override) / task_id / "public"
    return public_task_material_dir(task_id)


@lru_cache(maxsize=4)
def load_public_instance(task_id: str) -> dict:
    """Return the served instance payload: public spec fields + the opaque circuit."""
    public = _materials_public_dir(task_id)
    spec = json.loads((public / "public_spec.json").read_text())
    circuit = json.loads(
        (public / spec.get("input_circuit_file", "input_circuit.json")).read_text()
    )
    return {"spec": spec, "circuit": circuit}
