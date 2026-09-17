"""Load and validate the exact public candidate bank executed by qsim."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from qiqcbench.qsim.qtypes.scheduled_transmon_gate_model.bank import (
    CandidateBank,
    load_candidate_bank,
)
from qiqcbench.qsim.task_materials import public_task_material_dir


def _public_dir(task_id: str) -> Path:
    override = os.environ.get("QIQCBENCH_TASK_MATERIALS_DIR")
    if override:
        return Path(override) / task_id / "public"
    return public_task_material_dir(task_id)


@lru_cache(maxsize=8)
def load_public_candidate_bank(
    task_id: str,
    filename: str,
    expected_digest: str,
) -> CandidateBank:
    root = _public_dir(task_id).resolve()
    source = (root / filename).resolve()
    if source.parent != root:
        raise ValueError("candidate bank path escapes the public task-material directory")
    return load_candidate_bank(source, expected_digest=expected_digest)


__all__ = ["load_public_candidate_bank"]
