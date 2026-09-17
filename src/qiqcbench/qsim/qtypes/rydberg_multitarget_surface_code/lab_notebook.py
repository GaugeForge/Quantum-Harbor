"""Agent-visible stale calibration notebook for native CZ2 memory experiments."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.rydberg_multitarget_surface_code.device import (
    HiddenRydbergMultitargetSurfaceCodeConfig,
)


class RydbergMultitargetLabNotebook(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["rydberg_multitarget_surface_code"] = "rydberg_multitarget_surface_code"
    device_id: str
    claimed_cz2_fidelity: str
    duration_advice: str
    phase_advice: str
    partition_advice: str
    note: str
    # Provenance only. Saying "stale", or naming what to characterize, would tell the agent
    # the answer to a question it must settle from evidence -- and "compare legal partitions
    # from data" is the task itself. State when the values were accepted; nothing more.
    reliability: str = "values accepted at the last scheduled calibration"


def build_rydberg_multitarget_lab_notebook(
    hidden: HiddenRydbergMultitargetSurfaceCodeConfig,
) -> RydbergMultitargetLabNotebook:
    nb = hidden.stale_lab_notebook
    return RydbergMultitargetLabNotebook(
        device_id=hidden.device_id,
        claimed_cz2_fidelity=nb.claimed_cz2_fidelity,
        duration_advice=nb.duration_advice,
        phase_advice=nb.phase_advice,
        partition_advice=nb.partition_advice,
        note=nb.note
        or "The previous calibration saw no target-pair dependence. Re-characterize before use.",
    )
