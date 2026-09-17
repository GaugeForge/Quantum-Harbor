"""Lab-notebook model and builder for the gmon-ring qtype.

Surfaces only the *stale* calibration claims to the agent: a drifted coupler
hopping ``g0_mhz`` and an optimistic *symmetric* readout-fidelity claim. The true
g0, the per-qubit *asymmetric* readout, T1/T2, and the RNG seed stay hidden.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.gmon_ring_3q.device import HiddenGmonRingConfig


class GmonRingLabNotebook(BaseModel):
    """Stale lab-notebook entry for the gmon-ring qtype.

    Contents are intentionally partial and possibly out of date. The agent must
    verify before relying on these values (Stage A recovers the true g0).
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["gmon_ring_3q"] = "gmon_ring_3q"
    device_id: str
    last_updated: str = "stale"
    g0_mhz_estimate: float | None = None
    readout_fidelity_claim: float | None = None
    notes: str = ""


def build_gmon_ring_3q_lab_notebook(hidden: HiddenGmonRingConfig) -> GmonRingLabNotebook:
    """Build the gmon-ring lab-notebook view from a hidden config.

    The recorded g0 is a stale claim (drifted from the true simulator value), and
    the readout fidelity is an optimistic symmetric figure — the real per-qubit
    readout is asymmetric.
    """
    nb = hidden.stale_lab_notebook
    return GmonRingLabNotebook(
        device_id=hidden.device_id,
        last_updated=nb.last_calibrated or "stale",
        g0_mhz_estimate=nb.g0_mhz_claim,
        readout_fidelity_claim=nb.readout_fidelity_claim,
        notes=(
            nb.notes
            or (
                "Coupler g0 was last calibrated ~3 weeks ago and may have drifted; "
                "verify it experimentally before relying on it. The readout fidelity "
                "is an optimistic symmetric estimate — per-qubit readout is asymmetric, "
                "so measure it before applying readout correction."
            )
        ),
    )


__all__ = ["GmonRingLabNotebook", "build_gmon_ring_3q_lab_notebook"]
