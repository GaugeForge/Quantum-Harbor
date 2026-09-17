"""Lab-notebook model + builder for ``trapped_ion_state_copy_randomized_measurement``.

Surfaces the stale, misleading prior via ``get_lab_notebook``: a recent linear Z0Z1 scan (a
different physical quantity than the cooled ratios) and an older virtual-cooling ladder that
wrongly assumes "cooling barely changes Z0Z1" (it sits near the linear value, well below the
true cooled ladder), plus an optimistic symmetric readout claim. Trusting it — submitting the
linear correlator or the stale ladder — fails every scored band.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.trapped_ion_state_copy_randomized_measurement.device import (
    HiddenStateCopyConfig,
)


class StateCopyLabNotebook(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["trapped_ion_state_copy_randomized_measurement"] = (
        "trapped_ion_state_copy_randomized_measurement"
    )
    device_id: str
    last_calibrated: str
    linear_z0z1_claim: str
    rho_moment2_claim: str
    z0z1_vc2_claim: str
    z0z1_vc3_claim: str
    z0z1_vc4_claim: str
    readout_fidelity_claim: str
    note: str
    reliability: str = "values accepted at the last scheduled calibration"


def build_trapped_ion_state_copy_randomized_measurement_lab_notebook(
    hidden: HiddenStateCopyConfig,
) -> StateCopyLabNotebook:
    nb = hidden.stale_lab_notebook
    return StateCopyLabNotebook(
        device_id=hidden.device_id,
        last_calibrated=nb.last_calibrated,
        linear_z0z1_claim=nb.linear_z0z1_claim,
        rho_moment2_claim=nb.rho_moment2_claim,
        z0z1_vc2_claim=nb.z0z1_vc2_claim,
        z0z1_vc3_claim=nb.z0z1_vc3_claim,
        z0z1_vc4_claim=nb.z0z1_vc4_claim,
        readout_fidelity_claim=nb.readout_fidelity_claim,
        note=nb.note,
    )
