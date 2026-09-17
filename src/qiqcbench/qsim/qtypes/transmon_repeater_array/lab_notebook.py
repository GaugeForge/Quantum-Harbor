"""Lab-notebook model + builder for the ``transmon_repeater_array`` qtype.

Surfaces an **honest but stale** calibration prior via ``get_lab_notebook``:
a *nominal/claimed* distributed-pair fidelity the agent must VERIFY, and an explicit
statement that the achievable purified-fidelity ceiling has NOT been characterized.
No misinformation is planted — there is no claim of an unachievable purified fidelity
and no protocol recommendation. The difficulty comes entirely from the hidden params the
agent measures (the real F0, the noise-floored F_sat, and which protocol wins).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.transmon_repeater_array.device import HiddenRepeaterConfig


class RepeaterLabNotebook(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["transmon_repeater_array"] = "transmon_repeater_array"
    device_id: str
    claimed_distributed_fidelity: float  # nominal/spec value — verify on the live device
    purified_fidelity_status: str  # honest: the achievable ceiling is uncharacterized
    transport_noise: str  # honest: nominal assumption only, never a structure claim
    last_calibrated: str
    note: str
    reliability: str = "stale; predates current operation — verify before relying on it"


def build_transmon_repeater_array_lab_notebook(
    hidden: HiddenRepeaterConfig,
) -> RepeaterLabNotebook:
    nb = hidden.stale_lab_notebook
    return RepeaterLabNotebook(
        device_id=hidden.device_id,
        claimed_distributed_fidelity=nb.claimed_distributed_fidelity,
        purified_fidelity_status="not characterized on the current wiring — measure it",
        transport_noise=(
            nb.transport_noise
            or "nominal spec assumes isotropic transport noise; not characterized on the "
            "current wiring — measure the delivered pair before designing"
        ),
        last_calibrated=nb.last_calibrated,
        note=(
            nb.note
            or "Nominal spec distributed-pair fidelity ~0.92 (datasheet, not a live measurement) "
            "— transport noise may make the real value lower, so verify it. The achievable "
            "purified-fidelity ceiling and the best purification protocol have NOT been "
            "characterized on this wiring; determine them experimentally. (Stale — verify everything.)"
        ),
    )
