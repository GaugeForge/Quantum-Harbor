"""Lab-notebook model + builder for the ``cycle_error_recon`` qtype.

Surfaces the recorded calibration prior and its in-world analysis provenance through
``get_lab_notebook`` without adding a maintainer verdict about its correctness.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.cycle_error_recon.device import HiddenCerConfig


class CerLabNotebook(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["cycle_error_recon"] = "cycle_error_recon"
    device_id: str
    parallel_cycle_fidelity: float
    serial_cycle_fidelity: float
    preferred_schedule: str
    coherent_error_detected: bool
    correlated_error_probability: float
    readout_model: str
    fold_model: str
    note: str
    reliability: str = "recorded at the last cycle-benchmarking block"


def build_cycle_error_recon_lab_notebook(hidden: HiddenCerConfig) -> CerLabNotebook:
    nb = hidden.stale_lab_notebook
    return CerLabNotebook(
        device_id=hidden.device_id,
        parallel_cycle_fidelity=nb.parallel_cycle_fidelity,
        serial_cycle_fidelity=nb.serial_cycle_fidelity,
        preferred_schedule=nb.preferred_schedule,
        coherent_error_detected=nb.coherent_error_detected,
        correlated_error_probability=nb.correlated_error_probability,
        readout_model=nb.readout_model,
        fold_model=nb.fold_model,
        note=(
            nb.note
            or "Recorded cycle-benchmarking block: both schedules ~1.5% error per cycle; serial "
            "was preferred. The analysis reported nearly depolarizing error, no coherent "
            "contribution above 0.01 rad, ~98% symmetric readout, and one shared linear-plus-"
            "quadratic fold model."
        ),
    )
