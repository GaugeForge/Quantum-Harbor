"""Lab-notebook model and builder for the transmon (pulse-level) qtype.

The canonical ``LabNotebook`` lives here; ``qsim.core.lab_notebook`` re-exports
it for backward compatibility and exposes the cross-qtype union alias.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.transmon_pulse.device import HiddenTransmonConfig

NOMINAL_QUBIT_FREQUENCY_HZ = 5.0e9


class LabNotebook(BaseModel):
    """Stale lab notebook entry for the transmon (pulse-level) Qtype.

    Contents are intentionally partial and possibly out of date. The benchmark
    agent must verify before relying on these values.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["transmon_pulse"] = "transmon_pulse"
    device_id: str
    last_updated: str = "stale"
    qubit_frequency_hz: float | None = None
    x180_amp: float | None = None
    x180_duration_ns: float | None = None
    x180_sigma_ns: float | None = None
    readout_threshold_i: float | None = None
    t1_s_estimate: float | None = None
    notes: str = ""


def build_transmon_lab_notebook(hidden: HiddenTransmonConfig) -> LabNotebook:
    """Build the transmon lab-notebook view from a hidden config.

    The qubit drive frequency is a nominal/stale public value, not the exact
    simulator truth. Everything else in ``stale_lab_notebook`` is partial and
    potentially out of date.
    """
    nb = hidden.stale_lab_notebook
    return LabNotebook(
        device_id=hidden.device_id,
        last_updated="2026-01-15 (likely stale)",
        qubit_frequency_hz=NOMINAL_QUBIT_FREQUENCY_HZ,
        x180_amp=nb.x180_amp,
        x180_duration_ns=nb.x180_duration_ns,
        x180_sigma_ns=nb.x180_sigma_ns,
        readout_threshold_i=nb.readout_threshold_i,
        t1_s_estimate=nb.t1_s,
        notes=(
            "Last calibration ~3 months old. The qubit frequency is nominal; "
            "verify before relying on it. T1 is known to drift; "
            "the recorded value here may be optimistic. The X180 pulse "
            "should still prepare |1> with good fidelity."
        ),
    )


__all__ = ["LabNotebook", "build_transmon_lab_notebook"]
