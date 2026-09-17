"""Stale lab-notebook for the tunable-coupler CZ qtype.

Surfaces the stale/optimistic calibration: a wrong anharmonicity/coupling, an
optimistic idle coupler flux ("J=0, ZZ negligible"), "a unipolar pulse is fine",
and "the flux line is pre-corrected / reuse last cooldown's filter". An agent
that trusts these mistargets the crossing, leaves residual idle ZZ, loses 1/f
immunity, and ignores the flux-line distortion.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.transmon_pair_tunable_coupler.device import (
    HiddenTunableCouplerConfig,
)


class TunableCouplerLabNotebook(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["transmon_pair_tunable_coupler"] = "transmon_pair_tunable_coupler"
    device_id: str
    last_updated: str = "unrecorded"
    alpha_2_mhz_claim: float | None = None
    coupling_j_mhz_claim: float | None = None
    g_mhz_claim: str | None = None
    coupler_idle_flux_claim: float | None = None
    pulse_advice: str | None = None
    line_transfer_function_claim: str | None = None
    readout_claim: str | None = None
    notes: str = ""


def build_tunable_coupler_lab_notebook(
    hidden: HiddenTunableCouplerConfig,
) -> TunableCouplerLabNotebook:
    nb = hidden.stale_lab_notebook
    return TunableCouplerLabNotebook(
        device_id=hidden.device_id,
        last_updated=nb.last_calibrated or "unrecorded",
        alpha_2_mhz_claim=nb.alpha_2_mhz_claim,
        coupling_j_mhz_claim=nb.coupling_j_mhz_claim,
        g_mhz_claim=nb.g_mhz_claim,
        coupler_idle_flux_claim=nb.coupler_idle_flux_claim,
        pulse_advice=nb.pulse_advice,
        line_transfer_function_claim=nb.line_transfer_function_claim,
        readout_claim=nb.readout_claim,
        notes=(
            nb.notes
            or (
                "Values were accepted at the previous cooldown's calibration. The "
                "flux-line correction filter was carried over from that cooldown's "
                "fit, so pulses can be programmed directly; the unipolar CZ pulse "
                "met spec at that calibration."
            )
        ),
    )


__all__ = ["TunableCouplerLabNotebook", "build_tunable_coupler_lab_notebook"]
