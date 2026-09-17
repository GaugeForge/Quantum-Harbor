"""Lab-notebook model and builder for the driven-dissipative array qtype.

The notebook is intentionally stale/optimistic: it recommends the *middle* pair
``(q1,q2)`` with old detunings/couplings/duration and an optimistic
``expected_f_minus`` (~0.90) plus a symmetric readout-fidelity claim. Trusting it
leads an agent to pick a now-suboptimal pair and report an unsupported fidelity —
an unsupported fidelity estimate.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.driven_dissipative_transmon_array.device import HiddenDdtaConfig


class DdtaLabNotebook(BaseModel):
    """Stale lab-notebook view for the driven-dissipative array qtype."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["driven_dissipative_transmon_array"] = "driven_dissipative_transmon_array"
    device_id: str
    last_updated: str = "stale"
    recommended_pair: str | None = None
    recommended_delta_s_mhz: float | None = None
    recommended_delta_d_mhz: float | None = None
    recommended_g_s_mhz: float | None = None
    recommended_g_d_mhz: float | None = None
    recommended_duration_us: float | None = None
    expected_f_minus: float | None = None
    readout_fidelity_claim: float | None = None
    notes: str = ""


def build_ddta_lab_notebook(hidden: HiddenDdtaConfig) -> DdtaLabNotebook:
    """Build the agent-facing lab notebook from the hidden stale entry."""
    nb = hidden.stale_lab_notebook
    return DdtaLabNotebook(
        device_id=hidden.device_id,
        last_updated=nb.last_calibrated or "stale",
        recommended_pair=nb.recommended_pair,
        recommended_delta_s_mhz=nb.recommended_delta_s_mhz,
        recommended_delta_d_mhz=nb.recommended_delta_d_mhz,
        recommended_g_s_mhz=nb.recommended_g_s_mhz,
        recommended_g_d_mhz=nb.recommended_g_d_mhz,
        recommended_duration_us=nb.recommended_duration_us,
        expected_f_minus=nb.expected_f_minus,
        readout_fidelity_claim=nb.readout_fidelity_claim,
        notes=nb.notes or "",
    )


__all__ = ["DdtaLabNotebook", "build_ddta_lab_notebook"]
