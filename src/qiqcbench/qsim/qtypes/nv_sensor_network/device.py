"""Public + hidden device config types for the NV sensor-network qtype.

An ``nv_sensor_network`` device is a network of ``d`` spatially separated
single-NV-center magnetometer nodes on a line. Each node runs a Ramsey sequence
sensing its local static field ``theta_i = gamma_e * B_i`` (rad/s), with a
mid-window echo pulse realizing a signed effective interrogation time. A central
station can distribute a GHZ across a chosen node subset by consuming heralded,
node-dependent-fidelity Bell pairs.

Public / hidden are STRUCTURALLY separated: both classes
use ``extra="forbid"``. The PUBLIC spec carries node positions, the gyromagnetic
ratio, control limits, and *claimed/stale* calibrations (homogenized T2*, link,
readout). The HIDDEN config holds the true per-node T2*, link fidelity, readout
asymmetry, fields, and the RNG seed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Public spec (visible to agent) ----------


class PublicNvNode(_Strict):
    id: int = Field(..., ge=0)
    position: float = Field(..., description="Node position x_i on the line (arb. units).")


class PublicNvSensorNetworkSpec(_Strict):
    """Hardware-manual-style public spec for the NV sensor-network qtype.

    Sensing model: per-node Ramsey phase ``theta_i = gamma_e * B_i`` per unit time,
    reported in rad/s. A signed effective interrogation time is set by echo placement.
    Entanglement distribution (GHZ over a node subset) is a costly, noisy resource.
    """

    schema_version: int = 1
    device_id: str
    qtype: Literal["nv_sensor_network"] = "nv_sensor_network"
    nodes: list[PublicNvNode] = Field(..., min_length=1)
    primary_node_count: int = Field(
        6, ge=1, description="Node count for the primary (Track-B) targets; n0..n5."
    )
    gyromagnetic_ratio_rad_per_s_per_t: float = Field(
        2.0 * 3.141592653589793 * 28.024e9,
        description="NV electron gyromagnetic ratio gamma_e (public constant).",
    )
    time_resolution_s: float = Field(
        10e-9, gt=0, description="Free-evolution timing grid (interrogation times on this grid)."
    )
    pulse_duration_s: float = Field(30e-9, gt=0, description="Nominal pi/2 / pi pulse duration.")
    max_field_hz: float = Field(
        500.0, gt=0, description="Per-node |theta_i|/2pi target range (Hz); exact values hidden."
    )
    total_interrogation_time_budget_s: float = Field(
        1.0, gt=0, description="Fixed cumulative interrogation-time budget T_tot per target."
    )
    max_shots: int = 100_000
    measurement_return: Literal["bitstring"] = "bitstring"
    # Claimed / stale calibrations (the true values are heterogeneous; see notebook).
    claimed_t2star_us: float = Field(400.0, gt=0, description="Stale homogenized per-node T2*.")
    claimed_link_fidelity: float = Field(
        0.99, ge=0, le=1, description="Stale uniform link fidelity."
    )
    claimed_readout_fidelity: float = Field(
        0.99, ge=0, le=1, description="Stale symmetric readout."
    )
    notes: str = (
        "Node calibrations may have drifted since last service and were recorded as a single "
        "representative value; verify per node before designing. Entanglement distribution is a "
        "limited, noisy resource."
    )


# ---------- Hidden truth (only inside qsim) ----------


class HiddenNvNode(_Strict):
    id: int = Field(..., ge=0)
    t2star_us: float = Field(..., gt=0, description="True per-node Gaussian dephasing time T2*.")
    link_fidelity: float = Field(..., ge=0, le=1, description="True heralded Bell-pair fidelity.")
    p_0_to_1: float = Field(..., ge=0, le=1)
    p_1_to_0: float = Field(..., ge=0, le=1)
    field_hz: float = Field(..., description="True local field theta_i/2pi in Hz.")


class StaleNvNotebookEntry(_Strict):
    """Optional stale/optimistic calibration written into the lab notebook."""

    last_calibrated: str | None = None
    t2star_us_claim: float | None = None
    link_fidelity_claim: float | None = None
    readout_fidelity_claim: float | None = None
    recommended_strategy: str | None = None
    notes: str = ""


class HiddenNvSensorNetworkConfig(_Strict):
    """Hidden truth for the NV sensor-network qtype. Lives only inside qsim."""

    schema_version: int = 1
    device_id: str
    qtype: Literal["nv_sensor_network"] = "nv_sensor_network"
    seed: int
    nodes: list[HiddenNvNode] = Field(..., min_length=1)
    stale_lab_notebook: StaleNvNotebookEntry = StaleNvNotebookEntry()


__all__ = [
    "HiddenNvNode",
    "HiddenNvSensorNetworkConfig",
    "PublicNvNode",
    "PublicNvSensorNetworkSpec",
    "StaleNvNotebookEntry",
]
