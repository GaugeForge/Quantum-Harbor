"""Public + hidden device config for the multilevel-transmon (pulse-level) qtype.

A ``transmon_multilevel_pulse`` device is a single fixed-frequency transmon kept
as a 4-level Duffing oscillator. The agent controls two drive quadratures
``Omega_x(t)``, ``Omega_y(t)`` (specified in DAC units, mapped to a Rabi rate by
the *hidden* amplitude calibration) and a drive detuning ``delta(t)`` (a direct
Hz frequency knob), and reads out level-resolved populations P0..P3.

Public / hidden are STRUCTURALLY separated: the public
spec carries only the control surface (grid, ceilings, bandwidth), never the
true anharmonicity, amplitude scale, coherence, or readout matrix. The realized
anharmonicity and amplitude calibration are the *answers* to Stage A and live
only in the hidden config; the stale/optimistic claims live in the lab notebook.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Public spec (visible to agent) ----------


class PublicTransmonMultilevelSpec(_Strict):
    """Hardware-manual-style public spec for the multilevel-transmon qtype.

    Hamiltonian (rotating frame at ``omega_01``, sign convention
    ``delta = omega_01 - omega_drive``)::

        H/hbar = alpha * diag(0,0,1,3)            # anharmonicity (hidden value)
               + delta(t) * N                      # drive detuning (Hz knob)
               + (1/2)[Omega_x(t)(a+a_dag) + Omega_y(t) i(a_dag-a)]

    Drive quadratures are specified in DAC units; the hidden amplitude
    calibration maps DAC -> Rabi rate. Detuning is a direct frequency knob.
    """

    schema_version: int = 1
    device_id: str
    qtype: Literal["transmon_multilevel_pulse"] = "transmon_multilevel_pulse"
    omega01_hz: float = Field(
        5.000e9, description="0->1 transition frequency (public exact = control setpoint)."
    )
    n_levels: int = Field(4, ge=3, description="Duffing levels modeled and read out (P0..P{n-1}).")
    sample_dt_ns: float = Field(
        0.4, gt=0, description="AWG sample spacing; pulses lie on this grid."
    )
    min_sample_count: int = Field(15, ge=1, description="Minimum samples in a sampled pulse.")
    max_amp_dac: float = Field(
        2.0, gt=0, description="Per-quadrature DAC amplitude ceiling (dimensionless)."
    )
    max_abs_detuning_hz: float = Field(
        250e6, gt=0, description="Drive-detuning ceiling |delta/2pi|."
    )
    analog_bandwidth_hz: float = Field(
        900e6, gt=0, description="Control-line bandwidth; faster content is unphysical."
    )
    max_pulse_duration_ns: float = Field(200.0, gt=0)
    max_segments: int = Field(8, ge=1, description="Max drive segments per sequence.")
    supported_shapes: list[str] = ["gaussian", "square"]
    detuning_sign_convention: Literal["omega_01_minus_omega_drive"] = "omega_01_minus_omega_drive"
    matrix_element_ratio_nominal: float = Field(
        1.4142135623730951, description="lambda = <2|a_dag|1>/<1|a_dag|0> = sqrt(2) (Duffing)."
    )
    measurement_return: Literal["level_population"] = "level_population"
    max_shots: int = 100_000
    notes: str = (
        "Anharmonicity and amplitude calibration were copied from an earlier "
        "configuration. Drive quadratures are in DAC units; detuning is in Hz."
    )


# ---------- Hidden truth (only inside qsim) ----------


class HiddenMultilevelReadout(_Strict):
    """Full level-resolved assignment matrix (rows=prepared, cols=reported)."""

    assignment: list[list[float]] = Field(
        ..., description="n_levels x n_levels row-stochastic assignment matrix."
    )


class StaleMultilevelNotebookEntry(_Strict):
    """Stale/optimistic calibration written into the lab notebook."""

    last_calibrated: str | None = None
    anharmonicity_mhz_claim: float | None = None
    rabi_mhz_per_dac_claim: float | None = None
    amplitude_linear_claim: bool | None = None
    drag_beta_claim: float | None = None
    detuning_claim: str | None = None
    level_model_claim: str | None = None
    notes: str = ""


class HiddenTransmonMultilevelConfig(_Strict):
    """Hidden truth for the multilevel-transmon qtype. Lives only inside qsim."""

    schema_version: int = 1
    device_id: str
    qtype: Literal["transmon_multilevel_pulse"] = "transmon_multilevel_pulse"
    seed: int
    anharmonicity_hz: float = Field(..., description="True alpha/2pi (signed, negative).")
    rabi_mhz_per_dac: float = Field(..., gt=0, description="True DAC->Rabi scale k (MHz/unit).")
    amp_compression: float = Field(
        0.0, ge=0, description="Compressive nonlinearity: Rabi = k*A/(1+comp*|A|^2)."
    )
    t1_s: float = Field(..., gt=0)
    t2_s: float = Field(..., gt=0)
    readout: HiddenMultilevelReadout
    stale_lab_notebook: StaleMultilevelNotebookEntry = StaleMultilevelNotebookEntry()


__all__ = [
    "HiddenMultilevelReadout",
    "HiddenTransmonMultilevelConfig",
    "PublicTransmonMultilevelSpec",
    "StaleMultilevelNotebookEntry",
]
