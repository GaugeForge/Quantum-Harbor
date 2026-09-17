"""Public + hidden device config types for the trapped_ion_chain qtype.

Mirrors the public/hidden split pattern of transmon_pulse/device.py and
digital_gate_model/device.py. extra='forbid' on both sides; loading a hidden
YAML into the public type must fail.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Public spec (visible to agent) ----------


class PublicIonChainQubit(_Strict):
    id: int = Field(..., ge=0)


class PublicGateSet(_Strict):
    one_qubit: list[str]
    entangling: list[str]


class PublicClaimedFidelities(_Strict):
    one_qubit_avg: float = Field(..., gt=0, le=1)
    entangling_ms: float = Field(..., gt=0, le=1)
    readout: float = Field(..., gt=0, le=1)


class PublicFreeEvolutionLimits(_Strict):
    min_us: float = Field(0.0, ge=0)
    max_us: float = Field(..., gt=0)
    resolution_us: float = Field(..., gt=0)


class PublicMsParams(_Strict):
    coupling_band_hz: tuple[float, float]
    duration_range_us: tuple[float, float]
    connectivity: Literal["all_to_all_shared_mode"] = "all_to_all_shared_mode"


class PublicTrappedIonChainSpec(_Strict):
    """Hardware-manual-style public spec for the trapped_ion_chain qtype."""

    schema_version: int = 1
    device_id: str
    qtype: Literal["trapped_ion_chain"] = "trapped_ion_chain"
    ion_species: str
    encoding: str
    qubits: list[PublicIonChainQubit] = Field(..., min_length=1)
    gate_set: PublicGateSet
    nominal_one_qubit_gate_duration_us: float = Field(..., gt=0)
    ms_params: PublicMsParams
    free_evolution: PublicFreeEvolutionLimits
    claimed_fidelities: PublicClaimedFidelities
    max_shots: int = Field(100_000, gt=0)
    notes: str = "Calibrations may be stale; verify before relying on them."


# ---------- Hidden truth (only inside qsim) ----------


class HiddenIonQubit(_Strict):
    id: int = Field(..., ge=0)
    frequency_hz: float
    detuning_hz: float = Field(
        ...,
        description=(
            "Per-ion static frequency offset applied during free evolution. "
            "Despite being a 'target' to estimate, the engine "
            "uses this as a device-physics property; estimation is a task-level "
            "concern not encoded in this field's name."
        ),
    )
    t2_s: float = Field(..., gt=0)
    dephasing_exponent: float = Field(
        ...,
        description="Currently must be exactly 1.0 (Markovian) or 2.0 (non-Markovian Zeno).",
    )
    tau_c_s: float = Field(
        ...,
        gt=0,
        description="Bath correlation time. Model-validity metadata; current quasi-static engine path does not consume it at runtime but it gates Zeno-regime validity in the scorer's sanity checks.",
    )
    t1_s: float | None = None

    @field_validator("dephasing_exponent")
    @classmethod
    def _check_dephasing_exponent_supported(cls, v: float) -> float:
        if v not in (1.0, 2.0):
            raise ValueError(
                f"dephasing_exponent={v} not supported; engine implements only "
                "1.0 (exponential) and 2.0 (Gaussian). Stretched-exponential is "
                "a recorded forward opening, not a live schema promise."
            )
        return v


class HiddenReadoutIon(_Strict):
    id: int = Field(..., ge=0)
    p_0_to_1: float = Field(..., ge=0, le=1)
    p_1_to_0: float = Field(..., ge=0, le=1)


class HiddenOneQubitGateErrors(_Strict):
    depolarizing: float = Field(..., ge=0, le=1)


class HiddenMsConfig(_Strict):
    coupling_j_hz: float = Field(..., gt=0)
    depolarizing_error: float = Field(..., ge=0, le=1)
    duration_us: float = Field(..., gt=0)


class StaleIonChainNotebookEntry(_Strict):
    """Stale calibration claims surfaced via get_lab_notebook.

    All fields nullable so partial stale entries are valid.
    """

    claimed_decay_model: Literal["exponential", "gaussian", "stretched_exponential"] | None = None
    claimed_t2_star_us: float | None = None
    claimed_ms_fidelity: float | None = None
    claimed_readout_fidelity: float | None = None
    claimed_readout_symmetric_p: float | None = None
    advisory_notes: str | None = None


class HiddenTrappedIonChainConfig(_Strict):
    """Hidden truth for the trapped_ion_chain qtype. Lives only inside the qsim container."""

    schema_version: int = 1
    device_id: str
    qtype: Literal["trapped_ion_chain"] = "trapped_ion_chain"
    seed: int
    ions: list[HiddenIonQubit] = Field(..., min_length=1)
    readout: list[HiddenReadoutIon] = Field(..., min_length=1)
    one_qubit_gate_errors: HiddenOneQubitGateErrors
    ms: HiddenMsConfig
    stale_lab_notebook: StaleIonChainNotebookEntry = StaleIonChainNotebookEntry()


__all__ = [
    "HiddenIonQubit",
    "HiddenMsConfig",
    "HiddenOneQubitGateErrors",
    "HiddenReadoutIon",
    "HiddenTrappedIonChainConfig",
    "PublicClaimedFidelities",
    "PublicFreeEvolutionLimits",
    "PublicGateSet",
    "PublicIonChainQubit",
    "PublicMsParams",
    "PublicTrappedIonChainSpec",
    "StaleIonChainNotebookEntry",
]
