"""Public + hidden device config types for the chain pulse-compiler qtype.

A 5-qubit linear-chain transmon ``q0 - q1 - q2 - q3 - q4`` with a native
continuous controlled-phase ``CPhase(theta)`` on each nearest-neighbour edge,
virtual-Z single-qubit phases (free), finite-duration ``Rx``/``Ry``, and
multilevel leakage to ``|2>`` during the entangling gate. The agent submits a
scheduled gate program and is scored by the SPAM-inclusive average-gate fidelity
of the implemented operation vs an ideal target unitary.

Public and hidden are STRUCTURALLY separated: both use
``extra="forbid"`` so a hidden-truth field can never be absorbed by the public
spec and vice versa. Mirrors the split in ``digital_gate_model/device.py`` and
``transmon_multilevel_pulse/device.py``.

What is public: the per-gate
durations and depolarizing floors, the well-tuned leakage floor, and the
heterogeneous ``T1``/``T2``. What is HIDDEN (the levers): the *shipped* entangling
pulse's coherent over-rotation factor and excess-leakage factor (the agent must
recover them in Stage A), the per-qubit *asymmetric* readout, the state-prep
error, and the RNG seed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Public spec (visible to agent) ----------


class PublicChainCompilerSpec(_Strict):
    """Hardware-manual-style public spec for the chain pulse-compiler qtype.

    The device noise model is largely public; the *stale* notebook claims and the
    shipped entangling-pulse calibration (a hidden coherent over-rotation) are the
    levers — see ``lab_notebook.py`` and the hidden config.
    """

    schema_version: int = 1
    device_id: str
    qtype: Literal["transmon_chain_pulse_compiler"] = "transmon_chain_pulse_compiler"

    n_qubits: int = Field(5, ge=2, description="Transmons in the chain.")
    connectivity: Literal["all_to_all"] = Field(
        "all_to_all",
        description=(
            "v1 exposes a controlled-phase between ANY qubit pair (the physical device "
            "is a chain, but routing on a strict line is a deferred v2 lever; the "
            "primary discriminators are native-CPhase vs CNOT-decomposition, pulse "
            "recalibration, virtual-Z, software bit-reversal, and scheduling)."
        ),
    )
    n_levels: int = Field(
        3, ge=3, description="Levels modeled/read out per transmon (|0>,|1>,|2> leakage)."
    )

    native_1q_gates: list[str] = ["rz", "rx", "ry"]
    native_2q_gate: Literal["cphase"] = "cphase"
    virtual_z: bool = Field(
        True, description="Rz is a virtual-Z frame change: zero duration, ~zero error."
    )

    rx_ry_duration_ns: float = Field(25.0, gt=0, description="Per single-qubit physical gate.")
    rx_ry_depolarizing: float = Field(3.0e-4, ge=0, description="Per single-qubit gate.")
    cphase_duration_ns: float = Field(40.0, gt=0, description="Per entangling gate.")
    cphase_depolarizing: float = Field(
        4.0e-3, ge=0, description="Purity-limited per-CPhase error floor (not removable)."
    )
    cphase_leakage_floor: float = Field(
        1.0e-3, ge=0, description="Achievable leakage to |2> of a well-tuned CPhase."
    )

    t1_us: list[float] = Field(..., min_length=2, description="Per-qubit T1 (us); public.")
    t2_us: list[float] = Field(..., min_length=2, description="Per-qubit T2 echo (us); public.")

    measurement_return: Literal["level_population"] = "level_population"
    max_shots: int = Field(100_000, gt=0)
    readout_note: str = (
        "Readout is per-qubit asymmetric (~1.5%) and is included in the primary "
        "SPAM-inclusive score; the exact rates are not advertised — measure them if "
        "you want to apply readout mitigation."
    )
    notes: str = (
        "Target: implement an ideal QFT_5. Native gates: virtual-Z Rz (free), "
        "finite-duration Rx/Ry, and a continuous CPhase(theta) on each edge (ONE "
        "entangling pulse per controlled-phase; you may recalibrate it). The "
        "entangling-gate pulse can be recalibrated via the run tool's 'calibration' "
        "field. You are scored on the achieved average-gate "
        "fidelity of the implemented QFT_5, higher is better."
    )


# ---------- Hidden truth (only inside qsim) ----------


class HiddenChainReadout(_Strict):
    """Per-qubit asymmetric computational-basis readout (the SPAM lever).

    ``p_0_to_1`` / ``p_1_to_0`` are the classical bit-flip confusion rates used as
    a CPTP channel in the scored fidelity. ``p_2_misread`` is the probability a true
    ``|2>`` (leakage) reads back as a computational level instead of '2' — only
    affects the agent's Stage-A leakage scan, not the scored computational fidelity.
    """

    id: int = Field(..., ge=0)
    p_0_to_1: float = Field(..., ge=0, le=1)
    p_1_to_0: float = Field(..., ge=0, le=1)
    p_2_misread: float = Field(0.0, ge=0, le=1)


class StaleChainNotebookEntry(_Strict):
    """Stale/optimistic claims surfaced via get_lab_notebook (the textbook trap)."""

    last_calibrated: str | None = None
    two_qubit_fidelity_claim: float | None = None
    controlled_phase_recipe: str | None = None
    rz_recipe: str | None = None
    swap_recipe: str | None = None
    approximation_advice: str | None = None
    entangling_pulse_claim: str | None = None
    readout_claim: str | None = None
    notes: str = ""


class HiddenChainCompilerConfig(_Strict):
    """Hidden truth for the chain pulse-compiler qtype. Lives only inside qsim."""

    schema_version: int = 1
    device_id: str
    qtype: Literal["transmon_chain_pulse_compiler"] = "transmon_chain_pulse_compiler"
    seed: int

    n_qubits: int = Field(5, ge=2)
    t1_us: list[float] = Field(..., min_length=2)
    t2_us: list[float] = Field(..., min_length=2)

    rx_ry_duration_ns: float = Field(25.0, gt=0)
    rx_ry_depolarizing: float = Field(3.0e-4, ge=0)
    cphase_duration_ns: float = Field(40.0, gt=0)
    cphase_depolarizing: float = Field(4.0e-3, ge=0)
    cphase_leakage_well_tuned: float = Field(
        1.0e-3, ge=0, description="Achievable leakage floor (well-tuned pulse)."
    )

    # --- the hidden entangling-pulse calibration (Stage-A recovery target) ---
    shipped_over_rotation_factor: float = Field(
        1.05,
        gt=0,
        description=(
            "Realized conditional phase = factor * intended (the shipped stale pulse "
            "carries a +5% coherent over-rotation). The agent recovers this in Stage A "
            "and submits a correction; a recalibrated pulse removes the coherent error."
        ),
    )
    shipped_leakage_factor: float = Field(
        2.0,
        ge=1,
        description=(
            "Realized leakage = well_tuned * factor when uncorrected (the shipped pulse "
            "has ~2x the leakage of a well-tuned pulse). The agent can correct toward 1."
        ),
    )

    prep_error: float = Field(
        1.0e-3, ge=0, description="Residual |1> population after state prep (SPAM)."
    )
    readout: list[HiddenChainReadout] = Field(..., min_length=2)

    stale_lab_notebook: StaleChainNotebookEntry = StaleChainNotebookEntry()


__all__ = [
    "HiddenChainCompilerConfig",
    "HiddenChainReadout",
    "PublicChainCompilerSpec",
    "StaleChainNotebookEntry",
]
