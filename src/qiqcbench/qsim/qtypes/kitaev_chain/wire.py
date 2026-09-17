"""kitaev_chain control + result wire schemas (qtype-local).

Three charge-readout primitives (gated on ``kitaev_charge_readout``) plus pulse
control (gated on ``majorana_pulse_control``) for a fixed-parity Majorana qubit.
The charge-readout primitives use the async job model:

* ``run_charge_stability`` -- base-band-pulse a 2-dot readout window onto one
  bond and scan the two dot detunings, returning **raw per-shot dual-channel
  parity assignments** (global quantum capacitance + local charge sensor). The
  agent forms the Pearson correlation rho_MR and reads the indistinguishability
  line's tilt -> the bond's |t|/|Delta| balance.
* ``run_protection_sweep`` -- global common-mode detuning; returns raw per-shot
  parity bits per detuning, from which the agent forms the parity polarization
  P_M (proportional to the edge-mode splitting) and fits the protection exponent.
* ``run_subchain_spectroscopy`` -- measure the bulk excitation gap of a
  contiguous sub-chain (a phase defect suppresses the gap of windows spanning it
  while leaving amplitude-balanced bonds untouched -- a non-local handle).

Parity bit arrays are packed losslessly as base64 of ``np.packbits`` (row-major
``[n_point][shots]``). Results carry no fitted parameter, splitting, exponent,
or defect identity -- only raw measurement evidence.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from qiqcbench.qsim.core.wire import SCHEMA_VERSION, _Strict, _StrictRequest

MAX_PULSE_DETUNING_UEV = 20.0

# ---------- requests ----------


class ChargeStabilityRequest(_StrictRequest):
    schema_version: int = SCHEMA_VERSION
    bond: int = Field(..., ge=0, description="Bond index b (the 2-dot window on dots b, b+1).")
    mu_ld_values: list[float] = Field(..., min_length=1, max_length=64)
    mu_rd_values: list[float] = Field(..., min_length=1, max_length=64)
    shots: int = Field(..., ge=1, le=20_000)


class ProtectionSweepRequest(_StrictRequest):
    schema_version: int = SCHEMA_VERSION
    mu_common_values: list[float] = Field(..., min_length=1, max_length=128)
    shots: int = Field(..., ge=1, le=50_000)


class SubchainSpectroscopyRequest(_StrictRequest):
    schema_version: int = SCHEMA_VERSION
    site_a: int = Field(..., ge=0)
    site_b: int = Field(..., ge=0)
    shots: int = Field(..., ge=1, le=50_000)


class PulseSegmentWire(_StrictRequest):
    """One piecewise-constant drive segment of a Majorana-qubit gate pulse.

    ``amp`` is the DAC drive amplitude (0..1); the realized Rabi rate is
    ``amp * drive_rate`` where ``drive_rate`` is a hidden device parameter and
    the laboratory-notebook estimate may be stale. ``phase_rad`` sets the
    rotation axis in the x-y plane, ``detuning_ueV`` produces a logical-Z
    control, and ``drag`` is a constant quadrature ratio. With zero detuning
    and ``drag=0``, positive amplitude at ``phase_rad=0`` follows
    ``U=exp(-i theta X/2)``. With zero amplitude, positive detuning follows
    ``U=exp(-i theta Z/2)``, for ``theta > 0``. The wire name is retained for
    compatibility; it does not imply a derivative-DRAG waveform.
    """

    amp: float = Field(..., ge=0.0, le=1.0)
    phase_rad: float = Field(0.0, allow_inf_nan=False)
    # Above-gap excursions require ramp, bandwidth, and continuum physics that
    # this piecewise-constant effective model deliberately does not contain.
    detuning_ueV: float = Field(
        0.0,
        ge=-MAX_PULSE_DETUNING_UEV,
        le=MAX_PULSE_DETUNING_UEV,
    )
    duration_ns: float = Field(..., gt=0.0, le=50.0)
    drag: float = Field(0.0, ge=-5.0, le=5.0)


class MajoranaPulseBatchRequest(_StrictRequest):
    """Run a batch of pulse sequences on the logical Majorana qubit from ``|0_L>``.

    Each entry of ``sequences`` is a list of drive segments applied in order; the
    engine integrates an effective model of the fixed-total-parity computational
    states plus one aggregate non-computational level and reads out the state per
    shot. Used for calibration and exploratory control. Credited randomized
    benchmarking uses ``MajoranaCliffordRBRequest`` so qsim owns the random draw.
    """

    schema_version: int = SCHEMA_VERSION
    sequences: list[list[PulseSegmentWire]] = Field(..., min_length=1, max_length=128)
    shots: int = Field(..., ge=1, le=10_000)

    @property
    def total_segments(self) -> int:
        return sum(len(s) for s in self.sequences)


class MajoranaCliffordRBRequest(_StrictRequest):
    """Run qsim-owned uniform single-qubit Clifford RB on submitted gates.

    qsim draws each random Clifford privately and compiles both it and the exact
    inverse with one canonical compiler over the submitted ``X90``/``Z90``
    primitives.  The short length window is deliberate: level-resolved readout
    then supports the leakage-aware linear estimator published by the task.
    """

    schema_version: int = SCHEMA_VERSION
    gate_x90_segments: list[PulseSegmentWire] = Field(..., min_length=1, max_length=16)
    gate_z90_segments: list[PulseSegmentWire] = Field(..., min_length=1, max_length=16)
    lengths: list[int] = Field(..., min_length=3, max_length=8)
    sequences_per_length: int = Field(..., ge=4, le=32)
    shots_per_sequence: int = Field(..., ge=64, le=2_000)

    @model_validator(mode="after")
    def _validate_design(self) -> MajoranaCliffordRBRequest:
        if self.lengths != sorted(set(self.lengths)):
            raise ValueError("lengths must be distinct and strictly increasing")
        if self.lengths[0] < 1 or self.lengths[-1] > 8:
            raise ValueError("short-sequence RB lengths must lie in [1,8]")
        if len(self.lengths) * self.sequences_per_length > 128:
            raise ValueError("lengths * sequences_per_length must not exceed 128")
        return self


# ---------- results ----------


class KitaevChargeStabilityData(_Strict):
    """Raw dual-channel per-shot parity assignments over a 2-dot detuning grid.

    Grid points are ordered with ``mu_ld_values`` as the outer (slow) axis and
    ``mu_rd_values`` as the inner (fast) axis: point index ``g = i_ld * len(rd) +
    i_rd``. ``global_parity_b64`` / ``local_parity_b64`` pack a ``[n_grid][shots]``
    uint8 0/1 array (row-major) via base64 of ``np.packbits``.
    """

    kind: Literal["kitaev_charge_stability"] = "kitaev_charge_stability"
    bond: int
    mu_ld_values: list[float]
    mu_rd_values: list[float]
    shots: int
    global_parity_b64: str
    local_parity_b64: str


class KitaevPolarizationData(_Strict):
    """Raw per-shot parity bits per common-mode detuning.

    ``parity_b64`` packs a ``[n_mu][shots]`` uint8 0/1 array (row-major). The
    agent forms P_M = <1 - 2*bit> per detuning; |P_M| grows with the edge-mode
    splitting, so its log-log slope vs detuning is the protection exponent.
    """

    kind: Literal["kitaev_polarization"] = "kitaev_polarization"
    mu_common_values: list[float]
    shots: int
    parity_b64: str


class KitaevSubchainGapData(_Strict):
    """Measured bulk excitation gap (micro-eV) of a contiguous sub-chain."""

    kind: Literal["kitaev_subchain_gap"] = "kitaev_subchain_gap"
    site_a: int
    site_b: int
    measured_gap_ueV: float


class KitaevPulseBatchData(_Strict):
    """Raw per-shot charge-readout outcomes for a batch of pulse sequences.

    ``levels_b64`` is base64 of a ``[n_sequences][shots]`` uint8 array (row-major,
    1 byte per shot) of charge-state outcomes: ``0`` = logical ``|0>``, ``1`` =
    logical ``|1>``, ``2`` = leaked (bulk quasiparticle, out of the qubit
    subspace). The agent forms per-sequence survival ``= mean(level == 0)`` and
    leakage ``= mean(level == 2)``. No fitted parameter is returned.
    """

    kind: Literal["kitaev_pulse_batch"] = "kitaev_pulse_batch"
    n_sequences: int
    shots: int
    levels_b64: str


class KitaevCliffordRBData(_Strict):
    """Raw level-resolved outcomes from qsim-owned uniform Clifford RB.

    ``levels_b64`` is a contiguous uint8 array with shape
    ``[len(lengths), sequences_per_length, shots_per_sequence]``.  The nested
    Clifford-index and primitive-count arrays follow the first two axes.
    Clifford indices address the canonical 24-element compiler used by qsim.
    """

    kind: Literal["kitaev_clifford_rb"] = "kitaev_clifford_rb"
    lengths: list[int]
    sequences_per_length: int
    shots_per_sequence: int
    clifford_indices: list[list[list[int]]]
    compiled_primitive_counts: list[list[int]]
    levels_b64: str


__all__ = [
    "ChargeStabilityRequest",
    "MAX_PULSE_DETUNING_UEV",
    "ProtectionSweepRequest",
    "SubchainSpectroscopyRequest",
    "PulseSegmentWire",
    "MajoranaPulseBatchRequest",
    "MajoranaCliffordRBRequest",
    "KitaevChargeStabilityData",
    "KitaevPolarizationData",
    "KitaevSubchainGapData",
    "KitaevPulseBatchData",
    "KitaevCliffordRBData",
]
