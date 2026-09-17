"""Public + hidden device config for the ``trapped_ion_state_copy_randomized_measurement`` qtype.

A six-ion trapped-ion long-range XX simulator that prepares independent identical copies of an
unknown mixed state ``rho`` on demand, exposing single-copy randomized measurements and
collective copy-block (cyclic-shift) measurements for estimating nonlinear functionals
``Tr(Z0Z1 rho^k)/Tr(rho^k)``.

Architecture invariant 2: the public/hidden split is structural — both classes use
``extra="forbid"``. The public spec exposes the state-copy interface, budgets, and target
observable only. It never exposes the Hamiltonian family, its parameters, the temperature, the
exact moments, or the real (asymmetric) readout — those live only in the hidden config. (Edition
v3 removed the public parameter ranges and regime promise: a model drawn from them passed the
scored gates 15.5% of the time without any measurement.)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------- public ----------------


class StateCopyBudgets(_Strict):
    max_state_copies: int = Field(..., ge=1)
    max_randomized_bases: int = Field(..., ge=1)
    max_copy_block_size: int = Field(4, ge=2, le=4)
    max_jobs: int = Field(80, ge=1)


class PublicStateCopySpec(_Strict):
    """Agent-visible device specification (no hidden truth)."""

    schema_version: int = 1
    device_id: str
    qtype: Literal["trapped_ion_state_copy_randomized_measurement"] = (
        "trapped_ion_state_copy_randomized_measurement"
    )
    task_id: str | None = None

    platform: str = "trapped_ion"
    n_ions: int = 6
    ion_labels: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    prepared_state: str = (
        "unknown reproducible mixed state rho (independent identical copies on demand)"
    )
    state_promise: str = (
        "every requested copy is the same mixed state rho; copies are prepared independently"
    )
    target_observable: Literal["Z0Z1"] = "Z0Z1"
    required_powers: list[int] = Field(default_factory=lambda: [2, 3])

    supported_randomization: str = (
        "local single-qubit Pauli-basis randomized measurements (ORM / shadow); collective "
        "copy-block cyclic-shift measurements up to block size 4"
    )
    measurement_return: str = (
        "raw bitstrings for single-copy randomized measurements; aligned bounded-real joint "
        "eigenvalues for copy-block measurements; no density matrix, exact probabilities, or "
        "hidden targets"
    )
    claimed_readout_fidelity: str = "0.995 per ion (value recorded at the last calibration)"

    budgets: StateCopyBudgets
    notes: str = ""


# ---------------- hidden ----------------


class HiddenStateModel(_Strict):
    """The hidden XX-thermal-plus-depolarizing state parameters (the answer generator)."""

    J0: float = 1.0
    alpha: float
    B_over_J0: float
    delta_B_over_J0: list[float]  # length 6, hidden site-field drifts (in units of J0)
    beta_J0: float
    depolarizing_lambda: float = Field(..., ge=0.0, le=0.5)


class HiddenReadout(_Strict):
    """Per-ion asymmetric readout error (hidden; the public claim is symmetric 0.995)."""

    p01: list[float]  # P(read 1 | prepared 0), length 6
    p10: list[float]  # P(read 0 | prepared 1), length 6


class StaleStateCopyNotebook(_Strict):
    model_config = ConfigDict(extra="forbid")

    last_calibrated: str = "2026-04-18"
    linear_z0z1_claim: str = "0.50 +/- 0.018"
    rho_moment2_claim: str = "0.16"
    z0z1_vc2_claim: str = "0.55"
    z0z1_vc3_claim: str = "0.57"
    z0z1_vc4_claim: str = "0.58"
    readout_fidelity_claim: str = "0.995, symmetric"
    note: str = (
        "Earlier shadow data suggested that virtual cooling only mildly changes Z0Z1. The latest "
        "direct Z0Z1 scan gave about 0.50. Readout has historically been symmetric at ~99.5%."
    )


class HiddenStateCopyConfig(_Strict):
    schema_version: int = 1
    device_id: str
    qtype: Literal["trapped_ion_state_copy_randomized_measurement"] = (
        "trapped_ion_state_copy_randomized_measurement"
    )
    seed: int

    state: HiddenStateModel
    readout: HiddenReadout
    max_state_copies: int = 32000
    max_randomized_bases: int = 40
    max_copy_block_size: int = 4

    stale_lab_notebook: StaleStateCopyNotebook = StaleStateCopyNotebook()
