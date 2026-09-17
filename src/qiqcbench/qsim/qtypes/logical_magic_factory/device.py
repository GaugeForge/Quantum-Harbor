"""Public + hidden device config types for the logical_magic_factory qtype.

A ``logical_magic_factory`` device is a black-box fault-tolerant Steane-code
([[7,1,3]]) magic-state factory (Goto, Sci. Rep. 6, 19578, 2016):
distillation-free preparation of the logical |H> magic state via a
fault-tolerant measurement of the encoded Hadamard operator plus
error-detecting teleportation. The agent draws noisy logical |H> copies from
the factory -- each charged against a run-long magic-state budget -- applies
transversal logical Clifford twirls and an optional two-copy joint
(Bell/SWAP-test) measurement, and must recover the factory's logical
infidelity epsilon = 1 - F to multiplicative precision.

Public / hidden are STRUCTURALLY separated: both
classes use ``extra="forbid"``. The PUBLIC spec advertises the logical control
set and *claimed* (not true) infidelity/acceptance ranges only; the true
noisy-state parameters (coherent rotation, depolarizing rate, syndrome
acceptance) and the emergent epsilon live only in the hidden config inside
qsim.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Cultivation line (optional second device family) ----------


class PublicCultivationSpec(_Strict):
    """Public control surface of the cultivation line (magic_cultivation_v0).

    Structure is fully disclosed; every error/retention number is a *claimed*
    value, never truth. The physical narrative (54 physical qubits, d=3 color
    code grafting into d=5) is descriptive dressing -- the engine is a
    logical-effective model.
    """

    stage_description: str = (
        "Cultivation line: non-fault-tolerant magic-state injection into a "
        "d=3 color-code patch (54-physical-qubit processor), repeated "
        "fault-tolerant H_L kickback cultivation rounds with interleaved "
        "d=3 QEC cycles, then escape/graft into a d=5 patch running "
        "escape_cycle_n cycles before delivery."
    )
    injection_theta_range_rad: tuple[float, float] = (0.0, 1.5707963267948966)
    cultivation_rounds_range: tuple[int, int] = (0, 4)
    qec_cycles_per_round_range: tuple[int, int] = (0, 4)
    escape_cycle_n_range: tuple[int, int] = (1, 8)
    detector_groups: list[str] = Field(
        default_factory=lambda: ["injection", "cultivation", "qec", "graft"]
    )
    measure_axes: list[str] = Field(
        default_factory=lambda: ["target_axis", "x", "y", "z"],
        description=(
            "Terminal logical measurement axis. 'target_axis' measures along "
            "the fixed ideal |H> preparation axis (injection angle pi/4), "
            "whatever injection angle the point requests; x/y/z are logical "
            "Pauli axes."
        ),
    )
    injection_budget: int = Field(50_000, ge=1, description="Total injection attempts for the run.")
    claimed_delivered_fidelity: str = "historically >= 0.9995 (see lab notebook)"
    claimed_retention: float = 0.45
    pass_ceiling_delivered_infidelity: float = Field(
        1.18e-2,
        description=(
            "Published pass line: the true delivered infidelity of the "
            "submitted schedule+mask must be at or below this."
        ),
    )
    pass_min_advantage: float = Field(
        4.0,
        description=(
            "Published pass line: required improvement factor over the raw "
            "injection-only infidelity."
        ),
    )
    min_accepted_verification_shots: int = Field(
        1500,
        description=(
            "Published evidence floor: mask-accepted shots measured on the "
            "submitted schedule needed to support the reported infidelity."
        ),
    )


class HiddenCultivationParams(_Strict):
    """Hidden per-stage truth of the cultivation line. Lives only inside qsim."""

    lam_inj: float
    d_inj: float
    c_inj: float
    delta_inj_rad: float
    lam_cult: float
    d_cult: float
    c_cult: float
    kappa_cult: float
    g_cult: float
    s_cult_good: float
    s_cult_bad: float
    c_cult_bad_scale: float
    lam_q3: float
    d_q3: float
    c_q3: float
    lam_ext: float
    d_ext: float
    c_ext: float
    graft_transient_rad: float
    graft_ringdown: float
    lam_q5: float
    d_q5: float
    c_q5: float
    drift_sigma_rad: float
    drift_ar_rho: float
    readout_p00: float
    readout_p11: float
    injection_budget: int = Field(50_000, ge=1)


# ---------- Public spec (visible to agent) ----------


class PublicMagicFactorySpec(_Strict):
    """Hardware-manual-style public spec for the logical magic-factory qtype."""

    schema_version: int = 1
    device_id: str
    qtype: Literal["logical_magic_factory"] = "logical_magic_factory"

    code: str = (
        "[[7,1,3]] Steane code (distance 3, detects up to 2 errors); all Clifford gates transversal"
    )
    factory_description: str = (
        "Distillation-free fault-tolerant preparation of the logical |H> "
        "magic state a la Goto (Sci. Rep. 6, 19578, 2016): a non-fault-"
        "tolerant magic-state encoding, a fault-tolerant measurement of the "
        "encoded Hadamard operator, then error-detecting teleportation with "
        "syndrome-free post-selection. ~25 physical qubits per emitted "
        "logical magic state."
    )
    ideal_target_state: str = (
        "Logical |H> = cos(pi/8)|0> + sin(pi/8)|1>, the +1 eigenstate of the "
        "logical Hadamard (Bloch vector (1/sqrt2, 0, 1/sqrt2))."
    )
    logical_operations: list[str] = Field(
        default_factory=lambda: [
            "H",
            "S",
            "X",
            "Y",
            "Z",
            "CNOT (between two logical copies)",
            "logical Pauli (computational-basis / X-basis / Y-basis) measurement",
        ]
    )
    twirl_primitive: str = (
        "Per-shot logical Clifford twirl over the |H> stabilizer group "
        "{I, H} (the transversal logical Hadamard), available as a "
        "twirl=true convenience flag on each drawn copy."
    )
    joint_measurement: str = (
        "Optional two-copy joint measurement: a fixed CNOT + logical-"
        "Hadamard + computational-basis-readout circuit applied to two "
        "drawn copies (n_copies=2)."
    )
    syndrome_note: str = (
        "Each consumed copy returns a per-shot syndrome flag "
        "(0=clean, 1=flagged); whether to post-select on it is the "
        "agent's choice."
    )
    magic_state_budget: int = Field(
        8000, ge=1, description="Public; the binding resource for the whole run."
    )
    claimed_logical_infidelity_range: tuple[float, float] = (1e-3, 2e-2)
    claimed_syndrome_acceptance: float = 0.90
    measurement_return: Literal["records"] = "records"
    max_shots: int = 100_000
    cultivation: PublicCultivationSpec | None = Field(
        None,
        description=(
            "Present only on cultivation-line devices (magic_cultivation_v0); "
            "None on the plain black-box factory."
        ),
    )
    notes: str = (
        "The factory emits logical |H> magic states on a [[7,1,3]] Steane "
        "code, prepared fault-tolerantly. Each logical magic state you "
        "consume is charged against a fixed budget; plan accordingly. "
        "The notebook fidelity predates the "
        "current factory and verification pass. Estimate the logical "
        "infidelity epsilon to multiplicative precision and report a "
        "calibrated 1-sigma error."
    )


# ---------- Hidden truth (only inside qsim) ----------


class StaleMagicNotebookEntry(_Strict):
    """Optional stale/optimistic calibration written into the lab notebook."""

    last_calibrated: str | None = None
    logical_fidelity_claim: float | None = None
    recommended_protocol: str | None = None
    recommended_shots: int | None = None
    syndrome_acceptance_claim: float | None = None
    readout_fidelity_claim: float | None = None
    # Cultivation-line stale entries (magic_cultivation_v0 only).
    noise_model_claim: str | None = None
    recommended_verification: str | None = None
    escape_default_n: int | None = None
    cultivation_rounds_default: int | None = None
    retention_claim: float | None = None
    notes: str = ""


class HiddenMagicFactoryConfig(_Strict):
    """Hidden truth for the logical_magic_factory qtype. Lives only inside qsim.

    The logical infidelity epsilon (the task answer) is NOT stored here as a
    constant: it is computed exactly (closed-form, from the constructed
    accepted-copy density matrix) at engine-build time, emergent from
    ``delta_rad`` / ``lambda_dep``.
    """

    schema_version: int = 1
    device_id: str
    qtype: Literal["logical_magic_factory"] = "logical_magic_factory"
    seed: int

    # Accepted-copy (syndrome-clean) noisy-state parameters: rho_acc =
    # D_lambda_dep(R_delta_rad |H><H| R_delta_rad^dagger).
    delta_rad: float = Field(
        ..., description="Residual coherent over-rotation (rad) for accepted copies."
    )
    lambda_dep: float = Field(
        ..., ge=0.0, le=1.0, description="Depolarizing parameter for accepted copies."
    )

    # Flagged/rejected-copy noisy-state parameters (worse than accepted).
    delta_rej_rad: float = Field(
        ..., description="Coherent over-rotation (rad) for flagged copies."
    )
    lambda_dep_rej: float = Field(
        ..., ge=0.0, le=1.0, description="Depolarizing parameter for flagged copies."
    )

    syndrome_acceptance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of drawn copies that are syndrome-clean (accepted).",
    )
    readout_confusion_diag: tuple[float, float] = Field(
        ..., description="(P(measure 0 | prepared 0), P(measure 1 | prepared 1)); asymmetric."
    )
    magic_state_budget: int = Field(8000, ge=1)

    cultivation: HiddenCultivationParams | None = Field(
        None, description="Present only on cultivation-line devices."
    )

    stale_lab_notebook: StaleMagicNotebookEntry = StaleMagicNotebookEntry()


__all__ = [
    "HiddenCultivationParams",
    "HiddenMagicFactoryConfig",
    "PublicCultivationSpec",
    "PublicMagicFactorySpec",
    "StaleMagicNotebookEntry",
]
