"""blackbox_boundedgate_circuit control + result wire schemas (qtype-local).

One experiment primitive, run-long budgeted:

- ``run_basis_shots`` — a batched job of measurement blocks. Each block picks
  one input ``x`` for the hidden bounded-gate circuit and one or more
  measurement settings; each setting is either a fixed length-n per-qubit
  basis string over {x, y, z} or the literal ``random_pauli`` (per-shot
  uniformly random per-qubit bases, returned alongside the outcomes). The
  device returns raw per-shot bitstrings — never expectation values, never
  the circuit.

Budget accounting (total shots, jobs) accumulates across every call in the
run and is echoed back in each result; the evidence-log copy is authoritative
for the verifier. Requests are admitted atomically: an over-budget or
malformed request is rejected as a whole and consumes nothing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from qiqcbench.qsim.core.wire import SCHEMA_VERSION, _Strict, _StrictRequest

RANDOM_PAULI = "random_pauli"


class BasisShotSetting(_StrictRequest):
    """One measurement setting: a fixed per-qubit basis string or random_pauli."""

    basis: str = Field(
        ...,
        pattern=r"^(random_pauli|[xyz]+)$",
        description="Length-n string over {x,y,z}, or 'random_pauli'.",
    )
    shots: int = Field(..., ge=1, le=100_000)


class BasisShotBlock(_StrictRequest):
    """One circuit input plus the settings to measure it in."""

    x: list[float] = Field(..., min_length=1, max_length=64)
    # No separate per-block settings limit: the only settings envelope an
    # agent must respect is the public per-job total (``max_settings_per_job``,
    # enforced by the engine). This ingress bound is a transport sanity cap
    # sized to the largest disclosed per-job total (192) so a single block may
    # carry a whole job's settings.
    settings: list[BasisShotSetting] = Field(..., min_length=1, max_length=192)


class BasisShotRequest(_StrictRequest):
    schema_version: int = SCHEMA_VERSION
    blocks: list[BasisShotBlock] = Field(..., min_length=1, max_length=64)


class BasisShotBudgetView(_Strict):
    """Cumulative run-long budget echoed in every result.

    ``elapsed_wall_clock_s`` is the device's own clock: seconds since the
    run-long engine started (shortly before the agent's first call). It is
    instrument state reported like ``shots_used``; ``None`` only in evidence
    recorded by engines that predate the clock.
    """

    shots_used: int
    total_shot_budget: int
    jobs_used: int
    max_jobs: int
    elapsed_wall_clock_s: float | None = Field(None, ge=0.0)


class BasisShotSettingResult(_Strict):
    basis: str
    shots: int
    # Raw per-shot outcomes; character i of each string is qubit i (0 / 1).
    bitstrings: list[str]
    # For random_pauli settings only: per-shot basis strings, aligned with bitstrings.
    random_bases: list[str] | None = None


class BasisShotBlockResult(_Strict):
    x: list[float]
    settings: list[BasisShotSettingResult]


class JobBasisShotData(_Strict):
    kind: Literal["basis_shots"] = "basis_shots"
    blocks: list[BasisShotBlockResult]
    budget: BasisShotBudgetView


class SealedTargetChallengeReveal(_Strict):
    """Synchronous receipt returned when measurements are irreversibly sealed."""

    schema_version: Literal[1] = 1
    challenge_id: str
    commitment_scheme: Literal["qiqcbench_sealed_target_sha256_v1"] = (
        "qiqcbench_sealed_target_sha256_v1"
    )
    target_inputs: list[list[float]]
    target_inputs_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    target_commitment_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    commitment_nonce: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    shots_used: int = Field(..., ge=0)
    jobs_used: int = Field(..., ge=0)
    last_admission_sequence: int = Field(..., ge=0)
    # Device clock at the irreversible seal (frozen; repeats return the same
    # receipt). Live time keeps arriving with every completed result's budget.
    elapsed_wall_clock_s: float | None = Field(None, ge=0.0)
    is_repeat: bool = False


__all__ = [
    "RANDOM_PAULI",
    "BasisShotSetting",
    "BasisShotBlock",
    "BasisShotRequest",
    "BasisShotBudgetView",
    "BasisShotSettingResult",
    "BasisShotBlockResult",
    "JobBasisShotData",
    "SealedTargetChallengeReveal",
]
