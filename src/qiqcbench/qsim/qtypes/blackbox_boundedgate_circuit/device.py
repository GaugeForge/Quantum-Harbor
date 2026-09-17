"""Public + hidden device config types for the blackbox bounded-gate qtype.

A fixed hidden n-qubit circuit U(x) built from Clifford gates plus a bounded
number of single-qubit non-Clifford rotations driven by a small input vector
x. The agent chooses inputs, per-qubit measurement bases, and shot counts
under a hard cumulative shot/job budget; it never sees the circuit.

Public and hidden are STRUCTURALLY separated: both use
``extra="forbid"`` so a hidden-truth field can never be absorbed by the public
spec and vice versa. The hidden lever is the entire circuit instance (rotation
placements/axes/angle assignment, Clifford scaffold) plus the per-qubit
measurement-noise process; the public spec discloses only the qubit/input
dimensions, rotation and harmonic bounds, and resource budgets.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.circuit import CircuitOp


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Public spec (visible to agent) ----------


class BoundedgateBudget(_Strict):
    """Hard per-call and cumulative resource budgets enforced by qsim."""

    total_shot_budget: int = Field(..., gt=0)
    max_jobs: int = Field(..., gt=0)
    max_blocks_per_job: int = Field(8, gt=0)
    max_settings_per_job: int = Field(24, gt=0, description="Total across a job's blocks.")
    max_shots_per_setting: int = Field(4000, gt=0)
    max_ingress_request_bytes: int = Field(..., gt=0)
    max_metadata_calls: int = Field(..., gt=0)
    max_job_result_polls: int = Field(..., gt=0)
    max_sealed_holdout_reveals: int = Field(..., gt=0)
    max_final_answer_serialized_bytes: int = Field(..., gt=0)
    max_final_answer_submissions: int = Field(..., gt=0)
    max_answer_string_characters: int = Field(..., gt=0)


SEALED_TARGET_COMMITMENT_SCHEME = "qiqcbench_sealed_target_sha256_v1"
SEALED_TARGET_INPUTS_DOMAIN = b"qiqcbench/sealed-target-inputs/v1\0"
SEALED_TARGET_COMMITMENT_DOMAIN = b"qiqcbench/sealed-target-commitment/v1\0"


def canonical_target_inputs_sha256(target_inputs: list[list[float]]) -> str:
    """Digest an ordered target-coordinate payload using canonical JSON."""
    payload = json.dumps(
        target_inputs,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(SEALED_TARGET_INPUTS_DOMAIN + payload).hexdigest()


def sealed_target_commitment_sha256(
    *, challenge_id: str, target_inputs_sha256: str, commitment_nonce: str
) -> str:
    """Commit to a target digest under a challenge domain and 256-bit opening."""
    envelope = {
        "challenge_id": challenge_id,
        "commitment_nonce": commitment_nonce,
        "target_inputs_sha256": target_inputs_sha256,
    }
    payload = json.dumps(
        envelope,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(SEALED_TARGET_COMMITMENT_DOMAIN + payload).hexdigest()


class PublicSealedTargetChallenge(_Strict):
    """Opaque precommitment for a target set revealed only after measurement seal."""

    schema_version: Literal[1] = 1
    challenge_id: str = Field(..., min_length=1, max_length=128)
    commitment_scheme: Literal["qiqcbench_sealed_target_sha256_v1"] = (
        SEALED_TARGET_COMMITMENT_SCHEME
    )
    target_count: int = Field(..., gt=0)
    input_dimension: int = Field(..., gt=0)
    target_commitment_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")


class PublicBoundedgateSpec(_Strict):
    """Hardware-manual-style public spec for the blackbox bounded-gate qtype."""

    schema_version: Literal[3]
    device_id: str
    qtype: Literal["blackbox_boundedgate_circuit"] = "blackbox_boundedgate_circuit"
    n_qubits: int = Field(..., gt=1)
    input_dimension: int = Field(..., gt=0, description="Length d of the input vector x.")
    max_rotation_gates: int = Field(
        ..., gt=0, description="Disclosed upper bound R on non-Clifford rotation gates."
    )
    max_harmonic_degree_per_angle: int = Field(
        ...,
        gt=0,
        description=(
            "Disclosed structural promise: every measured expectation value is a "
            "trigonometric polynomial in x with at most this per-variable harmonic degree."
        ),
    )
    budget: BoundedgateBudget
    target_challenge: PublicSealedTargetChallenge | None = None
    measurement_return: Literal["bitstring"] = "bitstring"
    notes: str = (
        "The circuit is fixed for the whole run. Internal measurement behavior is "
        "guaranteed only as stated by the task. Measurement settings are per-qubit "
        "Pauli bases (or per-shot randomized bases)."
    )

    @model_validator(mode="after")
    def _check_sealed_challenge_schema_version(self) -> PublicBoundedgateSpec:
        if (
            self.target_challenge is not None
            and self.target_challenge.input_dimension != self.input_dimension
        ):
            raise ValueError("target_challenge input_dimension must match the public device")
        return self


# ---------- Hidden truth (only inside qsim) ----------


class StaleBoundedgateNotebookEntry(_Strict):
    """Stale characterization claims surfaced via get_lab_notebook."""

    previous_fit_summary: str | None = None
    suggested_degrees: list[int] | None = None
    suggested_protocol: str | None = None
    last_characterized: str | None = None
    notes: str | None = None


class HiddenSealedTargetChallenge(_Strict):
    """Qsim-only target payload and opening for a public sealed challenge."""

    schema_version: Literal[1] = 1
    challenge_id: str = Field(..., min_length=1, max_length=128)
    commitment_scheme: Literal["qiqcbench_sealed_target_sha256_v1"] = (
        SEALED_TARGET_COMMITMENT_SCHEME
    )
    target_inputs: list[list[float]] = Field(..., min_length=1)
    target_inputs_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    target_commitment_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    commitment_nonce: str = Field(
        ...,
        pattern=r"^[0-9a-f]{64}$",
        description="A 256-bit opening encoded as 64 lowercase hexadecimal characters.",
    )

    @model_validator(mode="after")
    def _check_commitment(self) -> HiddenSealedTargetChallenge:
        if any(not row for row in self.target_inputs):
            raise ValueError("every sealed target input must be non-empty")
        if any(not math.isfinite(value) for row in self.target_inputs for value in row):
            raise ValueError("sealed target inputs must contain only finite values")
        inputs_digest = canonical_target_inputs_sha256(self.target_inputs)
        if inputs_digest != self.target_inputs_sha256:
            raise ValueError("target_inputs_sha256 does not match target_inputs")
        commitment = sealed_target_commitment_sha256(
            challenge_id=self.challenge_id,
            target_inputs_sha256=inputs_digest,
            commitment_nonce=self.commitment_nonce,
        )
        if commitment != self.target_commitment_sha256:
            raise ValueError("target_commitment_sha256 does not match the sealed target opening")
        return self


class HiddenBoundedgateConfig(_Strict):
    """Hidden truth for the blackbox bounded-gate qtype. Lives only inside qsim."""

    schema_version: int = 1
    device_id: str
    qtype: Literal["blackbox_boundedgate_circuit"] = "blackbox_boundedgate_circuit"
    seed: int
    input_dimension: int = Field(..., gt=0)
    circuit: list[CircuitOp] = Field(..., min_length=1)
    depolarizing_flip_prob: list[float] = Field(
        ...,
        min_length=2,
        description=(
            "Per-qubit outcome-flip probability e_i (== depolarizing strength 2*e_i "
            "before measurement) at the START of a run; its length defines the "
            "qubit count."
        ),
    )
    readout_drift_per_qubit: list[float] | None = Field(
        None,
        description=(
            "Optional per-qubit readout drift d_i. The effective flip probability "
            "is e_i(u) = e_i + d_i * u, where u in [0, 1] is the fraction of the "
            "run's total shot budget already consumed when the job was admitted "
            "(a usage clock, not wall time). u = 0 for the first job, so the "
            "device starts exactly at its declared e_i calibration. Omitted or "
            "all-zero means a stationary device."
        ),
    )
    max_bond: int = Field(
        128,
        gt=0,
        description="Exact-MPS bond-dimension guard; exceeding it fails the run loudly.",
    )
    target_challenge: HiddenSealedTargetChallenge | None = None
    stale_lab_notebook: StaleBoundedgateNotebookEntry = StaleBoundedgateNotebookEntry()
    edition_binding: dict[str, str] | None = Field(
        None,
        description=(
            "Optional task-edition binding digests (opaque to the engine). Because the "
            "runtime hidden-model commitment is an HMAC over this whole config, digests "
            "recorded here (e.g. the task instance digest and the scorer policy digest) "
            "are transitively execution-bound; the task verifier recomputes and checks "
            "them against its baked materials."
        ),
    )

    @property
    def n_qubits(self) -> int:
        return len(self.depolarizing_flip_prob)

    @model_validator(mode="after")
    def _check_instance(self) -> HiddenBoundedgateConfig:
        if self.target_challenge is not None and self.schema_version != 2:
            raise ValueError("target_challenge requires hidden device schema_version 2")
        n = len(self.depolarizing_flip_prob)
        for e in self.depolarizing_flip_prob:
            if not 0.0 <= e < 0.5:
                raise ValueError("depolarizing_flip_prob entries must lie in [0, 0.5)")
        if self.readout_drift_per_qubit is not None:
            if len(self.readout_drift_per_qubit) != n:
                raise ValueError("readout_drift_per_qubit must have one entry per qubit")
            # The drifted probability must stay a probability over the whole run.
            for e, d in zip(self.depolarizing_flip_prob, self.readout_drift_per_qubit, strict=True):
                if not 0.0 <= e + d < 0.5:
                    raise ValueError(
                        "drifted flip probability e_i + d_i must lie in [0, 0.5) at u = 1"
                    )
        for op in self.circuit:
            hi = max(op.q, op.q2 if op.q2 is not None else 0)
            if hi >= n:
                raise ValueError(f"circuit op touches qubit {hi} >= n_qubits {n}")
            if op.gate == "rot" and op.angle is not None and op.angle >= self.input_dimension:
                raise ValueError(
                    f"rot angle index {op.angle} >= input_dimension {self.input_dimension}"
                )
        if self.target_challenge is not None:
            for row in self.target_challenge.target_inputs:
                if len(row) != self.input_dimension:
                    raise ValueError(
                        f"every sealed target input must have length {self.input_dimension}"
                    )
                if any(not -math.pi <= value <= math.pi for value in row):
                    raise ValueError("every sealed target component must lie in [-pi, pi]")
        return self


__all__ = [
    "BoundedgateBudget",
    "HiddenSealedTargetChallenge",
    "HiddenBoundedgateConfig",
    "PublicSealedTargetChallenge",
    "PublicBoundedgateSpec",
    "SEALED_TARGET_COMMITMENT_SCHEME",
    "SEALED_TARGET_COMMITMENT_DOMAIN",
    "SEALED_TARGET_INPUTS_DOMAIN",
    "StaleBoundedgateNotebookEntry",
    "canonical_target_inputs_sha256",
    "sealed_target_commitment_sha256",
]
