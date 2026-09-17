"""Run-long stateful runtime for the heralded-leakage decoding capability.

Three instruments on one engine:

- **calibration**: leakage memory experiments (detectors + heralds + per-shot logical
  truth) charged to the run-long shot budget, drawn from the calibration RNG stream;
- **challenge**: one fixed sealed challenge set (detectors + heralds ONLY) built in a
  single deterministic pass from a *disjoint* RNG stream, cached, and served in
  byte-identical chunks — the verifier regenerates the identical tensors through
  :func:`build_challenge` and cross-checks the served digest;
- **predictions**: canonical, digest-bound acceptance of per-shot logical-flip
  predictions (sync; no truth-derived feedback of any kind).

Seed lineage (deliberately disjoint list-seeds; magic_bell per-rep-independence lesson):
calibration draws from ``[seed, CALIB_RNG_TAG, instance_seed]`` and the challenge from
``[seed, CHALLENGE_RNG_TAG, instance_seed]``, so no calibration call sequence can
reproduce challenge bits.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from qiqcbench.qsim.qtypes.surface_code_memory import leakage as LK
from qiqcbench.qsim.qtypes.surface_code_memory import surface_code as SC
from qiqcbench.qsim.qtypes.surface_code_memory.device import (
    HiddenSurfaceCodeConfig,
    PublicSurfaceCodeSpec,
)
from qiqcbench.qsim.qtypes.surface_code_memory.wire import (
    ChallengeBatchRequest,
    JobLeakageChallengeData,
    JobLeakageMemoryData,
    LeakageMemoryRequest,
    _BudgetView,
)

CALIB_RNG_TAG = 977_101
CHALLENGE_RNG_TAG = 977_431


def base_dem_params(hidden: HiddenSurfaceCodeConfig) -> SC.HiddenDemParams:
    n = hidden.noise
    return SC.HiddenDemParams(
        p_data=n.p_data,
        p_meas=n.p_meas,
        hook_base=n.hook_base,
        hook_multipliers=list(n.hook_multipliers),
        p_leak=n.p_leak,
        leak_pairs=[(lp.s1, lp.s2, lp.dt, int(lp.flips_observable)) for lp in n.leak_pairs],
    )


def leak_params(hidden: HiddenSurfaceCodeConfig) -> LK.HeraldedLeakageParams:
    hl = hidden.heralded_leakage
    if hl is None:
        raise ValueError("device has no hidden heralded_leakage parameters")
    n_data = hidden.distance * hidden.distance
    fp = np.full(n_data, hl.herald_false_positive, dtype=float)
    for q in hl.fp_hot_qubits:
        fp[q] = hl.fp_hot_rate
    return LK.HeraldedLeakageParams(
        p_leak_per_lru=hl.p_leak_per_lru,
        b_self=hl.b_self,
        b_meas=hl.b_meas,
        b_partner=hl.b_partner,
        b_decay_second_round=hl.b_decay_second_round,
        herald_false_negative=hl.herald_false_negative,
        herald_false_positive=fp,
    )


# Task-local evidence schema: versionless events are the historical
# pre-commitment shape; version 1 events must carry the opaque instance
# commitment recomputed and exactly matched by the private scorer.
HERALDED_EVIDENCE_SCHEMA_VERSION = 1
HERALDED_INSTANCE_COMMITMENT_NAME = "qiqcbench_heralded_leakage_instance_hmac_v1"
_HERALDED_INSTANCE_KEY_NAME = "qiqcbench_heralded_leakage_instance_key_v1"
_CHALLENGE_SET_HMAC_NAME = "qiqcbench_heralded_leakage_challenge_set_hmac_v1"
_CHALLENGE_SET_KEY_NAME = "qiqcbench_heralded_leakage_challenge_set_key_v1"


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def instance_evidence_commitment(
    hidden: HiddenSurfaceCodeConfig, *, task_id: str | None, instance_seed: int
) -> dict[str, Any]:
    """Opaque equality proof binding evidence to the graded run inputs.

    This task's runtime injects no execution-scoped commitment secret, so the
    HMAC key is derived from the full canonical hidden device dump instead: the
    preimage space is the entire hidden model, never the brute-forceable
    (hidden.seed, instance_seed) pair, so the agent-readable digest reveals
    neither seed.  Deterministic for one env+config so honest rescoring agrees.
    """
    key_envelope = {
        "name": _HERALDED_INSTANCE_KEY_NAME,
        "schema_version": 1,
        "task_id": task_id,
        "hidden_device": hidden.model_dump(mode="json"),
    }
    key = hashlib.sha256(_canonical_json_bytes(key_envelope)).digest()
    envelope = {
        "name": HERALDED_INSTANCE_COMMITMENT_NAME,
        "schema_version": 1,
        "task_id": task_id,
        "instance_seed": int(instance_seed),
    }
    digest = hmac.new(key, _canonical_json_bytes(envelope), hashlib.sha256).hexdigest()
    return {
        "name": HERALDED_INSTANCE_COMMITMENT_NAME,
        "schema_version": 1,
        "hmac_sha256": digest,
    }


def challenge_set_id(hidden: HiddenSurfaceCodeConfig, instance_seed: int) -> str:
    """Seed-free sealed-set identity: the v1 format embedded the raw
    graded instance seed (``inst{seed}``) into an agent-visible identifier.

    The suffix is an HMAC keyed from the full canonical hidden device dump
    (same preimage argument as :func:`instance_evidence_commitment`, own
    domain labels; no task_id because this derivation lives below the task
    layer, shared by the engine and the verifier's pure regeneration), so it
    is deterministic across restarts of one env+config and distinct across
    instance seeds without revealing them.  Scoring compares the cited and
    served strings byte-for-byte and never parses the format.
    """
    key_envelope = {
        "name": _CHALLENGE_SET_KEY_NAME,
        "schema_version": 1,
        "hidden_device": hidden.model_dump(mode="json"),
    }
    key = hashlib.sha256(_canonical_json_bytes(key_envelope)).digest()
    envelope = {
        "name": _CHALLENGE_SET_HMAC_NAME,
        "schema_version": 1,
        "instance_seed": int(instance_seed),
    }
    digest = hmac.new(key, _canonical_json_bytes(envelope), hashlib.sha256).hexdigest()
    return f"{hidden.device_id}:challenge_v2:{digest[:16]}"


@dataclass
class ChallengeData:
    set_id: str
    digest: str
    rounds: int
    syn: np.ndarray  # [shots, n_det] uint8
    heralds: np.ndarray  # [shots, n_data, n_slots] uint8
    obs: np.ndarray  # [shots] uint8 — withheld truth (never serialized agent-side)


def build_challenge(
    hidden: HiddenSurfaceCodeConfig, public: PublicSurfaceCodeSpec, instance_seed: int
) -> ChallengeData:
    """One full deterministic pass; the engine and the verifier call this identically."""
    spec = public.heralded_leakage
    if spec is None:
        raise ValueError("device has no public heralded_leakage instrument")
    geo = SC.build_geometry(hidden.distance)
    adj = LK.build_adjacency(geo)
    dem = SC.build_hidden_dem(geo, base_dem_params(hidden), spec.challenge_rounds)
    rng = np.random.default_rng([hidden.seed, CHALLENGE_RNG_TAG, instance_seed])
    syn, heralds, obs = LK.sample_leakage_run(
        geo, adj, dem, leak_params(hidden), spec.challenge_rounds, spec.challenge_shots, rng
    )
    return ChallengeData(
        set_id=challenge_set_id(hidden, instance_seed),
        digest=LK.challenge_digest(syn, heralds),
        rounds=spec.challenge_rounds,
        syn=syn,
        heralds=heralds,
        obs=obs,
    )


class LeakageEngine:
    def __init__(
        self,
        hidden: HiddenSurfaceCodeConfig,
        public: PublicSurfaceCodeSpec,
        instance_seed: int,
    ):
        spec = public.heralded_leakage
        if spec is None:
            raise ValueError("device has no public heralded_leakage instrument")
        self._hidden = hidden
        self._public = public
        self._spec = spec
        self._instance_seed = instance_seed
        self._lock = threading.Lock()
        self._geo = SC.build_geometry(hidden.distance)
        self._adj = LK.build_adjacency(self._geo)
        self._base = base_dem_params(hidden)
        self._leak = leak_params(hidden)
        self._rng = np.random.default_rng([hidden.seed, CALIB_RNG_TAG, instance_seed])
        self._shots_cap = public.budgets.shot_budget
        self._shots_used = 0
        self._dem_cache: dict[int, list[tuple[int, int, float, int]]] = {}
        self._challenge: ChallengeData | None = None
        self._predictions: dict[int, np.ndarray] = {}

    @property
    def instance_seed(self) -> int:
        return self._instance_seed

    @property
    def shots_used(self) -> int:
        return self._shots_used

    def _dem(self, rounds: int) -> list[tuple[int, int, float, int]]:
        if rounds not in self._dem_cache:
            self._dem_cache[rounds] = SC.build_hidden_dem(self._geo, self._base, rounds)
        return self._dem_cache[rounds]

    def _budget_view(self) -> _BudgetView:
        return _BudgetView(shots_used=self._shots_used, shots_cap=self._shots_cap)

    # ---------------- calibration ----------------

    def run_calibration(self, request: LeakageMemoryRequest) -> JobLeakageMemoryData | str:
        with self._lock:
            if request.rounds > self._spec.calibration_rounds_max:
                return (
                    f"rounds {request.rounds} exceeds calibration_rounds_max "
                    f"{self._spec.calibration_rounds_max}"
                )
            if request.shots > self._spec.max_shots_per_call:
                return (
                    f"shots {request.shots} exceeds max_shots_per_call "
                    f"{self._spec.max_shots_per_call}"
                )
            if self._shots_used + request.shots > self._shots_cap:
                return (
                    f"shot budget exhausted: {self._shots_used}+{request.shots} "
                    f"> cap {self._shots_cap}"
                )
            rounds = request.rounds
            syn, heralds, obs = LK.sample_leakage_run(
                self._geo,
                self._adj,
                self._dem(rounds),
                self._leak,
                rounds,
                request.shots,
                self._rng,
            )
            self._shots_used += request.shots
            return JobLeakageMemoryData(
                rounds=rounds,
                n_detectors=self._geo.n_checks * rounds,
                shots=request.shots,
                n_herald_slots=LK.n_herald_slots(rounds),
                detection_events_b64=LK.pack_bits_b64(syn),
                herald_events_b64=LK.pack_bits_b64(heralds),
                logical_outcomes_b64=LK.pack_bits_b64(obs),
                budget=self._budget_view(),
            )

    # ---------------- challenge ----------------

    def _challenge_data(self) -> ChallengeData:
        if self._challenge is None:
            self._challenge = build_challenge(self._hidden, self._public, self._instance_seed)
        return self._challenge

    def n_chunks(self) -> int:
        return self._spec.challenge_shots // self._spec.challenge_chunk_shots

    def challenge_chunk(self, request: ChallengeBatchRequest) -> JobLeakageChallengeData | str:
        with self._lock:
            n_chunks = self.n_chunks()
            if request.chunk_index >= n_chunks:
                return f"chunk_index {request.chunk_index} out of range [0, {n_chunks})"
            ch = self._challenge_data()
            cs = self._spec.challenge_chunk_shots
            start = request.chunk_index * cs
            return JobLeakageChallengeData(
                challenge_set_id=ch.set_id,
                challenge_digest=ch.digest,
                chunk_index=request.chunk_index,
                n_chunks=n_chunks,
                chunk_shots=cs,
                start_shot=start,
                rounds=ch.rounds,
                n_detectors=ch.syn.shape[1],
                n_herald_slots=ch.heralds.shape[2],
                detection_events_b64=LK.pack_bits_b64(ch.syn[start : start + cs]),
                herald_events_b64=LK.pack_bits_b64(ch.heralds[start : start + cs]),
            )

    # ---------------- predictions ----------------

    def accept_predictions(self, chunk_index: int, predictions_b64: str) -> dict:
        """Canonicalize + record one prediction chunk (last submission per chunk wins).

        Returns the evidence payload. Raises ValueError on malformed input. The return
        carries NOTHING truth-derived — digests, coverage, and chunk bookkeeping only.
        """
        with self._lock:
            n_chunks = self.n_chunks()
            if not (0 <= chunk_index < n_chunks):
                raise ValueError(f"chunk_index {chunk_index} out of range [0, {n_chunks})")
            cs = self._spec.challenge_chunk_shots
            bits = LK.unpack_bits_b64(predictions_b64, cs)
            self._predictions[chunk_index] = bits.astype(np.uint8)
            accepted = sorted(self._predictions)
            combined = np.concatenate([self._predictions[i] for i in accepted])
            ch = self._challenge_data()
            return {
                "challenge_set_id": ch.set_id,
                "chunk_index": chunk_index,
                "chunk_shots": cs,
                "predictions_b64": LK.pack_bits_b64(bits),
                "chunk_sha256": LK.predictions_digest(bits),
                "chunks_accepted": accepted,
                "coverage_shots": int(len(combined)),
                "combined_sha256": LK.predictions_digest(combined),
                "complete": len(accepted) == n_chunks,
            }


__all__ = [
    "CALIB_RNG_TAG",
    "CHALLENGE_RNG_TAG",
    "HERALDED_EVIDENCE_SCHEMA_VERSION",
    "HERALDED_INSTANCE_COMMITMENT_NAME",
    "ChallengeData",
    "LeakageEngine",
    "base_dem_params",
    "build_challenge",
    "challenge_set_id",
    "instance_evidence_commitment",
    "leak_params",
]
