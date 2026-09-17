"""Causal role-level located-erasure control for two Steane memories.

The runtime tracks exact erasure support and separates carrier occupancy from
quantum damage. A replacement fills a carrier; only exact-code recovery can
clear damage. Task-local private populations and score thresholds remain under
``hidden_dynamics``.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import threading
import zlib
from dataclasses import dataclass, field
from typing import Any

from qiqcbench.qsim.qtypes.neutral_atom_logical_processor.capabilities.located_erasure_repair.wire import (
    CONTROLLER_FIELD_LIMITS,
    CausalDecisionList,
    ControllerPredicate,
    ErasureRepairEpisodeRequest,
    JobErasureRepairEpisodeData,
)
from qiqcbench.qsim.qtypes.neutral_atom_logical_processor.device import (
    HiddenNeutralAtomConfig,
    PublicNeutralAtomSpec,
)
from qiqcbench.qsim.qtypes.neutral_atom_logical_processor.steane import (
    N_DATA,
    steane_erasure_correctable,
)

_ALL_ROLES = (1 << N_DATA) - 1
TRACE_SCHEMA_VERSION = 2


class RawTraceSizeError(ValueError):
    """A valid public request produced a trace above the declared byte cap."""


def canonical_controller(controller: CausalDecisionList) -> dict[str, Any]:
    """Return behavior-relevant canonical controller data.

    Rule and preference order remain behavioral. Predicate order within one
    conjunction is canonicalized because logical conjunction is commutative.
    """

    rules = []
    for rule in controller.rules:
        predicates = sorted(
            (predicate.model_dump(mode="json") for predicate in rule.when_all),
            key=lambda item: (item["field"], item["op"], item["value"]),
        )
        rules.append({"when_all": predicates, "preference": list(rule.preference)})
    return {
        "schema_version": controller.schema_version,
        "kind": controller.kind,
        "rules": rules,
        "default_preference": list(controller.default_preference),
    }


def controller_digest(controller: CausalDecisionList) -> str:
    payload = json.dumps(
        canonical_controller(controller),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def episode_request_digest(request: ErasureRepairEpisodeRequest) -> str:
    """Bind the complete validated public request, including controller behavior."""

    payload = json.dumps(
        request.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _strict_int(value: Any, *, name: str, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def _require_exact_keys(value: Any, keys: set[str], *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} does not match the exact raw trace schema")
    return value


def _validate_trace_event(event: Any, *, horizon: int) -> None:
    if not isinstance(event, dict) or not isinstance(event.get("event"), str):
        raise ValueError("trace event must be an object with an event kind")
    kind = event["event"]
    schemas = {
        "transport_ack": {"epoch", "event", "block", "role", "success"},
        "fixed_recovery_complete": {"epoch", "event", "block"},
        "recovery_output_loss": {"epoch", "event", "block", "role"},
        "ldu_observation": {
            "epoch",
            "event",
            "block",
            "role",
            "detected_vacancy",
        },
        "controller_decision": {
            "epoch",
            "event",
            "public_state",
            "matched_rule_index",
            "action",
        },
        "dispatch": {
            "epoch",
            "event",
            "block",
            "vacancy_mask",
            "reserves_after",
        },
        "terminal_occupancy_audit": {
            "epoch",
            "event",
            "block",
            "newly_located_mask",
        },
    }
    if kind not in schemas or set(event) != schemas[kind]:
        raise ValueError(f"raw trace event {kind!r} does not match its exact schema")
    epoch = _strict_int(event["epoch"], name="event epoch", lower=0, upper=horizon)
    if "block" in event and event["block"] not in {"A", "B"}:
        raise ValueError("trace event block must be A or B")
    if "role" in event:
        _strict_int(event["role"], name="event role", lower=0, upper=N_DATA - 1)
    for binary_field in ("success", "detected_vacancy"):
        if binary_field in event:
            _strict_int(event[binary_field], name=binary_field, lower=0, upper=1)
    for mask_field in ("vacancy_mask", "newly_located_mask"):
        if mask_field in event:
            _strict_int(event[mask_field], name=mask_field, lower=0, upper=_ALL_ROLES)
    if "reserves_after" in event:
        _strict_int(event["reserves_after"], name="reserves_after", lower=0, upper=8)
    if kind == "controller_decision":
        if event["action"] not in {"dispatch_A", "dispatch_B", "hold"}:
            raise ValueError("controller decision contains an unknown action")
        _strict_int(
            event["matched_rule_index"],
            name="matched_rule_index",
            lower=-1,
            upper=127,
        )
        state = _require_exact_keys(
            event["public_state"],
            set(CONTROLLER_FIELD_LIMITS),
            name="controller public state",
        )
        for field, (lower, upper) in CONTROLLER_FIELD_LIMITS.items():
            _strict_int(state[field], name=field, lower=lower, upper=upper)
        if state["epoch_index"] != epoch or state["epochs_remaining"] != horizon - epoch:
            raise ValueError("controller public state is not bound to its event epoch")
    if kind == "terminal_occupancy_audit" and epoch != horizon:
        raise ValueError("terminal occupancy audit must occur at the horizon")


def validate_episode_trace(
    value: Any,
    *,
    expected_horizon: int | None = None,
    expected_episodes: int | None = None,
) -> dict[str, Any]:
    """Validate the complete raw episode/event schema and terminal invariants."""

    trace = _require_exact_keys(value, {"schema_version", "episodes"}, name="trace")
    if trace["schema_version"] != TRACE_SCHEMA_VERSION or not isinstance(trace["episodes"], list):
        raise ValueError("raw trace schema version or episode array is invalid")
    if expected_episodes is not None and len(trace["episodes"]) != expected_episodes:
        raise ValueError("raw trace episode count does not match its completed job")
    for index, episode_value in enumerate(trace["episodes"]):
        episode = _require_exact_keys(
            episode_value,
            {
                "horizon",
                "events",
                "terminal_blocks",
                "joint_logical_survival_bit",
                "reserves_remaining",
            },
            name=f"episode {index}",
        )
        horizon = _strict_int(episode["horizon"], name="episode horizon", lower=1, upper=40)
        if expected_horizon is not None and horizon != expected_horizon:
            raise ValueError("raw trace horizon does not match its completed job")
        if not isinstance(episode["events"], list):
            raise ValueError("episode events must be an array")
        for event in episode["events"]:
            _validate_trace_event(event, horizon=horizon)
        if not isinstance(episode["terminal_blocks"], list) or len(episode["terminal_blocks"]) != 2:
            raise ValueError("episode must contain exactly two terminal block records")
        terminal_bits: dict[str, int] = {}
        for terminal_value in episode["terminal_blocks"]:
            terminal = _require_exact_keys(
                terminal_value,
                {
                    "block",
                    "logical_survival_bit",
                    "terminal_damage_mask",
                    "terminal_located_damage_mask",
                    "terminal_occupied_mask",
                },
                name="terminal block",
            )
            block = terminal["block"]
            if block not in {"A", "B"} or block in terminal_bits:
                raise ValueError("terminal block identities must be exactly A and B")
            terminal_bits[block] = _strict_int(
                terminal["logical_survival_bit"],
                name="logical_survival_bit",
                lower=0,
                upper=1,
            )
            for mask_field in (
                "terminal_damage_mask",
                "terminal_located_damage_mask",
                "terminal_occupied_mask",
            ):
                _strict_int(
                    terminal[mask_field],
                    name=mask_field,
                    lower=0,
                    upper=_ALL_ROLES,
                )
        if set(terminal_bits) != {"A", "B"}:
            raise ValueError("terminal block identities must be exactly A and B")
        joint = _strict_int(
            episode["joint_logical_survival_bit"],
            name="joint_logical_survival_bit",
            lower=0,
            upper=1,
        )
        if joint != terminal_bits["A"] & terminal_bits["B"]:
            raise ValueError("joint terminal bit is not the conjunction of A and B")
        _strict_int(
            episode["reserves_remaining"],
            name="reserves_remaining",
            lower=0,
            upper=8,
        )
    return trace


def decode_episode_trace(
    payload: str,
    *,
    max_uncompressed_bytes: int = 268_435_456,
    expected_horizon: int | None = None,
    expected_episodes: int | None = None,
) -> dict[str, Any]:
    """Boundedly decode, parse, and validate a public raw episode payload."""

    if not isinstance(payload, str):
        raise ValueError("trace payload must be base64 text")
    if (
        isinstance(max_uncompressed_bytes, bool)
        or not isinstance(max_uncompressed_bytes, int)
        or max_uncompressed_bytes < 1
    ):
        raise ValueError("trace byte limit must be a positive integer")
    max_encoded_bytes = ((max_uncompressed_bytes + 1024) * 4 // 3) + 8
    if len(payload.encode("ascii", errors="ignore")) > max_encoded_bytes:
        raise ValueError("encoded trace exceeds its public byte limit")
    try:
        compressed = base64.b64decode(payload.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error) as exc:
        raise ValueError("trace payload is not canonical base64") from exc
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(compressed, max_uncompressed_bytes + 1)
        if len(raw) <= max_uncompressed_bytes:
            raw += decoder.flush(max_uncompressed_bytes + 1 - len(raw))
    except zlib.error as exc:
        raise ValueError("trace payload is not a valid zlib stream") from exc
    if len(raw) > max_uncompressed_bytes or decoder.unconsumed_tail:
        raise ValueError("decoded trace exceeds its public byte limit")
    if not decoder.eof or decoder.unused_data:
        raise ValueError("trace payload has a truncated or trailing zlib stream")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("decoded trace is not UTF-8 JSON") from exc
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if canonical != raw:
        raise ValueError("decoded trace is not canonically encoded")
    return validate_episode_trace(
        value,
        expected_horizon=expected_horizon,
        expected_episodes=expected_episodes,
    )


def validate_raw_episode_record(
    value: Any,
    *,
    expected_task_id: str | None = None,
    expected_device_id: str | None = None,
    expected_job_id: str | None = None,
    expected_execution_context_id: str | None = None,
    expected_public_device_model_commitment: dict[str, Any] | None = None,
    max_uncompressed_bytes: int = 268_435_456,
) -> dict[str, Any]:
    """Validate one complete public artifact and return its decoded trace."""

    record = _require_exact_keys(
        value,
        {
            "artifact_kind",
            "schema_version",
            "complete",
            "kind",
            "task_id",
            "device_id",
            "job_id",
            "tool",
            "execution_context_id",
            "public_device_model_commitment",
            "request_digest",
            "request",
            "horizon",
            "episodes",
            "controller_digest",
            "raw_schema_version",
            "trace_encoding",
            "trace_payload_b64",
        },
        name="raw episode artifact",
    )
    literals = {
        "artifact_kind": "located_erasure_control_episode_record_v1",
        "schema_version": 1,
        "complete": True,
        "kind": "located_erasure_control_episode",
        "tool": "run_located_erasure_control_episodes",
        "raw_schema_version": TRACE_SCHEMA_VERSION,
        "trace_encoding": "zlib_json_base64",
    }
    if any(record[key] != expected for key, expected in literals.items()):
        raise ValueError("raw episode artifact literal or completeness marker is invalid")
    for field_name in ("task_id", "device_id", "job_id"):
        if not isinstance(record[field_name], str) or not record[field_name]:
            raise ValueError(f"raw episode artifact {field_name} is invalid")
    if expected_task_id is not None and record["task_id"] != expected_task_id:
        raise ValueError("raw episode artifact task ID is not bound to this run")
    if expected_device_id is not None and record["device_id"] != expected_device_id:
        raise ValueError("raw episode artifact device ID is not bound to this run")
    if expected_job_id is not None and record["job_id"] != expected_job_id:
        raise ValueError("raw episode artifact job ID is not bound to this result")
    if record["execution_context_id"] != expected_execution_context_id:
        raise ValueError("raw episode artifact execution context is not bound to this run")
    if (
        expected_public_device_model_commitment is not None
        and record["public_device_model_commitment"] != expected_public_device_model_commitment
    ):
        raise ValueError("raw episode artifact public commitment is not bound to this run")
    if not isinstance(record["request_digest"], str) or len(record["request_digest"]) != 64:
        raise ValueError("raw episode artifact request digest is invalid")
    request = ErasureRepairEpisodeRequest.model_validate(record["request"])
    if episode_request_digest(request) != record["request_digest"]:
        raise ValueError("raw episode artifact request digest does not match its request")
    if controller_digest(request.controller) != record["controller_digest"]:
        raise ValueError("raw episode artifact controller digest does not match its request")
    if request.horizon != record["horizon"] or request.episodes != record["episodes"]:
        raise ValueError("raw episode artifact request/result dimensions differ")
    return decode_episode_trace(
        record["trace_payload_b64"],
        max_uncompressed_bytes=max_uncompressed_bytes,
        expected_horizon=request.horizon,
        expected_episodes=request.episodes,
    )


def _encode_episode_trace(
    value: dict[str, Any], *, max_uncompressed_bytes: int = 268_435_456
) -> str:
    validate_episode_trace(value)
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if len(raw) > max_uncompressed_bytes:
        raise RawTraceSizeError(
            "raw episode trace exceeds max_raw_result_uncompressed_bytes; "
            "split characterization across smaller jobs"
        )
    return base64.b64encode(zlib.compress(raw, level=9)).decode("ascii")


def _predicate_matches(predicate: ControllerPredicate, state: dict[str, int]) -> bool:
    observed = state[predicate.field]
    value = predicate.value
    if predicate.op == "eq":
        return observed == value
    if predicate.op == "ne":
        return observed != value
    if predicate.op == "lt":
        return observed < value
    if predicate.op == "le":
        return observed <= value
    if predicate.op == "gt":
        return observed > value
    if predicate.op == "ge":
        return observed >= value
    if predicate.op == "contains_all_bits":
        return observed & value == value
    if predicate.op == "contains_any_bits":
        return observed & value != 0
    raise AssertionError(f"unhandled predicate operator {predicate.op!r}")


def evaluate_controller(
    controller: CausalDecisionList,
    state: dict[str, int],
    *,
    legal_actions: set[str],
) -> tuple[str, int]:
    """Evaluate one immutable public state and return action plus rule index.

    Index ``-1`` denotes the default preference. Every valid preference ends in
    ``hold``, and hold is always legal, so evaluation is total.
    """

    preference = controller.default_preference
    rule_index = -1
    for index, rule in enumerate(controller.rules):
        if all(_predicate_matches(predicate, state) for predicate in rule.when_all):
            preference = rule.preference
            rule_index = index
            break
    for action in preference:
        if action in legal_actions:
            return action, rule_index
    raise AssertionError("validated controller has no legal fallback")


@dataclass
class BlockErasureState:
    """Private physical state with a separately derivable public projection."""

    name: str
    occupied_mask: int = _ALL_ROLES
    damage_mask: int = 0
    located_damage_mask: int = 0
    detected_vacancy_mask: int = 0
    first_loss_epoch_by_role: list[int | None] = field(default_factory=lambda: [None] * N_DATA)
    first_detection_epoch_by_role: list[int | None] = field(default_factory=lambda: [None] * N_DATA)
    epochs_since_last_recovery: int = 0
    cumulative_detection_count: int = 0
    cumulative_transport_failure_count: int = 0
    completed_recovery_count: int = 0
    last_detection_epoch: int | None = None
    logical_alive: bool = True


def record_role_loss(block: BlockErasureState, *, role: int, epoch: int) -> bool:
    """Apply loss to one occupied role and return whether state changed."""

    if not 0 <= role < N_DATA:
        raise ValueError("Steane role is out of range")
    role_bit = 1 << role
    if block.occupied_mask & role_bit == 0:
        return False
    block.occupied_mask &= ~role_bit
    block.damage_mask |= role_bit
    if block.first_loss_epoch_by_role[role] is None:
        block.first_loss_epoch_by_role[role] = epoch
    if not steane_erasure_correctable(block.damage_mask):
        block.logical_alive = False
    return True


def locate_vacancy(block: BlockErasureState, *, role: int, epoch: int) -> bool:
    """Publish one existing vacancy as located damage; return whether newly located."""

    if not 0 <= role < N_DATA:
        raise ValueError("Steane role is out of range")
    role_bit = 1 << role
    if block.occupied_mask & role_bit:
        return False
    if block.detected_vacancy_mask & role_bit:
        return False
    block.damage_mask |= role_bit
    block.located_damage_mask |= role_bit
    block.detected_vacancy_mask |= role_bit
    if block.first_detection_epoch_by_role[role] is None:
        block.first_detection_epoch_by_role[role] = epoch
    block.cumulative_detection_count += 1
    block.last_detection_epoch = epoch
    return True


def complete_replacement(
    block: BlockErasureState,
    *,
    attempted_mask: int,
    successful_mask: int,
) -> None:
    """Fill successful carriers without clearing their quantum damage."""

    if attempted_mask & ~_ALL_ROLES or successful_mask & ~attempted_mask:
        raise ValueError("replacement masks are inconsistent")
    if attempted_mask & ~block.detected_vacancy_mask:
        raise ValueError("replacement attempted a vacancy not exposed to the controller")
    block.occupied_mask |= successful_mask
    block.detected_vacancy_mask &= ~successful_mask
    block.cumulative_transport_failure_count += (attempted_mask & ~successful_mask).bit_count()


def try_fixed_recovery(block: BlockErasureState) -> bool:
    """Clear exactly a fully located, occupied, correctable live damage support."""

    if block.damage_mask == 0:
        return False
    if not block.logical_alive:
        return False
    if block.damage_mask != block.located_damage_mask:
        return False
    if block.occupied_mask != _ALL_ROLES or block.detected_vacancy_mask:
        return False
    if not steane_erasure_correctable(block.damage_mask):
        block.logical_alive = False
        return False
    block.damage_mask = 0
    block.located_damage_mask = 0
    block.detected_vacancy_mask = 0
    block.first_loss_epoch_by_role = [None] * N_DATA
    block.first_detection_epoch_by_role = [None] * N_DATA
    block.epochs_since_last_recovery = 0
    block.completed_recovery_count += 1
    return True


def terminal_occupancy_audit(block: BlockErasureState, *, epoch: int) -> int:
    """Locate every currently absent carrier in the destructive terminal audit."""

    newly_located = 0
    absent = _ALL_ROLES & ~block.occupied_mask
    for role in range(N_DATA):
        role_bit = 1 << role
        if absent & role_bit and locate_vacancy(block, role=role, epoch=epoch):
            newly_located |= role_bit
    return newly_located


def _oldest_detection_age(block: BlockErasureState, epoch: int) -> int:
    epochs = [
        detected_epoch
        for role, detected_epoch in enumerate(block.first_detection_epoch_by_role)
        if detected_epoch is not None and block.located_damage_mask & (1 << role)
    ]
    return 0 if not epochs else max(0, epoch - min(epochs))


def public_controller_state(
    blocks: list[BlockErasureState],
    *,
    epoch: int,
    horizon: int,
    reserves: int,
) -> dict[str, int]:
    state = {
        "epoch_index": epoch,
        "epochs_remaining": horizon - epoch,
        "reserves_available": reserves,
    }
    for block in blocks:
        prefix = block.name
        state[f"{prefix}.located_damage_mask"] = block.located_damage_mask
        state[f"{prefix}.detected_vacancy_mask"] = block.detected_vacancy_mask
        state[f"{prefix}.located_damage_count"] = block.located_damage_mask.bit_count()
        state[f"{prefix}.detected_vacancy_count"] = block.detected_vacancy_mask.bit_count()
        state[f"{prefix}.oldest_detection_age_epochs"] = _oldest_detection_age(block, epoch)
        state[f"{prefix}.epochs_since_last_recovery"] = min(
            horizon, block.epochs_since_last_recovery
        )
        state[f"{prefix}.cumulative_detection_count"] = min(255, block.cumulative_detection_count)
        state[f"{prefix}.cumulative_transport_failure_count"] = min(
            255, block.cumulative_transport_failure_count
        )
        state[f"{prefix}.completed_recovery_count"] = min(255, block.completed_recovery_count)
        state[f"{prefix}.epochs_since_last_detection"] = (
            horizon
            if block.last_detection_epoch is None
            else min(horizon, epoch - block.last_detection_epoch)
        )
    return state


@dataclass
class _LaneTransaction:
    block: str
    vacancy_mask: int
    epochs_remaining: int


class LocatedErasureRepairEngine:
    """Run-long two-memory engine with atomic public-episode admission."""

    def __init__(
        self,
        hidden: HiddenNeutralAtomConfig,
        public: PublicNeutralAtomSpec,
        rng: Any | None = None,
        *,
        entropy_key: bytes | None = None,
    ):
        if public.erasure_repair is None or hidden.erasure_repair_noise is None:
            raise ValueError(
                "located_erasure_repair requires public and hidden repair-service material"
            )
        self.hidden = hidden
        self.public = public
        # ``rng`` is accepted temporarily for source compatibility but never
        # consumed. Addressed draws prevent controller-dependent random-number
        # consumption or worker scheduling from changing exogenous events.
        if entropy_key is None:
            entropy_material = f"development-seed:{hidden.seed}".encode()
        elif not isinstance(entropy_key, bytes) or len(entropy_key) < 16:
            raise ValueError("located-erasure entropy key must contain at least 128 bits")
        else:
            entropy_material = entropy_key
        self._entropy_key = hashlib.sha256(
            b"located-erasure-control-addressed-entropy-v3\0" + entropy_material
        ).digest()
        self.used_episodes = 0
        self._budget_lock = threading.Lock()
        self._reservations: set[int] = set()
        self._next_reservation = 0

    def _uniform(self, namespace: str, *address: object) -> float:
        payload = json.dumps(
            [namespace, *address],
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest = hashlib.blake2b(payload, key=self._entropy_key, digest_size=8).digest()
        return int.from_bytes(digest, "big") / 2**64

    def reserve_episodes(self, episodes: int) -> int:
        repair = self.public.erasure_repair
        assert repair is not None
        with self._budget_lock:
            if self.used_episodes + episodes > repair.episode_budget:
                raise ValueError("run-long repair episode budget exceeded")
            self.used_episodes += episodes
            self._next_reservation += 1
            token = self._next_reservation
            self._reservations.add(token)
            return token

    def discard_reservation(self, token: int, *, episodes: int) -> None:
        with self._budget_lock:
            if token in self._reservations:
                self._reservations.remove(token)
                self.used_episodes -= episodes

    def _consume_reservation(self, token: int | None, *, episodes: int) -> None:
        if token is None:
            token = self.reserve_episodes(episodes)
        with self._budget_lock:
            if token not in self._reservations:
                raise ValueError("repair episode reservation is missing or already consumed")
            self._reservations.remove(token)

    def _complete_lane(
        self,
        lane: _LaneTransaction,
        blocks: list[BlockErasureState],
        *,
        epoch: int,
        episode_index: int,
        entropy_namespace: str,
        events: list[dict[str, Any]],
    ) -> None:
        noise = self.hidden.erasure_repair_noise
        assert noise is not None
        block = next(item for item in blocks if item.name == lane.block)
        block_index = 0 if block.name == "A" else 1
        recovery_role_loss_probability = noise.recovery_role_loss_probability_by_block[block_index]
        successful_mask = 0
        for role in range(N_DATA):
            role_bit = 1 << role
            if lane.vacancy_mask & role_bit == 0:
                continue
            succeeded = bool(
                self._uniform(
                    entropy_namespace,
                    "transport",
                    episode_index,
                    epoch,
                    block.name,
                    role,
                    block.completed_recovery_count,
                )
                >= noise.transport_failure_probability
            )
            if succeeded:
                successful_mask |= role_bit
            events.append(
                {
                    "epoch": epoch,
                    "event": "transport_ack",
                    "block": block.name,
                    "role": role,
                    "success": int(succeeded),
                }
            )
        complete_replacement(
            block,
            attempted_mask=lane.vacancy_mask,
            successful_mask=successful_mask,
        )
        if try_fixed_recovery(block):
            events.append(
                {
                    "epoch": epoch,
                    "event": "fixed_recovery_complete",
                    "block": block.name,
                }
            )
            recovery_index = block.completed_recovery_count
            for role in range(N_DATA):
                role_bit = 1 << role
                if block.occupied_mask & role_bit == 0:
                    continue
                lost = (
                    self._uniform(
                        entropy_namespace,
                        "recovery_erasure",
                        episode_index,
                        epoch,
                        block.name,
                        role,
                        recovery_index,
                    )
                    < recovery_role_loss_probability
                )
                if lost:
                    record_role_loss(block, role=role, epoch=epoch)
                    locate_vacancy(block, role=role, epoch=epoch)
                    events.append(
                        {
                            "epoch": epoch,
                            "event": "recovery_output_loss",
                            "block": block.name,
                            "role": role,
                        }
                    )

    def _sample_role_losses(
        self,
        blocks: list[BlockErasureState],
        *,
        epoch: int,
        episode_index: int,
        entropy_namespace: str,
    ) -> None:
        noise = self.hidden.erasure_repair_noise
        assert noise is not None
        burst = bool(
            self._uniform(entropy_namespace, "burst", episode_index, epoch)
            < noise.burst_probability
        )
        base_probability = noise.burst_role_loss_per_epoch if burst else noise.role_loss_per_epoch
        for block_index, block in enumerate(blocks):
            multiplier = noise.block_loss_multipliers[block_index]
            probability = min(1.0, base_probability * multiplier)
            for role in range(N_DATA):
                role_bit = 1 << role
                if (
                    block.occupied_mask & role_bit
                    and self._uniform(
                        entropy_namespace,
                        "memory_loss",
                        episode_index,
                        epoch,
                        block.name,
                        role,
                    )
                    < probability
                ):
                    record_role_loss(block, role=role, epoch=epoch)

    def _run_ldu(
        self,
        blocks: list[BlockErasureState],
        *,
        epoch: int,
        episode_index: int,
        entropy_namespace: str,
        events: list[dict[str, Any]],
    ) -> None:
        noise = self.hidden.erasure_repair_noise
        assert noise is not None
        for block in blocks:
            undetected_vacancies = _ALL_ROLES & ~block.occupied_mask & ~block.detected_vacancy_mask
            for role in range(N_DATA):
                role_bit = 1 << role
                if undetected_vacancies & role_bit == 0:
                    continue
                detected = bool(
                    self._uniform(
                        entropy_namespace,
                        "ldu",
                        episode_index,
                        epoch,
                        block.name,
                        role,
                    )
                    >= noise.missed_detection_probability
                )
                events.append(
                    {
                        "epoch": epoch,
                        "event": "ldu_observation",
                        "block": block.name,
                        "role": role,
                        "detected_vacancy": int(detected),
                    }
                )
                if detected:
                    locate_vacancy(block, role=role, epoch=epoch)

    @staticmethod
    def _legal_actions(blocks: list[BlockErasureState], reserves: int) -> set[str]:
        legal = {"hold"}
        for block in blocks:
            vacancies = block.detected_vacancy_mask.bit_count()
            if vacancies and vacancies <= reserves:
                legal.add(f"dispatch_{block.name}")
        return legal

    def _run_one(
        self,
        horizon: int,
        controller: CausalDecisionList,
        *,
        episode_index: int,
        entropy_namespace: str,
    ) -> dict[str, Any]:
        repair = self.public.erasure_repair
        assert repair is not None
        blocks = [BlockErasureState("A"), BlockErasureState("B")]
        reserves = repair.reserve_atoms
        lane: _LaneTransaction | None = None
        events: list[dict[str, Any]] = []

        for epoch in range(horizon):
            if lane is not None:
                lane.epochs_remaining -= 1
                if lane.epochs_remaining == 0:
                    self._complete_lane(
                        lane,
                        blocks,
                        epoch=epoch,
                        episode_index=episode_index,
                        entropy_namespace=entropy_namespace,
                        events=events,
                    )
                    lane = None

            self._sample_role_losses(
                blocks,
                epoch=epoch,
                episode_index=episode_index,
                entropy_namespace=entropy_namespace,
            )
            if (epoch + 1) % repair.ldu_cadence_epochs == 0:
                self._run_ldu(
                    blocks,
                    epoch=epoch,
                    episode_index=episode_index,
                    entropy_namespace=entropy_namespace,
                    events=events,
                )

            if lane is None:
                state = public_controller_state(
                    blocks,
                    epoch=epoch,
                    horizon=horizon,
                    reserves=reserves,
                )
                legal_actions = self._legal_actions(blocks, reserves)
                action, rule_index = evaluate_controller(
                    controller, state, legal_actions=legal_actions
                )
                events.append(
                    {
                        "epoch": epoch,
                        "event": "controller_decision",
                        "public_state": state,
                        "matched_rule_index": rule_index,
                        "action": action,
                    }
                )
                if action != "hold":
                    block_name = action[-1]
                    block = next(item for item in blocks if item.name == block_name)
                    vacancy_mask = block.detected_vacancy_mask
                    allocated = vacancy_mask.bit_count()
                    reserves -= allocated
                    lane = _LaneTransaction(
                        block=block_name,
                        vacancy_mask=vacancy_mask,
                        epochs_remaining=repair.replacement_epochs_per_atom * allocated,
                    )
                    events.append(
                        {
                            "epoch": epoch,
                            "event": "dispatch",
                            "block": block_name,
                            "vacancy_mask": vacancy_mask,
                            "reserves_after": reserves,
                        }
                    )

            for block in blocks:
                block.epochs_since_last_recovery += 1

        terminal: list[dict[str, Any]] = []
        for block in blocks:
            newly_located = terminal_occupancy_audit(block, epoch=horizon)
            if newly_located:
                events.append(
                    {
                        "epoch": horizon,
                        "event": "terminal_occupancy_audit",
                        "block": block.name,
                        "newly_located_mask": newly_located,
                    }
                )
            survived = int(block.logical_alive and steane_erasure_correctable(block.damage_mask))
            terminal.append(
                {
                    "block": block.name,
                    "logical_survival_bit": survived,
                    "terminal_damage_mask": block.damage_mask,
                    "terminal_located_damage_mask": block.located_damage_mask,
                    "terminal_occupied_mask": block.occupied_mask,
                }
            )
        joint = int(all(item["logical_survival_bit"] == 1 for item in terminal))
        return {
            "horizon": horizon,
            "events": events,
            "terminal_blocks": terminal,
            "joint_logical_survival_bit": joint,
            "reserves_remaining": reserves,
        }

    def run(
        self,
        request: ErasureRepairEpisodeRequest,
        *,
        reservation_token: int | None = None,
        entropy_namespace: str = "deterministic-regression",
    ) -> JobErasureRepairEpisodeData | str:
        repair = self.public.erasure_repair
        assert repair is not None
        if request.horizon not in repair.public_horizons:
            return "horizon is not in the public family"
        if request.episodes > repair.max_episodes_per_call:
            return "episodes exceeds max_episodes_per_call"
        try:
            self._consume_reservation(reservation_token, episodes=request.episodes)
        except ValueError as exc:
            return str(exc)
        traces = [
            self._run_one(
                request.horizon,
                request.controller,
                episode_index=episode_index,
                entropy_namespace=entropy_namespace,
            )
            for episode_index in range(request.episodes)
        ]
        try:
            trace_payload = _encode_episode_trace(
                {"schema_version": TRACE_SCHEMA_VERSION, "episodes": traces},
                max_uncompressed_bytes=repair.max_raw_result_uncompressed_bytes,
            )
        except RawTraceSizeError as exc:
            return str(exc)
        return JobErasureRepairEpisodeData(
            horizon=request.horizon,
            episodes=request.episodes,
            controller_digest=controller_digest(request.controller),
            request_digest=episode_request_digest(request),
            raw_schema_version=TRACE_SCHEMA_VERSION,
            trace_payload_b64=trace_payload,
        )

    def assess_joint_survival(
        self,
        controller: CausalDecisionList,
        *,
        horizon: int,
        episodes: int,
        entropy_namespace: str,
    ) -> int:
        """Count joint survival without constructing a public aggregate artifact."""

        if episodes < 1:
            raise ValueError("assessment requires at least one episode")
        successes = 0
        for episode_index in range(episodes):
            trace = self._run_one(
                horizon,
                controller,
                episode_index=episode_index,
                entropy_namespace=entropy_namespace,
            )
            successes += int(trace["joint_logical_survival_bit"])
        return successes


def build_raw_episode_record(
    request: ErasureRepairEpisodeRequest,
    traces: list[dict[str, Any]],
    *,
    max_uncompressed_bytes: int = 268_435_456,
) -> dict[str, Any]:
    """Build the raw-only qsim artifact shape used by tests and action delivery."""

    return {
        "kind": "located_erasure_control_episode",
        "raw_schema_version": 2,
        "horizon": request.horizon,
        "episodes": request.episodes,
        "controller_digest": controller_digest(request.controller),
        "request_digest": episode_request_digest(request),
        "trace_encoding": "zlib_json_base64",
        "trace_payload_b64": _encode_episode_trace(
            {"schema_version": TRACE_SCHEMA_VERSION, "episodes": traces},
            max_uncompressed_bytes=max_uncompressed_bytes,
        ),
    }


__all__ = [
    "BlockErasureState",
    "LocatedErasureRepairEngine",
    "build_raw_episode_record",
    "canonical_controller",
    "complete_replacement",
    "controller_digest",
    "decode_episode_trace",
    "episode_request_digest",
    "evaluate_controller",
    "locate_vacancy",
    "public_controller_state",
    "record_role_loss",
    "terminal_occupancy_audit",
    "try_fixed_recovery",
    "validate_episode_trace",
    "validate_raw_episode_record",
]
