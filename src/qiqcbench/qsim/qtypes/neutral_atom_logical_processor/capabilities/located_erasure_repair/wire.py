"""Strict wire contract for causal located-erasure control episodes."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from qiqcbench.qsim.core.wire import SCHEMA_VERSION

ControllerField = Literal[
    "epoch_index",
    "epochs_remaining",
    "reserves_available",
    "A.located_damage_mask",
    "A.detected_vacancy_mask",
    "A.located_damage_count",
    "A.detected_vacancy_count",
    "A.oldest_detection_age_epochs",
    "A.epochs_since_last_recovery",
    "A.cumulative_detection_count",
    "A.cumulative_transport_failure_count",
    "A.completed_recovery_count",
    "A.epochs_since_last_detection",
    "B.located_damage_mask",
    "B.detected_vacancy_mask",
    "B.located_damage_count",
    "B.detected_vacancy_count",
    "B.oldest_detection_age_epochs",
    "B.epochs_since_last_recovery",
    "B.cumulative_detection_count",
    "B.cumulative_transport_failure_count",
    "B.completed_recovery_count",
    "B.epochs_since_last_detection",
]
PredicateOperator = Literal[
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "contains_all_bits",
    "contains_any_bits",
]
ControllerAction = Literal["dispatch_A", "dispatch_B", "hold"]

MAX_CONTROLLER_RULES = 128
MAX_PREDICATES_PER_RULE = 16
MAX_CONTROLLER_PREDICATES = 1_024
MAX_CANONICAL_CONTROLLER_BYTES = 65_536

_MASK_FIELDS = {
    "A.located_damage_mask",
    "A.detected_vacancy_mask",
    "B.located_damage_mask",
    "B.detected_vacancy_mask",
}
CONTROLLER_FIELD_LIMITS: dict[str, tuple[int, int]] = {
    "epoch_index": (0, 39),
    "epochs_remaining": (1, 40),
    "reserves_available": (0, 4),
    "A.located_damage_mask": (0, 127),
    "A.detected_vacancy_mask": (0, 127),
    "A.located_damage_count": (0, 7),
    "A.detected_vacancy_count": (0, 7),
    "A.oldest_detection_age_epochs": (0, 40),
    "A.epochs_since_last_recovery": (0, 40),
    "A.cumulative_detection_count": (0, 255),
    "A.cumulative_transport_failure_count": (0, 255),
    "A.completed_recovery_count": (0, 255),
    "A.epochs_since_last_detection": (0, 40),
    "B.located_damage_mask": (0, 127),
    "B.detected_vacancy_mask": (0, 127),
    "B.located_damage_count": (0, 7),
    "B.detected_vacancy_count": (0, 7),
    "B.oldest_detection_age_epochs": (0, 40),
    "B.epochs_since_last_recovery": (0, 40),
    "B.cumulative_detection_count": (0, 255),
    "B.cumulative_transport_failure_count": (0, 255),
    "B.completed_recovery_count": (0, 255),
    "B.epochs_since_last_detection": (0, 40),
}


class _RepairStrict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _RepairStrictRequest(_RepairStrict):
    """_RepairStrict with NaN/Infinity refused on every float field."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class ControllerPredicate(_RepairStrictRequest):
    field: ControllerField
    op: PredicateOperator
    value: StrictInt

    @model_validator(mode="after")
    def _operator_and_value_match_field(self) -> ControllerPredicate:
        if self.op.startswith("contains_") and self.field not in _MASK_FIELDS:
            raise ValueError("mask operator is valid only for a public mask field")
        lower, upper = CONTROLLER_FIELD_LIMITS[self.field]
        if not lower <= self.value <= upper:
            raise ValueError(f"predicate value for {self.field!r} must be in [{lower}, {upper}]")
        return self


class ControllerRule(_RepairStrictRequest):
    when_all: list[ControllerPredicate] = Field(
        default_factory=list, max_length=MAX_PREDICATES_PER_RULE
    )
    preference: list[ControllerAction] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def _unambiguous_rule(self) -> ControllerRule:
        predicate_keys = [
            (predicate.field, predicate.op, predicate.value) for predicate in self.when_all
        ]
        if len(predicate_keys) != len(set(predicate_keys)):
            raise ValueError("duplicate predicate in one when_all conjunction")
        if len(self.preference) != len(set(self.preference)):
            raise ValueError("action preferences must be unique")
        if self.preference[-1] != "hold":
            raise ValueError("every action preference list must end in hold")
        return self


class CausalDecisionList(_RepairStrictRequest):
    schema_version: Literal[2] = 2
    kind: Literal["causal_decision_list"] = "causal_decision_list"
    rules: list[ControllerRule] = Field(default_factory=list, max_length=MAX_CONTROLLER_RULES)
    default_preference: list[ControllerAction] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def _bounded_controller(self) -> CausalDecisionList:
        if len(self.default_preference) != len(set(self.default_preference)):
            raise ValueError("default action preferences must be unique")
        if self.default_preference[-1] != "hold":
            raise ValueError("default action preference must end in hold")
        predicate_count = sum(len(rule.when_all) for rule in self.rules)
        if predicate_count > MAX_CONTROLLER_PREDICATES:
            raise ValueError("controller exceeds the total predicate limit")
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if len(payload) > MAX_CANONICAL_CONTROLLER_BYTES:
            raise ValueError("canonical controller exceeds 65536 bytes")
        return self


class ErasureRepairEpisodeRequest(_RepairStrictRequest):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    horizon: Literal[24, 32, 40]
    episodes: StrictInt = Field(ge=1, le=40_000)
    controller: CausalDecisionList


class JobErasureRepairEpisodeData(_RepairStrict):
    kind: Literal["located_erasure_control_episode"] = "located_erasure_control_episode"
    horizon: StrictInt
    episodes: StrictInt
    controller_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_schema_version: Literal[2] = 2
    trace_encoding: Literal["zlib_json_base64"] = "zlib_json_base64"
    raw_data_file: str | None = None
    raw_data_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    raw_data_size_bytes: StrictInt | None = Field(default=None, gt=0)
    # Transient qsim-internal carrier. The action persists it atomically and
    # clears it before returning the completed result to the agent.
    trace_payload_b64: str | None = None


__all__ = [
    "CausalDecisionList",
    "CONTROLLER_FIELD_LIMITS",
    "ControllerAction",
    "ControllerField",
    "ControllerPredicate",
    "ControllerRule",
    "ErasureRepairEpisodeRequest",
    "JobErasureRepairEpisodeData",
    "MAX_CANONICAL_CONTROLLER_BYTES",
    "MAX_CONTROLLER_PREDICATES",
    "MAX_CONTROLLER_RULES",
    "MAX_PREDICATES_PER_RULE",
    "PredicateOperator",
]
