"""Shared strict primitives for versioned evaluation artifacts."""

from __future__ import annotations

import math
import re
from collections.abc import Collection, Mapping
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, StringConstraints, model_validator

NonEmptyStr = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, pattern=r"\S"),
]
Identifier = Annotated[
    str,
    StringConstraints(strict=True, min_length=1, pattern=r"^\S+$"),
]
Digest = Annotated[
    str,
    StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
]

_ID_KIND = re.compile(r"[a-z][a-z0-9_]*\Z")


def _immutable_collection_error(*_args: Any, **_kwargs: Any) -> None:
    raise TypeError("evaluation contract collections are immutable")


class _FrozenDict(dict[Any, Any]):
    """JSON-compatible dictionary that rejects in-place mutation."""

    __setitem__ = _immutable_collection_error
    __delitem__ = _immutable_collection_error
    clear = _immutable_collection_error
    pop = _immutable_collection_error
    popitem = _immutable_collection_error
    setdefault = _immutable_collection_error
    update = _immutable_collection_error
    __ior__ = _immutable_collection_error

    def __copy__(self) -> _FrozenDict:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenDict:
        memo[id(self)] = self
        return self


class _FrozenList(list[Any]):
    """JSON-compatible list that rejects in-place mutation."""

    __setitem__ = _immutable_collection_error
    __delitem__ = _immutable_collection_error
    __iadd__ = _immutable_collection_error
    __imul__ = _immutable_collection_error
    append = _immutable_collection_error
    clear = _immutable_collection_error
    extend = _immutable_collection_error
    insert = _immutable_collection_error
    pop = _immutable_collection_error
    remove = _immutable_collection_error
    reverse = _immutable_collection_error
    sort = _immutable_collection_error

    def __copy__(self) -> _FrozenList:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenList:
        memo[id(self)] = self
        return self


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, _FrozenDict | _FrozenList):
        return value
    if isinstance(value, dict):
        return _FrozenDict({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_deep_freeze(item) for item in value)
    return value


class StrictModel(BaseModel):
    """Base for immutable, fail-closed evaluation contracts."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )

    @staticmethod
    def _freeze_contract_value(value: Any) -> Any:
        return _deep_freeze(value)

    @model_validator(mode="after")
    def _eagerly_freeze_collections(self) -> StrictModel:
        for field_name in type(self).model_fields:
            object.__setattr__(
                self,
                field_name,
                _deep_freeze(getattr(self, field_name)),
            )
        return self

    def __getattribute__(self, name: str) -> Any:
        value = super().__getattribute__(name)
        model_fields = type(self).model_fields
        if name in model_fields:
            frozen = _deep_freeze(value)
            if frozen is not value:
                object.__setattr__(self, name, frozen)
                return frozen
        return value


def semantic_digest(
    payload: Mapping[str, Any],
    *,
    unordered_fields: Collection[str] = (),
) -> str:
    """Hash an already-selected semantic mapping through canonical IO.

    Identity models construct ``payload`` from explicit field allowlists.  This
    wrapper keeps that selection next to each contract while ensuring all
    identities share the canonical serializer.
    """

    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("semantic identity payload must be a non-empty mapping")
    # Keep contract imports independent of the IO package's public facade.  The
    # local import also prevents a future artifact-reader export from creating
    # a contracts <-> IO import cycle.
    from qiqcbench.eval.io.canonical import semantic_digest as canonical_semantic_digest

    return canonical_semantic_digest(
        payload,
        include_fields=tuple(payload),
        unordered_fields=unordered_fields,
    )


def semantic_id(
    kind: str,
    payload: Mapping[str, Any],
    *,
    unordered_fields: Collection[str] = (),
) -> str:
    """Return ``<kind>_<full SHA-256>`` for an explicit semantic payload."""

    if not isinstance(kind, str) or _ID_KIND.fullmatch(kind) is None:
        raise ValueError(f"invalid semantic ID kind {kind!r}")
    return f"{kind}_{semantic_digest(payload, unordered_fields=unordered_fields)}"


def ensure_finite_json(value: Any, *, path: str = "value") -> Any:
    """Validate that ``value`` is finite JSON data without coercion."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite numbers")
        return value
    if isinstance(value, list):
        for index, item in enumerate(value):
            ensure_finite_json(item, path=f"{path}[{index}]")
        return value
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key or not key.strip():
                raise ValueError(f"{path} keys must be non-empty strings")
            ensure_finite_json(item, path=f"{path}.{key}")
        return value
    raise ValueError(f"{path} must contain only JSON values")


__all__ = [
    "Digest",
    "Identifier",
    "NonEmptyStr",
    "StrictModel",
    "ensure_finite_json",
    "semantic_digest",
    "semantic_id",
]
