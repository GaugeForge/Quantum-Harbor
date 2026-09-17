"""Canonical JSON, semantic digests, and immutable artifact writes.

Evaluation identities must be independent of mapping order and must never
silently admit non-JSON values such as NaN.  This module is intentionally
task-agnostic: callers provide the explicit semantic field allowlist for an
identity, while artifact writers operate only on already selected content.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_SAFE_SUFFIX = re.compile(r"\.[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        # ``model_copy(update=...)`` is a trusted Pydantic escape hatch that
        # skips validators.  Persistence is not: rebuild from dumped fields so
        # stale content-derived IDs and invalid trusted copies fail closed.
        validated = type(value).model_validate(value.model_dump(mode="python"))
        return validated.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical UTF-8 JSON representation of ``value``.

    Canonical evaluation JSON has sorted object keys, compact separators, and
    literal Unicode.  ``allow_nan=False`` makes NaN and infinities hard errors
    instead of emitting JavaScript-only spellings that are not valid JSON.
    """

    return json.dumps(
        _json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    """Return a full lowercase SHA-256 digest for exact bytes."""

    if not isinstance(payload, bytes):
        raise TypeError("sha256_bytes requires bytes")
    return hashlib.sha256(payload).hexdigest()


def canonical_digest(value: Any) -> str:
    """Return the SHA-256 digest of canonical JSON content."""

    return sha256_bytes(canonical_json_bytes(value))


def _normalized_unordered(value: Any, *, field: str) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(value, Collection):
        raise TypeError(f"unordered semantic field {field!r} must be a collection")

    by_bytes: dict[bytes, Any] = {}
    for item in value:
        normalized = _json_value(item)
        encoded = canonical_json_bytes(normalized)
        by_bytes.setdefault(encoded, normalized)
    return [by_bytes[key] for key in sorted(by_bytes)]


def semantic_digest(
    value: Mapping[str, Any] | BaseModel,
    *,
    include_fields: Sequence[str],
    unordered_fields: Collection[str] = (),
) -> str:
    """Hash an explicit allowlist of semantic fields.

    Requiring ``include_fields`` prevents a raw environment/configuration dump
    from accidentally entering a permanent identity.  Fields declared
    unordered are sorted by their own canonical bytes and de-duplicated before
    hashing; callers should use that option only for set-like semantics.
    """

    if isinstance(value, BaseModel):
        source = _json_value(value)
    elif isinstance(value, Mapping):
        source = value
    else:
        raise TypeError("semantic_digest requires a mapping or Pydantic model")

    fields = tuple(include_fields)
    if not fields or any(not isinstance(field, str) or not field for field in fields):
        raise ValueError("semantic_digest requires a non-empty semantic field allowlist")
    if len(fields) != len(set(fields)):
        raise ValueError("semantic field allowlist contains duplicates")

    unordered = set(unordered_fields)
    outside_allowlist = unordered.difference(fields)
    if outside_allowlist:
        names = ", ".join(sorted(outside_allowlist))
        raise ValueError(f"unordered field(s) {names} not in the semantic allowlist")

    missing = [field for field in fields if field not in source]
    if missing:
        raise KeyError(f"semantic payload is missing field(s): {', '.join(missing)}")

    selected: dict[str, Any] = {}
    for field in fields:
        field_value = source[field]
        selected[field] = (
            _normalized_unordered(field_value, field=field) if field in unordered else field_value
        )
    return canonical_digest(selected)


def _fsync_directory(path: Path) -> None:
    """Best-effort durability barrier for a directory entry update."""

    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_temp_file(directory: Path, payload: bytes, *, prefix: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(dir=directory, prefix=prefix, suffix=".tmp")
    path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    """Atomically replace ``path`` from a temporary file in the same directory."""

    if not isinstance(payload, bytes):
        raise TypeError("atomic_write_bytes requires bytes")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _write_temp_file(destination.parent, payload, prefix=f".{destination.name}.")
    try:
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def atomic_write_json(path: str | Path, value: Any) -> Path:
    """Atomically replace ``path`` with canonical JSON bytes."""

    return atomic_write_bytes(path, canonical_json_bytes(value))


def _validate_suffix(suffix: str) -> str:
    if not isinstance(suffix, str) or _SAFE_SUFFIX.fullmatch(suffix) is None:
        raise ValueError("content-addressed suffix must be a safe dot-prefixed filename suffix")
    return suffix


def write_content_addressed_bytes(
    directory: str | Path,
    payload: bytes,
    *,
    suffix: str,
) -> Path:
    """Write immutable bytes at ``<sha256><suffix>`` without overwriting.

    A temporary file and hard link make publication atomic and no-clobber.  A
    byte-identical pre-existing artifact is an idempotent no-op; any different
    bytes at the claimed digest path are treated as a collision/corruption.
    """

    if not isinstance(payload, bytes):
        raise TypeError("write_content_addressed_bytes requires bytes")
    safe_suffix = _validate_suffix(suffix)
    store = Path(directory)
    store.mkdir(parents=True, exist_ok=True)
    destination = store / f"{sha256_bytes(payload)}{safe_suffix}"

    if destination.exists():
        if destination.read_bytes() != payload:
            raise FileExistsError(f"content-addressed path contains different bytes: {destination}")
        return destination

    temporary = _write_temp_file(store, payload, prefix=f".{destination.name}.")
    try:
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != payload:
                raise FileExistsError(
                    f"content-addressed path contains different bytes: {destination}"
                ) from None
        _fsync_directory(store)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_content_addressed_json(
    directory: str | Path,
    value: Any,
    *,
    suffix: str = ".json",
) -> Path:
    """Write canonical JSON to an immutable content-addressed path."""

    return write_content_addressed_bytes(
        directory,
        canonical_json_bytes(value),
        suffix=suffix,
    )


__all__ = [
    "atomic_write_bytes",
    "atomic_write_json",
    "canonical_digest",
    "canonical_json_bytes",
    "semantic_digest",
    "sha256_bytes",
    "write_content_addressed_bytes",
    "write_content_addressed_json",
]
