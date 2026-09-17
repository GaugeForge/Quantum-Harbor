"""Bounded delivery and verifier hydration for public qsim raw evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

MAX_PUBLIC_RAW_RECORD_BYTES = 256 * 1024 * 1024
_SAFE_JOB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def canonical_public_raw_bytes(record: dict[str, Any]) -> bytes:
    """Return the one accepted byte representation for a public raw record."""

    return json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _require_real_directory(path: Path, *, description: str) -> None:
    if os.path.lexists(path):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{description} must be a real directory")


def _require_regular_file(path: Path, *, description: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"{description} is missing") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{description} must be a regular non-symlink file")
    return metadata


def persist_public_raw_record(
    log_dir: Path | None,
    *,
    job_id: str,
    record: dict[str, Any],
    max_bytes: int = MAX_PUBLIC_RAW_RECORD_BYTES,
) -> str | None:
    """Immutably persist a bounded canonical record and return its relative path.

    The qsim worker owns ``job_id`` and the evidence root. Publication uses a
    temporary regular file followed by an atomic hard link, so an existing job
    record is never overwritten. An idempotent retry may reuse identical
    bytes; a conflicting retry fails closed.
    """

    if log_dir is None:
        return None
    if not isinstance(job_id, str) or _SAFE_JOB_ID.fullmatch(job_id) is None:
        raise ValueError("raw evidence job ID is not path-safe")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("raw evidence byte limit must be a positive integer")
    if not isinstance(record, dict):
        raise ValueError("raw evidence record must be an object")

    payload = canonical_public_raw_bytes(record)
    if len(payload) > max_bytes:
        raise ValueError(f"raw evidence record exceeds its {max_bytes}-byte public limit")

    root = Path(log_dir).resolve()
    raw_dir = root / "raw_results"
    _require_real_directory(raw_dir, description="raw evidence directory")
    raw_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    resolved_dir = raw_dir.resolve()
    try:
        resolved_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("raw evidence directory escapes the qsim artifact root") from exc

    target = resolved_dir / f"{job_id}.json"
    if os.path.lexists(target):
        _require_regular_file(target, description="raw evidence target")
        if target.read_bytes() != payload:
            raise ValueError("conflicting immutable raw evidence for one job ID")
        return target.relative_to(root).as_posix()

    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "wb",
            dir=resolved_dir,
            prefix=".public_raw_record.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o644)
        try:
            os.link(temporary, target)
        except FileExistsError:
            _require_regular_file(target, description="raw evidence target")
            if target.read_bytes() != payload:
                raise ValueError("conflicting immutable raw evidence for one job ID") from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    target.chmod(0o644)
    return target.relative_to(root).as_posix()


def hydrate_public_raw_record(
    record: dict[str, Any],
    *,
    evidence_dir: Path | None,
    max_bytes: int = MAX_PUBLIC_RAW_RECORD_BYTES,
) -> dict[str, Any]:
    """Load one canonical raw record with path, size, digest, and summary closure."""

    raw_data_file = record.get("raw_data_file")
    if not raw_data_file:
        return record
    if not isinstance(raw_data_file, str) or evidence_dir is None:
        raise ValueError("completed evidence references an unavailable raw data file")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError("raw evidence byte limit must be a positive integer")

    relative = Path(raw_data_file)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("raw evidence path must be a confined relative path")
    root = evidence_dir.resolve()
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if os.path.lexists(cursor) and cursor.is_symlink():
            raise ValueError("raw evidence path must not contain symlinks")
    candidate = root / relative
    try:
        candidate.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError("raw evidence path escapes the qsim artifact directory") from exc

    metadata = _require_regular_file(candidate, description="raw evidence file")
    declared_size = record.get("raw_data_size_bytes")
    if declared_size is not None:
        if (
            isinstance(declared_size, bool)
            or not isinstance(declared_size, int)
            or declared_size < 1
        ):
            raise ValueError("raw evidence declared size is invalid")
        if metadata.st_size != declared_size:
            raise ValueError("raw evidence size does not match its completed job summary")
    if metadata.st_size > max_bytes:
        raise ValueError("raw evidence file exceeds its public byte limit")

    try:
        with candidate.open("rb") as handle:
            raw_bytes = handle.read(max_bytes + 1)
    except OSError as exc:
        raise ValueError(f"could not load raw evidence file: {exc}") from exc
    if len(raw_bytes) != metadata.st_size:
        raise ValueError("raw evidence size changed while reading")

    declared_digest = record.get("raw_data_sha256")
    if declared_digest is not None:
        if (
            not isinstance(declared_digest, str)
            or hashlib.sha256(raw_bytes).hexdigest() != declared_digest
        ):
            raise ValueError("raw evidence digest does not match its completed job summary")
    try:
        raw = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not parse raw evidence file: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("raw evidence record must be an object")
    if canonical_public_raw_bytes(raw) != raw_bytes:
        raise ValueError("raw evidence record is not canonically encoded")
    if raw.get("kind") != record.get("kind"):
        raise ValueError("raw evidence record does not match its completed job summary")
    for key in (
        "protocol_id",
        "system_size",
        "shots",
        "horizon",
        "episodes",
        "policy_digest",
        "controller_digest",
        "request_digest",
        "raw_schema_version",
        "trace_encoding",
    ):
        if key in record and record.get(key) is not None and raw.get(key) != record[key]:
            raise ValueError(f"raw evidence field {key!r} does not match its job summary")
    return raw


__all__ = [
    "MAX_PUBLIC_RAW_RECORD_BYTES",
    "canonical_public_raw_bytes",
    "hydrate_public_raw_record",
    "persist_public_raw_record",
]
