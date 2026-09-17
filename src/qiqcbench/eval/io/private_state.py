"""Secure no-clobber storage for named private evaluation state.

Unlike the content-addressed private archive, this small primitive is for the
few resumable files whose names are part of an offline workflow.  Every path is
opened from absolute directory descriptors with ``O_NOFOLLOW``.  Existing
nodes are validated, never permission-repaired.
"""

from __future__ import annotations

import errno
import hashlib
import os
import re
import secrets
import stat
import time
from pathlib import Path, PurePosixPath

_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)
_TRANSIENT_LINK_RETRIES = 100
_TRANSIENT_LINK_RETRY_SECONDS = 0.005


class PrivateStateError(RuntimeError):
    """Base error for unsafe or unstable named private state."""


class PrivateStatePathError(PrivateStateError, PermissionError):
    """A private-state path contains an unsafe node or permission."""


class PrivateStateIntegrityError(PrivateStateError):
    """A named private file changed while it was being consumed."""


def _absolute_path(value: str | Path) -> Path:
    if not isinstance(value, str | Path):
        raise TypeError("private state root must be a string or Path")
    path = Path(os.path.abspath(os.fspath(value)))
    if path == Path(os.sep):
        raise PrivateStatePathError("private state root cannot be the filesystem root")
    return path


def _safe_relative_file(value: str | PurePosixPath) -> PurePosixPath:
    if not isinstance(value, str | PurePosixPath):
        raise TypeError("private state reference must be a string or PurePosixPath")
    reference = PurePosixPath(value)
    if (
        reference.is_absolute()
        or not reference.parts
        or any(part in {"", ".", ".."} or "/" in part for part in reference.parts)
    ):
        raise ValueError("private state reference must be a normalized relative file path")
    return reference


def _open_directory_component(parent_fd: int, component: str, *, label: str) -> int:
    try:
        descriptor = os.open(
            component,
            _DIRECTORY_FLAGS | _NOFOLLOW,
            dir_fd=parent_fd,
        )
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise PrivateStatePathError(f"{label} is a symlink or not a directory") from exc
        raise
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):  # pragma: no cover - O_DIRECTORY.
        os.close(descriptor)
        raise PrivateStatePathError(f"{label} is not a directory")
    return descriptor


def _require_current_owner(metadata: os.stat_result, *, label: str) -> None:
    getuid = getattr(os, "getuid", None)
    if getuid is not None and metadata.st_uid != getuid():
        raise PrivateStatePathError(f"{label} must be owned by the current user")


def _validate_private_directory(descriptor: int, *, label: str) -> None:
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        raise PrivateStatePathError(f"{label} is not a directory")
    _require_current_owner(metadata, label=label)
    if stat.S_IMODE(metadata.st_mode) != 0o700:
        raise PrivateStatePathError(f"{label} must have exact mode 0700")


def _validate_private_file(descriptor: int, *, label: str) -> os.stat_result:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise PrivateStatePathError(f"{label} is not a regular file")
    _require_current_owner(metadata, label=label)
    if stat.S_IMODE(metadata.st_mode) != 0o600:
        raise PrivateStatePathError(f"{label} must have exact mode 0600")
    if metadata.st_nlink != 1:
        raise PrivateStatePathError(f"{label} cannot have hardlink aliases")
    return metadata


def _open_absolute_directory(path: Path) -> int:
    """Open an existing absolute directory without following any ancestor link."""

    descriptor = os.open(os.sep, _DIRECTORY_FLAGS)
    try:
        for component in path.parts[1:]:
            next_descriptor = _open_directory_component(
                descriptor,
                component,
                label=f"absolute path component {component!r}",
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_or_create_private_child(parent_fd: int, name: str, *, label: str) -> int:
    created = False
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
        created = True
    except FileExistsError:
        pass
    descriptor = _open_directory_component(parent_fd, name, label=label)
    try:
        if created:
            os.fchmod(descriptor, 0o700)
            os.fsync(descriptor)
        _validate_private_directory(descriptor, label=label)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _read_all(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:  # pragma: no cover - regular writes progress or raise.
            raise OSError("private state write made no progress")
        written += count


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _only_writer_temporary_aliases(
    parent_fd: int,
    *,
    destination_name: str,
    metadata: os.stat_result,
) -> bool:
    """Recognize only this primitive's short-lived pre-unlink writer alias."""

    temporary = re.compile(rf"\.{re.escape(destination_name)}\.[0-9a-f]{{32}}\.tmp\Z")
    matching: list[str] = []
    for name in os.listdir(parent_fd):
        try:
            candidate = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            continue
        if (candidate.st_dev, candidate.st_ino) == (metadata.st_dev, metadata.st_ino):
            matching.append(name)
    return (
        len(matching) == metadata.st_nlink
        and destination_name in matching
        and all(name == destination_name or temporary.fullmatch(name) for name in matching)
    )


class PrivateStateStream:
    """Descriptor-anchored 0600 streaming writer published without clobbering."""

    def __init__(self, state: NamedPrivateState, reference: str | PurePosixPath) -> None:
        self._state = state
        self._reference = _safe_relative_file(reference)
        self._parent_fd = state._open_parent(self._reference, create=True)
        self._temporary_name = f".{self._reference.name}.{secrets.token_hex(16)}.tmp"
        self._descriptor: int | None = None
        self._closed = False
        self._size = 0
        self._digest = hashlib.sha256()
        try:
            self._descriptor = os.open(
                self._temporary_name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
                0o600,
                dir_fd=self._parent_fd,
            )
            os.fchmod(self._descriptor, 0o600)
        except BaseException:
            os.close(self._parent_fd)
            raise

    @property
    def reference(self) -> str:
        return str(self._reference)

    def write(self, payload: bytes) -> None:
        if not isinstance(payload, bytes):
            raise TypeError("private state stream payload must be bytes")
        if self._closed or self._descriptor is None:
            raise PrivateStateError("private state stream is closed")
        _write_all(self._descriptor, payload)
        self._size += len(payload)
        self._digest.update(payload)

    def seal(self) -> tuple[int, str]:
        """Fsync and atomically publish, returning the exact byte fact."""

        if self._closed or self._descriptor is None:
            raise PrivateStateError("private state stream is closed")
        descriptor, self._descriptor = self._descriptor, None
        published = False
        try:
            _validate_private_file(descriptor, label="private state stream")
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            os.link(
                self._temporary_name,
                self._reference.name,
                src_dir_fd=self._parent_fd,
                dst_dir_fd=self._parent_fd,
                follow_symlinks=False,
            )
            published = True
            os.fsync(self._parent_fd)
            return self._size, self._digest.hexdigest()
        finally:
            self._closed = True
            if descriptor != -1:
                os.close(descriptor)
            if self._descriptor is not None:
                os.close(self._descriptor)
                self._descriptor = None
            try:
                os.unlink(self._temporary_name, dir_fd=self._parent_fd)
                os.fsync(self._parent_fd)
            except FileNotFoundError:
                if not published:
                    raise PrivateStateIntegrityError(
                        "private state stream temporary file disappeared before publication"
                    ) from None
            finally:
                os.close(self._parent_fd)

    def abort(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._descriptor is not None:
            os.close(self._descriptor)
            self._descriptor = None
        try:
            os.unlink(self._temporary_name, dir_fd=self._parent_fd)
            os.fsync(self._parent_fd)
        except FileNotFoundError:
            pass
        finally:
            os.close(self._parent_fd)


class PrivateStateLiveLog:
    """Descriptor-anchored, immediately visible 0600 append-only live log."""

    def __init__(self, state: NamedPrivateState, reference: str | PurePosixPath) -> None:
        self._reference = _safe_relative_file(reference)
        self._parent_fd = state._open_parent(self._reference, create=True)
        self._descriptor: int | None = None
        self._closed = False
        try:
            self._descriptor = os.open(
                self._reference.name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
                0o600,
                dir_fd=self._parent_fd,
            )
            os.fchmod(self._descriptor, 0o600)
            _validate_private_file(self._descriptor, label="private live log")
            os.fsync(self._parent_fd)
        except BaseException:
            if self._descriptor is not None:
                os.close(self._descriptor)
            os.close(self._parent_fd)
            raise

    @property
    def reference(self) -> str:
        return str(self._reference)

    def write(self, payload: bytes) -> None:
        if not isinstance(payload, bytes):
            raise TypeError("private live log payload must be bytes")
        if self._closed or self._descriptor is None:
            raise PrivateStateError("private live log is closed")
        _write_all(self._descriptor, payload)

    def fileno(self) -> int:
        if self._closed or self._descriptor is None:
            raise PrivateStateError("private live log is closed")
        return self._descriptor

    def seal(self) -> tuple[int, str]:
        if self._closed or self._descriptor is None:
            raise PrivateStateError("private live log is closed")
        descriptor, self._descriptor = self._descriptor, None
        self._closed = True
        try:
            metadata = _validate_private_file(descriptor, label="private live log")
            os.fsync(descriptor)
            return metadata.st_size, _sha256_descriptor(descriptor)
        finally:
            os.close(descriptor)
            os.close(self._parent_fd)

    def close(self) -> None:
        """Flush and release the parent descriptor while a child retains its copy."""

        if self._closed:
            return
        self._closed = True
        try:
            if self._descriptor is not None:
                os.fsync(self._descriptor)
                os.close(self._descriptor)
                self._descriptor = None
        finally:
            os.close(self._parent_fd)


def _sha256_descriptor(descriptor: int) -> str:
    """Hash a regular descriptor without relying on its path after creation."""

    position = os.lseek(descriptor, 0, os.SEEK_CUR)
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.lseek(descriptor, position, os.SEEK_SET)


def _list_private_regular_files(descriptor: int, *, prefix: PurePosixPath) -> tuple[PurePosixPath, ...]:
    """Enumerate only owner-private regular files without traversing path aliases."""

    discovered: list[PurePosixPath] = []
    for name in os.listdir(descriptor):
        if name.startswith("."):
            continue
        try:
            metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise PrivateStateIntegrityError("private state directory changed while listed") from exc
        reference = prefix / name
        if stat.S_ISDIR(metadata.st_mode):
            child = _open_directory_component(descriptor, name, label=f"private state directory {name!r}")
            try:
                _validate_private_directory(child, label=f"private state directory {name!r}")
                discovered.extend(_list_private_regular_files(child, prefix=reference))
            finally:
                os.close(child)
        elif stat.S_ISREG(metadata.st_mode):
            child = os.open(name, os.O_RDONLY | _NOFOLLOW | _NONBLOCK, dir_fd=descriptor)
            try:
                _validate_private_file(child, label=f"private state file {reference!s}")
                after = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                if _file_identity(metadata) != _file_identity(after):
                    raise PrivateStateIntegrityError("private state file changed while listed")
            finally:
                os.close(child)
            discovered.append(reference)
        else:
            raise PrivateStatePathError(f"private state entry {reference!s} is not a regular file or directory")
    return tuple(sorted(discovered))


class NamedPrivateState:
    """Owner-only named state below one exact-mode private root."""

    def __init__(self, root: str | Path) -> None:
        self.root = _absolute_path(root)

    def _open_root(self, *, create: bool) -> int:
        parent_fd = _open_absolute_directory(self.root.parent)
        try:
            if create:
                return _open_or_create_private_child(
                    parent_fd,
                    self.root.name,
                    label="private state root",
                )
            root_fd = _open_directory_component(
                parent_fd,
                self.root.name,
                label="private state root",
            )
            try:
                _validate_private_directory(root_fd, label="private state root")
            except BaseException:
                os.close(root_fd)
                raise
            return root_fd
        finally:
            os.close(parent_fd)

    def _open_parent(self, reference: PurePosixPath, *, create: bool) -> int:
        descriptor = self._open_root(create=create)
        try:
            for component in reference.parts[:-1]:
                if create:
                    next_descriptor = _open_or_create_private_child(
                        descriptor,
                        component,
                        label=f"private state directory {component!r}",
                    )
                else:
                    next_descriptor = _open_directory_component(
                        descriptor,
                        component,
                        label=f"private state directory {component!r}",
                    )
                    try:
                        _validate_private_directory(
                            next_descriptor,
                            label=f"private state directory {component!r}",
                        )
                    except BaseException:
                        os.close(next_descriptor)
                        raise
                os.close(descriptor)
                descriptor = next_descriptor
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def read(self, reference: str | PurePosixPath) -> bytes:
        """Read exact bytes after file-descriptor and named-path stability checks."""

        relative = _safe_relative_file(reference)
        parent_fd = self._open_parent(relative, create=False)
        descriptor: int | None = None
        try:
            before_fd: os.stat_result | None = None
            for retry in range(_TRANSIENT_LINK_RETRIES + 1):
                try:
                    before_path = os.stat(
                        relative.name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                    descriptor = os.open(
                        relative.name,
                        os.O_RDONLY | _NOFOLLOW | _NONBLOCK,
                        dir_fd=parent_fd,
                    )
                except OSError as exc:
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                        raise PrivateStatePathError("private state file is a symlink") from exc
                    raise
                opened = os.fstat(descriptor)
                if not stat.S_ISREG(opened.st_mode):
                    raise PrivateStatePathError("private state file is not a regular file")
                _require_current_owner(opened, label="private state file")
                if stat.S_IMODE(opened.st_mode) != 0o600:
                    raise PrivateStatePathError("private state file must have exact mode 0600")
                if (before_path.st_dev, before_path.st_ino) != (
                    opened.st_dev,
                    opened.st_ino,
                ):
                    os.close(descriptor)
                    descriptor = None
                    raise PrivateStateIntegrityError(
                        "private state file changed while it was opened"
                    )
                if opened.st_nlink == 1 and before_path.st_nlink == 1:
                    before_fd = opened
                    break
                if not _only_writer_temporary_aliases(
                    parent_fd,
                    destination_name=relative.name,
                    metadata=opened,
                ):
                    refreshed = os.fstat(descriptor)
                    if refreshed.st_nlink != 1:
                        raise PrivateStatePathError(
                            "private state file cannot have external hardlink aliases"
                        )
                os.close(descriptor)
                descriptor = None
                if retry == _TRANSIENT_LINK_RETRIES:
                    raise PrivateStateIntegrityError("private state writer alias did not settle")
                time.sleep(_TRANSIENT_LINK_RETRY_SECONDS)
            assert descriptor is not None and before_fd is not None
            payload = _read_all(descriptor)
            after_fd = _validate_private_file(descriptor, label="private state file")
            try:
                after_path = os.stat(
                    relative.name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError as exc:
                raise PrivateStateIntegrityError(
                    "private state file disappeared while it was read"
                ) from exc
            if not (
                _file_identity(before_fd) == _file_identity(after_fd) == _file_identity(after_path)
            ):
                raise PrivateStateIntegrityError("private state file changed while it was read")
            return payload
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent_fd)

    def list_regular_files(self) -> tuple[PurePosixPath, ...]:
        """Return validated relative regular-file names without following links.

        The descriptor-anchored walk intentionally validates every directory and
        file at the same owner/mode/no-hardlink boundary as ``read``.
        """

        descriptor = self._open_root(create=False)
        try:
            return _list_private_regular_files(descriptor, prefix=PurePosixPath())
        finally:
            os.close(descriptor)

    def validate_file(self, reference: str | PurePosixPath) -> None:
        """Validate one existing private regular file without consuming mutable live bytes."""

        relative = _safe_relative_file(reference)
        parent_fd = self._open_parent(relative, create=False)
        descriptor: int | None = None
        try:
            before = os.stat(relative.name, dir_fd=parent_fd, follow_symlinks=False)
            descriptor = os.open(relative.name, os.O_RDONLY | _NOFOLLOW | _NONBLOCK, dir_fd=parent_fd)
            opened = _validate_private_file(descriptor, label="private state file")
            after = os.stat(relative.name, dir_fd=parent_fd, follow_symlinks=False)
            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino) or (
                after.st_dev,
                after.st_ino,
            ) != (opened.st_dev, opened.st_ino):
                raise PrivateStateIntegrityError("private state file changed while it was opened")
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent_fd)

    def write_no_clobber(self, reference: str | PurePosixPath, payload: bytes) -> bool:
        """Atomically publish one exact-mode file, returning false if its name exists."""

        if not isinstance(payload, bytes):
            raise TypeError("named private state payload must be bytes")
        relative = _safe_relative_file(reference)
        parent_fd = self._open_parent(relative, create=True)
        temporary_name = f".{relative.name}.{secrets.token_hex(16)}.tmp"
        descriptor: int | None = None
        published = False
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
            os.fchmod(descriptor, 0o600)
            _write_all(descriptor, payload)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            try:
                os.link(
                    temporary_name,
                    relative.name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                published = True
                os.fsync(parent_fd)
            except FileExistsError:
                return False
            return True
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except FileNotFoundError:  # pragma: no cover - only an external race can remove it.
                if not published:
                    raise PrivateStateIntegrityError(
                        "private state temporary file disappeared before publication"
                    ) from None
            finally:
                os.close(parent_fd)

    def open_stream_no_clobber(self, reference: str | PurePosixPath) -> PrivateStateStream:
        """Open an owner-private streaming writer for one as-yet-unpublished name."""

        return PrivateStateStream(self, reference)

    def open_live_log_no_clobber(self, reference: str | PurePosixPath) -> PrivateStateLiveLog:
        """Create one visible owner-private log without replacing an existing name."""

        return PrivateStateLiveLog(self, reference)


__all__ = [
    "NamedPrivateState",
    "PrivateStateLiveLog",
    "PrivateStateStream",
    "PrivateStateError",
    "PrivateStateIntegrityError",
    "PrivateStatePathError",
]
