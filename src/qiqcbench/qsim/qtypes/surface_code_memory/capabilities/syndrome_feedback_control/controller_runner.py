"""Fail-closed streaming runner for agent-authored syndrome controllers.

The parent process owns the simulator and hidden state.  The child receives
only public initialization metadata and the raw detector bits from epochs that
have already completed. It runs as uid/gid 65534 inside a package-free chroot
whose native networking and FFI extensions are removed; the scorer never
imports the submitted source.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import select
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

CHROOT_EXECUTABLE = "/usr/sbin/chroot"
DEFAULT_RUNNER_ROOT = "/controller_runner_root"
_DOMAIN = b"qiqcbench-syndrome-controller-bundle-v1\0"
_RUNNER_LOCK = threading.Lock()
_PREFLIGHT_SENTINEL = "qiqcbench-controller-sandbox-preflight-ok"


class ControllerSubmissionError(RuntimeError):
    """A model-owned source, execution, or protocol defect."""


class ControllerRunnerInfrastructureError(RuntimeError):
    """A verifier/qsim-owned sandbox failure."""


@dataclass(frozen=True)
class ControllerBundleManifest:
    sha256: str
    files: tuple[str, ...]
    size_bytes: int


def _resolved_inside(path: Path, root: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ControllerRunnerInfrastructureError(
            f"controller sandbox path escapes or is missing: {path}"
        ) from exc
    return resolved


def validate_controller_sandbox(
    runner_root: str | Path = DEFAULT_RUNNER_ROOT,
    *,
    chroot_executable: str | Path = CHROOT_EXECUTABLE,
) -> None:
    """Fail closed unless the production controller chroot is usable.

    This check is deliberately independent of any submitted controller. The
    production verifier calls it before classifying a missing or malformed
    answer, so a broken sandbox can never be hidden by a model disposition.
    """

    root = Path(runner_root)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise ControllerRunnerInfrastructureError("controller sandbox root is missing or unsafe")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise ControllerRunnerInfrastructureError(
            "controller sandbox root is missing or unreadable"
        ) from exc
    if resolved_root == Path("/"):
        raise ControllerRunnerInfrastructureError("controller sandbox root must not be /")

    chroot = Path(chroot_executable)
    if (
        not chroot.is_absolute()
        or not chroot.is_file()
        or chroot.is_symlink()
        or not os.access(chroot, os.X_OK)
    ):
        raise ControllerRunnerInfrastructureError("controller sandbox chroot executable is missing")

    python = root / "usr/local/bin/python"
    resolved_python = _resolved_inside(python, resolved_root)
    if not resolved_python.is_file() or not os.access(resolved_python, os.X_OK):
        raise ControllerRunnerInfrastructureError(
            "controller sandbox Python is missing or not executable"
        )

    for name in ("work", "tmp"):
        writable = root / name
        if not writable.is_dir() or writable.is_symlink():
            raise ControllerRunnerInfrastructureError(
                f"controller sandbox {name} directory is missing or unsafe"
            )
        _resolved_inside(writable, resolved_root)
    submission = root / "submission"
    if submission.exists() or submission.is_symlink():
        if not submission.is_dir() or submission.is_symlink():
            raise ControllerRunnerInfrastructureError(
                "controller sandbox submission path is unsafe"
            )
        _resolved_inside(submission, resolved_root)

    smoke = (
        "import os\n"
        "if os.getuid() != 65534 or os.getgid() != 65534:\n"
        "    raise SystemExit(10)\n"
        "for name in ('qiqcbench', 'numpy', 'socket', 'ssl', 'ctypes'):\n"
        "    try:\n"
        "        __import__(name)\n"
        "    except ImportError:\n"
        "        continue\n"
        "    raise SystemExit(20)\n"
        f"print({_PREFLIGHT_SENTINEL!r})\n"
    )
    command = [
        str(chroot),
        "--userspec=65534:65534",
        str(resolved_root),
        "/usr/local/bin/python",
        "-I",
        "-S",
        "-B",
        "-c",
        smoke,
    ]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env={
                "HOME": "/tmp",
                "LANG": "C.UTF-8",
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "PYTHONNOUSERSITE": "1",
            },
            check=False,
            timeout=10.0,
            text=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ControllerRunnerInfrastructureError(
            f"controller sandbox preflight could not run: {type(exc).__name__}: {exc}"
        ) from exc
    if completed.returncode != 0 or completed.stdout.strip() != _PREFLIGHT_SENTINEL:
        stderr = completed.stderr[-1_000:].strip()
        raise ControllerRunnerInfrastructureError(
            "controller sandbox preflight failed"
            + (f": {stderr}" if stderr else f" (status {completed.returncode})")
        )


def controller_bundle_manifest(
    source_dir: str | Path,
    *,
    maximum_files: int,
    maximum_bytes: int,
) -> ControllerBundleManifest:
    """Validate a regular-file tree and hash paths plus exact bytes."""

    root = Path(source_dir)
    if not root.is_dir() or root.is_symlink():
        raise ControllerSubmissionError("controller source directory is missing or unsafe")
    files: list[Path] = []
    total = 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ControllerSubmissionError(f"symbolic links are forbidden: {relative}")
        if path.is_dir():
            if path.name == "__pycache__":
                raise ControllerSubmissionError("__pycache__ directories are forbidden")
            continue
        mode = path.stat(follow_symlinks=False).st_mode
        if not stat.S_ISREG(mode):
            raise ControllerSubmissionError(f"non-regular file is forbidden: {relative}")
        if path.suffix in {".pyc", ".pyo"}:
            raise ControllerSubmissionError("compiled Python files are forbidden")
        files.append(path)
        total += path.stat().st_size
        if len(files) > maximum_files:
            raise ControllerSubmissionError("controller bundle file-count limit exceeded")
        if total > maximum_bytes:
            raise ControllerSubmissionError("controller bundle byte-size limit exceeded")
    entrypoint = root / "run.py"
    if entrypoint not in files:
        raise ControllerSubmissionError("controller bundle must contain a regular run.py")
    digest = hashlib.sha256()
    digest.update(_DOMAIN)
    relative_names: list[str] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        encoded = relative.encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        relative_names.append(relative)
    return ControllerBundleManifest(digest.hexdigest(), tuple(relative_names), total)


def _chmod_tree_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_dir() and not child.is_symlink():
            child.chmod(0o700)
    path.chmod(0o700)
    shutil.rmtree(path)


def _reset_writable(path: Path, *, mode: int) -> None:
    _remove_tree(path)
    path.mkdir(parents=True, mode=mode)
    path.chmod(mode)


def _limits(cpu_seconds: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_AS, (512_000_000, 512_000_000))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1_000_000, 1_000_000))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
    except (OSError, ValueError):
        pass


def _sandbox_process_ids(uid: int = 65_534) -> list[int]:
    process_ids: list[int] = []
    for status_path in Path("/proc").glob("[0-9]*/status"):
        try:
            lines = status_path.read_text(encoding="utf-8").splitlines()
            uid_line = next(line for line in lines if line.startswith("Uid:"))
            state_line = next(line for line in lines if line.startswith("State:"))
            if int(uid_line.split()[1]) == uid and "Z" not in state_line.split()[1]:
                process_ids.append(int(status_path.parent.name))
        except (OSError, StopIteration, ValueError):
            continue
    return process_ids


def _kill_sandbox_processes(process_group_id: int) -> None:
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 5.0
    while True:
        process_ids = _sandbox_process_ids()
        if not process_ids:
            return
        for process_id in process_ids:
            try:
                os.kill(process_id, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    if _sandbox_process_ids():
        raise ControllerRunnerInfrastructureError(
            "controller sandbox child processes survived forced cleanup"
        )


class ControllerProcess:
    """One stateful controller process for one continuous trajectory."""

    def __init__(
        self,
        source_dir: str | Path,
        *,
        expected_manifest_sha256: str,
        runner_root: str | Path = DEFAULT_RUNNER_ROOT,
        maximum_files: int,
        maximum_bytes: int,
        maximum_response_bytes: int,
        step_timeout_s: float,
        trajectory_timeout_s: float,
        max_requested_trim_abs: float,
    ) -> None:
        self.source_dir = Path(source_dir)
        self.expected_manifest_sha256 = expected_manifest_sha256
        self.runner_root = Path(runner_root)
        self.maximum_files = maximum_files
        self.maximum_bytes = maximum_bytes
        self.maximum_response_bytes = maximum_response_bytes
        self.step_timeout_s = step_timeout_s
        self.trajectory_timeout_s = trajectory_timeout_s
        self.max_requested_trim_abs = max_requested_trim_abs
        self._process: subprocess.Popen[bytes] | None = None
        self._stderr_file: Any = None
        self._stdout_buffer = bytearray()
        self._started_at = 0.0
        self._lock_held = False

    def __enter__(self) -> ControllerProcess:
        _RUNNER_LOCK.acquire()
        self._lock_held = True
        try:
            self._start()
        except Exception:
            try:
                self.close()
            finally:
                self._release_lock()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        try:
            self.close()
        finally:
            self._release_lock()

    def _release_lock(self) -> None:
        if self._lock_held:
            _RUNNER_LOCK.release()
            self._lock_held = False

    def _start(self) -> None:
        manifest = controller_bundle_manifest(
            self.source_dir,
            maximum_files=self.maximum_files,
            maximum_bytes=self.maximum_bytes,
        )
        if manifest.sha256 != self.expected_manifest_sha256:
            raise ControllerSubmissionError(
                "controller_manifest_sha256 does not bind the staged source bundle"
            )
        if not (self.runner_root / "usr/local/bin/python").is_file():
            raise ControllerRunnerInfrastructureError("controller sandbox Python is missing")
        staged_submission = self.runner_root / "submission"
        staged_work = self.runner_root / "work"
        staged_tmp = self.runner_root / "tmp"
        _remove_tree(staged_submission)
        shutil.copytree(self.source_dir, staged_submission)
        copied = controller_bundle_manifest(
            staged_submission,
            maximum_files=self.maximum_files,
            maximum_bytes=self.maximum_bytes,
        )
        if copied.sha256 != manifest.sha256:
            raise ControllerRunnerInfrastructureError("controller source changed while staging")
        _chmod_tree_read_only(staged_submission)
        _reset_writable(staged_work, mode=0o777)
        _reset_writable(staged_tmp, mode=0o1777)
        command = [
            CHROOT_EXECUTABLE,
            "--userspec=65534:65534",
            str(self.runner_root),
            "/usr/local/bin/python",
            "-E",
            "-s",
            "-B",
            "/submission/run.py",
        ]
        environment = {
            "HOME": "/tmp",
            "LANG": "C.UTF-8",
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PYTHONNOUSERSITE": "1",
        }
        self._stderr_file = tempfile.TemporaryFile()
        cpu_seconds = max(1, int(math.ceil(self.trajectory_timeout_s)))
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            env=environment,
            start_new_session=True,
            preexec_fn=partial(_limits, cpu_seconds),
        )
        assert self._process.stdout is not None
        os.set_blocking(self._process.stdout.fileno(), False)
        self._started_at = time.monotonic()

    def _stderr_tail(self) -> str:
        if self._stderr_file is None:
            return ""
        self._stderr_file.flush()
        self._stderr_file.seek(0)
        return self._stderr_file.read()[-2_000:].decode("utf-8", errors="replace")

    def _readline(self) -> dict[str, Any]:
        assert self._process is not None and self._process.stdout is not None
        deadline = min(
            time.monotonic() + self.step_timeout_s,
            self._started_at + self.trajectory_timeout_s,
        )
        fd = self._process.stdout.fileno()
        while True:
            newline = self._stdout_buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self._stdout_buffer[:newline])
                del self._stdout_buffer[: newline + 1]
                break
            if len(self._stdout_buffer) > self.maximum_response_bytes:
                raise ControllerSubmissionError("controller response exceeds the byte limit")
            if self._process.poll() is not None:
                raise ControllerSubmissionError(
                    f"controller exited before replying: {self._stderr_tail()}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ControllerSubmissionError("controller response timeout")
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                raise ControllerSubmissionError("controller response timeout")
            chunk = os.read(fd, min(4096, self.maximum_response_bytes + 1))
            if not chunk:
                raise ControllerSubmissionError(
                    f"controller closed stdout before replying: {self._stderr_tail()}"
                )
            self._stdout_buffer.extend(chunk)
        if len(line) > self.maximum_response_bytes:
            raise ControllerSubmissionError("controller response exceeds the byte limit")
        try:
            payload = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ControllerSubmissionError("controller response is not one JSON object") from exc
        if not isinstance(payload, dict):
            raise ControllerSubmissionError("controller response must be a JSON object")
        return payload

    def _exchange(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self._process is not None and self._process.stdin is not None
        encoded = (
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8")
        try:
            self._process.stdin.write(encoded)
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ControllerSubmissionError(
                f"controller stdin closed unexpectedly: {self._stderr_tail()}"
            ) from exc
        return self._readline()

    def initialize(self, payload: dict[str, Any]) -> None:
        response = self._exchange(payload)
        if response != {"ready": True}:
            raise ControllerSubmissionError('initialize response must be exactly {"ready":true}')

    def step(self, payload: dict[str, Any]) -> list[float]:
        response = self._exchange(payload)
        if set(response) != {"requested_trim"}:
            raise ControllerSubmissionError("step response must contain only requested_trim")
        requested = response["requested_trim"]
        if not isinstance(requested, list) or len(requested) != 2:
            raise ControllerSubmissionError("requested_trim must be a two-element JSON array")
        parsed: list[float] = []
        for value in requested:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ControllerSubmissionError("requested_trim entries must be numbers")
            number = float(value)
            if not math.isfinite(number) or abs(number) > self.max_requested_trim_abs:
                raise ControllerSubmissionError("requested_trim entry is non-finite or too large")
            parsed.append(number)
        return parsed

    def close(self) -> None:
        process = self._process
        self._process = None
        try:
            if process is not None:
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
                _kill_sandbox_processes(process.pid)
                if process.poll() is None:
                    process.wait(timeout=5)
        finally:
            if self._stderr_file is not None:
                self._stderr_file.close()
                self._stderr_file = None
            for name, mode in (("work", 0o777), ("tmp", 0o1777)):
                _reset_writable(self.runner_root / name, mode=mode)
            _remove_tree(self.runner_root / "submission")


class HostSmokeControllerProcess(ControllerProcess):
    """Host-only protocol smoke runner used when a chroot is unavailable.

    Production qsim and separate-verifier scoring always use
    :class:`ControllerProcess`. This class exists only so the verifier skill can
    exercise identical JSONL/source semantics on macOS before running the
    authoritative container smoke. It provides process/resource separation but
    intentionally makes no no-network claim.
    """

    def _start(self) -> None:
        manifest = controller_bundle_manifest(
            self.source_dir,
            maximum_files=self.maximum_files,
            maximum_bytes=self.maximum_bytes,
        )
        if manifest.sha256 != self.expected_manifest_sha256:
            raise ControllerSubmissionError(
                "controller_manifest_sha256 does not bind the staged source bundle"
            )
        self._stderr_file = tempfile.TemporaryFile()
        self._process = subprocess.Popen(
            [sys.executable, "-I", "-S", "-B", str(self.source_dir / "run.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            env={
                "HOME": tempfile.gettempdir(),
                "LANG": "C.UTF-8",
                "PATH": "/usr/bin:/bin",
                "PYTHONNOUSERSITE": "1",
            },
            start_new_session=True,
        )
        assert self._process.stdout is not None
        os.set_blocking(self._process.stdout.fileno(), False)
        self._started_at = time.monotonic()

    def close(self) -> None:
        process = self._process
        self._process = None
        try:
            if process is not None:
                if process.stdin is not None:
                    try:
                        process.stdin.close()
                    except OSError:
                        pass
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                if process.poll() is None:
                    process.wait(timeout=5)
        finally:
            if self._stderr_file is not None:
                self._stderr_file.close()
                self._stderr_file = None


__all__ = [
    "ControllerBundleManifest",
    "ControllerProcess",
    "ControllerRunnerInfrastructureError",
    "ControllerSubmissionError",
    "DEFAULT_RUNNER_ROOT",
    "HostSmokeControllerProcess",
    "controller_bundle_manifest",
    "validate_controller_sandbox",
]
