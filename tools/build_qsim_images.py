"""Build qiqcbench-qsim and verifier images declared in harbor_tasks/*/task.toml.

Requires a git checkout and Docker. Images carry SOURCE_REVISION (with a
``-dirty`` suffix for working-tree edits); standalone verifier contexts also
carry CONTENT_DIGEST. The builder checks qsim-derived layers against the qsim
image and standalone context labels against their source revision and digest.

    uv run python tools/build_qsim_images.py                  # qsim + all verifiers
    uv run python tools/build_qsim_images.py --tasks a,b,c    # qsim + those verifiers
    uv run python tools/build_qsim_images.py --qsim-only

"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

QSIM_IMAGE = "qiqcbench-qsim:latest"


def source_revision(root: Path, run=subprocess.run) -> str:
    """The checked-out commit, suffixed ``-dirty`` when the tree has local edits.

    Untracked files count as dirty: ``.dockerignore`` still lets an untracked
    ``src/`` module into the build context, so it can change what the image runs.
    """
    head = run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    status = run(
        ["git", "-C", str(root), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{head}-dirty" if status else head


def verifier_images(root: Path) -> dict[str, list[str]]:
    """Map each declared verifier image tag to the task bundles that use it.

    Several bundles may share one tag, so the tag is the
    build unit; the first bundle's ``tests/`` directory supplies the context.
    """
    images: dict[str, list[str]] = {}
    for task_toml in sorted((root / "harbor_tasks").glob("*/task.toml")):
        config = tomllib.loads(task_toml.read_text(encoding="utf-8"))
        image = config.get("verifier", {}).get("environment", {}).get("docker_image")
        if image:
            images.setdefault(image, []).append(task_toml.parent.name)
    return images


def _safe_manifest_path(value: Any, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"verifier context {field} must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe verifier context {field}: {value!r}")
    return path


def _manifest_entries(root: Path, context: Path) -> list[tuple[Path, PurePosixPath, int]] | None:
    manifest = context / "verifier-context.json"
    if not manifest.exists():
        return None
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError(f"verifier context manifest is not a regular file: {manifest}")
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid verifier context manifest {manifest}: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "files"}:
        raise ValueError("verifier context manifest must contain exactly schema_version and files")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("verifier context manifest schema_version must be 1")
    files = raw["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("verifier context manifest files must be a non-empty list")

    root = root.resolve()
    entries: list[tuple[Path, PurePosixPath, int]] = []
    targets: set[PurePosixPath] = set()
    for index, entry in enumerate(files):
        if not isinstance(entry, dict) or set(entry) != {"source", "target", "mode"}:
            raise ValueError(
                f"verifier context entry {index} must contain exactly source, target, and mode"
            )
        source_relative = _safe_manifest_path(entry["source"], field=f"source[{index}]")
        target = _safe_manifest_path(entry["target"], field=f"target[{index}]")
        if target in targets:
            raise ValueError(f"duplicate verifier context target: {target}")
        targets.add(target)
        mode_text = entry["mode"]
        if mode_text not in {"0400", "0644", "0755"}:
            raise ValueError(f"unsupported verifier context mode at entry {index}: {mode_text!r}")

        source = root.joinpath(*source_relative.parts)
        cursor = root
        for part in source_relative.parts:
            cursor /= part
            if cursor.is_symlink():
                raise ValueError(f"verifier context source traverses a symlink: {source_relative}")
        try:
            resolved = source.resolve(strict=True)
            resolved.relative_to(root)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"verifier context source escapes or is missing: {source_relative}"
            ) from exc
        if not resolved.is_file():
            raise ValueError(f"verifier context source is not a regular file: {source_relative}")
        entries.append((resolved, target, int(mode_text, 8)))
    return entries


def _entries_digest(entries: list[tuple[Path, PurePosixPath, int]]) -> str:
    digest = hashlib.sha256()
    for source, target, mode in sorted(entries, key=lambda item: item[1].as_posix()):
        digest.update(target.as_posix().encode("utf-8"))
        digest.update(b"\x00")
        digest.update(f"{mode:04o}".encode("ascii"))
        digest.update(b"\x00")
        digest.update(source.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


def verifier_content_digest(context: Path, *, root: Path | None = None) -> str:
    """Deterministic digest over the bytes supplied to a verifier build.

    SHA-256 over each regular file's repo-relative name and content, in sorted
    order. Stamped into the image as the ``qiqcbench.verifier.content_digest``
    label so preflight can prove the image was built from exactly the current
    context bytes (standalone minimal verifiers have no qsim base layers to
    compare against). For a manifest-backed context, source paths may live
    elsewhere in the repo and target paths describe the staged Docker context.
    """
    context = Path(context)
    if root is None:
        # Canonical layout: <repo>/harbor_tasks/<task>/tests.
        root = context.resolve().parents[2]
    entries = _manifest_entries(root, context)
    if entries is not None:
        return _entries_digest(entries)
    digest = hashlib.sha256()
    for path in sorted(context.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(context).as_posix().encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()


@contextmanager
def prepared_verifier_context(root: Path, context: Path) -> Iterator[tuple[Path, str]]:
    """Yield the exact Docker context and the digest stamped into its image."""

    entries = _manifest_entries(root, context)
    if entries is None:
        yield context, verifier_content_digest(context, root=root)
        return
    with tempfile.TemporaryDirectory(prefix=f"qiqcbench-{context.parent.name}-verifier-") as temp:
        staged = Path(temp)
        staged_entries: list[tuple[Path, PurePosixPath, int]] = []
        for source, target, mode in entries:
            destination = staged.joinpath(*target.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            os.chmod(destination, mode)
            staged_entries.append((destination, target, mode))

        expected_modes = {target.as_posix(): mode for _, target, mode in staged_entries}
        expected_targets = set(expected_modes)

        def assert_closed_context() -> None:
            actual_targets: set[str] = set()
            for path in staged.rglob("*"):
                if path.is_symlink():
                    raise ValueError(f"staged verifier context contains a symlink: {path}")
                if path.is_file():
                    relative = path.relative_to(staged).as_posix()
                    actual_targets.add(relative)
                    if relative in expected_modes and not _mode_matches(
                        path.stat().st_mode, expected_modes[relative]
                    ):
                        raise ValueError(f"staged verifier context mode changed: {relative}")
            if actual_targets != expected_targets:
                raise ValueError("staged verifier context differs from its manifest allowlist")

        # Hash the bytes after copying, not the source files before copying, so
        # the label describes the context Docker actually receives even if an
        # editor changes a source while staging is in progress.
        assert_closed_context()
        content_digest = _entries_digest(staged_entries)
        yield staged, content_digest
        # Docker reads the temporary directory during the yielded build. Catch
        # any accidental mutation before reporting the image as successfully
        # built under the original digest label.
        assert_closed_context()
        if _entries_digest(staged_entries) != content_digest:
            raise ValueError("staged verifier context changed during the Docker build")


def _mode_matches(actual_mode: int, expected_mode: int) -> bool:
    """Compare a staged file's permission bits with its manifest mode.

    Windows keeps only a read-only attribute, so there the check compares the
    owner-write bit. The verifier Dockerfile sets every file mode with
    ``COPY --chmod``, so the image does not depend on host permission bits.
    """
    if os.name == "nt":
        return bool(actual_mode & 0o200) == bool(expected_mode & 0o200)
    return actual_mode & 0o777 == expected_mode


def _build(args: list[str], root: Path, run=subprocess.run) -> None:
    print("+ " + " ".join(args), flush=True)
    run(args, cwd=str(root), check=True)


def main(argv: list[str] | None = None, run=subprocess.run) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tasks", default="", help="Comma-separated task ids (default: all)")
    parser.add_argument("--qsim-only", action="store_true", help="Rebuild only the qsim sidecar")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    revision = source_revision(root, run=run)
    if revision.endswith("-dirty"):
        print(f"WARNING: building from a DIRTY tree; images stamp {revision}", file=sys.stderr)

    _build(
        [
            "docker",
            "build",
            "--build-arg",
            f"SOURCE_REVISION={revision}",
            "-t",
            QSIM_IMAGE,
            "-f",
            "docker/qsim/Dockerfile",
            ".",
        ],
        root,
        run=run,
    )
    if args.qsim_only:
        return 0

    selected = [task for task in args.tasks.split(",") if task]
    for image, tasks in verifier_images(root).items():
        if selected and not set(tasks) & set(selected):
            continue
        context = root / "harbor_tasks" / tasks[0] / "tests"
        with prepared_verifier_context(root, context) as (build_context, content_digest):
            display_context = (
                f"harbor_tasks/{tasks[0]}/tests/"
                if build_context == context
                else str(build_context)
            )
            _build(
                [
                    "docker",
                    "build",
                    "--build-arg",
                    f"SOURCE_REVISION={revision}",
                    "--build-arg",
                    f"CONTENT_DIGEST={content_digest}",
                    "-t",
                    image,
                    display_context,
                ],
                root,
                run=run,
            )

    print(f"\nBuilt qsim + verifier images at revision {revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
