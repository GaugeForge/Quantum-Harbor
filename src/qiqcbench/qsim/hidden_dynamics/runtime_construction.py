"""Boot-time hidden-apparatus construction hook.

``docker/qsim/entrypoint.sh`` calls this module on every boot. No released task
declares a runtime-constructed apparatus; the hook leaves the configured hidden
path unchanged. The entrypoint continues to resolve a mounted hidden device file
or the local-development example.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from types import MappingProxyType

# Every runtime-constructed instance lands under qsim's private log volume.
# ``/qsim_logs/private`` is never mounted into the agent container, and the only
# nested path main may see is the sanctioned ``public_job_results`` surface.
PRIVATE_INSTANCE_ROOT = "/qsim_logs/private"

# task_id -> the exact container path that task's instance is written to.  The
# task's builder is ``qiqcbench.qsim.hidden_dynamics.<task_id>.construction``,
# by the same convention that places every other task-specific hidden module.
# Frozen: nothing inside the running container may add or move a declaration.
RUNTIME_CONSTRUCTED_HIDDEN_PATHS: Mapping[str, str] = MappingProxyType(
    {}
)


def construct_runtime_instance(task_id: str, hidden_path: str) -> str | None:
    """Draw ``task_id``'s hidden apparatus, returning its digest, or None.

    ``None`` means nothing was constructed and the caller must leave the
    configured hidden path exactly as it found it, so the entrypoint's existing
    resolution still governs.  That happens in two cases.

    The task declares no runtime construction, which is the ordinary one.

    Or ``hidden_path`` is empty.  An unset ``QIQCBENCH_HIDDEN_PATH`` is the
    documented local-development invocation, where the entrypoint falls back to
    the conventional bind mount or the shipped example; a launch always pins the
    path from its bundle.  Constructing into the declared *container* path would
    be wrong outside a container, and refusing to start would break local
    development for no gain. What keeps a bundle from losing
    its pin is a test over the shipped bundles, not a runtime guess here.

    A non-empty ``hidden_path`` is a real launch, and it has to match the
    declaration exactly: a bundle that pins somewhere else would either score
    against the wrong file or drop the instance outside the private volume, and
    both must stop the container rather than produce a plausible run.
    """

    declared = RUNTIME_CONSTRUCTED_HIDDEN_PATHS.get(task_id)
    if declared is None or not hidden_path:
        return None
    if hidden_path != declared:
        raise RuntimeError(
            f"task {task_id!r} constructs its hidden apparatus at {declared}, but the "
            f"launch configured QIQCBENCH_HIDDEN_PATH={hidden_path!r}"
        )
    module = import_module(f"qiqcbench.qsim.hidden_dynamics.{task_id}.construction")
    return module.write_runtime_instance(Path(declared))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--hidden-path", default="")
    args = parser.parse_args()
    digest = construct_runtime_instance(args.task_id, args.hidden_path)
    if digest is None:
        return
    # Non-secret provenance only.  The seed and the drawn parameters stay in the
    # verifier-only artifact and never reach the agent transcript.
    print(
        json.dumps(
            {"runtime_instance_created": True, "runtime_instance_sha256": digest},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
