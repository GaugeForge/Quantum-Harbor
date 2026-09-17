from __future__ import annotations

from collections.abc import Iterable
from typing import cast, get_args

from qiqcbench.qsim.tasks import AgentSurface, TaskSpec

_KNOWN_SURFACES: frozenset[str] = frozenset(str(surface) for surface in get_args(AgentSurface))


def _normalize(name_list: Iterable[str], *, label: str) -> tuple[AgentSurface, ...]:
    seen: set[str] = set()
    out: list[AgentSurface] = []
    for raw in name_list:
        if raw not in _KNOWN_SURFACES:
            raise ValueError(f"Unknown surface {raw!r}")
        if raw in seen:
            raise ValueError(f"{label} contains duplicate surface {raw!r}")
        seen.add(raw)
        out.append(cast(AgentSurface, raw))
    if not out:
        raise ValueError(f"{label} must be non-empty")
    return tuple(out)


def resolve_active_surfaces(
    spec: TaskSpec,
    *,
    override: Iterable[str] | None,
    supported_surfaces: set[str],
) -> tuple[AgentSurface, ...]:
    """Resolve a task's active agent surfaces.

    Task surface policy is intentionally asymmetric: ``default_surfaces`` are
    used as-is when no override is supplied, while ``allowed_surfaces`` gates
    explicit overrides only. Defaults do not need to be a subset of allowed
    surfaces; this lets legacy MCP defaults coexist with opt-in SDK surfaces.
    """
    if override is None:
        active = _normalize(spec.default_surfaces, label="default_surfaces")
    else:
        # ``allowed_surfaces`` is not a full permission universe. It gates only
        # caller-supplied overrides; task defaults bypass this check by design.
        active = _normalize(override, label="override surfaces")
        for name in active:
            if name not in spec.allowed_surfaces:
                raise ValueError(f"Task {spec.task_id!r} does not allow surface {name!r}")
    for name in active:
        if name not in supported_surfaces:
            raise ValueError(f"Surface {name!r} is not supported by this qtype")
    return active
