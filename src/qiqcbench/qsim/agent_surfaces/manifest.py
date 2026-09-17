from __future__ import annotations


def build_surface_manifest(
    *,
    task_id: str | None,
    qtype: str,
    backend_mode: str,
    default_surfaces: tuple[str, ...],
    allowed_surfaces: tuple[str, ...],
    active_surfaces: tuple[str, ...],
    override_source: str | None,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "qtype": qtype,
        "backend_mode": backend_mode,
        "default_surfaces": list(default_surfaces),
        "allowed_surfaces": list(allowed_surfaces),
        "active_surfaces": list(active_surfaces),
        "override_source": override_source,
    }
