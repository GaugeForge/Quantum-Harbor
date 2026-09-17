from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SurfaceRuntime:
    name: str
    route_prefix: str
