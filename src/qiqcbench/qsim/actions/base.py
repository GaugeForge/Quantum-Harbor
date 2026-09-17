from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActionContext:
    state: Any
    surface: str


@dataclass(frozen=True)
class ActionDescriptor:
    name: str
    handler: Callable[..., Any]
