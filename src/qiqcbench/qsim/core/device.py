"""Qtype-agnostic device union surface.

Concrete per-qtype schemas live in the qtype packages and are registered via
``qtypes/registry.py``. This module builds the discriminated-union *type
aliases* keyed on ``qtype`` dynamically from the registry, so adding a qtype
never edits this file.

``PublicDeviceSpec`` and ``HiddenDeviceConfig`` are aliases, not concrete
classes; they cannot be ``.model_validate(...)``-ed directly. Use
``TypeAdapter(...).validate_python(...)`` (see ``qsim/devices.py``).
"""

from __future__ import annotations

from functools import reduce
from operator import or_
from typing import Annotated as _Annotated

from pydantic import Field as _Field

from qiqcbench.qsim.qtypes.registry import all_descriptors


def _discriminated_union(models: tuple[type, ...]):
    """Build ``Annotated[m0 | m1 | ..., Field(discriminator="qtype")]``.

    Members are taken in the registry's deterministic (qtype-sorted) order so
    generated JSON schemas and validation error messages are reproducible.
    """
    if not models:
        raise RuntimeError("no qtype device models registered")
    if len(models) == 1:
        # A single member cannot form a discriminated union; the lone model
        # already validates by itself.
        return models[0]
    return _Annotated[reduce(or_, models), _Field(discriminator="qtype")]


_descriptors = all_descriptors()

PublicDeviceSpec = _discriminated_union(tuple(d.public_model for d in _descriptors))
HiddenDeviceConfig = _discriminated_union(tuple(d.hidden_model for d in _descriptors))

__all__ = ["HiddenDeviceConfig", "PublicDeviceSpec"]
