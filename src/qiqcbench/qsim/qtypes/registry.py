"""Single qtype descriptor registry.

The registry is the only allowed central qtype map. Top-level runtime modules
(``runner``, ``devices``, ``backends.factory``, ``mcp.server``) may ask the
registry for per-qtype capabilities but must not hardcode their own
``if qtype == ...`` ladders.

Qtypes are **auto-discovered**: every package directly under ``qsim/qtypes/``
whose name does not start with ``_`` must expose a ``QtypeDescriptor`` as
``DESCRIPTOR`` in its ``qtype.py``. Adding a qtype is therefore dropping a
``qtypes/<name>/`` folder — no central file is edited here. Shared helper
packages (not qtypes) must be named with a leading underscore (e.g.
``_shared``) so the scanner skips them.

The descriptors are import-light (schema eager, runtime callables lazy), so
building the registry does not import engines/backends/mcp. That keeps the
dynamic device / lab-notebook / capability / ``JobData`` unions free of import
cycles.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor

_QTYPES_DIR = Path(__file__).resolve().parent
_QTYPES_PKG = "qiqcbench.qsim.qtypes"


def _discover_descriptors() -> dict[str, QtypeDescriptor]:
    """Import every ``qtypes/<name>/qtype.py`` and collect its ``DESCRIPTOR``.

    Fails loud (never silently skips) on a malformed qtype package so a broken
    contribution surfaces at import time rather than as a mysterious
    ``Unsupported qtype`` later.
    """
    registry: dict[str, QtypeDescriptor] = {}
    for module_info in sorted(pkgutil.iter_modules([str(_QTYPES_DIR)]), key=lambda m: m.name):
        if not module_info.ispkg or module_info.name.startswith("_"):
            continue
        package = module_info.name
        qtype_module = f"{_QTYPES_PKG}.{package}.qtype"
        try:
            module = importlib.import_module(qtype_module)
        except ModuleNotFoundError as exc:
            # A non-underscore package with no qtype.py is a mistake: either it
            # is a qtype (add qtype.py) or a shared helper (prefix with '_').
            if exc.name == qtype_module:
                raise ValueError(
                    f"qtype package {package!r} has no {package}/qtype.py; "
                    "add one exposing DESCRIPTOR, or rename the package with a "
                    "leading underscore if it is a shared helper, not a qtype."
                ) from exc
            raise
        try:
            descriptor = module.DESCRIPTOR
        except AttributeError as exc:
            raise ValueError(f"{qtype_module} must define a module-level DESCRIPTOR") from exc
        if not isinstance(descriptor, QtypeDescriptor):
            raise TypeError(
                f"{qtype_module}.DESCRIPTOR must be a QtypeDescriptor, "
                f"got {type(descriptor).__name__}"
            )
        if descriptor.qtype != package:
            raise ValueError(
                f"qtype package {package!r} declares mismatched "
                f"descriptor qtype {descriptor.qtype!r}; the two must match."
            )
        if descriptor.qtype in registry:
            raise ValueError(f"Duplicate qtype descriptor: {descriptor.qtype!r}")
        registry[descriptor.qtype] = descriptor
    if not registry:
        raise RuntimeError(f"No qtype descriptors discovered under {_QTYPES_PKG}")
    return registry


_REGISTRY: dict[str, QtypeDescriptor] = _discover_descriptors()


def descriptor_for_qtype(qtype: str) -> QtypeDescriptor:
    """Return the ``QtypeDescriptor`` for ``qtype`` or raise ``ValueError``."""
    try:
        return _REGISTRY[qtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported qtype: {qtype!r}") from exc


def all_descriptors() -> tuple[QtypeDescriptor, ...]:
    """Return every registered descriptor, ordered by qtype for determinism.

    The dynamic device / lab-notebook / capability / ``JobData`` unions are
    built from this, so a stable order keeps generated JSON schemas and error
    messages reproducible.
    """
    return tuple(_REGISTRY[qtype] for qtype in sorted(_REGISTRY))


__all__ = ["QtypeDescriptor", "all_descriptors", "descriptor_for_qtype"]
