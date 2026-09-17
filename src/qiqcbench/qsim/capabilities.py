"""Capability-family resolution.

Capability-family names are contributed by qtypes through their descriptors
(``QtypeDescriptor.supported_capabilities``); the known set is the union across
all registered qtypes. Adding a capability for a new qtype therefore happens in
that qtype's ``qtype.py`` — this module is never edited.

``CapabilityFamily`` is kept as a readable alias for ``str``; validation is
performed dynamically against the registry-derived known set plus the per-qtype
supported set, not a static ``Literal``.
"""

from __future__ import annotations

from typing import cast

from qiqcbench.qsim.qtypes.registry import all_descriptors

CapabilityFamily = str


def known_capabilities() -> frozenset[str]:
    """Union of every registered qtype's supported capability families."""
    known: set[str] = set()
    for descriptor in all_descriptors():
        known |= set(descriptor.supported_capabilities)
    return frozenset(known)


_KNOWN_CAPABILITIES = known_capabilities()


def resolve_enabled_capabilities(
    enabled: tuple[str, ...],
    *,
    supported_capabilities: set[str],
) -> tuple[CapabilityFamily, ...]:
    if not enabled:
        raise ValueError("enabled_capabilities must be non-empty")

    resolved: list[CapabilityFamily] = []
    seen: set[str] = set()
    for capability in enabled:
        if capability not in _KNOWN_CAPABILITIES:
            raise ValueError(f"Unknown capability {capability!r}")
        if capability in seen:
            raise ValueError(f"Duplicate capability {capability!r}")
        if capability not in supported_capabilities:
            raise ValueError(f"capability {capability!r} is not supported by this qtype")
        seen.add(capability)
        resolved.append(cast(CapabilityFamily, capability))

    return tuple(resolved)
