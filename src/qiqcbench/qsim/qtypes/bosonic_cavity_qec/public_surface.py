"""Versioned qtype- and device-keyed public surfaces for ``bosonic_cavity_qec``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from ruamel.yaml import YAML

from qiqcbench.qsim.qtypes.bosonic_cavity_qec.lab_notebook import (
    BosonicCavityLabNotebook,
)

_yaml = YAML(typ="safe")
_yaml.allow_duplicate_keys = False


class BosonicCavityQtypeAgentSurface(BaseModel):
    """Exact task-neutral instructions rendered by the qtype MCP server."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    qtype: Literal["bosonic_cavity_qec"] = "bosonic_cavity_qec"
    mcp_instructions: str = Field(min_length=1)


def load_bosonic_cavity_qtype_agent_surface(
    path: str | Path | None = None,
) -> BosonicCavityQtypeAgentSurface:
    """Load task-neutral qtype instructions from canonical public material."""

    if path is None:
        # Local import avoids a device-registry cycle during qtype discovery.
        from qiqcbench.qsim.devices import configs_root

        path = configs_root() / "qtypes" / "bosonic_cavity_qec.agent_surface.yaml"
    with Path(path).open(encoding="utf-8") as stream:
        payload = _yaml.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("bosonic-cavity qtype agent surface must be a YAML mapping")
    return BosonicCavityQtypeAgentSurface.model_validate(payload)


def load_bosonic_cavity_device_notebook(
    device_id: str,
    path: str | Path | None = None,
) -> BosonicCavityLabNotebook:
    """Load one device-keyed public notebook without consulting hidden truth."""

    if not isinstance(device_id, str) or not device_id:
        raise ValueError("bosonic-cavity public notebook requires a device ID")
    if path is None:
        from qiqcbench.qsim.devices import configs_root

        path = configs_root() / "devices" / f"{device_id}.notebook.yaml"
    with Path(path).open(encoding="utf-8") as stream:
        payload = _yaml.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("bosonic-cavity public notebook must be a YAML mapping")
    notebook = BosonicCavityLabNotebook.model_validate(payload)
    if notebook.device_id != device_id:
        raise ValueError("bosonic-cavity public notebook device_id does not match its catalog key")
    return notebook


__all__ = [
    "BosonicCavityQtypeAgentSurface",
    "load_bosonic_cavity_device_notebook",
    "load_bosonic_cavity_qtype_agent_surface",
]
