"""Versioned qtype- and device-keyed neutral-atom public surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from ruamel.yaml import YAML

from qiqcbench.qsim.qtypes.neutral_atom_logical_processor.lab_notebook import (
    NeutralAtomLabNotebook,
)

_yaml = YAML(typ="safe")
_yaml.allow_duplicate_keys = False


class NeutralAtomQtypeAgentSurface(BaseModel):
    """Exact task-neutral instructions rendered by the qtype MCP server."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    qtype: Literal["neutral_atom_logical_processor"] = "neutral_atom_logical_processor"
    mcp_instructions: str = Field(min_length=1)


def load_neutral_atom_qtype_agent_surface(
    path: str | Path | None = None,
) -> NeutralAtomQtypeAgentSurface:
    """Load task-neutral qtype instructions from canonical public material."""

    if path is None:
        # Local import avoids a device-registry cycle during qtype discovery.
        from qiqcbench.qsim.devices import configs_root

        path = configs_root() / "qtypes" / "neutral_atom_logical_processor.agent_surface.yaml"
    with Path(path).open(encoding="utf-8") as stream:
        payload = _yaml.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("neutral-atom qtype agent surface must be a YAML mapping")
    return NeutralAtomQtypeAgentSurface.model_validate(payload)


def load_neutral_atom_device_notebook(
    device_id: str,
    path: str | Path | None = None,
) -> NeutralAtomLabNotebook:
    """Load one device-keyed public notebook without consulting hidden truth."""

    if not isinstance(device_id, str) or not device_id:
        raise ValueError("neutral-atom public notebook requires a device ID")
    if path is None:
        from qiqcbench.qsim.devices import configs_root

        path = configs_root() / "devices" / f"{device_id}.notebook.yaml"
    with Path(path).open(encoding="utf-8") as stream:
        payload = _yaml.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("neutral-atom public notebook must be a YAML mapping")
    notebook = NeutralAtomLabNotebook.model_validate(payload)
    if notebook.device_id != device_id:
        raise ValueError("neutral-atom public notebook device_id does not match its catalog key")
    return notebook


__all__ = [
    "NeutralAtomQtypeAgentSurface",
    "load_neutral_atom_device_notebook",
    "load_neutral_atom_qtype_agent_surface",
]
