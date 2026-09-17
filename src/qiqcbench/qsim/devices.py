"""Device + task config loaders.

Public configs ship in `configs/devices/*.public.yaml`. Hidden truth comes
from a path supplied at server start (typically a Docker bind mount in the
qsim container, never accessible from the agent container).
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import TypeAdapter
from ruamel.yaml import YAML

from qiqcbench.qsim.core.device import HiddenDeviceConfig, PublicDeviceSpec
from qiqcbench.qsim.core.lab_notebook import AnyLabNotebook, DigitalLabNotebook, LabNotebook
from qiqcbench.qsim.qtypes.registry import descriptor_for_qtype

_yaml = YAML(typ="safe")

# `HiddenDeviceConfig` and `PublicDeviceSpec` are now discriminated-union type
# aliases (see core/device.py); validation goes through TypeAdapter rather than
# `.model_validate(...)`.
_PUBLIC_ADAPTER = TypeAdapter(PublicDeviceSpec)
_HIDDEN_ADAPTER = TypeAdapter(HiddenDeviceConfig)

__all__ = [
    "AnyLabNotebook",
    "DigitalLabNotebook",
    "LabNotebook",
    "configs_root",
    "lab_notebook_from_hidden",
    "list_known_devices",
    "load_hidden_config",
    "load_public_spec",
]


def configs_root() -> Path:
    """Resolve the configs/ directory.

    Order:
      1. $QIQCBENCH_CONFIGS env var (used inside containers).
      2. Repo-relative: <package parent>/../configs.
    """
    if env := os.environ.get("QIQCBENCH_CONFIGS"):
        return Path(env)
    here = Path(__file__).resolve()
    # src/qiqcbench/qsim/devices.py -> repo root
    return here.parents[3] / "configs"


def list_known_devices() -> list[str]:
    root = configs_root() / "devices"
    if not root.exists():
        return []
    return sorted(p.stem.removesuffix(".public") for p in root.glob("*.public.yaml"))


def load_public_spec(device_id: str):
    path = configs_root() / "devices" / f"{device_id}.public.yaml"
    with path.open() as f:
        data = _yaml.load(f)
    return _PUBLIC_ADAPTER.validate_python(data)


def load_hidden_config(path: str | Path):
    with Path(path).open() as f:
        data = _yaml.load(f)
    return _HIDDEN_ADAPTER.validate_python(data)


def lab_notebook_from_hidden(hidden) -> AnyLabNotebook:
    """Build the lab-notebook view from the hidden config via the qtype registry."""
    desc = descriptor_for_qtype(hidden.qtype)
    return desc.build_lab_notebook(hidden)
