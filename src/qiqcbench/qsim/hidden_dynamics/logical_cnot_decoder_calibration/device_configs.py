"""Materializers for ``logical_cnot_decoder_calibration``: device YAMLs + task materials.

Public materials (agent-facing, mounted at ``/task_materials`` in the Harbor bundle) live
under ``configs/task_materials/<task>/public/``: the public device spec, the stale
notebook, and the machine-readable per-layout detector-index + matching-graph adjacency
JSONs. The hidden noise model lives under ``.../hidden/`` (qsim-only) and in
``configs/devices/<id>.hidden.example.yaml``.
"""

from __future__ import annotations

import json
from pathlib import Path

from ruamel.yaml import YAML

from qiqcbench.qsim.hidden_dynamics.logical_cnot_decoder_calibration import construction as C
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery import metadata as MD

_yaml = YAML()
_yaml.default_flow_style = False
_yaml.width = 4096


def _dump_yaml(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        _yaml.dump(obj, fh)


def _dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True))


def write_device_configs(devices_dir: str | Path) -> dict[str, Path]:
    devices = Path(devices_dir)
    public_path = devices / f"{C.DEVICE_ID}.public.yaml"
    hidden_path = devices / f"{C.DEVICE_ID}.hidden.example.yaml"
    _dump_yaml(C.build_public_spec(), public_path)
    _dump_yaml(C.build_hidden_config().model_dump(), hidden_path)
    return {"public": public_path, "hidden": hidden_path}


def write_public_materials(public_dir: str | Path) -> dict[str, Path]:
    public = Path(public_dir)
    out: dict[str, Path] = {}
    _dump_yaml(C.build_public_spec(), public / "public_spec.yaml")
    out["public_spec"] = public / "public_spec.yaml"
    _dump_yaml(C.stale_notebook_payload(), public / "stale_notebook.yaml")
    out["stale_notebook"] = public / "stale_notebook.yaml"

    for layout in ("d3_control", "d3_intermediate", "d3_target", "d5"):
        path = public / f"memory_{layout}_scoring.json"
        _dump_json(MD.memory_metadata(layout, C.MEMORY_SCORING_ROUNDS), path)
        out[f"memory_{layout}"] = path
    for window in ("zz", "xx"):
        path = public / f"merged_{window}_transitions.json"
        _dump_json(MD.merged_metadata(window, 3, True), path)
        out[f"merged_{window}"] = path
    path = public / "cnot_superset.json"
    _dump_json(MD.cnot_superset_metadata(), path)
    out["cnot_superset"] = path
    return out


def write_hidden_materials(hidden_dir: str | Path) -> dict[str, Path]:
    hidden = Path(hidden_dir)
    path = hidden / "hidden_noise_model.yaml"
    _dump_yaml(C.build_hidden_config().model_dump(), path)
    return {"hidden_noise_model": path}


def write_all(materials_root: str | Path, devices_dir: str | Path) -> dict[str, Path]:
    root = Path(materials_root)
    out = write_device_configs(devices_dir)
    out.update(write_public_materials(root / "public"))
    out.update(write_hidden_materials(root / "hidden"))
    return out
