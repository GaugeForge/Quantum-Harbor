"""Deterministic generator for ``adaptive_clustered_clbcs_h2o`` task materials.

Writes the compact hidden scorer YAML under
``configs/task_materials/adaptive_clustered_clbcs_h2o/hidden/hidden_scorer.yaml``.

The public Hamiltonian CSV and the full audit reference JSON are checked in
directly. The hidden state itself lives in the qsim ``hidden_dynamics`` module
(``construction.py``), not in a material file.

"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ruamel.yaml import YAML

from qiqcbench.qsim.hidden_dynamics.adaptive_clustered_clbcs_h2o import construction as C

TASK_ID = "adaptive_clustered_clbcs_h2o"
ROOT = Path(__file__).resolve().parents[2] / "configs" / "task_materials" / TASK_ID

_yaml = YAML()
_yaml.default_flow_style = False
_yaml.width = 4096


def build_scorer_payload() -> dict:
    public_dir = ROOT / "public"
    csv_path = public_dir / C.HAMILTONIAN_FILENAME
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    return {
        "schema_version": 3,
        "task_id": TASK_ID,
        "public_hamiltonian_sha256": sha,
        "num_components": 16,
        "coverage_floor": 1.0e-9,
        "production_settings": 192,
        "production_shots": 256,
        "max_control_variates": 192,
        "pilot_max_settings": 96,
        "pilot_max_shots": 24576,
        "track_a_thresholds": {"pass": 600.0, "bronze": 590.0, "silver": 582.0, "gold": 575.0},
        "track_b_thresholds": {"pass": 0.0440, "bronze": 0.0420, "silver": 0.0400, "gold": 0.0380},
        "track_a_comparison_decimals": 2,
        "track_b_comparison_decimals": 4,
        "energy_report_abs_tolerance_hartree": 0.0001,
        "uncertainty_report_abs_tolerance_hartree": 0.0001,
        "points_validity": 10,
        "points_track_a": 20,
        "points_track_b": 50,
        "points_energy": 10,
        "points_uncertainty": 10,
        "best_known_track_a_v_haar": 570.28886401966963,
        "best_known_track_b_sigma": 0.0367446746005135,
    }


def main() -> None:
    hidden_dir = ROOT / "hidden"
    scorer = build_scorer_payload()
    hidden_dir.mkdir(parents=True, exist_ok=True)
    scorer_path = hidden_dir / "hidden_scorer.yaml"
    if scorer_path.is_file():
        current = YAML(typ="safe").load(scorer_path.read_text(encoding="utf-8"))
        if current == scorer:
            print(
                f"already current: {scorer_path} "
                f"(sha={scorer['public_hamiltonian_sha256'][:12]}...)"
            )
            return
    with scorer_path.open("w", encoding="utf-8") as f:
        _yaml.dump(scorer, f)
    print(f"wrote {scorer_path} (sha={scorer['public_hamiltonian_sha256'][:12]}...)")


if __name__ == "__main__":
    main()
