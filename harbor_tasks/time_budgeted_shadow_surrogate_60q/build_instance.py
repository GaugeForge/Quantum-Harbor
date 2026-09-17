#!/usr/bin/env python3
"""Validate or materialize the single tracked Shadow apparatus.

No private study state, cached answer table, certificate or slot selection is
required. This historical command name is retained for contributor workflows;
it does not resample or choose a different physical instance.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ruamel.yaml import YAML

from qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q.runtime import (
    HIDDEN_DEVICE_RELATIVE,
    RUNTIME_FILES,
    build_runtime_truth,
)
from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.device import HiddenBoundedgateConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Optional NEW configs directory for exact copies of the four runtime files.",
    )
    args = parser.parse_args()
    configs = Path(__file__).resolve().parents[2] / "configs"
    hidden = HiddenBoundedgateConfig.model_validate(
        YAML(typ="safe").load((configs / HIDDEN_DEVICE_RELATIVE).read_text())
    )
    truth = build_runtime_truth(hidden)
    if args.output_root is not None:
        # Exclusive creation: never overwrite another task/run's material.
        args.output_root.mkdir(parents=True, exist_ok=False)
        for relative in RUNTIME_FILES:
            destination = args.output_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(configs / relative, destination)
    print(
        json.dumps(
            {
                "task_id": truth["task_id"],
                "runtime_files": len(RUNTIME_FILES),
                "targets": len(truth["labels"]),
                "responses_per_target": len(truth["labels"][0]),
                "reference_source": "bound physical device; computed in memory",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
