"""Deterministic generator for the ``time_budgeted_hamlearn_10q`` materials.

Materializes everything downstream of the fixed hidden instance:

  * public + hidden task materials under
    ``configs/task_materials/time_budgeted_hamlearn_10q/`` (the digest-bound
    public dictionary / spec / notebook and the hidden scorer);
  * the analog-qtype device configs under ``configs/devices/`` (the public
    spec returned by ``get_device_spec`` and the hidden ``*.example.yaml``
    carrying the true coefficients).

The hidden scorer is deliberately NOT copied into the agent-facing Harbor
bundle or its generic trial artifacts. The separate verifier image receives it
only through the allowlisted build context declared by
``tests/verifier-context.json``; local development resolves the tracked
``configs/task_materials`` copy through the scorer's repo-relative fallback.

All construction logic and invariant enforcement live inside the qsim
hidden-dynamics module; this script is a thin, deterministic wrapper.
"""

from __future__ import annotations

from pathlib import Path

from qiqcbench.qsim.hidden_dynamics.time_budgeted_hamlearn_10q.construction import (
    write_materials,
)
from qiqcbench.qsim.hidden_dynamics.time_budgeted_hamlearn_10q.device_configs import (
    write_device_configs,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TASK_ID = "time_budgeted_hamlearn_10q"
_MATERIALS_ROOT = _REPO_ROOT / "configs" / "task_materials" / _TASK_ID
_DEVICES_DIR = _REPO_ROOT / "configs" / "devices"


def main() -> None:
    write_materials(_MATERIALS_ROOT)
    write_device_configs(_DEVICES_DIR)


if __name__ == "__main__":
    main()
