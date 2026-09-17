"""Synchronous backend for the ``neutral_atom_ftqc_compiler`` qtype.

There is **no engine, no shots, no async job**: the "backend" serves the public
compilation target. ``get_target_circuit`` returns the fixed modular-multiplier
Clifford+T netlist + dependency DAG for this run's instance seed. The cost model
is fully public (in the device spec), so the agent computes its own
physical-qubit-seconds; the verifier independently recomputes the authoritative
score from the submitted compiled program. The backend therefore exposes no cost
oracle.
"""

from __future__ import annotations

import os
from typing import Any

from qiqcbench.qsim.qtypes.neutral_atom_ftqc_compiler.device import (
    HiddenNeutralAtomConfig,
    PublicNeutralAtomSpec,
)
from qiqcbench.qsim.qtypes.neutral_atom_ftqc_compiler.instance import load_target_netlist

__all__ = ["NeutralAtomCompilerBackend", "build_neutral_atom_simulator_backend"]


def _resolve_instance_seed(env: dict[str, str] | None = None) -> int:
    env_map = os.environ if env is None else env
    raw = env_map.get("QIQCBENCH_INSTANCE_SEED")
    if raw is None or raw == "":
        return 0
    return int(raw)


class NeutralAtomCompilerBackend:
    """Serves the public compilation-target netlist (synchronous, no jobs)."""

    def __init__(self, hidden: HiddenNeutralAtomConfig, public: PublicNeutralAtomSpec):
        self._public = public
        self._hidden = hidden
        self._seed = _resolve_instance_seed()

    @property
    def instance_seed(self) -> int:
        return self._seed

    def target_circuit(self) -> dict[str, Any]:
        """Return the fixed Clifford+T netlist + dependency DAG for this instance."""
        netlist = load_target_netlist(self._public.task_id, self._seed)
        return netlist


def build_neutral_atom_simulator_backend(
    hidden: HiddenNeutralAtomConfig, public: PublicNeutralAtomSpec
) -> NeutralAtomCompilerBackend:
    return NeutralAtomCompilerBackend(hidden, public)
