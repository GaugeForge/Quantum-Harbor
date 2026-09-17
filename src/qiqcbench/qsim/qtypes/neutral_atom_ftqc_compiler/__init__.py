"""``neutral_atom_ftqc_compiler`` qtype — fault-tolerant neutral-atom compilation.

A device-less / static-instance qtype: no quantum dynamics, no
shots, no async job model. A single synchronous calculator tool
(``get_target_circuit``) serves a fixed Clifford+T compilation target over a
public, authoritative deterministic resource/error/timing cost model (surface-code
tiles, transversal vs lattice-surgery CNOTs, magic-state factories, AOD movement
constraints).
"""

from __future__ import annotations

from qiqcbench.qsim.qtypes.neutral_atom_ftqc_compiler.backend import (
    NeutralAtomCompilerBackend,
    build_neutral_atom_simulator_backend,
)
from qiqcbench.qsim.qtypes.neutral_atom_ftqc_compiler.device import (
    HiddenNeutralAtomConfig,
    PublicNeutralAtomSpec,
)
from qiqcbench.qsim.qtypes.neutral_atom_ftqc_compiler.lab_notebook import (
    NeutralAtomLabNotebook,
    build_neutral_atom_lab_notebook,
)

__all__ = [
    "HiddenNeutralAtomConfig",
    "NeutralAtomCompilerBackend",
    "NeutralAtomLabNotebook",
    "PublicNeutralAtomSpec",
    "build_neutral_atom_lab_notebook",
    "build_neutral_atom_simulator_backend",
]
