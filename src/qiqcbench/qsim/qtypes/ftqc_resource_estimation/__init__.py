"""``ftqc_resource_estimation`` qtype — classical FTQC resource estimation.

A device-less / static-instance qtype: no quantum dynamics, no shots, no async
job model. Synchronous calculator tools cover physical CCZ2T resource estimation
and MPS-initialized QPE resource planning.
"""

from __future__ import annotations

from qiqcbench.qsim.qtypes.ftqc_resource_estimation.backend import (
    GenericFtqcEvaluatorBackend,
    build_ftqc_simulator_backend,
)
from qiqcbench.qsim.qtypes.ftqc_resource_estimation.device import (
    HiddenFtqcConfig,
    PublicFtqcSpec,
)
from qiqcbench.qsim.qtypes.ftqc_resource_estimation.lab_notebook import (
    FtqcLabNotebook,
    build_ftqc_lab_notebook,
)

__all__ = [
    "GenericFtqcEvaluatorBackend",
    "FtqcLabNotebook",
    "HiddenFtqcConfig",
    "PublicFtqcSpec",
    "build_ftqc_lab_notebook",
    "build_ftqc_simulator_backend",
]
