"""surface_code_memory qtype descriptor (discovered by qtypes/registry.py).

A rotated distance-d surface-code memory experiment. One batched experiment tool
(run_memory_experiment) on the async job model, gated on the
``surface_code_memory_calibration`` capability. Simulator-only, MCP-only; the engine is
run-long stateful (the shot budget accumulates across calls). No standard
run_pulse/run_circuit runners — the descriptor leaves those slots fail-closed by default.
"""

from __future__ import annotations

from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.surface_code_memory.device import (
    HiddenSurfaceCodeConfig,
    PublicSurfaceCodeSpec,
)
from qiqcbench.qsim.qtypes.surface_code_memory.lab_notebook import (
    SurfaceCodeLabNotebook,
    build_surface_code_memory_lab_notebook,
)
from qiqcbench.qsim.qtypes.surface_code_memory.wire import (
    JobDetectorData,
    JobDriftControlData,
    JobDriftStreamData,
    JobLeakageChallengeData,
    JobLeakageMemoryData,
    JobSyndromeControllerProgramData,
    JobSyndromeControlProbeDataV3,
)

_PKG = "qiqcbench.qsim.qtypes.surface_code_memory"

DESCRIPTOR = QtypeDescriptor(
    qtype="surface_code_memory",
    public_model=PublicSurfaceCodeSpec,
    hidden_model=HiddenSurfaceCodeConfig,
    notebook_model=SurfaceCodeLabNotebook,
    build_lab_notebook=build_surface_code_memory_lab_notebook,
    result_data_models=(
        JobDetectorData,
        JobSyndromeControlProbeDataV3,
        JobSyndromeControllerProgramData,
        JobDriftStreamData,
        JobDriftControlData,
        JobLeakageMemoryData,
        JobLeakageChallengeData,
    ),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset(
        {
            "surface_code_memory_calibration",
            "syndrome_feedback_control",
            "drift_recalibration_scheduling",
            "heralded_leakage_decoding",
        }
    ),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_surface_code_memory_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
