"""nv_sensor_network qtype descriptor (discovered by qtypes/registry.py)."""

from __future__ import annotations

from qiqcbench.qsim.core.wire import JobNvSensingData
from qiqcbench.qsim.qtypes.descriptor import QtypeDescriptor, lazy, lazy_instructions
from qiqcbench.qsim.qtypes.nv_sensor_network.device import (
    HiddenNvSensorNetworkConfig,
    PublicNvSensorNetworkSpec,
)
from qiqcbench.qsim.qtypes.nv_sensor_network.lab_notebook import (
    NvSensorNetworkLabNotebook,
    build_nv_sensor_network_lab_notebook,
)

_PKG = "qiqcbench.qsim.qtypes.nv_sensor_network"

DESCRIPTOR = QtypeDescriptor(
    qtype="nv_sensor_network",
    public_model=PublicNvSensorNetworkSpec,
    hidden_model=HiddenNvSensorNetworkConfig,
    notebook_model=NvSensorNetworkLabNotebook,
    build_lab_notebook=build_nv_sensor_network_lab_notebook,
    result_data_models=(JobNvSensingData,),
    supported_backend_modes=frozenset({"simulator"}),
    supported_surfaces=frozenset({"mcp"}),
    supported_capabilities=frozenset({"basic_measurement", "network_field_sensing"}),
    build_simulator_backend=lazy(f"{_PKG}.backend", "build_nv_sensor_network_simulator_backend"),
    register_mcp_tools=lazy(f"{_PKG}.mcp", "register_mcp_tools"),
    instructions=lazy_instructions(f"{_PKG}.mcp", "INSTRUCTIONS"),
)
