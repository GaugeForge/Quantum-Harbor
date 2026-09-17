from __future__ import annotations

from qiqcbench.qsim.actions import (
    bose_hubbard,
    common,
    digital,
    nv_sensor_network,
    transmon,
)
from qiqcbench.qsim.actions.base import ActionDescriptor

COMMON_ACTIONS: dict[str, ActionDescriptor] = {
    "list_devices": ActionDescriptor("list_devices", common.list_devices),
    "get_device_spec": ActionDescriptor("get_device_spec", common.get_device_spec),
    "get_lab_notebook": ActionDescriptor("get_lab_notebook", common.get_lab_notebook),
    "get_job_result": ActionDescriptor("get_job_result", common.get_job_result),
    "submit_final_answer": ActionDescriptor("submit_final_answer", common.submit_final_answer),
}

DIGITAL_ACTIONS: dict[str, ActionDescriptor] = {
    "submit_circuit": ActionDescriptor("submit_circuit", digital.submit_circuit),
    "submit_circuit_sweep": ActionDescriptor("submit_circuit_sweep", digital.submit_circuit_sweep),
}

TRANSMON_ACTIONS: dict[str, ActionDescriptor] = {
    "submit_pulse_sequence": ActionDescriptor(
        "submit_pulse_sequence", transmon.submit_pulse_sequence
    ),
    "submit_pulse_sweep": ActionDescriptor("submit_pulse_sweep", transmon.submit_pulse_sweep),
}

BOSE_HUBBARD_ACTIONS: dict[str, ActionDescriptor] = {
    "submit_evolution": ActionDescriptor("submit_evolution", bose_hubbard.submit_evolution),
    "submit_evolution_sweep": ActionDescriptor(
        "submit_evolution_sweep", bose_hubbard.submit_evolution_sweep
    ),
}

NV_SENSOR_NETWORK_ACTIONS: dict[str, ActionDescriptor] = {
    "submit_sensing_probe": ActionDescriptor(
        "submit_sensing_probe", nv_sensor_network.submit_sensing_probe
    ),
    "submit_sensing_sweep": ActionDescriptor(
        "submit_sensing_sweep", nv_sensor_network.submit_sensing_sweep
    ),
}

ACTIONS: dict[str, ActionDescriptor] = {
    **COMMON_ACTIONS,
    **DIGITAL_ACTIONS,
    **TRANSMON_ACTIONS,
    **BOSE_HUBBARD_ACTIONS,
    **NV_SENSOR_NETWORK_ACTIONS,
}


__all__ = [
    "ACTIONS",
    "BOSE_HUBBARD_ACTIONS",
    "COMMON_ACTIONS",
    "DIGITAL_ACTIONS",
    "NV_SENSOR_NETWORK_ACTIONS",
    "TRANSMON_ACTIONS",
]
