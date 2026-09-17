"""MCP tool registration for ion_trap_gate_model.

``run_ion_circuit`` and ``run_randomized_measurement_batch`` are registered iff
the ``randomized_measurement`` capability is active. Always-on tools
(list_devices, get_device_spec, get_lab_notebook, get_job_result,
submit_final_answer) are registered by the central MCP server.
"""

from __future__ import annotations

from typing import Any

from qiqcbench.qsim.qtypes.ion_trap_gate_model.wire import (
    IonCircuitRequest,
    RandomizedMeasurementBatchRequestV2,
)

INSTRUCTIONS = """\
You operate a 16-qubit all-to-all trapped-ion gate-model processor.
Native gates (qubit 0 = least-significant bit):
  single-qubit: rx(theta), ry(theta), rz(theta) (universal; GPI/GPI2-equivalent),
                plus x, y, z, h, s, sdg.
  two-qubit:    ms(theta) on a chosen PAIR [i, j] implementing
                exp(-i (theta/2) X_i X_j). This is the ONLY entangler; its
                depolarizing error dominates, so your MS-gate count sets your
                achieved fidelity.
Connectivity is all-to-all (no routing/SWAP needed). You receive raw per-shot
bitstrings; thresholding / post-processing / estimator choices are yours.

Tools (async: each returns {job_id, status}; poll get_job_result until complete):
  run_ion_circuit(request)
      request = {shots, circuit: [GateOp...], measured_qubits: [int...]}
      Run one native-gate circuit; returns raw bitstrings (bit i = measured_qubits[i]).
  run_randomized_measurement_batch(request)
      request = {prepare_circuit: [GateOp...], n_unitaries, shots_per_unitary,
                 subsystem_qubits: [int...], experiment_tag: object|null,
                 protocol_version: 2}
      For each of n_unitaries settings, prepares the noisy state and applies a
      fresh qsim-private Haar-random single-qubit basis on every subsystem qubit,
      then returns shots_per_unitary raw subsystem bitstrings. Optional task tags
      label a condition before execution; consult instruction.md for task-specific
      tag and estimator requirements.
A GateOp is {"kind":"gate", "name": <gate>, "qubits": [int...], "params": [float...]}.
"""


def register_mcp_tools(server, state, **_kwargs: Any) -> None:
    if "randomized_measurement" not in state.active_capabilities:
        return

    from qiqcbench.qsim.actions.base import ActionContext
    from qiqcbench.qsim.actions.randomized_measurement import (
        submit_ion_circuit,
        submit_randomized_measurement_batch,
    )

    @server.tool(
        name="run_ion_circuit",
        description="Run one native-gate circuit and read out the requested qubits.",
    )
    def _run_ion_circuit(request: IonCircuitRequest) -> dict[str, Any]:
        """Run one native-gate circuit and read out the requested qubits."""
        return submit_ion_circuit(ActionContext(state=state, surface="mcp"), request)

    @server.tool(
        name="run_randomized_measurement_batch",
        description=(
            "Prepare a state once and sample N_U CUE-random local bases on a "
            "subsystem (second-Renyi-entropy / classical-shadow protocol)."
        ),
    )
    def _run_rm_batch(request: RandomizedMeasurementBatchRequestV2) -> dict[str, Any]:
        """Submit a randomized-measurement batch for entropy estimation."""
        return submit_randomized_measurement_batch(
            ActionContext(state=state, surface="mcp"), request
        )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
