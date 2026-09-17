"""MCP tool registration for the ``logical_circuit_optimizer`` qtype.

Registers one **synchronous** calculator tool (no ``job_id``, no polling), gated on
the ``clifford_t_optimization`` capability. The task runs no quantum dynamics; the
agent reads the opaque circuit and submits an optimized one via ``submit_final_answer``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import logical_circuit_optimizer as lco_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Logical Clifford+T circuit-optimization MCP server. This is a classical logical-"
    "layer optimization task, not a quantum experiment: there are no shots and no async "
    "jobs; the one tool is a synchronous calculator. Read get_device_spec, "
    "get_lab_notebook, and get_optimization_instance to obtain the opaque circuit, fixed "
    "unitary gate set, and public qubit/gate-entry limits. Return a unitary circuit that "
    "implements the same operation on the data qubits with fewer T/T-dagger gates. Extra "
    "qubits start in |0> and are discarded. The verifier checks the submitted endpoint "
    "circuit and recounts T/T-dagger; lower is better. Submit with submit_final_answer."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    ctx = ActionContext(state=state, surface="mcp")

    if "clifford_t_optimization" in state.active_capabilities:

        @mcp.tool()
        def get_optimization_instance(device_id: str) -> dict:
            """Return the opaque Clifford+T circuit to optimize + the objective.

            Synchronous (no job_id). Returns the circuit (canonical
            qiqc_clifford_t_unitary_v2
            gate list, also mounted at /task_materials/input_circuit.json), n_qubits,
            allowed gate_set, public resource limits, initial T+T-dagger count, and the
            objective. There is no optimization or equivalence oracle.
            """
            return lco_actions.get_optimization_instance(ctx, device_id=device_id)


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
