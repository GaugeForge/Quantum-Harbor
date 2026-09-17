"""MCP tool registration for the ``neutral_atom_ftqc_compiler`` qtype.

Registers one **synchronous** calculator tool (gated on the
``neutral_atom_ftqc_compilation`` capability): ``get_target_circuit``. It returns
the compilation-target netlist directly — no ``job_id``, no polling — mirroring
the always-on synchronous ``submit_final_answer`` / ``get_device_spec`` tools.
The async-job invariant governs ``run_*`` *experiment* tools only; this qtype
runs no quantum dynamics.

The full deterministic cost model is public (returned by ``get_device_spec``), so
the agent computes its own physical-qubit-seconds and there is deliberately no
cost oracle: the engineering challenge is producing a correct, AOD-legal,
in-budget compilation, not querying a black box.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import neutral_atom_ftqc as na_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover - import-only typing aid
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Fault-tolerant neutral-atom FTQC compilation MCP server. The task is "
    "compilation/resource-estimation, not a quantum experiment: there are no shots "
    "and no async jobs. get_device_spec returns the PUBLIC, authoritative cost model "
    "(surface-code tiles of 2*d^2 atoms; logical-error model P(d)=a*(p/p_star)^"
    "floor((d+1)/2); transversal logical CNOT = move + 1 global CZ + O(1) rounds vs "
    "lattice surgery = d rounds; 15-to-1 magic-state factories; AOD movement "
    "constraints; the logical-error budget eps). get_target_circuit (available when "
    "the neutral_atom_ftqc_compilation capability is active) returns the fixed "
    "controlled modular-multiplier Clifford+T netlist + dependency DAG you must "
    "compile, plus the maintainer-pinned budget reference depth n_ref_cycles. You are "
    "scored on physical-qubit-seconds of your compiled program (LOWER is better), "
    "gated by FT-correctness + AOD-legality + the error budget. Choose the code "
    "distance, transversal-vs-lattice-surgery per CNOT, magic-state factory count, "
    "AOD-legal movement, and an explicit FT syndrome + T-teleport schedule, then "
    "submit_final_answer. The verifier recomputes the score from your schedule; "
    "self-reported numbers are not trusted."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    """Attach the synchronous calculator tool to ``mcp`` for the qsim state."""
    ctx = ActionContext(state=state, surface="mcp")

    if "neutral_atom_ftqc_compilation" in state.active_capabilities:

        @mcp.tool()
        def get_target_circuit(device_id: str) -> dict:
            """Return the fixed compilation-target netlist (gates + dependency DAG).

            Synchronous (no job_id). The payload is the controlled modular-multiplier
            Clifford+T netlist for this run's instance: ``gates`` (ordered
            ``{id, op, qubits}``), ``edges`` (``[pred_id, succ_id]`` dependency DAG),
            ``n_data_qubits``, ``n_ref_cycles`` (the maintainer-pinned budget reference
            depth), and summary counts (``t_count``, ``cx_count``, ``t_depth``). Your
            compiled program must implement this netlist fault-tolerantly; the cost
            model is in get_device_spec.
            """
            return na_actions.get_target_circuit(ctx, device_id=device_id)


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
