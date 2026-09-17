"""MCP tools for the ``qccd_ion_compiler`` qtype.

Two **synchronous calculator** tools (no async job, no shots), gated on the
``qccd_compilation`` capability:

* ``get_compilation_instance(device_id)`` — the target QAOA circuit instance
  (cluster membership, weighted edges with per-layer ``γ``, per-layer ``β``,
  ``p``) for this run's ``QIQCBENCH_INSTANCE_SEED``, plus a machine-readable copy
  of the public cost-model coefficients.
* ``evaluate_schedule(device_id, schedule)`` — a deterministic cost oracle. It
  recomputes legality, circuit-correctness, and (only if both pass) the total
  estimated circuit infidelity from the public cost model, using the SAME code
  the verifier uses. Returns specific legality/correctness errors (decidable
  constraints, not physics hints) so you can iterate to a runnable schedule.
  Capped at the device ``evaluator_call_cap`` (a wall-clock/abuse bound — the
  cost model is public, so this guards nothing secret).

Both delegate to ``qsim/actions/qccd_ion_compiler.py`` (imported inside the
registrar to keep module import light during qtype discovery).
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

INSTRUCTIONS = (
    "Zoned-QCCD trapped-ion compilation MCP server. The task is to COMPILE a "
    "QAOA (MaxCut, depth p) circuit onto a zoned QCCD ion trap and is scored by "
    "the total estimated circuit infidelity of the compiled schedule (LOWER is "
    "better) — there are no shots, no async jobs, and no quantum dynamics; the "
    "device is a deterministic, fully PUBLIC resource/error/timing cost model. "
    "get_compilation_instance(device_id) returns the target QAOA circuit (cluster "
    "membership, the weighted edges with per-layer gamma angles, the per-layer "
    "beta, and p) for this run plus the public cost-model coefficients. Connectivity "
    "is realized by ION TRANSPORT (shuttle / split-merge / swap / junction "
    "traversal), and transport HEATS the shared motional mode, which raises the "
    "MS-gate error eps_MS = floor(mode) + s_angle*(|theta|/(pi/2)) + "
    "kappa(mode)*(2*nbar+1). Co-locate each graph cluster's ions (cluster-aware "
    "placement), use the native arbitrary-angle ZZ(gamma) gate (one gate per edge, "
    "not two full gates), route with low-heating primitives at a per-move "
    "optimized/fast speed, recool (exchange: fast/partial; sympathetic: slow/deep) "
    "just before hot gates, pick the normal mode (COM cold / stretch hot) per gate, "
    "dynamically decouple idle qubits at the interior-optimal pulse density, and "
    "batch gates across the 3 gate zones. evaluate_schedule(device_id, schedule) is "
    "a deterministic cost oracle (capped) that recomputes legality, "
    "circuit-correctness, and the total infidelity from the public model. The lab "
    "notebook's recommendations are stale traps. Use submit_final_answer when ready."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    """Attach the synchronous calculator tools to ``mcp`` for the qsim state."""
    from qiqcbench.qsim.actions import qccd_ion_compiler as qccd_actions
    from qiqcbench.qsim.actions.base import ActionContext

    ctx = ActionContext(state=state, surface="mcp")

    if "qccd_compilation" in state.active_capabilities:

        @mcp.tool()
        def get_compilation_instance(device_id: str) -> dict:
            """Return the target QAOA circuit instance + public cost model.

            The instance (graph cluster membership, the weighted edges with
            per-layer ``γ`` angles, the per-layer ``β``, and ``p``) is a
            deterministic function of this run's instance seed. Synchronous:
            returns the payload directly (no job_id).
            """
            return qccd_actions.get_compilation_instance(ctx, device_id=device_id)

        @mcp.tool()
        def evaluate_schedule(device_id: str, schedule: dict) -> dict:
            """Deterministic cost oracle for a candidate compiled schedule.

            ``schedule`` is the Stage-B compiled schedule (the ``op_schedule`` plus
            the Stage-A ``ion_placement``). Returns ``legal`` / ``circuit_correct``
            booleans with SPECIFIC error reasons, and — only when both pass — the
            ``total_infidelity`` recomputed from the public cost model (the same
            computation the verifier runs). Capped at ``evaluator_call_cap`` calls
            (a wall-clock/abuse bound; the cost model is public so this guards
            nothing secret). Synchronous: returns the payload directly.
            """
            return qccd_actions.evaluate_schedule(ctx, device_id=device_id, schedule=schedule)
