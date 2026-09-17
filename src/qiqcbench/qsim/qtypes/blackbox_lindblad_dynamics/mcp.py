"""MCP tool registration for the blackbox Lindblad-dynamics qtype.

Owns the qtype-local MCP surface: the instructions string and
``register_mcp_tools``, which attaches the capability-gated
``run_lindblad_probe_batch`` tool to a ``FastMCP`` instance. The always-on tools
(get_device_spec, get_lab_notebook, get_job_result, submit_final_answer,
list_devices) are registered by the top-level server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import lindblad as lindblad_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover - import-only typing aid
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Quantum simulator MCP server. Operate a black-box open-system "
    "(Lindblad/GKSL) device: learn an unknown generator dρ/dt = -i[H,ρ] + "
    "Σ_ab c_ab (P_a ρ P_b - 1/2{P_b P_a, ρ}) with H = (1/2) Σ_P h_P P (the two "
    "public dictionaries are exposed via get_device_spec; the coefficients are "
    "hidden) by submitting probe batches. Each probe prepares a product state, "
    "evolves under the master equation for an on-grid time, and measures a Pauli "
    "observable, returning raw +-1 outcomes. The scarce resource is total "
    "accepted evolution time, which accumulates across batches until the device "
    "budget is exhausted. "
    "run_lindblad_probe_batch is available when the lindblad_probe capability is "
    "active. Poll get_job_result; off-grid or over-budget rows are rejected and "
    "consume no budget. Reported single-site dissipator blocks must be physical "
    "(positive semidefinite). Use submit_final_answer when ready."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    """Attach the probe tool to ``mcp`` for the given qsim state.

    ``state`` is the ``QsimState`` instance; typed ``Any`` so the qtype module
    does not import runtime state just to register tools.
    """

    ctx = ActionContext(state=state, surface="mcp")

    if "lindblad_probe" in state.active_capabilities:

        @mcp.tool()
        def run_lindblad_probe_batch(device_id: str, rows: list[dict]) -> dict:
            """Submit a batch of open-system probes. Returns a job_id immediately.

            ``rows`` is a list of probes, each:
              {"initial_state": ["z+","x-",...],  # one product-state label per qubit
               "evolve_time_us": <float on the evolve_time grid>,
               "observable_pauli": "IIZ...X"}     # length-n_qubits I/X/Y/Z string

            The internal repetition count and the total-evolution-time budget are
            fixed by the device (see get_device_spec). Poll get_job_result(job_id);
            the result data has kind="probe_outcome" with one outcome row per probe
            (status "accepted"|"rejected", raw_pauli_outcomes for accepted rows),
            plus cumulative budget_used_us / budget_remaining_us / accepted_row_count.
            Off-grid/out-of-range times, malformed states, and invalid observables
            are rejected and consume zero budget.
            """
            return lindblad_actions.submit_lindblad_probe_batch(ctx, device_id=device_id, rows=rows)


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
