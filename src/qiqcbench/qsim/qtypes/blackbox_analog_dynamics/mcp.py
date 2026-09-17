"""MCP tool registration for the blackbox analog-dynamics qtype.

Owns the qtype-local MCP surface: the instructions string and
``register_mcp_tools``, which attaches the capability-gated
``run_hamiltonian_probe_batch`` tool to a ``FastMCP`` instance. The always-on
tools (get_device_spec, get_lab_notebook, get_job_result, submit_final_answer,
list_devices) are registered by the top-level server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import analog as analog_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover - import-only typing aid
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Quantum simulator MCP server. Operate a black-box analog-dynamics device: "
    "learn an unknown Hamiltonian H = (1/2) sum_P omega_P P (the term dictionary is "
    "public via get_device_spec; the coefficients are hidden) by submitting "
    "probe batches. Each probe prepares a product state, evolves under "
    "U(t)=exp(-i t H) for an on-grid time, and measures a Pauli observable, "
    "returning raw +-1 outcomes. The scarce resource is total accepted "
    "evolution time, which accumulates across batches until the device budget "
    "is exhausted. run_hamiltonian_probe_batch is available when the "
    "hamiltonian_probe capability is active. Poll get_job_result; off-grid or "
    "over-budget rows are rejected and consume no budget. Use submit_final_answer "
    "when ready."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    """Attach the probe tool to ``mcp`` for the given qsim state.

    ``state`` is the ``QsimState`` instance; typed ``Any`` so the qtype module
    does not import runtime state just to register tools.
    """

    ctx = ActionContext(state=state, surface="mcp")

    # Execution binding: commit the exact hidden instance this server will
    # execute into the qsim-private log, once, before any agent traffic. The
    # separate verifier recomputes the same digest from its private scoring
    # truth and refuses to score a mismatched instance (see
    # device.hidden_instance_commitment).
    from qiqcbench.qsim.qtypes.blackbox_analog_dynamics.device import (
        hidden_instance_commitment,
    )

    state.log(
        {
            "surface": "system",
            "action": "hidden_instance_commitment",
            "qtype": "blackbox_analog_dynamics",
            "device_id": state.public.device_id,
            "task_id": state.task_id,
            "scheme": "sha256_omega_vector_v1",
            "commitment_sha256": hidden_instance_commitment(state.hidden, state.public),
        }
    )
    state._analog_instance_commitment_logged = True

    if "hamiltonian_probe" in state.active_capabilities:

        @mcp.tool()
        def run_hamiltonian_probe_batch(device_id: str, rows: list[dict]) -> dict:
            """Submit a batch of Hamiltonian probes. Returns a job_id immediately.

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
            return analog_actions.submit_hamiltonian_probe_batch(
                ctx, device_id=device_id, rows=rows
            )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
