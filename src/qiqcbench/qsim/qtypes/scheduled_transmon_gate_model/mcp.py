"""MCP surface for compiled-candidate mirror and readout-reference jobs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import compiled_candidate_evaluation as compilation_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Scheduled native-gate transmon simulator. The active compiled-candidate capability "
    "offers finite-bank public-seed-twirled mirror experiments and trusted "
    "computational-basis readout references. The public mirror seed selects a deterministic "
    "virtual coherent-axis plan; it neither inserts a physical Pauli gate nor keys hidden "
    "drift or shot noise. Both run_* tools are asynchronous and return a job_id; poll "
    "get_job_result. Successful jobs return only raw all-zero-comparison-basis bitstrings, "
    "with string position i corresponding to measured_qubits[i]. Inspect the public "
    "candidate bank and authoritative protocol materials, respect the shared cumulative "
    "shot/job limits, and submit the final answer through submit_final_answer. Qsim never "
    "returns candidate quality or rank."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    ctx = ActionContext(state=state, surface="mcp")
    if "compiled_candidate_evaluation" not in state.active_capabilities:
        return

    @mcp.tool()
    def run_compilation_mirror(
        device_id: str,
        candidate_id: str,
        mirror_seed: int,
        shots: int,
    ) -> dict:
        """Run one public-seed-twirled mirror for a frozen candidate.

        Returns a job_id immediately. Poll get_job_result; successful data is one
        raw-bitstring point in the ideal all-zero comparison basis. The public seed
        selects a deterministic virtual coherent-axis plan, not a physical Pauli gate
        or hidden randomness stream.
        """

        return compilation_actions.submit_compilation_mirror(
            ctx,
            device_id=device_id,
            candidate_id=candidate_id,
            mirror_seed=mirror_seed,
            shots=shots,
        )

    @mcp.tool()
    def run_compilation_readout_reference(
        device_id: str,
        prepared_bitstring: str,
        shots: int,
    ) -> dict:
        """Run a trusted zero or single-excitation readout reference.

        Returns a job_id immediately. Reference jobs consume the shared budgets and
        advance the run timeline. Poll get_job_result for raw bitstrings.
        """

        return compilation_actions.submit_compilation_readout_reference(
            ctx,
            device_id=device_id,
            prepared_bitstring=prepared_bitstring,
            shots=shots,
        )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
