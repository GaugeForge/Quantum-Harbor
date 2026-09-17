"""MCP surface for the site-resolved Rydberg many-body simulator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import stroboscopic_rydberg_evolution as actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Site-resolved 1D neutral-atom many-body simulator. The only experiment is the fixed public "
    "depth6_rydberg_ising_v2 protocol: run_stroboscopic_rydberg_evolution(device_id, system_size, shots). "
    "It returns an async job ID; once complete, get_job_result returns small metadata and a public "
    "raw_data_file relative to /qsim_logs. Read that JSON file with a script to decode its bit-packed "
    "load, mid-sequence and final occupancy images plus the two target-atom fluorescence bits; do not "
    "print its payload. Use public N=25,49,81 only."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    ctx = ActionContext(state=state, surface="mcp")
    if "stroboscopic_rydberg_evolution" in state.active_capabilities:

        @mcp.tool()
        def run_stroboscopic_rydberg_evolution(
            device_id: str,
            system_size: int,
            shots: int,
            protocol_id: str = "depth6_rydberg_ising_v2",
        ) -> dict:
            """Run the fixed public Rydberg brickwork experiment; returns a job_id immediately."""
            return actions.submit_stroboscopic_rydberg_evolution(
                ctx,
                device_id=device_id,
                protocol_id=protocol_id,
                system_size=system_size,
                shots=shots,
            )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
