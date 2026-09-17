"""MCP tools for frozen-controller transfer and trusted readout references."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import structured_control_probe as control_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Three-spin coherent-control simulator. The structured_control_probe capability "
    "runs one of four frozen controllers under a commanded multiplicative q0 X-drive "
    "scale fraction, or a trusted computational-basis readout reference. Both tools "
    "return a job_id immediately; poll get_job_result. Results contain only raw q0-q2 "
    "bitstrings. Respect the shared job/shot budget and use the public protocol to "
    "support a finite pre-readout p(001)>=0.90 certificate around zero."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    ctx = ActionContext(state=state, surface="mcp")
    if "structured_control_probe" not in state.active_capabilities:
        return

    @mcp.tool()
    def run_control_transfer(
        device_id: str,
        controller_id: str,
        x_drive_scale_fraction: float,
        shots: int,
    ) -> dict:
        """Run a frozen controller and return a job_id for raw three-bit outcomes."""

        return control_actions.submit_control_transfer(
            ctx,
            device_id=device_id,
            controller_id=controller_id,
            x_drive_scale_fraction=x_drive_scale_fraction,
            shots=shots,
        )

    @mcp.tool()
    def run_control_readout_reference(
        device_id: str,
        prepared_bitstring: str,
        shots: int,
    ) -> dict:
        """Prepare a trusted 000/100/010/001 reference and return a job_id."""

        return control_actions.submit_control_readout_reference(
            ctx,
            device_id=device_id,
            prepared_bitstring=prepared_bitstring,
            shots=shots,
        )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
