"""MCP tool registration for the ``neutral_atom_logical_processor`` qtype.

Gated on the ``rydberg_logical_control`` capability:
  - ``run_atom_program`` (async, returns a job_id): execute a scheduled atom program.
  - ``validate_schedule`` (sync, free): legality verdict + nominal schedule duration.
The agent polls ``get_job_result`` for the raw per-shot measurement bits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import neutral_atom_logical_processor as na_actions
from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.qtypes.neutral_atom_logical_processor.public_surface import (
    load_neutral_atom_qtype_agent_surface,
)

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = load_neutral_atom_qtype_agent_surface().mcp_instructions


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    ctx = ActionContext(state=state, surface="mcp")

    if "rydberg_logical_control" in state.active_capabilities:

        @mcp.tool()
        def run_atom_program(
            device_id: str,
            ops: list[dict],
            shots: int,
            layout: list[int] | None = None,
        ) -> dict:
            """Execute a scheduled atom program (see INSTRUCTIONS for op kinds). Returns a
            job_id immediately; poll get_job_result for the raw per-shot measurement bits."""
            return na_actions.submit_atom_program(
                ctx, device_id=device_id, ops=ops, shots=shots, layout=layout
            )

        @mcp.tool()
        def validate_schedule(
            device_id: str, ops: list[dict], layout: list[int] | None = None
        ) -> dict:
            """SYNC + FREE: check program legality (atom/op/gate constraints) and return the
            nominal serial schedule duration. Returns no fidelity (that needs shots)."""
            return na_actions.validate_schedule(ctx, device_id=device_id, ops=ops, layout=layout)

    if "located_erasure_repair" in state.active_capabilities:

        @mcp.tool()
        def run_located_erasure_control_episodes(
            device_id: str,
            horizon: int,
            episodes: int,
            controller: dict,
        ) -> dict:
            """Run raw located-erasure control episodes and return a job ID.

            ``controller`` is a bounded causal decision list over the public
            masks/resource/history fields. Poll ``get_job_result`` for the
            digest-bound ``data.raw_data_file`` below ``/qsim_logs``. The file
            contains per-episode observations, decisions, physical events, and
            raw terminal bits; the completed response contains no aggregate rate.
            """
            return na_actions.submit_located_erasure_control_episodes(
                ctx,
                device_id=device_id,
                horizon=horizon,
                episodes=episodes,
                controller=controller,
            )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
