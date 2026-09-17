"""MCP tool registration for the ``neutral_atom_dm_processor`` qtype.

Gated on the ``rydberg_dm_control`` capability:
  - ``run_atom_program`` (async, returns a job_id): execute a scheduled atom
    program at one (idle_scale, noise_scale) setting.
  - ``run_atom_program_sweep`` (async): the same ops at several values of
    ``noise_scale`` OR ``idle_scale``; raw per-point bits land in a public raw
    file referenced by ``data.raw_data_file`` — inspect it with a script.
  - ``validate_schedule`` (sync, free): legality verdict + nominal duration +
    atoms_peak + move_count (no shots, no fidelity).
The agent polls ``get_job_result`` for raw measurement records.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import neutral_atom_dm_processor as na_dm_actions
from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.qtypes.neutral_atom_dm_processor.public_surface import (
    load_neutral_atom_dm_qtype_agent_surface,
)

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = load_neutral_atom_dm_qtype_agent_surface().mcp_instructions


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    ctx = ActionContext(state=state, surface="mcp")

    if "rydberg_dm_control" in state.active_capabilities:

        @mcp.tool()
        def run_atom_program(
            device_id: str,
            ops: list[dict],
            shots: int,
            layout: list[int],
            idle_scale: float = 1.0,
            noise_scale: float = 1.0,
        ) -> dict:
            """Execute a scheduled atom program (see INSTRUCTIONS for op kinds). Returns a
            job_id immediately; poll get_job_result for the raw per-shot measurement bits."""
            return na_dm_actions.submit_atom_program(
                ctx,
                device_id=device_id,
                ops=ops,
                shots=shots,
                layout=layout,
                idle_scale=idle_scale,
                noise_scale=noise_scale,
            )

        @mcp.tool()
        def run_atom_program_sweep(
            device_id: str,
            ops: list[dict],
            shots: int,
            layout: list[int],
            sweep_parameter: str,
            sweep_values: list[float],
            idle_scale: float = 1.0,
            noise_scale: float = 1.0,
        ) -> dict:
            """Run the same program at several noise_scale OR idle_scale values (one job;
            budgets charge len(sweep_values) x shots). Poll get_job_result; per-point raw
            bits are referenced by data.raw_data_file below /qsim_logs."""
            return na_dm_actions.submit_atom_program_sweep(
                ctx,
                device_id=device_id,
                ops=ops,
                shots=shots,
                layout=layout,
                sweep_parameter=sweep_parameter,
                sweep_values=sweep_values,
                idle_scale=idle_scale,
                noise_scale=noise_scale,
            )

        @mcp.tool()
        def validate_schedule(device_id: str, ops: list[dict], layout: list[int]) -> dict:
            """SYNC + FREE: check program legality (zones, sites, atoms, gates) and return
            the nominal serial duration, atoms_peak, and move_count. No fidelity."""
            return na_dm_actions.validate_schedule(ctx, device_id=device_id, ops=ops, layout=layout)


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
