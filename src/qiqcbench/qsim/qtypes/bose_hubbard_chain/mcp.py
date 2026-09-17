"""MCP tool registration for the Bose-Hubbard chain qtype.

Owns the qtype-local MCP surface: the instructions string and a
``register_mcp_tools`` callable that attaches the analog ``run_evolution`` /
``run_evolution_sweep`` tools when the ``bose_hubbard_spectroscopy`` capability
is active. Always-on tools are registered by the server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import bose_hubbard as bose_hubbard_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover - import-only typing aid
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Quantum simulator MCP server. The device is a 1D Bose-Hubbard photon chain "
    "operated as an analog simulator: prepare sites in (|0>+|1>)/sqrt(2) with "
    "run_evolution / run_evolution_sweep, evolve under the fixed device "
    "Hamiltonian, and read out sites in the X or Y basis. Each shot returns the "
    "read-out occupation (0, 1, or 2) per measured site; build sigma correlators "
    "from the {0,1} outcomes. Use time-domain evolution data to infer the device "
    "spectrum and coherence. Poll get_job_result for results. Use "
    "submit_final_answer when ready."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    """Attach the analog evolve/measure MCP tools for the given qsim state."""

    ctx = ActionContext(state=state, surface="mcp")

    if "bose_hubbard_spectroscopy" in state.active_capabilities:

        @mcp.tool()
        def run_evolution(
            device_id: str,
            init_excited_sites: list[int],
            evolution_time_ns: float,
            measure: list[dict],
            shots: int = 1024,
        ) -> dict:
            """Prepare, evolve for one time, and read out. Returns a job_id.

            `init_excited_sites` are prepared in (|0>+|1>)/sqrt(2). Each entry of
            `measure` is {"site": int, "basis": "x"|"y"}. Poll get_job_result.
            """
            return bose_hubbard_actions.submit_evolution(
                ctx,
                device_id=device_id,
                init_excited_sites=init_excited_sites,
                evolution_time_ns=evolution_time_ns,
                measure=measure,
                shots=shots,
            )

        @mcp.tool()
        def run_evolution_sweep(
            device_id: str,
            init_excited_sites: list[int],
            time_grid_ns: list[float],
            measure: list[dict],
            shots: int = 512,
        ) -> dict:
            """Sweep the evolution time on a fixed prep/measure. Returns one job_id.

            One measurement record is produced per time in `time_grid_ns`, in
            order; the per-time sweep coordinates are returned in the job
            metadata as `sweep_coords["time_ns"]`.
            """
            return bose_hubbard_actions.submit_evolution_sweep(
                ctx,
                device_id=device_id,
                init_excited_sites=init_excited_sites,
                time_grid_ns=time_grid_ns,
                measure=measure,
                shots=shots,
            )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
