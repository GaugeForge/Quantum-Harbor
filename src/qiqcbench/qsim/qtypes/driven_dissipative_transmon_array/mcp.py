"""MCP tool registration for the driven-dissipative array qtype.

Owns the qtype-local MCP surface: the instructions string and a
``register_mcp_tools`` callable that attaches the ``run_stabilization`` /
``run_stabilization_sweep`` tools when the ``local_reservoir_stabilization``
capability is active. Always-on tools are registered by the server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import dissipative_bell as ddta_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover - import-only typing aid
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Quantum simulator MCP server. The device is an open chain of hard-core "
    "transmon sites operated as an OPEN-SYSTEM analog simulator. Choose one "
    "adjacent pair, prepare it (gg/ge/eg/ee), and turn on local energy-selective "
    "pump and loss reservoirs to autonomously stabilize the single-excitation "
    "Bell singlet |-> = (|ge>-|eg>)/sqrt(2). Use run_stabilization (one duration) "
    "and run_stabilization_sweep (a grid of durations) to drive the reservoirs "
    "(pump detuning delta_s, loss detuning delta_d relative to the pair's "
    "single-excitation resonance; couplings g_s, g_d) and read the requested "
    "sites out in the x/y/z analysis basis. Computational-basis preparation and "
    "analysis rotations are ideal; Born outcomes then pass through a hidden, "
    "independent, site-dependent, asymmetric binary assignment channel. Each "
    "returned shot is therefore a post-readout raw 0/1 per measured site. A "
    "duration-zero experiment retains the known preparation and may be used for "
    "readout characterization. The public device budget declares per-call and "
    "trial-cumulative job, raw-record, poll, metadata, ingress, and final-answer "
    "limits; these are runtime safety bounds, not score terms. The lab notebook "
    "may be stale. Poll get_job_result; use submit_final_answer when ready."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    """Attach the stabilize/measure MCP tools for the given qsim state."""

    ctx = ActionContext(state=state, surface="mcp")

    if "local_reservoir_stabilization" in state.active_capabilities:

        @mcp.tool()
        def run_stabilization(
            device_id: str,
            pair: str,
            initial_state: str,
            delta_s_mhz: float,
            delta_d_mhz: float,
            g_s_mhz: float,
            g_d_mhz: float,
            duration_us: float,
            measure: list[dict],
            shots: int = 2048,
        ) -> dict:
            """Prepare a pair, drive the reservoirs for one duration, read out. Returns a job_id.

            `pair` is e.g. "q0,q1"; `initial_state` is one of gg/ge/eg/ee (first
            char is the lower-index site). Each entry of `measure` is
            {"site": int, "basis": "x"|"y"|"z"}. Poll get_job_result.
            """
            return ddta_actions.submit_stabilization(
                ctx,
                device_id=device_id,
                pair=pair,
                initial_state=initial_state,
                delta_s_mhz=delta_s_mhz,
                delta_d_mhz=delta_d_mhz,
                g_s_mhz=g_s_mhz,
                g_d_mhz=g_d_mhz,
                duration_us=duration_us,
                measure=measure,
                shots=shots,
            )

        @mcp.tool()
        def run_stabilization_sweep(
            device_id: str,
            pair: str,
            initial_state: str,
            delta_s_mhz: float,
            delta_d_mhz: float,
            g_s_mhz: float,
            g_d_mhz: float,
            duration_grid_us: list[float],
            measure: list[dict],
            shots: int = 1024,
        ) -> dict:
            """Sweep the stabilization duration on a fixed prep/reservoir/measure. Returns one job_id.

            One measurement record is produced per duration in `duration_grid_us`,
            in order; the per-point durations are returned in the job metadata as
            `sweep_coords["duration_us"]`.
            """
            return ddta_actions.submit_stabilization_sweep(
                ctx,
                device_id=device_id,
                pair=pair,
                initial_state=initial_state,
                delta_s_mhz=delta_s_mhz,
                delta_d_mhz=delta_d_mhz,
                g_s_mhz=g_s_mhz,
                g_d_mhz=g_d_mhz,
                duration_grid_us=duration_grid_us,
                measure=measure,
                shots=shots,
            )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
