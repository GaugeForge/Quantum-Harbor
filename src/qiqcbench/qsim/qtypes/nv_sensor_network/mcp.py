"""MCP tool registration for the NV sensor-network qtype.

Owns the qtype-local MCP surface: the instructions string and a
``register_mcp_tools`` callable that attaches the ``run_sensing_probe`` /
``run_sensing_sweep`` tools when the ``network_field_sensing`` capability is
active. Always-on tools are registered by the server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import nv_sensor_network as nv_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover - import-only typing aid
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Quantum simulator MCP server. The device is a network of single-NV magnetometer "
    "nodes, each running a Ramsey sequence sensing a local static field "
    "theta_i = gamma_e*B_i (rad/s). Use run_sensing_probe / run_sensing_sweep to "
    "interrogate nodes: probe_type 'separable' measures each support node "
    "independently; 'ghz' distributes a GHZ across the support (the returned per-node "
    "bits carry the joint parity) consuming len(support)-1 heralded Bell pairs; "
    "'link_probe' entangles one node with the ideal central station to calibrate that "
    "node's Bell-pair fidelity. echo_sign in {-1,+1} sets the signed effective sensing "
    "time per node; interrogation times are seconds on the device 10 ns grid. Each shot "
    "returns raw per-node bits; build populations / parity yourself. All run_* tools are "
    "async: submit -> poll get_job_result. Use submit_final_answer when ready."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    """Attach the NV sensing MCP tools for the given qsim state."""

    ctx = ActionContext(state=state, surface="mcp")

    if "network_field_sensing" in state.active_capabilities:

        @mcp.tool()
        def run_sensing_probe(
            device_id: str,
            probe_type: str,
            support: list[int],
            interrogation_time_s: list[float],
            echo_sign: list[int],
            analysis_phase_rad: float = 0.0,
            shots: int = 1024,
        ) -> dict:
            """Run one NV sensing probe. Returns a job_id; poll get_job_result.

            `probe_type` is 'separable' | 'ghz' | 'link_probe'. `support`,
            `interrogation_time_s`, and `echo_sign` are aligned per node. Cumulative
            interrogation time charged = shots * sum(interrogation_time_s).
            """
            return nv_actions.submit_sensing_probe(
                ctx,
                device_id=device_id,
                probe_type=probe_type,
                support=support,
                interrogation_time_s=interrogation_time_s,
                echo_sign=echo_sign,
                analysis_phase_rad=analysis_phase_rad,
                shots=shots,
            )

        @mcp.tool()
        def run_sensing_sweep(
            device_id: str,
            probe_type: str,
            support: list[int],
            interrogation_time_grid_s: list[float],
            echo_sign: list[int],
            analysis_phase_rad: float = 0.0,
            shots: int = 512,
        ) -> dict:
            """Sweep a single interrogation time (applied to every support node).

            One measurement record per grid value, in order; the sweep coordinates are
            returned in the job metadata as `sweep_coords["interrogation_time_s"]`.
            """
            return nv_actions.submit_sensing_sweep(
                ctx,
                device_id=device_id,
                probe_type=probe_type,
                support=support,
                interrogation_time_grid_s=interrogation_time_grid_s,
                echo_sign=echo_sign,
                analysis_phase_rad=analysis_phase_rad,
                shots=shots,
            )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
