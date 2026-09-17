"""MCP registration for Rydberg stabilizer calibration/readout."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import multitarget_stabilizer_readout as actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Native simultaneous-CZ2 unrotated distance-three surface-code memory. "
    "When multitarget_stabilizer_readout is active, "
    "run_cz2_echo_characterization(echo_protocol, repetitions, duration_scale, "
    "target_phase_compensation_rad, shots) is async; poll get_job_result for raw three-role "
    "fluorescence bitstrings. Every public weight-four bulk check in a submitted cycle is "
    "partitioned into two disjoint target-role pairs. "
    "validate_multitarget_stabilizer_cycle is free and returns only syntax, canonical digest, "
    "and nominal public time. run_multitarget_stabilizer_memory is async and returns raw packed "
    "ancilla and final-data measurement bitstrings. You may submit no DEM, "
    "decoder, weights, arbitrary program, or serial four-CZ fallback. Across calibration and "
    "memory calls, the resolved public device spec states the run-wide shot and job limits."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    if "multitarget_stabilizer_readout" not in state.active_capabilities:
        return
    ctx = ActionContext(state=state, surface="mcp")

    @mcp.tool()
    def run_cz2_echo_characterization(
        device_id: str,
        echo_protocol: str,
        repetitions: int,
        duration_scale: float,
        target_phase_compensation_rad: float,
        shots: int,
    ) -> dict:
        """Async Rydberg echo characterization; poll for raw fluorescence bits."""
        return actions.submit_cz2_echo_characterization(
            ctx,
            device_id=device_id,
            echo_protocol=echo_protocol,
            repetitions=repetitions,
            duration_scale=duration_scale,
            target_phase_compensation_rad=target_phase_compensation_rad,
            shots=shots,
        )

    @mcp.tool()
    def validate_multitarget_stabilizer_cycle(device_id: str, cycle: dict) -> dict:
        """Free canonical-cycle syntax/digest/time validation, without performance data."""
        return actions.validate_multitarget_stabilizer_cycle(ctx, device_id=device_id, cycle=cycle)

    @mcp.tool()
    def run_multitarget_stabilizer_memory(
        device_id: str,
        decoded_basis: str,
        cycle: dict,
        duration_scale: float,
        target_phase_compensation_rad: float,
        rounds: int,
        shots: int,
    ) -> dict:
        """Async repeated-stabilizer memory experiment returning raw measurements."""
        return actions.submit_multitarget_stabilizer_memory(
            ctx,
            device_id=device_id,
            decoded_basis=decoded_basis,
            cycle=cycle,
            duration_scale=duration_scale,
            target_phase_compensation_rad=target_phase_compensation_rad,
            rounds=rounds,
            shots=shots,
        )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
