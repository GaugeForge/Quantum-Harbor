"""MCP tool registration for the multilevel-transmon qtype.

Owns the qtype-local MCP surface: the instructions string and a
``register_mcp_tools`` callable that attaches ``run_pulse_sequence`` /
``run_pulse_sweep`` when the ``multilevel_pulse_control`` capability is active.
Always-on tools (list_devices, get_device_spec, get_lab_notebook,
get_job_result, submit_final_answer) are registered by the server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import transmon_multilevel as ml_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover - import-only typing aid
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Quantum simulator MCP server. The device is a single fixed-frequency transmon "
    "kept as a 4-level Duffing oscillator (rotating frame at omega_01; detuning sign "
    "delta = omega_01 - omega_drive). You control two drive quadratures Omega_x, Omega_y "
    "(in DAC units) and a per-sample drive detuning (Hz), and read out level-resolved "
    "outcomes (0..3) per shot. Build pulses with run_pulse_sequence (a list of drive "
    "segments, each 'analytic' {shape,amp_dac,duration_ns,sigma_ns,phase_rad,"
    "carrier_detuning_hz} or 'sampled' {omega_x_dac[],omega_y_dac[],detuning_hz[],"
    "sample_dt_ns}) executed from |0>, and run_pulse_sweep to sweep a named scalar "
    "placeholder '$name' across analytic segments. Anharmonicity and amplitude "
    "calibration in the lab notebook are stale; remeasure them. All run_* tools are "
    "async: submit, then poll get_job_result. Use submit_final_answer when ready."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    """Attach the multilevel-pulse MCP tools for the given qsim state."""

    ctx = ActionContext(state=state, surface="mcp")

    if "multilevel_pulse_control" in state.active_capabilities:

        @mcp.tool()
        def run_pulse_sequence(
            device_id: str,
            segments: list[dict],
            shots: int = 2048,
        ) -> dict:
            """Run a drive sequence from |0> and read out level-resolved. Returns a job_id.

            `segments` is a list of drive segments executed back-to-back. Each is
            either {"kind":"analytic","shape":"gaussian"|"square","amp_dac":float,
            "duration_ns":float,"sigma_ns":float,"phase_rad":float,
            "carrier_detuning_hz":float} or {"kind":"sampled","omega_x_dac":[...],
            "omega_y_dac":[...],"detuning_hz":[...],"sample_dt_ns":float}. Each shot
            returns the reported level (0..3). Poll get_job_result.
            """
            return ml_actions.submit_pulse_sequence(
                ctx, device_id=device_id, segments=segments, shots=shots
            )

        @mcp.tool()
        def run_pulse_sweep(
            device_id: str,
            template_segments: list[dict],
            sweep: dict[str, list[float]],
            shots: int = 1024,
            mode: str = "product",
        ) -> dict:
            """Sweep named scalar placeholders across analytic segments. Returns a job_id.

            Reference a swept scalar in an analytic segment as the string "$name"
            (e.g. carrier_detuning_hz="$det"), and pass sweep={"det":[...]}. One
            measurement record is produced per sweep point, in order; per-point
            coordinates are returned in the job metadata `sweep_coords`.
            """
            return ml_actions.submit_pulse_sweep(
                ctx,
                device_id=device_id,
                template_segments=template_segments,
                sweep=sweep,
                shots=shots,
                mode=mode,
            )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
