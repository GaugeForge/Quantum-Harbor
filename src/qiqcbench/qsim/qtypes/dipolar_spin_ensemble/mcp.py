"""MCP tool registration for the dipolar-spin-ensemble qtype.

Owns the qtype-local MCP surface: the instructions string and a
``register_mcp_tools`` callable that attaches the ``run_pulse_train`` tool when
the ``toggling_frame_control`` capability is active. Always-on tools
(get_device_spec / get_lab_notebook / get_job_result / submit_final_answer) are
registered by the server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import dipolar as dipolar_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover - import-only typing aid
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Quantum simulator MCP server. The device is a globally-controlled, "
    "disorder-dominated dipolar spin ensemble (NV centers): you apply global "
    "Rx/Ry rotations (pi/2 pulses about +-x, +-y) separated by free evolution "
    "and read out the COLLECTIVE magnetization <Sx>,<Sy>,<Sz> (no single-spin "
    "addressing). There is large static on-site disorder (W) and a much weaker "
    "traceless secular dipolar interaction (J). Use run_pulse_train to run a "
    "periodic base sequence stroboscopically: each shot returns a +-1 collective "
    "readout per requested axis ('1'=+1, '0'=-1), one record per cycle 0..n_cycles. "
    "Free-induction decay reveals W; a disorder-refocusing echo reveals J; "
    "toggling-frame sequences engineer target average Hamiltonians. Poll "
    "get_job_result. Use submit_final_answer when ready."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    """Attach the dipolar pulse-train MCP tool for the given qsim state."""

    ctx = ActionContext(state=state, surface="mcp")

    if "toggling_frame_control" in state.active_capabilities:

        @mcp.tool()
        def run_pulse_train(
            device_id: str,
            sequence: list[dict],
            init_axis: str = "+x",
            n_cycles: int = 1,
            measure_axes: list[str] | None = None,
            shots: int = 2048,
            inject_rotation_error: bool = False,
        ) -> dict:
            """Run a periodic global pulse sequence and read out stroboscopically.

            `sequence` is the base block as an ordered ops list. Each op is either
            {"op":"free","duration_ns":float} (free evolution under the native
            Hamiltonian) or {"op":"pulse","axis":"x"|"y","sign":"+"|"-",
            "angle_deg":float} (global rotation; default angle 90 = pi/2). The
            block is repeated `n_cycles` times; the collective magnetization on
            each axis in `measure_axes` (subset of ["x","y","z"]) is read out
            after every cycle (records 0..n_cycles). `init_axis` is the initial
            collective polarization (+x/-x/+y/-y/+z/-z). Set
            `inject_rotation_error` to apply the device's systematic pulse-angle
            error (for robustness tests). Returns a job_id; poll get_job_result.
            Each shot string has one char per measured axis ('1'=+1, '0'=-1) in
            the order of `measure_axes`; per-cycle times are in the job metadata
            `sweep_coords["cycle_time_ns"]`.
            """
            return dipolar_actions.submit_pulse_train(
                ctx,
                device_id=device_id,
                sequence=sequence,
                init_axis=init_axis,
                n_cycles=n_cycles,
                measure_axes=measure_axes,
                shots=shots,
                inject_rotation_error=inject_rotation_error,
            )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
