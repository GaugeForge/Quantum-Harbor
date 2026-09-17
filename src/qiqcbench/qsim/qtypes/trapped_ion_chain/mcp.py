"""MCP tool registration for trapped_ion_chain qtype.

``run_ramsey_experiment`` and ``run_ramsey_sweep`` are registered iff the
``ramsey_sensing`` capability is active for this run, per the task spec's
``enabled_capabilities``. Always-on tools (``list_devices``,
``get_device_spec``, ``get_lab_notebook``, ``get_job_result``,
``submit_final_answer``) are registered elsewhere by the central MCP server.
"""

from __future__ import annotations

from typing import Any

from qiqcbench.qsim.core.wire import RamseyExperimentRequest, RamseySweepRequest

INSTRUCTIONS = """\
You control a simulated linear chain of trapped 171Yb+ ions. Native 1q gates:
i, rx, ry, rz, x, y, z, h, s, sdg. Global entangling gate ms(theta) implements
exp(-i (theta/2) sum_{i<j} X_i X_j) over the active ion subset.

Free evolution: select an interrogation time in microseconds (on the 1 us
grid) and a phase_source. phase_source='reference' applies the
agent-supplied reference_detuning_hz (Hz) -- useful for calibration sweeps.
phase_source='target' applies the hidden device detuning -- use for actual
estimation runs.

Unit ladder: wire reference_detuning_hz is FREQUENCY (Hz), not angular
frequency. The engine converts internally. Answer-schema delta_omega_*
fields are in rad/s.

Tools:
  run_ramsey_experiment(request)  -> {job_id, status}
  run_ramsey_sweep(request)       -> {job_id, status}
  get_job_result(job_id)          -> JobResult dict
"""


def register_mcp_tools(server, state, **_kwargs: Any) -> None:
    """Register Ramsey tools on the FastMCP server.

    Gated on ``state.active_capabilities`` containing ``"ramsey_sensing"``.
    Deferred imports avoid requiring Task 5's action module at module-import
    time.
    """
    if "ramsey_sensing" not in state.active_capabilities:
        return

    from qiqcbench.qsim.actions.base import ActionContext
    from qiqcbench.qsim.actions.ramsey_sensing import (
        submit_ramsey_experiment,
        submit_ramsey_sweep,
    )

    @server.tool(
        name="run_ramsey_experiment",
        description="Submit a single-point Ramsey experiment.",
    )
    def _run_ramsey_experiment(request: RamseyExperimentRequest) -> dict[str, Any]:
        """Submit a single-point Ramsey experiment."""
        ctx = ActionContext(state=state, surface="mcp")
        return submit_ramsey_experiment(ctx, request)

    @server.tool(
        name="run_ramsey_sweep",
        description="Submit a Ramsey sweep over interrogation times.",
    )
    def _run_ramsey_sweep(request: RamseySweepRequest) -> dict[str, Any]:
        """Submit a Ramsey sweep over interrogation times."""
        ctx = ActionContext(state=state, surface="mcp")
        return submit_ramsey_sweep(ctx, request)


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
