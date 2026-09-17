"""MCP tool registration for the transmon (pulse-level) qtype.

This module owns the qtype-local MCP surface: the instructions string the
server advertises and a ``register_mcp_tools`` callable that attaches
``run_pulse_sequence`` and ``run_sweep`` to a ``FastMCP`` instance.
The top-level ``qiqcbench.qsim.mcp.server`` invokes this registrar through the
qtype registry after registering the always-on tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import transmon as transmon_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover - import-only typing aid
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Quantum simulator MCP server. Operate the transmon via "
    "run_pulse_sequence and run_sweep. Poll get_job_result for results. "
    "Simulator mode returns raw IQ data; provider-backed modes return "
    "per-shot bitstrings. Use submit_final_answer when ready."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    """Attach the pulse-level MCP tools to ``mcp`` for the given qsim state.

    ``state`` is the ``QsimState`` instance defined by
    ``qiqcbench.qsim.state``; we type it as ``Any`` here to keep qtype tool
    registration decoupled from runtime state imports.
    """

    ctx = ActionContext(state=state, surface="mcp")

    @mcp.tool()
    def run_pulse_sequence(device_id: str, sequence: list[dict], shots: int = 1024) -> dict:
        """Submit a pulse sequence. Returns a job_id immediately.

        Each op in `sequence` is a dict with `kind` in {pulse, delay, measure}.
        Pulse ops require {channel, shape, amp, duration_ns} and sigma_ns for
        gaussian. Delay ops require {duration_ns}. Measure ops require {qubits}.
        Poll get_job_result(job_id) until status is 'complete' or 'failed'.
        """
        return transmon_actions.submit_pulse_sequence(
            ctx, device_id=device_id, sequence=sequence, shots=shots
        )

    @mcp.tool()
    def run_sweep(
        device_id: str,
        template_sequence: list[dict],
        sweep: dict[str, list[float]],
        shots: int = 512,
        mode: str = "product",
    ) -> dict:
        """Submit a parameter sweep. Returns a single job_id covering all points.

        Placeholders in `template_sequence` (e.g. delay duration_ns="$delay_ns")
        are substituted with values from `sweep`. mode="product" runs the
        Cartesian product of sweep keys; mode="zip" requires equal-length
        arrays and pairs them.
        """
        return transmon_actions.submit_pulse_sweep(
            ctx,
            device_id=device_id,
            template_sequence=template_sequence,
            sweep=sweep,
            shots=shots,
            mode=mode,
        )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
