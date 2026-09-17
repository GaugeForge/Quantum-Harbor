"""MCP tool registration for the logical_magic_factory qtype.

Attaches ``run_magic_benchmark_batch`` when the ``logical_magic_benchmarking``
capability is active. Always-on tools are registered by the server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import magic_factory as magic_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover - import-only typing aid
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Quantum simulator MCP server. The device is a black-box fault-tolerant "
    "logical magic-state factory on a [[7,1,3]] Steane code, emitting noisy "
    "logical |H> copies. Use run_magic_benchmark_batch to draw copies and run "
    "single- or two-copy circuits; each call is charged against the run's "
    "magic-state budget (see get_device_spec). Each point in a batch: "
    "{n_copies: 1 or 2, twirl: bool, basis: 'x'/'y'/'z' (n_copies=1 only), "
    "shots: int}. Each shot returns a per-copy syndrome flag (0=clean, "
    "1=flagged) and the measured logical outcome bit(s), plus the updated "
    "budget meter. On cultivation-line devices the active tool is instead "
    "run_cultivation_batch (stage-scheduled injection attempts with per-shot "
    "detector-group flags; every attempt is charged against the injection "
    "budget). Poll get_job_result; submit_final_answer when ready."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    """Attach the magic-benchmark tool for the given qsim state."""

    ctx = ActionContext(state=state, surface="mcp")

    if "logical_magic_benchmarking" in state.active_capabilities:

        @mcp.tool()
        def run_magic_benchmark_batch(device_id: str, points: list[dict]) -> dict:
            """Draw magic states from the factory and run single- or two-copy circuits.

            Each point: {"n_copies": 1|2, "twirl": bool, "basis": "x"|"y"|"z"
            (required iff n_copies==1), "shots": int}. n_copies=1 consumes
            `shots` magic states; n_copies=2 consumes `2*shots`. Charged
            against the run's magic-state budget: points that would exceed
            the remaining budget are rejected (not charged, zero-length
            outcome arrays) — check `budget_remaining` on each returned
            point. Each shot returns a syndrome flag (1=flagged, 0=clean)
            and the measured logical outcome bit(s); for n_copies=2 both
            copies' flags/outcomes are returned. Poll get_job_result.
            """
            return magic_actions.submit_magic_benchmark_batch(
                ctx, device_id=device_id, points=points
            )

    if "magic_state_cultivation" in state.active_capabilities:

        @mcp.tool()
        def run_cultivation_batch(device_id: str, points: list[dict]) -> dict:
            """Run batches of cultivation-line injection attempts.

            Each point: {"injection_theta_rad": float, "cultivation_rounds":
            int, "qec_cycles_per_round": int, "escape_cycle_n": int,
            "measure_axis": "target_axis"|"x"|"y"|"z", "shots": int}. Every
            shot is one injection attempt charged against the run's injection
            budget (kept or not); points that would exceed the remaining
            budget are rejected (not charged) — check `budget_remaining`.
            Each shot returns four detector-group flags (injection/
            cultivation/qec/graft; 1 = a detector in that group fired) and
            the terminal logical measurement outcome bit. The engine never
            pre-filters: post-selection is your choice, client-side. Poll
            get_job_result.
            """
            return magic_actions.submit_cultivation_batch(ctx, device_id=device_id, points=points)


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
