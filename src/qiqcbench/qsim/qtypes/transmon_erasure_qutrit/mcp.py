"""MCP tool registration for the g-f erasure-qutrit qtype.

Attaches ``run_logical_memory`` / ``run_logical_memory_sweep`` when the
``mid_circuit_erasure_detection`` capability is active. Always-on tools are
registered by the server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import erasure_qutrit as erasure_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover - import-only typing aid
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Quantum simulator MCP server. The device is a transmon qutrit operated as a "
    "g-f erasure qubit (|0_L>=|g>, |1_L>=|f>, |e>=erasure/leakage) with a "
    "neighbouring ancilla for mid-circuit erasure detection. Use "
    "run_logical_memory / run_logical_memory_sweep to prepare a logical Z- or "
    "X-basis state, interleave dynamical decoupling with erasure-detection rounds "
    "at a chosen cycle time, and read out the data qutrit. Each shot returns "
    "the ordered post-readout ancilla syndrome bits for every round and the same "
    "shot's final post-readout qutrit assignment (0=g, 1=e, 2=f). Cadences "
    "must lie on the public range/resolution grid. Poll "
    "get_job_result; submit_final_answer when ready."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    """Attach the erasure-qutrit logical-memory tools for the given qsim state."""

    ctx = ActionContext(state=state, surface="mcp")

    if "mid_circuit_erasure_detection" in state.active_capabilities:

        @mcp.tool()
        def run_logical_memory(
            device_id: str,
            prep_state: str,
            n_rounds: int,
            cycle_time_us: float,
            prep_basis: str = "z",
            measure_basis: str = "z",
            dd: str = "xy4",
            shots: int = 2048,
        ) -> dict:
            """Prepare a logical state, run `n_rounds` of DD + erasure detection at
            `cycle_time_us`, then read out. Returns a job_id.

            `prep_state` is "0L"/"1L" (prep_basis "z") or "+X"/"-X" (prep_basis
            "x"); `measure_basis` must match `prep_basis`. `dd` is "xy4" or
            "none". Each shot in the result reports one ordered ancilla syndrome
            bit per detection round and its final qutrit assignment (0=g, 1=e,
            2=f). Poll get_job_result.
            """
            return erasure_actions.submit_logical_memory(
                ctx,
                device_id=device_id,
                prep_basis=prep_basis,
                prep_state=prep_state,
                measure_basis=measure_basis,
                n_rounds=n_rounds,
                cycle_time_us=cycle_time_us,
                dd=dd,
                shots=shots,
            )

        @mcp.tool()
        def run_logical_memory_sweep(
            device_id: str,
            prep_state: str,
            n_rounds_grid: list[int],
            cycle_time_us: float,
            prep_basis: str = "z",
            measure_basis: str = "z",
            dd: str = "xy4",
            shots: int = 1024,
        ) -> dict:
            """Sweep the number of erasure-detection rounds on a fixed prep/measure.

            `measure_basis` must match `prep_basis`. One measurement record is
            produced per entry of `n_rounds_grid`, in order; total evolution time
            of a point is `n_rounds * cycle_time_us`. The per-point sweep
            coordinates are returned in the job metadata as
            `sweep_coords["total_evolution_us"]`. Returns one job_id.
            """
            return erasure_actions.submit_logical_memory_sweep(
                ctx,
                device_id=device_id,
                prep_basis=prep_basis,
                prep_state=prep_state,
                measure_basis=measure_basis,
                n_rounds_grid=n_rounds_grid,
                cycle_time_us=cycle_time_us,
                dd=dd,
                shots=shots,
            )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
