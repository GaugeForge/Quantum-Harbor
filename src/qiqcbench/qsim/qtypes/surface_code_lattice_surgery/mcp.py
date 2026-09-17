"""MCP tool registration for the ``surface_code_lattice_surgery`` qtype.

Registers the three experiment tools (gated on the ``lattice_surgery_calibration``
capability): ``run_memory_experiment``, ``run_merged_memory``,
``run_lattice_surgery_cnot``. Each returns a ``job_id`` immediately (async job model);
the agent polls ``get_job_result``. Tool documentation stays neutral and factual (what a
tool returns, its parameters and limits) — the discovery narrative is the agent's job.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import surface_code_lattice_surgery as ls_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Rotated surface-code patches on a fixed transmon-style 2D grid with a HIDDEN "
    "circuit-level noise model; calibrate decoders for the d=5 memory AND the fixed "
    "lattice-surgery CNOT schedule. run_memory_experiment(layout in {d3_control, "
    "d3_intermediate, d3_target, d5}, rounds, shots) runs single-patch Z-basis memory; "
    "run_merged_memory(window in {zz, xx}, rounds, shots, include_transitions) holds the "
    "corresponding merged configuration (with include_transitions=true the run is a full "
    "split->merge->hold->split cycle); run_lattice_surgery_cnot(config in {z, x, bell_zz, "
    "bell_xx}, logical_input, shots) runs the fixed public CNOT schedule. All return RAW "
    "per-shot detection events (packed bit arrays; column order per the public detector "
    "metadata) plus raw readout evidence, never the noise model or decoded results. One "
    "shot budget and one shot-rounds budget accumulate across ALL calls. "
    "Submit a d=5 memory DEM + a CNOT spacetime "
    "DEM + decoder configs; both are scored by replay on FRESH hidden shots."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    ctx = ActionContext(state=state, surface="mcp")

    if "lattice_surgery_calibration" in state.active_capabilities:

        @mcp.tool()
        def run_memory_experiment(device_id: str, layout: str, rounds: int, shots: int) -> dict:
            """Run a single-patch surface-code memory experiment. Returns a job_id.

            ``layout`` in {d3_control, d3_intermediate, d3_target, d5}; ``rounds`` (1-32)
            syndrome-extraction rounds; ``shots`` (1-100000) drawn from the run-long shot
            + shot-rounds budgets. Result (via get_job_result) carries raw per-shot
            detection events + observable flips and the cumulative budgets.
            """
            return ls_actions.submit_memory_experiment(
                ctx, device_id=device_id, layout=layout, rounds=rounds, shots=shots
            )

        @mcp.tool()
        def run_merged_memory(
            device_id: str,
            window: str,
            rounds: int,
            shots: int,
            include_transitions: bool = False,
        ) -> dict:
            """Run a merged-configuration memory experiment. Returns a job_id.

            ``window`` in {zz, xx}. With ``include_transitions=false`` the merged code
            (patches + routing region as one code) is held for ``rounds`` (1-16) cycles;
            with ``include_transitions=true`` the run is a full split->merge->hold->split
            cycle (2 separate rounds, ``rounds`` merged rounds, split, 2 separate rounds)
            so the transition-round detectors appear. Raw detection events over the
            merged detector set.
            """
            return ls_actions.submit_merged_memory(
                ctx,
                device_id=device_id,
                window=window,
                rounds=rounds,
                shots=shots,
                include_transitions=include_transitions,
            )

        @mcp.tool()
        def run_lattice_surgery_cnot(
            device_id: str, config: str, shots: int, logical_input: str = "00"
        ) -> dict:
            """Run the fixed public lattice-surgery CNOT schedule. Returns a job_id.

            ``config`` in {z, x, bell_zz, bell_xx} selects the reporting configuration
            (preparation/readout bases); ``logical_input`` ("00".."11") selects the
            logical input for the z/x configurations (ignored for bell configs — their
            prep is fixed). Result carries raw detection events over the CNOT spacetime
            detector set, per-shot raw m_zz/m_xx/m_z_int parities, and the output
            observable flips.
            """
            return ls_actions.submit_lattice_surgery_cnot(
                ctx,
                device_id=device_id,
                config=config,
                logical_input=logical_input,
                shots=shots,
            )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
