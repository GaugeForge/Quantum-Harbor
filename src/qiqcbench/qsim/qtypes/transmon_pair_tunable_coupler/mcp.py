"""MCP tool registration for the tunable-coupler CZ qtype."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import transmon_pair_tunable_coupler as cz_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Quantum simulator MCP server for two transmons (q1 flux-tunable, q2 fixed) "
    "coupled by a tunable coupler. Build a controlled-Z via a NET-ZERO adiabatic "
    "flux pulse on the |11>-|02> avoided crossing. You PROGRAM DAC samples for "
    "q1.flux (Phi_0, on the 0.4 ns AWG grid); the flux the qubit actually sees is "
    "your programmed waveform DISTORTED by the flux line (short-time settling) — "
    "characterize it (cryoscope on the quadratic sweet-spot arc) and predistort. "
    "Set a constant coupler bias (it sets both J_eff and the static ZZ; park the "
    "idle coupler where BOTH vanish). Use run_flux_pulse with prep_ops / post_ops "
    "(single-qubit rotations for state prep and pre-measure tomography), the "
    "programmed flux, the coupler bias, and an optional idle (for static-ZZ). "
    "Read out level-resolved per-qubit outcomes (0/1/2). Judge the CZ by the "
    "conditional phase phi_2Q = phi_00 - phi_01 - phi_10 + phi_11 (target pi) AFTER "
    "the single-qubit virtual-Z corrections you report. All run_* tools are async: "
    "submit, then poll get_job_result. Use submit_final_answer when ready."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    ctx = ActionContext(state=state, surface="mcp")
    if "netzero_flux_control" not in state.active_capabilities:
        return

    @mcp.tool()
    def run_flux_pulse(
        device_id: str,
        coupler_flux: float,
        programmed_flux_q1: list[float] | None = None,
        sample_dt_ns: float = 0.4,
        prep_ops: list[dict] | None = None,
        post_ops: list[dict] | None = None,
        idle_ns: float = 0.0,
        measure_qubits: list[int] | None = None,
        shots: int = 4096,
    ) -> dict:
        """Run a q1 flux pulse at a fixed coupler bias and read out. Returns a job_id.

        - `programmed_flux_q1`: DAC samples for q1.flux (Phi_0) on the AWG grid; the
          qubit sees the flux-line-distorted version (predistort to compensate).
        - `coupler_flux`: coupler bias in Phi_0 (sets J_eff and the static ZZ).
        - `prep_ops` / `post_ops`: single-qubit rotations
          {"qubit": 1|2, "axis": "x"|"y"|"z", "angle_rad": <float>} before / after
          the pulse (state prep + pre-measure tomography).
        - `idle_ns`: idle evolution with no flux pulse (for static-ZZ Ramsey).
        - `measure_qubits`: which qubits to read out (default [1, 2]), level-resolved.

        Poll get_job_result(job_id); data.kind = "pair_level_outcome" (per-shot
        per-qubit levels 0/1/2). Useful experiments: |11>-|02> chevron (prep |11>,
        scan the pulse, watch q2 -> level 2); conditional-phase Ramsey (prep q1 in
        |+> via ry(pi/2), post ry(-pi/2)); static-ZZ (prep |+>|+>, idle); cryoscope
        (truncated pulse + Ramsey); CZ verification (computational prep + tomography).
        """
        return cz_actions.submit_flux_pulse(
            ctx,
            device_id=device_id,
            coupler_flux=coupler_flux,
            programmed_flux_q1=programmed_flux_q1,
            sample_dt_ns=sample_dt_ns,
            prep_ops=prep_ops,
            post_ops=post_ops,
            idle_ns=idle_ns,
            measure_qubits=measure_qubits,
            shots=shots,
        )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
