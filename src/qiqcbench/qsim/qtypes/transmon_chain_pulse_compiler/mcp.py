"""MCP tool registration for the chain pulse-compiler qtype.

Owns the qtype-local MCP surface: the instructions string the server advertises
plus a capability-gated ``register_mcp_tools`` that attaches
``run_compiled_circuit`` when the ``chain_pulse_compilation`` capability is
active. The top-level ``qiqcbench.qsim.mcp.server`` invokes this registrar
through the qtype registry after registering the always-on tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import transmon_chain_pulse_compiler as chain_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover - import-only typing aid
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Quantum simulator MCP server for a 5-qubit transmon chain "
    "(q0-q1-q2-q3-q4). Native gates: virtual-Z Rz (free frame change), "
    "finite-duration Rx/Ry, and a continuous CPhase(theta) available between ANY "
    "qubit pair (ONE entangling pulse per controlled-phase; CZ = CPhase(pi)). "
    "Build a scheduled gate program and run it with run_compiled_circuit, then "
    "poll get_job_result for raw per-shot, per-qubit level-resolved outcomes "
    "(0/1/2; 2 = leakage). The entangling-gate pulse can be recalibrated via the "
    "circuit's optional 'calibration' field. All run_* tools are async: submit, then poll "
    "get_job_result. Use submit_final_answer when ready."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    """Attach the chain pulse-compiler MCP tools, gated on ``chain_pulse_compilation``."""
    ctx = ActionContext(state=state, surface="mcp")

    if "chain_pulse_compilation" not in state.active_capabilities:
        return

    @mcp.tool()
    def run_compiled_circuit(
        device_id: str,
        circuit: list[dict],
        shots: int = 2048,
        calibration: dict | None = None,
        measure_qubits: list[int] | None = None,
    ) -> dict:
        """Run a scheduled gate program from |0...0>. Returns a job_id immediately.

        `circuit` is an ordered list of gate ops, each:
          {"type": "rz"|"rx"|"ry"|"cphase", "qubits": [q] or [a, b],
           "angle": <rad>, "virtual": <bool, rz only>, "layer": <int>}
        - rz with virtual=true is a free frame change (zero duration, ~zero error);
          virtual=false is charged as a finite physical pulse.
        - rx/ry are finite-duration single-qubit rotations.
        - cphase(theta) is the native entangling gate on any qubit pair [a, b]
          (CZ = cphase with angle=pi). One pulse per controlled-phase.
        `layer` is a schedule annotation with a validity constraint (no two
        gates in one layer may share a qubit).

        `calibration` (optional) recalibrates the native CPhase pulse:
          {"over_rotation_correction": <float>, "leakage_correction": <float>}
        Identity (omit, or both 1.0) replays the shipped pulse;
        the realized conditional phase is shipped_over_rotation * correction *
        intended.

        `measure_qubits` (optional) selects qubits to read out (default all),
        always in the computational (Z) basis. Append your own basis-rotation
        gates for X/Y measurements. Poll get_job_result(job_id); result data is
        raw per-shot, per-qubit level strings (data.kind="chain_level_outcome").
        """
        return chain_actions.submit_chain_circuit(
            ctx,
            device_id=device_id,
            circuit=circuit,
            shots=shots,
            calibration=calibration,
            measure_qubits=measure_qubits,
        )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
