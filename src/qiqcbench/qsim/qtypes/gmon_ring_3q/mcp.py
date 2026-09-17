"""MCP tool registration for the gmon-ring qtype.

Owns the qtype-local MCP surface: the instructions string the server advertises
plus a capability-gated ``register_mcp_tools`` that attaches
``run_gmon_sequence`` and ``run_gmon_sweep`` to a ``FastMCP`` instance when the
``ring_modulation`` capability is active. The top-level
``qiqcbench.qsim.mcp.server`` invokes this registrar through the qtype registry
after registering the always-on tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import gmon_ring_3q as gmon_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover - import-only typing aid
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Quantum simulator MCP server for a 3-qubit superconducting gmon ring "
    "(Q1, Q2, Q3) with tunable couplers CP12/CP23/CP31. Qubit frequencies are "
    "known and fixed; you control single-qubit XY rotations and each coupler's "
    "parametric modulation g_jk(t)=amp*env(t)*cos(2*pi*freq_hz*t + phase_rad). "
    "Submit control with run_gmon_sequence (and run_gmon_sweep), poll "
    "get_job_result for raw IQ shots. Resonant hopping between detuned qubits "
    "appears only when freq_hz matches the qubit frequency difference; the "
    "synthetic flux is the sum of modulation phases around the ring. Use "
    "submit_final_answer when ready."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    """Attach the gmon-ring MCP tools to ``mcp`` for the given qsim state.

    ``state`` is the ``QsimState`` instance; typed as ``Any`` so the qtype
    module does not import runtime state just to register tools. Tools are
    gated on the ``ring_modulation`` capability.
    """
    ctx = ActionContext(state=state, surface="mcp")

    if "ring_modulation" not in state.active_capabilities:
        return

    @mcp.tool()
    def run_gmon_sequence(device_id: str, sequence: list[dict], shots: int = 1024) -> dict:
        """Submit a gmon control sequence. Returns a job_id immediately.

        `sequence` is a list of ops:
          - {"kind": "xy_rot", "qubit": "Q1", "axis": "x"|"y"|"z", "angle_rad": <float>}
            A single-qubit rotation (state prep / pre-measure tomography).
          - {"kind": "evolve", "duration_ns": <float>, "couplers": [...],
             "envelope": "constant"|"raised_cosine", "ramp_ns": <float|null>}
            Evolve the ring under concurrent coupler modulations. Each coupler
            setting is {"coupler": "CP12", "amp": <0..1>, "freq_hz": <float>,
            "phase_rad": <float>}; couplers omitted are off. Use envelope
            "raised_cosine" with ramp_ns for adiabatic ramps.
          - {"kind": "delay", "duration_ns": <float>}  # idle (couplers off)
          - {"kind": "gmon_measure", "qubits": ["Q1", ...]}  # computational-basis readout
        A sequence must contain a gmon_measure op. Poll get_job_result(job_id);
        successful result data is raw per-qubit IQ shots (data.kind="iq").
        """
        return gmon_actions.submit_gmon_sequence(
            ctx, device_id=device_id, sequence=sequence, shots=shots
        )

    @mcp.tool()
    def run_gmon_sweep(
        device_id: str,
        template_sequence: list[dict],
        sweep: dict[str, list[float]],
        shots: int = 512,
        mode: str = "product",
    ) -> dict:
        """Submit a gmon control sweep. Returns one job_id covering all points.

        Sweepable numeric fields in `template_sequence` may be string
        placeholders (e.g. duration_ns="$t", a coupler phase_rad="$phi") bound
        from `sweep`. mode="product" runs the Cartesian product of sweep keys;
        mode="zip" pairs equal-length arrays. Result data is per-qubit IQ with
        an outer list per sweep point; metadata.sweep_coords records the values.
        """
        return gmon_actions.submit_gmon_sweep(
            ctx,
            device_id=device_id,
            template_sequence=template_sequence,
            sweep=sweep,
            shots=shots,
            mode=mode,
        )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
