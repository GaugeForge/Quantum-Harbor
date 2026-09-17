"""MCP tool registration for the bosonic_cavity_qec qtype.

Attaches ``run_bosonic_program`` / ``run_bosonic_program_sweep`` when the
``bosonic_qec_control`` capability is active. Always-on tools (list_devices,
get_device_spec, get_lab_notebook, get_job_result, submit_final_answer) are
registered by the server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import bosonic_cavity_qec as bosonic_actions
from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.qtypes.bosonic_cavity_qec.public_surface import (
    load_bosonic_cavity_qtype_agent_surface,
)

if TYPE_CHECKING:  # pragma: no cover - import-only typing aid
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = load_bosonic_cavity_qtype_agent_surface().mcp_instructions


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    """Attach the bosonic-cavity QEC program tools for the given qsim state."""

    ctx = ActionContext(state=state, surface="mcp")

    if "bosonic_qec_control" in state.active_capabilities:

        @mcp.tool()
        def run_bosonic_program(
            device_id: str,
            ops: list[dict],
            shots: int = 2048,
            evidence_tag: dict[str, Any] | None = None,
        ) -> dict:
            """Run one bosonic program (ordered list of primitive ops) for `shots`
            shots. Returns a job_id; poll get_job_result.

            Each op is a dict with a `kind` key:
              {"kind": "displace", "alpha_re": .., "alpha_im": ..}
              {"kind": "snap", "thetas": [..]}                  # phase on Fock level n
              {"kind": "ancilla_rotate", "theta": .., "phi": ..}
              {"kind": "dispersive_wait", "duration_ns": ..}    # parity-map primitive
              {"kind": "ancilla_measure", "reset": true}        # records a syndrome bit
              {"kind": "ancilla_measure", "reset": true, "record": false}  # measure+reset, no bit,
                                                                #   no branch cost, not conditionable
              {"kind": "photon_number_measure"}                 # records a photon number
              {"kind": "idle", "duration_ns": ..}               # storage (photon loss)
              {"kind": "conditional", "on_index": k, "value": 1, "op": {..}}  # feedback
            `on_index` indexes the k-th ancilla_measure (0-based). The nested op may
            control/storage operations but not a measurement, and the indexed
            measurement must occur before the conditional. Each shot in the result
            returns the ordered list of recorded measurement outcomes. See the public
            device spec for per-call, run-wide, outstanding-job, and retained-result
            budgets. A program must contain at least one measurement.

            `evidence_tag` is optional and does not change execution. Use it only
            when the active task instruction defines an exact pre-execution
            scientific-evidence contract.
            """
            return bosonic_actions.submit_bosonic_program(
                ctx,
                device_id=device_id,
                ops=ops,
                shots=shots,
                evidence_tag=evidence_tag,
            )

        @mcp.tool()
        def run_bosonic_program_sweep(
            device_id: str,
            ops: list[dict],
            sweep_op_index: int,
            sweep_field: str,
            sweep_values: list[float],
            shots: int = 1024,
            evidence_tag: dict[str, Any] | None = None,
        ) -> dict:
            """Run a bosonic program once per sweep value, overriding one scalar
            field (`sweep_field`, e.g. "duration_ns" or "alpha_re") of
            `ops[sweep_op_index]`. One result point per value, in order. Returns a
            job_id. Useful for storage-time decay curves (sweep an idle duration)
            and chi calibration (sweep a dispersive_wait duration).

            A task-defined `evidence_tag` applies to the whole sweep. The exact
            point coordinate is committed separately, so a task may derive a
            point's physical label from the materialized program.
            """
            return bosonic_actions.submit_bosonic_program_sweep(
                ctx,
                device_id=device_id,
                ops=ops,
                sweep_op_index=sweep_op_index,
                sweep_field=sweep_field,
                sweep_values=sweep_values,
                shots=shots,
                evidence_tag=evidence_tag,
            )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
