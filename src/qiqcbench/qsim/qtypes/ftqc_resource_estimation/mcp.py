"""MCP tool registration for the ``ftqc_resource_estimation`` qtype.

Registers **synchronous** calculator tools (no ``job_id``, no polling — mirroring
the always-on synchronous ``submit_final_answer`` / ``get_device_spec`` tools).
The async-job invariant governs ``run_*`` *experiment* tools only; this qtype runs
no quantum dynamics.

Two task families, each gated on its own capability (fail-closed):

* ``ftqc_physical_estimation`` → ``get_algorithm_instance`` +
  ``evaluate_factory_design`` (physical CCZ2T spacetime-volume estimation).
* ``mps_qpe_resource_planning`` → ``get_mps_qpe_instance`` +
  ``evaluate_mps_qpe_plan`` (MPS-initialized logical QPE planning).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import ftqc as ftqc_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover - import-only typing aid
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Fault-tolerant quantum resource-estimation MCP server. The task is classical "
    "resource estimation, not a quantum experiment: there are no shots and no async "
    "jobs — the tools are synchronous calculators. The available tools depend on the "
    "active capability. Read get_device_spec and get_lab_notebook first (the notebook "
    "holds operator notes carried over from prior studies; verify anything you rely "
    "on against the device). Evaluators are capped "
    "at evaluator_call_cap calls and return only the objective + a feasibility "
    "boolean. Apply the active task's public contract locally, then use the evaluator "
    "to verify a final candidate. Use "
    "submit_final_answer when ready."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    """Attach the synchronous calculator tools to ``mcp`` for the qsim state."""
    ctx = ActionContext(state=state, surface="mcp")

    if "mps_qpe_resource_planning" in state.active_capabilities:

        @mcp.tool()
        def get_mps_qpe_instance(device_id: str) -> dict:
            """Return public MPS-QPE case metadata and material handles.

            This is a synchronous static-resource task. The complete public case
            materials are mounted under ``/task_materials``; no hidden optimum or
            source PDF is returned.
            """
            return ftqc_actions.get_mps_qpe_instance(ctx, device_id=device_id)

        @mcp.tool()
        def evaluate_mps_qpe_plan(
            device_id: str,
            selected_block_encoding_id: str,
            candidate_methods: list[str],
            mps_descriptor_ids: list[str],
        ) -> dict:
            """Evaluate one method and MPS-descriptor selection per candidate.

            Returns validity, total logical Toffolis, and logical-qubit high-water.
            It does not expose the optimum, overlap, sub-costs, or hidden cases.
            The synchronous evaluator is capped by ``evaluator_call_cap``.
            """
            return ftqc_actions.evaluate_mps_qpe_plan(
                ctx,
                device_id=device_id,
                selected_block_encoding_id=selected_block_encoding_id,
                candidate_methods=candidate_methods,
                mps_descriptor_ids=mps_descriptor_ids,
            )

    if "ftqc_physical_estimation" in state.active_capabilities:

        @mcp.tool()
        def get_algorithm_instance(device_id: str) -> dict:
            """Return the fixed logical algorithm summary + public CCZ2T cost model.

            Synchronous (no job_id). Returns the algorithm's logical counts
            (n_algo_qubits, n_t, n_ccz — also mounted at /task_materials), the public
            Qualtran 0.7.0 Gidney-Fowler CCZ2T cost-model identity, the design domains
            (l1/l2 distances, n_factory choices, data-block distance), the failure
            budget + physical-qubit cap, and broad priors on the held-out hardware
            (phys_err, cycle_time_us). qualtran is preinstalled in this container.
            """
            return ftqc_actions.get_algorithm_instance(ctx, device_id=device_id)

        @mcp.tool()
        def evaluate_factory_design(
            device_id: str,
            distillation_l1_d: int,
            distillation_l2_d: int,
            n_factories: int,
            data_block_d: int,
        ) -> dict:
            """Cost oracle for a CCZ2T magic-state-factory + data-block design.

            Synchronous deterministic calculator (no job_id). Returns only the
            objective ``spacetime_volume_qubit_seconds`` and a single feasibility
            boolean ``valid`` (failure prob within budget AND footprint within cap) —
            never the footprint, failure probability, duration, or which constraint
            failed. The footprint and cycle count are computable from the public
            algorithm + design alone, so one reading pins the hidden cycle_time_us;
            ``valid`` bounds the hidden phys_err via the code-distance suppression law.
            Capped at evaluator_call_cap; once exhausted it returns accepted=false.
            """
            return ftqc_actions.evaluate_factory_design(
                ctx,
                device_id=device_id,
                distillation_l1_d=distillation_l1_d,
                distillation_l2_d=distillation_l2_d,
                n_factories=n_factories,
                data_block_d=data_block_d,
            )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
