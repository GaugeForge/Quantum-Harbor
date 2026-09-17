"""MCP tool registration for ``composite_measurement_estimation``.

Synchronous calculators (``get_observable_spec``, ``evaluate_composite_scheme``)
return results directly. Async experiments (``run_pilot_measurements``,
``execute_locked_composite_scheme``) return ``{job_id, status}`` and the agent
polls ``get_job_result``. All gated on the ``composite_measurement`` capability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import composite_measurement_estimation as cme_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Adaptive composite-measurement design for a fixed 14-qubit H2O/STO-3G Pauli "
    "Hamiltonian (public, mounted at /task_materials) and one fixed hidden state. "
    "get_observable_spec returns conventions, budgets, and the pinned estimator. "
    "Track A: design a 16-component C-LBCS scheme minimizing the public state-"
    "independent V_Haar; evaluate_composite_scheme (capped at evaluator_call_cap) "
    "returns the exact V_Haar + coverage quantiles. Track B: use run_pilot_measurements "
    "(<=96 settings, <=24576 shots) to learn the hidden state, then call "
    "execute_locked_composite_scheme ONCE with a locked 16-component scheme + <=192 "
    "sparse control variates; production draws 192 independent bases x 256 shots and "
    "returns raw counts + a request_digest. Report the energy with the pinned inverse-"
    "coverage control-variate estimator and a CLUSTER-correct 1-sigma uncertainty "
    "(treating the 49152 shots as IID is wrong). The verifier scores the locked scheme "
    "and control variates as objects from completed qsim evidence, not your reported "
    "scalars. Use submit_final_answer when ready."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    ctx = ActionContext(state=state, surface="mcp")

    if "composite_measurement" in state.active_capabilities:

        @mcp.tool()
        def get_observable_spec(device_id: str) -> dict:
            """Public Hamiltonian conventions, budgets, and the pinned estimator formula.

            Synchronous: returns the payload directly. The Hamiltonian CSV and the stale
            notebook (with a warm-start scheme) are mounted read-only at /task_materials.
            """
            return cme_actions.get_observable_spec(ctx, device_id=device_id)

        @mcp.tool()
        def evaluate_composite_scheme(
            device_id: str,
            mixture_weights: list[float],
            local_basis_probabilities_xyz: list[list[list[float]]],
        ) -> dict:
            """Exact public V_Haar + coverage quantiles for a proposed C-LBCS scheme.

            Synchronous calculator (no job_id), capped at evaluator_call_cap calls.
            ``mixture_weights`` is length 16; ``local_basis_probabilities_xyz`` is
            [16][14][3] with last-axis order X, Y, Z. Returns no hidden-state quantities.
            """
            return cme_actions.evaluate_composite_scheme(
                ctx,
                device_id=device_id,
                mixture_weights=mixture_weights,
                local_basis_probabilities_xyz=local_basis_probabilities_xyz,
            )

        @mcp.tool()
        def run_pilot_measurements(device_id: str, settings: list[dict]) -> dict:
            """Run product-Pauli pilot settings on the hidden state; return raw counts.

            Async: returns {job_id, status}; poll get_job_result. ``settings`` is a list
            of {"basis": <14-char XYZ string>, "shots": <multiple of 64 in [64,512]>}.
            Budget accumulates across calls (<=96 settings, <=24576 shots total).
            """
            return cme_actions.run_pilot_measurements(ctx, device_id=device_id, settings=settings)

        @mcp.tool()
        def execute_locked_composite_scheme(
            device_id: str,
            mixture_weights: list[float],
            local_basis_probabilities_xyz: list[list[list[float]]],
            control_variate_entries: list[dict],
        ) -> dict:
            """Lock a Track-B scheme + sparse control variates and run production ONCE.

            Async: returns {job_id, status}; poll get_job_result. The completed result has
            192 settings x 256 raw-count shots and a request_digest of the locked design.
            Coverage (h_j >= 1e-9) is validated at lock time and fails closed before
            consuming the single production run; a failed lock returns a generic invalid
            message and counts against a small lock-attempt budget. ``control_variate_entries``
            is a list of {"term_index": <int>, "mean": <float in [-1,1]>}, <=192 entries,
            no duplicates, identity index 0 forbidden.
            """
            return cme_actions.execute_locked_composite_scheme(
                ctx,
                device_id=device_id,
                mixture_weights=mixture_weights,
                local_basis_probabilities_xyz=local_basis_probabilities_xyz,
                control_variate_entries=control_variate_entries,
            )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
