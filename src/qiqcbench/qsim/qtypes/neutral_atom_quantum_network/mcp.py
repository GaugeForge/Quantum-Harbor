"""MCP tool registration for the ``neutral_atom_quantum_network`` qtype.

Registers, gated on the ``vfl_distributed_estimation`` capability:
* ``run_distributed_estimation`` — async (returns a ``job_id``; poll ``get_job_result``):
  estimate the cross-node association over the photonic link, paying the communication budget.

The framing is deliberately neutral (privacy-preserving federated health analytics); the device
does not name or rank a winning protocol, and it never returns the true association / floor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import neutral_atom_quantum_network as net_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Two neutral-atom processing nodes joined by a photonic interconnect (a low-rate, lossy, "
    "error-detected QUANTUM channel + a high-bandwidth CLASSICAL channel). Over the same N "
    "patients, Node A holds a binary exposure indicator x and Node B holds a binary outcome "
    "indicator y; neither node may ship its raw record vector. Estimate the ASSOCIATION "
    "(correlation) between x and y to the target additive accuracy, verified against the hidden "
    "ground truth. ALL communication crossing the link -- qubit-transmissions on the quantum "
    "channel AND classical bits -- counts against one budget (comm_budget). "
    "run_distributed_estimation(estimator, precision, n_index_bits) estimates the association "
    "over the link, paying the link's communication cost (returned per call), and returns the "
    "raw noisy estimate; the budget accumulates across the run. Submit your "
    "association estimate + the protocol you used (estimator, precision) + your communication "
    "accounting; the verifier RE-EXECUTES your declared protocol on fresh device draws, "
    "recomputes the estimate and the communication actually crossing the link, and passes iff "
    "the estimate is accurate AND the total communication is within budget."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    ctx = ActionContext(state=state, surface="mcp")

    if "vfl_distributed_estimation" in state.active_capabilities:

        @mcp.tool()
        def run_distributed_estimation(
            device_id: str,
            estimator: str,
            precision: int,
            n_index_bits: int = 18,
        ) -> dict:
            """Estimate the cross-node association over the photonic link. Returns a job_id.

            ``estimator`` is one of ``estimator_a`` (classical channel) or ``estimator_b`` /
            ``estimator_c`` (quantum channel). ``precision`` is the selected estimator's
            precision setting; inspect the public device spec for its communication-cost
            formula. The estimate is drawn from the run-long communication budget. The result
            (via get_job_result) carries the raw estimate, the exact communication charged, and
            the cumulative budget. A reduced ``n_index_bits`` estimates a cheaper, SEPARATE
            calibration sub-instance (for measuring achievable accuracy / scaling), not the
            production data.
            """
            return net_actions.submit_distributed_estimation(
                ctx,
                device_id=device_id,
                estimator=estimator,
                precision=precision,
                n_index_bits=n_index_bits,
            )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
