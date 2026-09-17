"""MCP tool registration for the ``cycle_error_recon`` qtype.

Registers the two batched experiment tools (gated on the ``folded_cer`` capability):
``run_readout_calibration_batch`` and ``run_folded_cer_batch``. Both return a ``job_id``
immediately (async job model); the agent polls ``get_job_result``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import cycle_error_recon as cer_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Contextual folded cycle-error-reconstruction device (5 qubits, line connectivity, "
    "native CZ). Two schedules (parallel_2, serial_4) implement the SAME ideal hard cycle "
    "G = CZ01 CZ12 CZ23 CZ34 (G^2 = I) but have different physical error channels. Per "
    "schedule, reconstruct: a sparse intrinsic Pauli channel over the fixed 30-term "
    "dictionary, at most one coherent Pauli rotation U_Q(theta)=exp(-i theta Q/2) from the "
    "candidates {none,X2,Z1Z2,Z2Z3,Z1Z3}, and asymmetric per-qubit readout. Tools: "
    "run_readout_calibration_batch (trusted ideal-prep calibration -> raw 5-bit counts), "
    "run_folded_cer_batch (folded CER -> raw 5-bit counts per randomization + ideal parity "
    "sign; probe P starts in its +1 product eigenstate, measures in the P basis, and each "
    "randomization is reduced to parity on P's non-I support times ideal_parity_sign; for fold "
    "x the hard block is (noisy cycle)^x with x odd so the ideal action stays G, randomized "
    "Pauli dressing between folded blocks). The backend returns raw counts "
    "only -- never fidelities, decays, exact probabilities, or hidden parameters. Budgets "
    "(hard-cycle exposure, raw shots, rows) accumulate across the run. Submit one structured "
    "final answer with both predictive channel models and the workload choice."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    ctx = ActionContext(state=state, surface="mcp")

    if "folded_cer" in state.active_capabilities:

        @mcp.tool()
        def run_readout_calibration_batch(device_id: str, rows: list[dict]) -> dict:
            """Trusted-basis (ideal prep) readout calibration. Returns a job_id immediately.

            Each row: ``prepared_bitstring`` (5 chars of 0/1, leftmost = q0), ``shots``
            (64-4096). Counts toward the raw-shot budget, not hard-cycle exposure.
            """
            return cer_actions.submit_readout_calibration_batch(ctx, device_id=device_id, rows=rows)

        @mcp.tool()
        def run_folded_cer_batch(device_id: str, rows: list[dict]) -> dict:
            """Folded cycle-error-reconstruction batch. Returns a job_id immediately.

            Each row: ``schedule`` (parallel_2|serial_4), ``probe_pauli`` (5-char I/X/Y/Z,
            weight 1-3), ``fold_factor`` (1,3,5,7), ``block_repetitions`` (2,4,8,16,32),
            ``num_randomizations`` (1-32), ``shots_per_randomization`` (64-1024). Returns raw
            5-bit counts per randomization plus ``ideal_parity_sign``. Probe P starts in its
            +1 product eigenstate (I positions use |0>) and is measured in the P basis (I
            positions use Z). The full bitstring distribution includes the tensor-product
            asymmetric readout channel. Reduce each count map to parity on P's non-I support and
            multiply by ``ideal_parity_sign``. After readout correction, the randomization mean is
            the product of folded-block fidelities along P's orbit under G, with one factor per
            block repetition; it is not generally one Pauli fidelity raised to that power.
            """
            return cer_actions.submit_folded_cer_batch(ctx, device_id=device_id, rows=rows)


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
