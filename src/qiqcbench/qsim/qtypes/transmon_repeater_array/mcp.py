"""MCP tool registration for the ``transmon_repeater_array`` qtype.

Registers one batched experiment tool (gated on ``entanglement_purification``):
``run_purification_experiment``. It returns a ``job_id`` immediately (async job model);
the agent polls ``get_job_result``. The instructions describe the device and the tool
contract only -- they do not name a best circuit, an optimal depth, or the noise floor
(those are the science under test). The agent designs the local unitaries; the score is
the re-run fidelity of the submitted design.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import transmon_repeater_array as rep_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Transmon-array quantum repeater. Bell pairs |Phi+> = (|00>+|11>)/sqrt(2) are generated "
    "locally in the centre of the array and SWAP-transported to two distant ends, Alice (left) "
    "and Bob (right); transport leaves them noisy, and the transport noise is not characterized "
    "in the notebook and need not be isotropic -- characterize the delivered pair before you "
    "design. Each end may apply ANY local unitary to its "
    "two held qubits; the device then measures the second pair and keeps the first pair when the "
    "two outcomes agree, converting two noisy pairs into one with some success probability, and "
    "the kept pairs feed the next round. Nothing may act across the two ends. Each end applies "
    "ONE fixed gate list, identically on every purification round (the two-to-one recurrence "
    "family of Bennett et al. quant-ph/9511027 and Deutsch et al. quant-ph/9604039). DESIGN the "
    "local unitaries and the number of rounds that maximize the FINAL fidelity of the surviving "
    "pair. Estimate fidelity from the three Pauli correlators: F = (1 + <ZZ> + <XX> - <YY>)/4. "
    "Tool (async, returns job_id; poll get_job_result): run_purification_experiment(device_id, "
    "rows). Each row = {depth, alice_circuit, bob_circuit, measure_basis, shots}: 'depth' = "
    "number of purification rounds, 0-7 (each surviving pair at depth d costs about "
    "2^d / P_success distributed pairs, P_success being the cumulative post-selection success "
    "probability returned with the row; depth=0 returns the bare distributed pair); "
    "'alice_circuit'/'bob_circuit' = the gate list that party applies to its two qubits each "
    "round; 'measure_basis' = any of the nine two-letter Pauli settings XX, XY, XZ, YX, YY, "
    "YZ, ZX, ZY, ZZ (first letter Alice's basis, second Bob's; the four 2-bit outcome counts "
    "give the correlator and both single-qubit marginals -- nine settings = full two-qubit "
    "tomography; F uses the three same-basis settings); 'shots' 64-20000. A gate is {gate, q, angle_deg} "
    "for rx/ry/rz, {gate, q} for h/s/sdg/x/y/z, or {gate, control, target} for cnot; party-local "
    "qubit indices are 0 (your half of pair-1 = the kept pair) and 1 (your half of pair-2 = the "
    'measured pair). Results give raw 2-bit counts ("ab", a=Alice, b=Bob), the success '
    "probability, the pairs charged, and a running pair-budget meter; rows are rejected once "
    "the budget is exhausted. Goal: submit your design (alice_circuit, bob_circuit, "
    "optimal_rounds) together with your own measurement of it; the verifier re-evaluates the "
    "submitted design on the device's true noise model and checks your reported numbers against "
    "your logged measurements."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    ctx = ActionContext(state=state, surface="mcp")

    if "entanglement_purification" in state.active_capabilities:

        @mcp.tool()
        def run_purification_experiment(device_id: str, rows: list[dict]) -> dict:
            """Run a batch of purification experiments. Returns a job_id immediately.

            Each row: ``depth`` (0-7 purification rounds); ``alice_circuit`` / ``bob_circuit``
            (the local unitary that party applies to its two qubits each round, as a list of
            gates -- ``{gate, q, angle_deg}`` for ``rx``/``ry``/``rz``, ``{gate, q}`` for
            ``h``/``s``/``sdg``/``x``/``y``/``z``, ``{gate, control, target}`` for ``cnot`` --
            on party-local qubits 0/1); ``measure_basis`` (``XX`` | ``XY`` | ``XZ`` | ``YX`` |
            ``YY`` | ``YZ`` | ``ZX`` | ``ZY`` | ``ZZ``);
            ``shots`` (64-20000). Each round applies your local unitaries, measures the second
            pair, and keeps the first when the outcomes agree; iterated to ``depth``. Returns raw
            2-bit outcome counts (``"ab"``), the success probability, the pairs charged, and the
            cumulative pair budget. The score is the re-run fidelity of the design you submit.
            """
            return rep_actions.submit_purification_batch(ctx, device_id=device_id, rows=rows)
