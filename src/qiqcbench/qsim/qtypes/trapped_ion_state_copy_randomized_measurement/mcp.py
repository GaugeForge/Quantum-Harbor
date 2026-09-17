"""MCP tool registration for ``trapped_ion_state_copy_randomized_measurement``.

Registers, gated on the ``nonlinear_randomized_measurement`` capability, three async experiment
tools (each returns a ``job_id``; poll ``get_job_result``):
* ``run_readout_calibration`` — prepare a computational calibration state, return raw bitstrings.
* ``run_local_pauli_batch`` — single-copy random local-Pauli measurements: raw bitstrings.
* ``run_copy_block_batch`` — collective copy-block cyclic-shift measurements: raw weighted-cycle
  joint eigenvalues for every power ``j <= block_size``.

The framing names the technique (nonlinear randomized measurement), never the answer: it does
not rank the estimators, name the regime, or reveal the correlated-ratio trick. The copy-block
tool has no design defaults (block size, block count, and measurement family are the agent's
experimental-design choices); the per-family copy cost and the same-index alignment of a
simultaneous batch are instrument contract and stay public.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import trapped_ion_state_copy as tsc_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "A six-ion trapped-ion simulator prepares independent identical copies of one unknown mixed "
    "state rho on demand. Estimate the virtually cooled correlators "
    "C_k = Tr(Z0Z1 rho^k)/Tr(rho^k) for k=2 and k=3, together with the numerators Tr(Z0Z1 rho^k), "
    "the denominators Tr(rho^k), and a 1-sigma uncertainty on each ratio. "
    "run_copy_block_batch(block_size, observable, num_blocks, measurement_family) performs "
    "collective copy-block measurements and returns bounded-real joint eigenvalue records for "
    "every power j<=block_size; run_local_pauli_batch(...) performs single-copy randomized "
    "measurements (raw bitstrings); run_readout_calibration(state_label, num_shots) returns raw "
    "bitstrings from a known computational state. State copies and randomized bases are run-long "
    "budgets that accumulate across calls and fail closed. Every run_* tool returns a job_id; "
    "poll get_job_result until it is complete. Submit with submit_final_answer."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    ctx = ActionContext(state=state, surface="mcp")

    if "nonlinear_randomized_measurement" not in state.active_capabilities:
        return

    @mcp.tool()
    def run_readout_calibration(
        device_id: str, state_label: str = "all_zero", num_shots: int = 400
    ) -> dict:
        """Prepare a computational calibration state and return raw bitstrings. Returns a job_id.

        ``state_label`` is a 6-bit string (e.g. ``"000000"``) or ``"all_zero"``/``"all_one"``.
        Consumes ``num_shots`` state copies and one randomized basis.
        """
        return tsc_actions.submit_readout_calibration(
            ctx, device_id=device_id, state_label=state_label, num_shots=num_shots
        )

    @mcp.tool()
    def run_local_pauli_batch(
        device_id: str,
        basis_family: str = "random_local_pauli",
        observable: str = "Z0Z1",
        num_bases: int = 8,
        shots_per_basis: int = 600,
        random_seed_label: str | None = None,
    ) -> dict:
        """Single-copy random local-Pauli measurements. Returns a job_id.

        ``basis_family="random_local_pauli"`` (the only family): each basis draws an independent
        uniformly random Pauli axis per ion, rotates, and reads out; the result carries the raw
        bitstrings + the basis used, with string position ``i`` = ion ``i`` = ``basis[i]``.
        Consumes ``num_bases * shots_per_basis`` state copies and ``num_bases`` randomized
        bases. The agent forms any estimator in post-processing.
        """
        return tsc_actions.submit_local_pauli_batch(
            ctx,
            device_id=device_id,
            basis_family=basis_family,
            observable=observable,
            num_bases=num_bases,
            shots_per_basis=shots_per_basis,
            random_seed_label=random_seed_label,
        )

    @mcp.tool()
    def run_copy_block_batch(
        device_id: str,
        block_size: int,
        num_blocks: int,
        measurement_family: str,
        observable: str = "Z0Z1",
    ) -> dict:
        """Collective copy-block (cyclic-shift) measurements. Returns a job_id.

        ``block_size`` is 2, 3, or 4 copies per block. For each block the result returns one
        bounded-real joint eigenvalue record of the commuting weighted-cycle observables for
        every power ``j <= block_size``: a denominator record (estimating ``Tr(rho^j)``) and, for
        ``observable="Z0Z1"``, a numerator record (estimating ``Tr(Z0Z1 rho^j)``).

        ``measurement_family="simultaneous_weighted_cycle"``: numerator and denominator records
        come from the same block; the same list index across every ``j``/``num``/``den`` column
        is one joint outcome. Consumes ``block_size * num_blocks`` state copies.
        ``measurement_family="separate_cyclic_shift"``: numerator and denominator records come
        from independent block families. Consumes ``2 * block_size * num_blocks`` copies.
        ``observable="I"`` returns denominator records only (``block_size * num_blocks`` copies).
        """
        return tsc_actions.submit_copy_block_batch(
            ctx,
            device_id=device_id,
            block_size=block_size,
            observable=observable,
            num_blocks=num_blocks,
            measurement_family=measurement_family,
        )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
