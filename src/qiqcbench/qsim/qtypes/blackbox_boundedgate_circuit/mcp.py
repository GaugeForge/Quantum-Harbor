"""MCP tool registration for the blackbox bounded-gate qtype.

Owns the qtype-local MCP surface: the instructions string and
``register_mcp_tools``, which attaches the capability-gated ``run_basis_shots``
tool to a ``FastMCP`` instance. The always-on tools (get_device_spec,
get_lab_notebook, get_job_result, submit_final_answer, list_devices) are
registered by the top-level server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import boundedgate as boundedgate_actions
from qiqcbench.qsim.actions import sealed_holdout as sealed_holdout_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover - import-only typing aid
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Quantum simulator MCP server. Operate a black-box bounded-gate device: a "
    "fixed hidden n-qubit circuit U(x) built from Clifford gates plus a bounded "
    "number of single-qubit rotations driven by a small input vector x (bounds "
    "and budgets: get_device_spec). Submit run_basis_shots jobs choosing the "
    "input x in [-pi, pi]^d, per-qubit measurement bases (a fixed length-n "
    "string over x/y/z, or 'random_pauli' for per-shot random bases), and shot "
    "counts; poll get_job_result for raw per-shot bitstrings. The total shot and "
    "job budgets accumulate across the whole run and rejected requests consume "
    "nothing. A task may expose a sealed holdout: when present, "
    "lock_measurements_and_reveal_challenge irreversibly closes further measurement "
    "admission and returns its precommitted target inputs. The circuit and internal "
    "measurement behavior are hidden; get_lab_notebook returns the public "
    "characterization record. Use submit_final_answer when ready."
)


def _validate_sealed_holdout_state(state: Any) -> None:
    """Fail startup when an enabled reveal capability has no coherent opening."""
    if sealed_holdout_actions.CAPABILITY not in state.active_capabilities:
        return
    public_challenge = state.public.target_challenge
    hidden_challenge = state.hidden.target_challenge
    if public_challenge is None or hidden_challenge is None:
        raise RuntimeError(
            "sealed_holdout_reveal requires public and hidden target_challenge configs"
        )
    if public_challenge.challenge_id != hidden_challenge.challenge_id:
        raise RuntimeError("public/hidden sealed challenge IDs disagree")
    if public_challenge.commitment_scheme != hidden_challenge.commitment_scheme:
        raise RuntimeError("public/hidden sealed challenge schemes disagree")
    if public_challenge.target_commitment_sha256 != hidden_challenge.target_commitment_sha256:
        raise RuntimeError("public/hidden sealed challenge commitments disagree")
    if public_challenge.target_count != len(hidden_challenge.target_inputs):
        raise RuntimeError("public sealed target count disagrees with hidden target inputs")
    if public_challenge.input_dimension != state.public.input_dimension:
        raise RuntimeError("public sealed challenge input dimension disagrees with the device")


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    """Attach the basis-shot tool to ``mcp`` for the given qsim state.

    ``state`` is the ``QsimState`` instance; typed ``Any`` so the qtype module
    does not import runtime state just to register tools.
    """

    _validate_sealed_holdout_state(state)
    ctx = ActionContext(state=state, surface="mcp")

    if boundedgate_actions.CAPABILITY in state.active_capabilities:

        @mcp.tool()
        def run_basis_shots(device_id: str, blocks: list[dict]) -> dict:
            """Measure the hidden circuit. Returns a job_id immediately.

            ``blocks`` is a list of measurement blocks (see get_device_spec for
            the per-job block/setting caps), each:
              {"x": [x1, ..., xd],       # circuit input, each component in [-pi, pi]
               "settings": [
                 {"basis": "zzz...z",    # length-n per-qubit bases over x/y/z,
                                         # or "random_pauli" for per-shot random bases
                  "shots": <int>}]}      # per-setting shot count

            Poll get_job_result(job_id); the result data has kind="basis_shots"
            with raw per-shot bitstrings per block/setting (character i =
            qubit i; outcome b means measured eigenvalue (-1)^b), random_bases
            strings for random_pauli settings, and the cumulative shots/jobs
            budget with the device clock (elapsed_wall_clock_s, seconds since
            the device started). Large results are offloaded to
            /qsim_logs/public_job_results/results/<job_id>.json (a public_warnings entry
            says so). Over-budget or malformed requests are rejected atomically
            and consume no budget.
            """
            return boundedgate_actions.submit_basis_shots(ctx, device_id=device_id, blocks=blocks)

    if sealed_holdout_actions.CAPABILITY in state.active_capabilities:

        @mcp.tool()
        def lock_measurements_and_reveal_challenge(device_id: str) -> dict:
            """Irreversibly seal measurements and reveal precommitted targets.

            All basis-shot jobs admitted before this call may finish, but no
            later run_basis_shots request can be admitted. The response opens
            the public target commitment and reports the exact admission cutoff,
            the qsim-owned budget, and the device clock at the seal. Repeating
            the call returns the same opening.
            """
            return sealed_holdout_actions.lock_measurements_and_reveal_challenge(
                ctx, device_id=device_id
            )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
