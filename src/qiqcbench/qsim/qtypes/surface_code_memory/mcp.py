"""MCP tool registration for the ``surface_code_memory`` qtype.

Registers the single experiment tool (gated on the ``surface_code_memory_calibration``
capability): ``run_memory_experiment``. It returns a ``job_id`` immediately (async job
model); the agent polls ``get_job_result``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import surface_code_memory as sc_actions
from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.qtypes.surface_code_memory.capabilities.drift_recalibration_scheduling import (
    actions as drift_actions,
)
from qiqcbench.qsim.qtypes.surface_code_memory.capabilities.heralded_leakage_decoding import (
    actions as leakage_actions,
)
from qiqcbench.qsim.qtypes.surface_code_memory.capabilities.syndrome_feedback_control import (
    actions as syndrome_feedback_actions,
)

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP


# Qtype default served at MCP initialize. Per-device override: a device's hidden
# config may set ``mcp_instructions`` (HiddenSurfaceCodeConfig) so each task on
# this qtype carries an accurate contract line; empty keeps this default, which
# describes the base surface_code_memory_calibration contract (surface_code_d5_v0).
# The qtype is DEM-level and its leak-pair mechanisms can span multiple rounds.
# Instructions describe only the instrument and deliverable; the task contract
# lives in instruction.md. The mechanism format allows 1 or 2 detectors.
INSTRUCTIONS = (
    "Rotated distance-d surface-code memory experiment (decoded Z basis). "
    "run_memory_experiment(rounds, shots) runs `rounds` syndrome-extraction rounds over "
    "`shots` shots under a HIDDEN noise model and returns RAW per-shot "
    "detection events (packed bit array, det_id = check*rounds + t) plus the final logical "
    "outcome per shot (for your own self-scoring; the device prepares logical |0> and "
    "measures Z-bar memory). The device returns raw detection events only -- "
    "never the noise model, weights, decoded result, or logical error rate. A single shot "
    "budget accumulates across the run. Submit a detector error model (list of mechanisms "
    "{detectors, p, flips_observable}: each mechanism flips 1 or 2 detectors) + a decoder "
    "config; it is scored by replay on FRESH hidden shots, pass iff the logical error rate "
    "per cycle is below the public floor."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    ctx = ActionContext(state=state, surface="mcp")

    if "surface_code_memory_calibration" in state.active_capabilities:

        @mcp.tool()
        def run_memory_experiment(device_id: str, rounds: int, shots: int) -> dict:
            """Run a surface-code memory experiment. Returns a job_id immediately.

            ``rounds`` (1-64) syndrome-extraction rounds; ``shots`` (1-200000) per call,
            drawn from the run-long shot budget. Result (via get_job_result) carries the raw
            per-shot detection events + final logical outcomes and the cumulative budget.
            """
            return sc_actions.submit_memory_experiment(
                ctx, device_id=device_id, rounds=rounds, shots=shots
            )

    if "syndrome_feedback_control" in state.active_capabilities:

        @mcp.tool()
        def run_syndrome_control_probe(
            device_id: str, constant_trim: list[int], trajectories: int
        ) -> dict:
            """Run a constant-trim characterization probe asynchronously.

            It returns only a job ID.  The completed record contains raw
            per-epoch detector and decoded-logical bitstrings, the applied
            trim history, and qsim-owned trajectory/cycle accounting. It
            returns no fitted probability or aggregate performance estimate.
            """
            return syndrome_feedback_actions.submit_syndrome_control_probe(
                ctx,
                device_id=device_id,
                constant_trim=constant_trim,
                trajectories=trajectories,
            )

        @mcp.tool()
        def validate_syndrome_controller_program(device_id: str) -> dict:
            """Validate and hash /submission/controller without evaluating quality."""
            return syndrome_feedback_actions.validate_syndrome_controller_program(
                ctx, device_id=device_id
            )

        @mcp.tool()
        def run_syndrome_controller_program(
            device_id: str, controller_manifest_sha256: str, trajectories: int
        ) -> dict:
            """Run the staged causal controller asynchronously; poll get_job_result."""
            return syndrome_feedback_actions.submit_syndrome_controller_program(
                ctx,
                device_id=device_id,
                controller_manifest_sha256=controller_manifest_sha256,
                trajectories=trajectories,
            )

    if "drift_recalibration_scheduling" in state.active_capabilities:

        @mcp.tool()
        def run_drift_syndrome_stream(device_id: str, windows: int) -> dict:
            """Stream syndrome windows from the operating tile asynchronously.

            Advances device time by ``windows`` windows. The completed record
            points to a lossless raw record containing per-cycle detector bits,
            the fixed-firmware-decoder logical-failure bit per window, and a
            down-mask. It also reports the run-long characterization budgets.
            """
            return drift_actions.submit_drift_syndrome_stream(
                ctx, device_id=device_id, windows=windows
            )

        @mcp.tool()
        def run_drift_control(device_id: str, action: str) -> dict:
            """Trigger a control action asynchronously.

            ``action`` is one of "recalibrate_operating_tile" (downtime on the
            memory), "recalibrate_idle_tile" (no downtime), or "relocate"
            (move the logical patch to the idle tile; requires it to be ready).
            Costs and completion windows are public device-spec constants.
            """
            return drift_actions.submit_drift_control(ctx, device_id=device_id, action=action)

        @mcp.tool()
        def validate_drift_policy(
            device_id: str, risk_model: dict, maintenance_policy: dict
        ) -> dict:
            """Check risk/policy grammar and return a composite digest, free of charge."""
            return drift_actions.validate_drift_policy(
                ctx,
                device_id=device_id,
                risk_model=risk_model,
                maintenance_policy=maintenance_policy,
            )

    if "heralded_leakage_decoding" in state.active_capabilities:

        @mcp.tool()
        def run_leakage_memory_experiment(device_id: str, rounds: int, shots: int) -> dict:
            """Run a calibration memory experiment with leakage heralds. Async: poll get_job_result.

            ``rounds`` (even, 2-24) syndrome-extraction rounds over ``shots`` shots, charged
            to the run-long shot budget. The completed record carries raw per-shot detection
            events, raw per-LRU-event herald bits, AND the true logical outcome per shot
            (calibration truth for your own decoder validation), plus the budget meter.
            """
            return leakage_actions.submit_leakage_memory_experiment(
                ctx, device_id=device_id, rounds=rounds, shots=shots
            )

        @mcp.tool()
        def fetch_challenge_batch(device_id: str, chunk_index: int) -> dict:
            """Fetch one sealed challenge chunk (free of charge). Async: poll get_job_result.

            The challenge set is fixed for the run; refetching a chunk returns byte-identical
            data. Chunks carry raw detection events + herald bits ONLY — the per-shot logical
            truth is withheld and scored by the verifier against your submitted predictions.
            """
            return leakage_actions.submit_challenge_batch(
                ctx, device_id=device_id, chunk_index=chunk_index
            )

        @mcp.tool()
        def submit_challenge_predictions(
            device_id: str, chunk_index: int, predictions_b64: str
        ) -> dict:
            """Record your per-shot logical-flip predictions for one challenge chunk.

            Synchronous and free of charge; returns digests + coverage bookkeeping only
            (never any correctness feedback). ``predictions_b64`` is base64(np.packbits) of
            chunk_shots 0/1 bits in shot order; zero padding to the byte boundary. The last
            submission per chunk wins. Missing chunks count as failures at scoring time.
            """
            return leakage_actions.submit_challenge_predictions(
                ctx,
                device_id=device_id,
                chunk_index=chunk_index,
                predictions_b64=predictions_b64,
            )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
