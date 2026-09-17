"""MCP tool registration for the digital (gate-model) qtype.

Owns the qtype-local MCP surface: the instructions string the server
advertises plus ``register_mcp_tools`` which attaches capability-gated digital
tools to a ``FastMCP`` instance.
The top-level ``qiqcbench.qsim.mcp.server`` invokes this registrar through the
qtype registry after registering the always-on tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from qiqcbench.qsim.actions import digital as digital_actions
from qiqcbench.qsim.actions import digital_vqe as digital_vqe_actions
from qiqcbench.qsim.actions import local_randomized_measurement as randomized_actions
from qiqcbench.qsim.actions.base import ActionContext

if TYPE_CHECKING:  # pragma: no cover - import-only typing aid
    from mcp.server.fastmcp import FastMCP


INSTRUCTIONS = (
    "Quantum simulator MCP server. Operate the digital (gate-model) device via "
    "the tools enabled for the active task capabilities. Generic circuit tools "
    "(run_circuit and run_circuit_sweep) are available when the "
    "digital_circuit_execution capability is active; circuits are lists of gate "
    "ops (kind='gate') ending in a measure op (kind='circuit_measure'). "
    "Observable-batch tools (run_observable_batch) are available when the "
    "digital_vqe capability is active for VQE-shaped per-Pauli observable "
    "evaluation. The run_local_randomized_measurement tool is available when "
    "the local_randomized_measurement capability is active; it returns fresh "
    "full-register raw-shot data at an agent-chosen circuit depth. Results are "
    "per-shot bitstrings. Poll get_job_result. Use "
    "submit_final_answer when ready."
)


def register_mcp_tools(mcp: FastMCP, state: Any) -> None:
    """Attach the circuit-level MCP tools to ``mcp`` for the given qsim state.

    ``state`` is the ``QsimState`` instance defined by
    ``qiqcbench.qsim.state``; typed as ``Any`` so the qtype module does not
    import runtime state just to register tools.
    """

    ctx = ActionContext(state=state, surface="mcp")

    if "digital_circuit_execution" in state.active_capabilities:

        @mcp.tool()
        def run_circuit(device_id: str, circuit: list[dict], shots: int = 1024) -> dict:
            """Submit a quantum circuit. Returns a job_id immediately.

            `circuit` is a list of ops:
              - {"kind": "gate", "name": "<gate>", "qubits": [...], "params": [...]}
                Supported gates: i, x, y, z, h, s, sdg, t, tdg, rx, ry, rz, cx, cz, swap.
                rx/ry/rz take one float param (radians).
              - {"kind": "circuit_measure", "qubits": [...], "classical": [...]}
                A circuit must end with a measure op.
            Poll get_job_result(job_id). Result data has shape:
              data.bitstrings = [[shot_str, ...]]   # outer length 1
              data.measured_qubits = [...]          # bit i in shot_str = value of measured_qubits[i]
            """
            return digital_actions.submit_circuit(
                ctx, device_id=device_id, circuit=circuit, shots=shots
            )

        @mcp.tool()
        def run_circuit_sweep(
            device_id: str,
            template_circuit: list[dict],
            parameters: list[str],
            sweep: dict[str, list[float]],
            shots: int = 512,
            mode: str = "product",
        ) -> dict:
            """Submit a parameterized circuit swept over named parameters.

            `template_circuit` is a circuit whose gate `params` may include
            string names (e.g. "phi") that are bound from `sweep`. Declared
            names must equal `parameters`. mode="product" runs the Cartesian
            product; mode="zip" requires equal-length arrays.

            Result data:
              data.bitstrings = [[shot_str, ...], ...]   # outer = sweep points
              data.measured_qubits = [...]
              metadata.sweep_coords = {param: [...]}
            """
            return digital_actions.submit_circuit_sweep(
                ctx,
                device_id=device_id,
                template_circuit=template_circuit,
                parameters=parameters,
                sweep=sweep,
                shots=shots,
                mode=mode,
            )

    if "digital_vqe" in state.active_capabilities:

        @mcp.tool()
        def run_observable_batch(
            profile: str,
            points: list[dict],
            shots_per_setting: int,
        ) -> dict:
            """Submit a VQE-shaped observable-batch job. Returns a job_id immediately.

            ``profile`` is ``"ideal"`` (noiseless) or ``"public_noise"`` (the
            task's public Qiskit-style noise model). ``points`` is a list of
            ``{"point_id": str, "values": [float, ...]}`` ansatz-parameter
            vectors using the task's declared ``parameter_convention``. Point
            IDs use the published safe transport format and must be unique
            within one call. The public ansatz material states the per-call and
            run-long evaluation limits and the exact noisy shot rule; admission
            reserves those limits before the job is queued.

            Poll ``get_job_result(job_id)`` until ``status`` is ``"complete"``
            or ``"failed"``. A successful result is a
            ``JobObservableBitstringData`` with raw bitstrings grouped by
            ``(point_id, pauli)``. Large results are stored under the
            agent-readable path named in ``metadata.public_warnings`` instead
            of being expanded inline. Poll every job whose result you intend to
            use or cite to a terminal status before final submission. The task
            instructions define how uncited exploration is treated.
            """
            return digital_vqe_actions.submit_observable_batch(
                ctx,
                device_id=state.public.device_id,
                profile=profile,
                points=points,
                shots_per_setting=shots_per_setting,
            )

    if randomized_actions.CAPABILITY in state.active_capabilities:

        @mcp.tool()
        def run_local_randomized_measurement(depth: int) -> dict:
            """Run one fresh local-randomized-measurement ensemble at ``depth``.

            The active task's public measurement protocol fixes the circuit
            family, number of random circuits, local measurement bases, shots,
            and readout-calibration records. qsim privately draws new circuit
            and basis randomness for every accepted job. Poll
            ``get_job_result(job_id)``. Large raw results are exposed through
            ``metadata.result_artifact`` and remain readable below
            ``/qsim_logs/public_job_results``.
            """

            return randomized_actions.submit_local_randomized_measurement(
                ctx,
                depth=depth,
            )


__all__ = ["INSTRUCTIONS", "register_mcp_tools"]
