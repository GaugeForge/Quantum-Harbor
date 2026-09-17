"""MCP server for qsim.

Always-on tools are list_devices, get_device_spec, get_lab_notebook,
get_job_result, and submit_final_answer. The active qtype registers its own
additional tools.

Transport: streamable-http on /mcp. Bind 0.0.0.0:<port> by default so the
agent container can reach it via the Compose DNS hostname.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from qiqcbench.qsim.actions import common as common_actions
from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.backends.factory import build_runtime
from qiqcbench.qsim.devices import (
    lab_notebook_from_hidden,
    load_hidden_config,
    load_public_spec,
)
from qiqcbench.qsim.execution_context import (
    load_execution_context_id,
    load_hidden_commitment_secret,
)
from qiqcbench.qsim.qtypes.registry import descriptor_for_qtype
from qiqcbench.qsim.state import QsimState
from qiqcbench.qsim.tasks import resolve_task_context


def _resolve_instructions(state: QsimState) -> str:
    """MCP instructions for this device: qtype default, per-device overridable.

    A qtype's hidden config may carry an optional ``mcp_instructions`` string
    (duck-typed; e.g. ``HiddenSurfaceCodeConfig``) so devices whose task
    contract differs from the qtype default serve an accurate initialize
    string. Empty/absent falls back to the descriptor's static instructions.
    """
    desc = descriptor_for_qtype(state.public.qtype)
    override = getattr(state.hidden, "mcp_instructions", "")
    return override or desc.instructions


def _build_app(state: QsimState) -> FastMCP:
    desc = descriptor_for_qtype(state.public.qtype)
    mcp = FastMCP("qiqcbench-qsim", instructions=_resolve_instructions(state))
    ctx = ActionContext(state=state, surface="mcp")

    # ---------------------- Always-on tools ----------------------

    @mcp.tool()
    def list_devices() -> list[str]:
        """Return device IDs hosted by this server."""
        return common_actions.list_devices(ctx)

    @mcp.tool()
    def get_device_spec(device_id: str) -> dict:
        """Return the public hardware-spec dict for a device."""
        return common_actions.get_device_spec(ctx, device_id=device_id)

    @mcp.tool()
    def get_lab_notebook(device_id: str) -> dict:
        """Return the (possibly stale) lab notebook for a device.

        Contents and meaning depend on the device's qtype (e.g. transmon
        notebooks include T1 estimates; digital notebooks include claimed
        gate fidelities). Calibrations may be stale — verify before relying.
        """
        return common_actions.get_lab_notebook(ctx, device_id=device_id)

    @mcp.tool()
    def get_job_result(job_id: str) -> dict:
        """Poll a job. Returns status='queued'|'running'|'complete'|'failed'.

        A completed result whose raw data is too large to return inline arrives
        with data=null and metadata.result_artifact set. The complete unmodified
        result, including that data, is then a file at
        /qsim_logs/public_job_results/results/<job_id>.json — read it with a script
        rather than printing it, and treat it as the raw evidence you would
        otherwise have received inline. No data is lost or aggregated.
        """
        return common_actions.get_job_result(ctx, job_id=job_id)

    # The file transport is advertised only where the agent
    # actually has the submission directory. Elsewhere the parameters are
    # absent from the schema entirely -- an offered-then-refused route wastes a
    # turn, and refusing it would have to describe the rig to explain itself.
    if common_actions.file_submission_enabled():

        @mcp.tool()
        def submit_final_answer(
            task_id: str,
            answer: dict | None = None,
            answer_file: str | None = None,
            answer_sha256: str | None = None,
        ) -> dict:
            """Submit your final answer for the current task.

            Answer shape is task-specific; see the task instructions. After
            submission the task is considered complete. Pass the answer exactly
            one of two ways:

            - Inline: ``answer``, the ordinary path for ordinarily sized
              answers.
            - As a file: write the answer dict as UTF-8 JSON to a single file
              directly in the agent-writable submission directory named in the
              task instructions (e.g. ``/submission``) -- a plain filename, no
              subdirectory -- then pass ``answer_file`` (that filename) plus
              ``answer_sha256`` (hex SHA-256 of the exact file bytes). Use this
              whenever the answer is too large to type into a tool call: build
              the file with a script and the payload never enters your context.
              The server validates the file content under the same limits and
              budgets as an inline answer and records the accepted bytes at
              submission time, so later edits to the file do not change what
              was submitted.
            """
            return common_actions.submit_final_answer(
                ctx,
                task_id=task_id,
                answer=answer,
                answer_file=answer_file,
                answer_sha256=answer_sha256,
            )

    else:

        @mcp.tool()
        def submit_final_answer(task_id: str, answer: dict) -> dict:
            """Submit your final answer for the current task.

            Answer shape is task-specific; see the task instructions. After
            submission the task is considered complete.
            """
            return common_actions.submit_final_answer(ctx, task_id=task_id, answer=answer)

    desc.register_mcp_tools(mcp, state)
    return mcp


def serve(
    host: str = "0.0.0.0",
    port: int = 8123,
    task_id: str | None = None,
    device_id: str | None = None,
    hidden_path: str | None = None,
    log_dir: str | None = None,
) -> None:
    """Start the FastMCP streamable-http server."""
    execution_context_id = load_execution_context_id()
    hidden_commitment_secret = load_hidden_commitment_secret()
    resolved_task_id = task_id if task_id is not None else os.environ.get("QIQCBENCH_TASK_ID")
    resolved_device_id = (
        device_id
        if device_id is not None
        else os.environ.get("QSIM_DEVICE_ID", "single_transmon_v0")
    )
    ctx = resolve_task_context(task_id=resolved_task_id, device_id=resolved_device_id)
    public = load_public_spec(ctx.device_id)
    if hidden_path is None:
        hidden_path = os.environ.get("QIQCBENCH_HIDDEN_PATH")
    if hidden_path is None:
        # Default: example file shipped with the repo.
        hidden_path = str(
            Path(__file__).resolve().parents[3].parent
            / "configs"
            / "devices"
            / f"{ctx.device_id}.hidden.example.yaml"
        )
    hidden = load_hidden_config(hidden_path)
    if hidden.device_id != public.device_id:
        raise ValueError(f"Hidden device_id {hidden.device_id!r} != public {public.device_id!r}")
    notebook = lab_notebook_from_hidden(hidden)
    log_dir_path = Path(log_dir) if log_dir else None

    runtime = build_runtime(hidden, public, log_dir=log_dir_path, task_spec=ctx.task_spec)
    state = QsimState(
        hidden,
        public,
        notebook,
        runtime.log_dir,
        runtime.backend,
        backend_mode=runtime.backend_mode,
        task_id=runtime.task_id,
        snapshot_hidden_truth=runtime.snapshot_hidden_truth,
        active_capabilities=runtime.active_capabilities,
        execution_context_id=execution_context_id,
        hidden_commitment_secret=hidden_commitment_secret,
    )
    mcp = _build_app(state)
    # Bind & disable DNS rebinding protection so non-localhost clients can reach us.
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.settings.transport_security.enable_dns_rebinding_protection = False
    print(f"[qsim] serving streamable-http on {host}:{port}/mcp", flush=True)
    mcp.run(transport="streamable-http")


def app_for_test(state: QsimState) -> Any:
    """Return the streamable_http_app for in-process testing."""
    mcp = _build_app(state)
    mcp.settings.transport_security.enable_dns_rebinding_protection = False
    return mcp.streamable_http_app()
