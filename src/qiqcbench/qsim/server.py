from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from qiqcbench.qsim.agent_surfaces.base import SurfaceRuntime
from qiqcbench.qsim.agent_surfaces.qcodes import routes as qcodes_routes
from qiqcbench.qsim.agent_surfaces.qiskit import routes as qiskit_routes
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
from qiqcbench.qsim.heartbeat import start_heartbeat_thread
from qiqcbench.qsim.mcp.server import _build_app as build_mcp_app
from qiqcbench.qsim.state import QsimState
from qiqcbench.qsim.tasks import resolve_task_context

_SURFACE_RUNTIMES: dict[str, tuple[SurfaceRuntime, Any]] = {
    "qiskit": (
        SurfaceRuntime(name="qiskit", route_prefix="/surfaces/qiskit"),
        qiskit_routes,
    ),
    "qcodes": (
        SurfaceRuntime(name="qcodes", route_prefix="/surfaces/qcodes"),
        qcodes_routes,
    ),
}


class _BoundedMcpRequestBody:
    """Prebuffer bounded POST bodies and reject entities beyond the public cap."""

    def __init__(self, app, *, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    @staticmethod
    def _is_bounded_path(path: str) -> bool:
        return path == "/mcp" or path.startswith("/mcp/") or path == "/surfaces/qiskit/vqe/jobs"

    async def _reject(self, scope, receive, send) -> None:
        response = JSONResponse(
            {
                "error": "request entity too large",
                "max_ingress_request_bytes": self.max_bytes,
            },
            status_code=413,
        )
        await response(scope, receive, send)

    async def __call__(self, scope: dict[str, Any], receive, send) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or not self._is_bounded_path(str(scope.get("path", "")))
        ):
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers", []):
            if name.lower() != b"content-length":
                continue
            try:
                declared_length = int(value.decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                break
            if declared_length > self.max_bytes:
                await self._reject(scope, receive, send)
                return
            break

        body_buffer = bytearray()
        saw_request = False
        interrupted_message: dict[str, Any] | None = None
        total = 0
        while True:
            message = await receive()
            if message.get("type") != "http.request":
                interrupted_message = message
                break
            saw_request = True
            body = message.get("body", b"")
            if not isinstance(body, bytes):
                await self._reject(scope, receive, send)
                return
            total += len(body)
            if total > self.max_bytes:
                await self._reject(scope, receive, send)
                return
            body_buffer.extend(body)
            if not message.get("more_body", False):
                break

        cursor = 0

        async def replay_receive():
            nonlocal cursor
            if saw_request and cursor == 0:
                cursor += 1
                return {
                    "type": "http.request",
                    "body": bytes(body_buffer),
                    "more_body": interrupted_message is not None,
                }
            if interrupted_message is not None and cursor == int(saw_request):
                cursor += 1
                return interrupted_message
            return await receive()

        await self.app(scope, replay_receive, send)


# The VQE qiskit surface predates the qtype-generic cap; both bound the same
# prebuffered request path, so the historical name stays importable.
_BoundedVQERequestBody = _BoundedMcpRequestBody


def _mcp_ingress_request_limit(state: QsimState) -> int | None:
    """Resolve an optional qtype-owned request cap without importing that qtype."""

    budget = getattr(getattr(state, "public", None), "budget", None)
    limit = getattr(budget, "max_ingress_request_bytes", None)
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise RuntimeError("public max_ingress_request_bytes must be a positive integer")
    return limit


def _vqe_ingress_request_limit(state: QsimState) -> int | None:
    if "digital_vqe" not in state.active_capabilities or not state.task_id:
        return None


class _ExactMcpPath:
    """Preserve exact /mcp compatibility for a root-path FastMCP mount."""

    def __init__(self, app: Starlette):
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive, send) -> None:
        mcp_scope = dict(scope)
        mcp_scope["path"] = "/"
        await self.app(mcp_scope, receive, send)


def _mcp_mount_app(state: QsimState) -> Starlette:
    mcp = build_mcp_app(state)
    mcp.settings.streamable_http_path = "/"
    mcp.settings.transport_security.enable_dns_rebinding_protection = False
    return mcp.streamable_http_app()


def _active_surface_mounts(state: QsimState) -> list[Mount]:
    mounts: list[Mount] = []
    for surface in state.active_surfaces:
        surface_runtime = _SURFACE_RUNTIMES.get(surface)
        if surface_runtime is None:
            continue
        runtime, route_factory = surface_runtime
        mounts.append(Mount(runtime.route_prefix, routes=route_factory(state)))
    return mounts


def build_sidecar_app(state: QsimState) -> Starlette:
    mcp_app = _mcp_mount_app(state)

    async def healthz(_request):
        return JSONResponse({"ok": True})

    @asynccontextmanager
    async def lifespan(_app):
        async with mcp_app.router.lifespan_context(mcp_app):
            yield

    routes = [
        Route("/healthz", healthz),
        # /health is the operator/ingest liveness probe path (alongside
        # heartbeat.jsonl); /healthz predates it
        # and stays for compatibility. Both answer only when the event loop is
        # responsive, which is exactly what they are probing.
        Route("/health", healthz),
        Route("/mcp", endpoint=_ExactMcpPath(mcp_app)),
        Mount("/mcp", app=mcp_app),
        *_active_surface_mounts(state),
    ]

    app = Starlette(
        routes=routes,
        lifespan=lifespan,
    )
    ingress_limit = _mcp_ingress_request_limit(state)
    if ingress_limit is None:
        ingress_limit = _vqe_ingress_request_limit(state)
    if ingress_limit is not None:
        app.add_middleware(_BoundedMcpRequestBody, max_bytes=ingress_limit)
    return app


def sidecar_app_for_test(state: QsimState) -> Starlette:
    return build_sidecar_app(state)


def _state_from_options(
    *,
    task_id: str | None,
    device_id: str | None,
    hidden_path: str | None,
    log_dir: str | None,
) -> QsimState:
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
        hidden_path = str(
            Path(__file__).resolve().parents[3]
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
        active_surfaces=runtime.active_surfaces,
        active_capabilities=runtime.active_capabilities,
        surface_manifest=runtime.surface_manifest,
        execution_context_id=execution_context_id,
        hidden_commitment_secret=hidden_commitment_secret,
    )
    return state


def serve_sidecar(
    host: str = "0.0.0.0",
    port: int = 8123,
    task_id: str | None = None,
    device_id: str | None = None,
    hidden_path: str | None = None,
    log_dir: str | None = None,
) -> None:
    state = _state_from_options(
        task_id=task_id,
        device_id=device_id,
        hidden_path=hidden_path,
        log_dir=log_dir,
    )
    app = build_sidecar_app(state)
    if state.log_dir is not None:
        # Liveness heartbeat: started before serving so the
        # file exists for the whole run, from a plain daemon thread so it keeps
        # beating even when the HTTP event loop is starved.
        start_heartbeat_thread(state.log_dir)
    print(f"[qsim] serving sidecar on {host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info")
