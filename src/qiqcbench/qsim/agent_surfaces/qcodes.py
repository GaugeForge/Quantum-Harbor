from __future__ import annotations

from json import JSONDecodeError
from typing import Any

from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.actions.common import get_job_result
from qiqcbench.qsim.actions.transmon import submit_pulse_sweep
from qiqcbench.qsim.core.wire import SweepRequest


def routes(state: Any) -> list[Route]:
    async def post_job(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except JSONDecodeError:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        if not isinstance(payload, dict):
            return JSONResponse({"error": "JSON body must be an object"}, status_code=422)

        missing = {
            key
            for key in ("device_id", "template_sequence", "sweep", "shots")
            if key not in payload
        }
        if missing:
            return JSONResponse(
                {"error": "missing required fields", "missing": sorted(missing)},
                status_code=422,
            )

        ctx = ActionContext(state=state, surface="qcodes")
        try:
            SweepRequest.model_validate(
                {
                    "template_sequence": payload["template_sequence"],
                    "sweep": payload["sweep"],
                    "shots": payload["shots"],
                    "mode": payload.get("mode", "zip"),
                }
            )
            delay_values = payload["sweep"].get("delay_ns")
            if not isinstance(delay_values, list) or not delay_values:
                raise ValueError("delay_ns sweep must be non-empty for qcodes T1 sweeps")
            result = submit_pulse_sweep(
                ctx,
                device_id=payload["device_id"],
                template_sequence=payload["template_sequence"],
                sweep=payload["sweep"],
                shots=payload["shots"],
                mode=payload.get("mode", "zip"),
            )
        except ValidationError as exc:
            return JSONResponse(
                {"error": "validation failed", "details": exc.errors(include_context=False)},
                status_code=422,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        return JSONResponse(result)

    async def get_job(request: Request) -> JSONResponse:
        ctx = ActionContext(state=state, surface="qcodes")
        try:
            result = get_job_result(ctx, job_id=request.path_params["job_id"])
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        return JSONResponse(result)

    return [
        Route("/jobs", post_job, methods=["POST"]),
        Route("/jobs/{job_id}", get_job, methods=["GET"]),
    ]
