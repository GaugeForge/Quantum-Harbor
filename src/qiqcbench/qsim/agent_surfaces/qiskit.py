from __future__ import annotations

from json import JSONDecodeError
from typing import Any

from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.actions.common import get_job_result
from qiqcbench.qsim.actions.digital import submit_circuit, submit_circuit_sweep
from qiqcbench.qsim.actions.digital_vqe import submit_observable_batch
from qiqcbench.qsim.core.wire import (
    CircuitRequest,
    CircuitSweepRequest,
    ObservableBatchRequest,
)


def _validate_sweep_shape(parameters: list[str], sweep: dict[str, list[float]]) -> None:
    if not sweep:
        raise ValueError("sweep must be non-empty for qiskit sweep requests")
    parameter_keys = set(parameters)
    sweep_keys = set(sweep)
    if parameter_keys != sweep_keys:
        raise ValueError("sweep keys must match declared qiskit parameters")
    empty_keys = [key for key, values in sweep.items() if not values]
    if empty_keys:
        raise ValueError("sweep values must be non-empty for qiskit sweep requests")


def routes(state: Any) -> list[Route]:
    async def post_job(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except JSONDecodeError:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        if not isinstance(payload, dict):
            return JSONResponse({"error": "JSON body must be an object"}, status_code=422)

        ctx = ActionContext(state=state, surface="qiskit")
        try:
            is_sweep_request = any(key in payload for key in ("parameters", "sweep", "mode"))
            if is_sweep_request:
                missing = {
                    key
                    for key in ("device_id", "circuit", "parameters", "sweep", "shots")
                    if key not in payload
                }
                if missing:
                    return JSONResponse(
                        {"error": "missing required fields", "missing": sorted(missing)},
                        status_code=422,
                    )
                if not payload.get("parameters"):
                    raise ValueError("parameters must be non-empty for qiskit sweep requests")
                sweep_request = CircuitSweepRequest.model_validate(
                    {
                        "template_circuit": payload["circuit"],
                        "parameters": payload["parameters"],
                        "sweep": payload["sweep"],
                        "shots": payload["shots"],
                        "mode": payload.get("mode", "product"),
                    }
                )
                _validate_sweep_shape(sweep_request.parameters, sweep_request.sweep)
                result = submit_circuit_sweep(
                    ctx,
                    device_id=payload["device_id"],
                    template_circuit=payload["circuit"],
                    parameters=payload["parameters"],
                    sweep=payload["sweep"],
                    shots=payload["shots"],
                    mode=payload.get("mode", "product"),
                )
            else:
                missing = {key for key in ("device_id", "circuit", "shots") if key not in payload}
                if missing:
                    return JSONResponse(
                        {"error": "missing required fields", "missing": sorted(missing)},
                        status_code=422,
                    )
                CircuitRequest.model_validate(
                    {"circuit": payload["circuit"], "shots": payload["shots"]}
                )
                result = submit_circuit(
                    ctx,
                    device_id=payload["device_id"],
                    circuit=payload["circuit"],
                    shots=payload["shots"],
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
        ctx = ActionContext(state=state, surface="qiskit")
        try:
            result = get_job_result(ctx, job_id=request.path_params["job_id"])
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        return JSONResponse(result)

    async def post_vqe_job(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except JSONDecodeError:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        if not isinstance(payload, dict):
            return JSONResponse({"error": "JSON body must be an object"}, status_code=422)

        missing = {
            key
            for key in (
                "device_id",
                "profile",
                "parameter_convention",
                "points",
                "shots_per_setting",
            )
            if key not in payload
        }
        if missing:
            return JSONResponse(
                {"error": "missing required fields", "missing": sorted(missing)},
                status_code=422,
            )

        ctx = ActionContext(state=state, surface="qiskit")
        try:
            request_model = ObservableBatchRequest.model_validate(
                {
                    "shots_per_setting": payload["shots_per_setting"],
                    "profile": payload["profile"],
                    "parameter_convention": payload["parameter_convention"],
                    "points": payload["points"],
                }
            )
            result = submit_observable_batch(
                ctx,
                device_id=payload["device_id"],
                profile=request_model.profile,
                parameter_convention=request_model.parameter_convention,
                points=[point.model_dump() for point in request_model.points],
                shots_per_setting=request_model.shots_per_setting,
            )
        except ValidationError as exc:
            return JSONResponse(
                {"error": "validation failed", "details": exc.errors(include_context=False)},
                status_code=422,
            )
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        return JSONResponse(result)

    routes_list: list[Route] = []
    if "digital_circuit_execution" in getattr(state, "active_capabilities", ()):
        routes_list.extend(
            [
                Route("/jobs", post_job, methods=["POST"]),
                Route("/jobs/{job_id}", get_job, methods=["GET"]),
            ]
        )
    if "digital_vqe" in getattr(state, "active_capabilities", ()):
        routes_list.extend(
            [
                Route("/vqe/jobs", post_vqe_job, methods=["POST"]),
                Route("/vqe/jobs/{job_id}", get_job, methods=["GET"]),
            ]
        )
    return routes_list
