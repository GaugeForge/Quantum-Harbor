"""Surface-neutral submit seam for Ramsey-sensing jobs.

Used by MCP (and any future surface) to enqueue a request through the
capability runtime. Fails closed on inactive capability. Logs the FULL
serialized request payload because the answer schema cites jobs by job_id, and
the verifier must recover the full protocol from the log to replay it.
"""

from __future__ import annotations

from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.backends.provider_artifacts import canonical_request_hash
from qiqcbench.qsim.core.wire import (
    JobResult,
    RamseyExperimentRequest,
    RamseySweepRequest,
)

__all__ = ["submit_ramsey_experiment", "submit_ramsey_sweep"]


def _active_device_id(ctx: ActionContext) -> str:
    return str(ctx.state.hidden.device_id)


def _validate_action_state(ctx: ActionContext, *, action: str) -> str:
    state = ctx.state
    if "ramsey_sensing" not in state.active_capabilities:
        raise ValueError(
            "ramsey_sensing capability is not active for this run; cannot submit Ramsey job"
        )
    qtype = getattr(state.hidden, "qtype", None)
    if qtype != "trapped_ion_chain":
        raise ValueError(f"{action} requires qtype='trapped_ion_chain'; got qtype={qtype!r}.")
    hidden_device_id = _active_device_id(ctx)
    public = getattr(state, "public", None)
    public_device_id = getattr(public, "device_id", hidden_device_id)
    if public_device_id != hidden_device_id:
        raise ValueError(
            f"state public device_id {public_device_id!r} does not match hidden device_id "
            f"{hidden_device_id!r}"
        )
    if not state.task_id:
        raise ValueError(f"{action} requires ctx.state.task_id")
    return hidden_device_id


def _build_runner_for_experiment(ctx: ActionContext, request: RamseyExperimentRequest):
    from qiqcbench.qsim.qtypes.trapped_ion_chain.backend import (
        build_ion_chain_simulator_backend,
    )
    from qiqcbench.qsim.qtypes.trapped_ion_chain.capabilities.ramsey_sensing.materials import (
        load_ramsey_sensing_materials,
    )
    from qiqcbench.qsim.qtypes.trapped_ion_chain.capabilities.ramsey_sensing.replay import (
        replay_ramsey_experiment_request,
    )
    from qiqcbench.qsim.qtypes.trapped_ion_chain.capabilities.ramsey_sensing.runtime import (
        run_ramsey_experiment_request,
    )
    from qiqcbench.qsim.qtypes.trapped_ion_chain.replay import build_ion_chain_replay_backend
    from qiqcbench.qsim.task_materials import public_task_material_dir

    task_id = ctx.state.task_id
    materials = load_ramsey_sensing_materials(public_task_material_dir(task_id))
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        if ctx.state.backend_mode == "provider_replay":
            replay_root = getattr(ctx.state.backend, "replay_root", None)
            if replay_root is None:
                return JobResult(
                    job_id=job_id,
                    device_id=_active_device_id(ctx),
                    status="failed",
                    shots=request.shots,
                    error="provider_replay requires backend with replay_root",
                )
            replay_backend = build_ion_chain_replay_backend(
                task_id=task_id,
                public=ctx.state.public,
                replay_root=replay_root,
                log_dir=getattr(ctx.state, "log_dir", None),
            )
            return replay_ramsey_experiment_request(
                request,
                replay_root=replay_backend.replay_root,
                job_id=job_id,
                device_id=_active_device_id(ctx),
            )
        backend = build_ion_chain_simulator_backend(ctx.state.hidden, public=ctx.state.public)
        return run_ramsey_experiment_request(
            backend,
            request,
            materials=materials,
            job_id=job_id,
            device_id=_active_device_id(ctx),
            salt=salt,
        )

    return runner


def _build_runner_for_sweep(ctx: ActionContext, request: RamseySweepRequest):
    from qiqcbench.qsim.qtypes.trapped_ion_chain.backend import (
        build_ion_chain_simulator_backend,
    )
    from qiqcbench.qsim.qtypes.trapped_ion_chain.capabilities.ramsey_sensing.materials import (
        load_ramsey_sensing_materials,
    )
    from qiqcbench.qsim.qtypes.trapped_ion_chain.capabilities.ramsey_sensing.replay import (
        replay_ramsey_sweep_request,
    )
    from qiqcbench.qsim.qtypes.trapped_ion_chain.capabilities.ramsey_sensing.runtime import (
        run_ramsey_sweep_request,
    )
    from qiqcbench.qsim.qtypes.trapped_ion_chain.replay import build_ion_chain_replay_backend
    from qiqcbench.qsim.task_materials import public_task_material_dir

    task_id = ctx.state.task_id
    materials = load_ramsey_sensing_materials(public_task_material_dir(task_id))
    salt = ctx.state.next_salt()

    def runner(job_id: str) -> JobResult:
        if ctx.state.backend_mode == "provider_replay":
            replay_root = getattr(ctx.state.backend, "replay_root", None)
            if replay_root is None:
                return JobResult(
                    job_id=job_id,
                    device_id=_active_device_id(ctx),
                    status="failed",
                    shots=request.shots,
                    error="provider_replay requires backend with replay_root",
                )
            replay_backend = build_ion_chain_replay_backend(
                task_id=task_id,
                public=ctx.state.public,
                replay_root=replay_root,
                log_dir=getattr(ctx.state, "log_dir", None),
            )
            return replay_ramsey_sweep_request(
                request,
                replay_root=replay_backend.replay_root,
                job_id=job_id,
                device_id=_active_device_id(ctx),
            )
        backend = build_ion_chain_simulator_backend(ctx.state.hidden, public=ctx.state.public)
        return run_ramsey_sweep_request(
            backend,
            request,
            materials=materials,
            job_id=job_id,
            device_id=_active_device_id(ctx),
            salt=salt,
        )

    return runner


def _log_submission(
    ctx: ActionContext,
    *,
    action: str,
    tool: str,
    device_id: str,
    job_id: str,
    request: RamseyExperimentRequest | RamseySweepRequest,
) -> None:
    request_dump = request.model_dump(mode="json")
    ctx.state.log(
        {
            "surface": ctx.surface,
            "action": action,
            "tool": tool,
            "device_id": device_id,
            "task_id": ctx.state.task_id,
            "job_id": job_id,
            "request": request_dump,
            "request_digest": canonical_request_hash(request),
        }
    )


def submit_ramsey_experiment(
    ctx: ActionContext,
    request: RamseyExperimentRequest,
) -> dict[str, Any]:
    """Submit a Ramsey single-point job."""
    device_id = _validate_action_state(ctx, action="submit_ramsey_experiment")

    runner = _build_runner_for_experiment(ctx, request)
    job_id = ctx.state.jobs.submit(runner)
    _log_submission(
        ctx,
        action="submit_ramsey_experiment",
        tool="run_ramsey_experiment",
        device_id=device_id,
        job_id=job_id,
        request=request,
    )
    return {"job_id": job_id, "status": "queued"}


def submit_ramsey_sweep(
    ctx: ActionContext,
    request: RamseySweepRequest,
) -> dict[str, Any]:
    """Submit a Ramsey sweep job."""
    device_id = _validate_action_state(ctx, action="submit_ramsey_sweep")

    runner = _build_runner_for_sweep(ctx, request)
    job_id = ctx.state.jobs.submit(runner)
    _log_submission(
        ctx,
        action="submit_ramsey_sweep",
        tool="run_ramsey_sweep",
        device_id=device_id,
        job_id=job_id,
        request=request,
    )
    return {"job_id": job_id, "status": "queued"}
