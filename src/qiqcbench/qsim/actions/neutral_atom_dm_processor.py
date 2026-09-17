"""Action seam for the ``neutral_atom_dm_processor`` qtype.

Routes ``run_atom_program`` / ``run_atom_program_sweep`` into the active
backend (async job model) and serves the synchronous, free
``validate_schedule`` legality/timing helper. Capability-gated and fail-closed
at this layer (defense in depth alongside MCP registration). Every completed
job persists a content-addressed lossless artifact (request + raw result) the
verifier binds to; the bounded JSONL event carries the reference. Sweep raw
bits additionally go to a public raw file below ``/qsim_logs`` and the
agent-facing job result carries only the pointer (context safety).
"""

from __future__ import annotations

import threading
from typing import Any

from qiqcbench.qsim.actions.base import ActionContext
from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.evidence import persist_public_raw_record
from qiqcbench.qsim.qtypes.neutral_atom_dm_processor.engine import (
    memory_budget_reason,
    program_peak_matrix_copies,
    program_state_bytes,
)
from qiqcbench.qsim.qtypes.neutral_atom_dm_processor.evidence import (
    MAX_NEUTRAL_ATOM_DM_RESULT_ARTIFACT_BYTES,
    NEUTRAL_ATOM_DM_EVIDENCE_CONTRACT,
    NEUTRAL_ATOM_DM_EVIDENCE_SCHEMA_VERSION,
    NEUTRAL_ATOM_DM_INFRASTRUCTURE_FAILURE,
    NeutralAtomDmResultArtifact,
    canonical_json_bytes,
    hidden_device_model_commitment,
    persist_result_artifact,
    program_measurement_labels,
    public_device_model_commitment,
    request_digest,
)
from qiqcbench.qsim.qtypes.neutral_atom_dm_processor.memory import (
    concurrent_state_budget_bytes,
)
from qiqcbench.qsim.qtypes.neutral_atom_dm_processor.wire import (
    AtomOp,
    AtomProgramRequest,
    AtomProgramSweepRequest,
    JobAtomDmShotData,
    JobAtomDmSweepData,
)

_CAP = "rydberg_dm_control"


def _state_budget_bytes(ctx: ActionContext) -> int:
    """The capacity the RUNNING engine enforces, so the free check cannot promise
    what submission will refuse. Falls back to the published contract when the
    backend exposes no engine (replay/live modes)."""

    engine = getattr(getattr(ctx.state, "backend", None), "engine", None)
    budget = getattr(engine, "memory", None)
    if budget is not None:
        return int(budget.capacity_bytes)
    declared = ctx.state.public.budgets.state_memory.budget_mib * 1024 * 1024
    return min(declared, concurrent_state_budget_bytes())


def _require(ctx: ActionContext, device_id: str) -> None:
    if _CAP not in ctx.state.active_capabilities:
        raise ValueError(f"{_CAP} capability is not active for this run")
    if device_id != ctx.state.hidden.device_id:
        raise ValueError(f"Unknown device {device_id!r}")


def _evidence_identity(ctx: ActionContext) -> dict[str, Any]:
    execution_context_id = getattr(ctx.state, "execution_context_id", None)
    commitment_secret = getattr(ctx.state, "hidden_commitment_secret", None)
    task_id = getattr(ctx.state, "task_id", None) or ctx.state.public.task_id
    identity = {
        "evidence_contract": NEUTRAL_ATOM_DM_EVIDENCE_CONTRACT,
        "evidence_schema_version": NEUTRAL_ATOM_DM_EVIDENCE_SCHEMA_VERSION,
        "public_device_model_commitment": public_device_model_commitment(ctx.state.public),
    }
    if execution_context_id is not None:
        if commitment_secret is None:
            raise RuntimeError("execution-bound atom evidence requires a hidden commitment secret")
        if not task_id:
            raise RuntimeError("execution-bound atom evidence requires an active task ID")
        identity["hidden_device_model_commitment"] = hidden_device_model_commitment(
            ctx.state.hidden,
            task_id=task_id,
            commitment_secret=commitment_secret,
        )
    return identity


def _mid_circuit_bit_counts(ops: list[AtomOp]) -> tuple[int, int]:
    """``(free, verified)`` recorded mid-circuit bits, mirroring the engine.

    FREE bits (measures without ``expect``) are branch doublings; VERIFIED bits
    (expected-outcome measures) park the contradicting branch instead of
    splitting the live tree, so they are not capped. A feedforward-conditioned
    measure always records (and cannot carry ``expect``).
    """
    from qiqcbench.qsim.qtypes.neutral_atom_dm_processor.engine import terminal_suffix_start

    free = verified = 0
    for op in ops[: terminal_suffix_start(ops)]:
        measured = op.then if op.type == "feedforward" and op.then is not None else op
        if measured.type != "measure":
            continue
        count = len(measured.atoms or [])
        if measured.expect is None:
            free += count
        else:
            verified += count
    return free, verified


def _mid_circuit_bits(ops: list[AtomOp]) -> int:
    """Free (branch-doubling) mid-circuit bits; see :func:`_mid_circuit_bit_counts`."""
    return _mid_circuit_bit_counts(ops)[0]


def _branch_cap_message(bits: int, max_branches: int) -> str:
    return (
        f"mid-circuit branching would exceed max_branches {max_branches}: "
        f"{bits} free recorded mid-circuit bits give up to 2**{bits} branches; "
        "condition on fewer bits, fold measurements into the terminal readout block, "
        "or declare verification readouts with 'expect' (heralded abort; not capped)"
    )


def _validate_result_capacity(
    ctx: ActionContext, request: AtomProgramRequest | AtomProgramSweepRequest
) -> None:
    budgets = ctx.state.public.budgets
    points = len(request.sweep_values) if isinstance(request, AtomProgramSweepRequest) else 1
    if len(request.ops) > budgets.max_ops:
        raise ValueError(f"program has {len(request.ops)} ops > max_ops {budgets.max_ops}")
    bits = _mid_circuit_bits(request.ops)
    if 2**bits > budgets.max_branches:
        raise ValueError(_branch_cap_message(bits, budgets.max_branches))
    if request.shots > budgets.max_shots_per_call:
        raise ValueError(
            f"shots {request.shots} exceeds max_shots_per_call {budgets.max_shots_per_call}"
        )
    if isinstance(request, AtomProgramSweepRequest) and points > budgets.max_sweep_points:
        raise ValueError(f"sweep has {points} points > max_sweep_points {budgets.max_sweep_points}")
    labels = program_measurement_labels(request.ops)
    recorded_bits = request.shots * points * len(labels)
    if recorded_bits > budgets.max_recorded_bits_per_call:
        raise ValueError(
            "raw result exceeds max_recorded_bits_per_call: "
            f"{recorded_bits} > {budgets.max_recorded_bits_per_call}; "
            "split the experiment into smaller jobs"
        )
    packed_bytes = (request.shots * len(labels) + 7) // 8
    encoded_bytes_per_field = 4 * ((packed_bytes + 2) // 3)
    aborted_mask_bytes = 4 * (((request.shots + 7) // 8 + 2) // 3)
    metadata_bytes = len(canonical_json_bytes(request.model_dump(mode="json"))) + len(
        canonical_json_bytes(labels)
    )
    conservative_artifact_bytes = (
        points * (2 * encoded_bytes_per_field + aborted_mask_bytes) + metadata_bytes + 4096
    )
    if conservative_artifact_bytes > MAX_NEUTRAL_ATOM_DM_RESULT_ARTIFACT_BYTES:
        raise ValueError(
            "raw result would exceed the neutral-atom-dm evidence artifact limit; "
            "split the experiment into smaller jobs"
        )


def _infrastructure_failure_result(
    ctx: ActionContext,
    *,
    job_id: str,
    device_id: str,
    tool: str,
    digest: str,
    identity: dict[str, Any],
    stage: str,
) -> JobResult:
    """Return and best-effort record one non-model qsim terminal failure."""

    try:
        ctx.state.log(
            {
                "surface": ctx.surface,
                "action": "atom_dm_infrastructure_failure",
                "tool": tool,
                "device_id": device_id,
                "job_id": job_id,
                "request_digest": digest,
                "failure_class": NEUTRAL_ATOM_DM_INFRASTRUCTURE_FAILURE,
                "failure_stage": stage,
                **identity,
            }
        )
    except Exception:
        pass
    return JobResult(
        job_id=job_id,
        device_id=device_id,
        status="failed",
        error=NEUTRAL_ATOM_DM_INFRASTRUCTURE_FAILURE,
    )


def _submit(
    ctx: ActionContext,
    *,
    device_id: str,
    request: AtomProgramRequest | AtomProgramSweepRequest,
    tool: str,
) -> dict[str, Any]:
    _validate_result_capacity(ctx, request)
    identity = _evidence_identity(ctx)
    reserve = getattr(ctx.state.backend, "reserve_program", None)
    reservation_token = reserve(request) if callable(reserve) else None
    salt = ctx.state.next_salt()
    dig = request_digest(request)
    is_sweep = isinstance(request, AtomProgramSweepRequest)
    submission_logged = threading.Event()

    def runner(job_id: str) -> JobResult:
        submission_logged.wait()
        try:
            backend_run = (
                ctx.state.backend.run_atom_program_sweep
                if is_sweep
                else ctx.state.backend.run_atom_program
            )
            if reservation_token is None:
                result = backend_run(request, job_id, salt)
            else:
                result = backend_run(request, job_id, salt, reservation_token=reservation_token)
        except Exception:
            return _infrastructure_failure_result(
                ctx,
                job_id=job_id,
                device_id=device_id,
                tool=tool,
                digest=dig,
                identity=identity,
                stage="backend_execution",
            )
        data = result.data
        if isinstance(data, (JobAtomDmShotData, JobAtomDmSweepData)):
            try:
                event: dict[str, Any] = {
                    "surface": ctx.surface,
                    "action": "atom_dm_program_result",
                    "tool": tool,
                    "device_id": device_id,
                    "job_id": job_id,
                    "status": result.status,
                    "n_ops": len(request.ops),
                    "request_digest": dig,
                    "layout": request.layout,
                    "shots": request.shots,
                    "idle_scale": request.idle_scale,
                    "noise_scale": request.noise_scale,
                    "n_recorded": data.n_recorded,
                    "measure_labels": data.measure_labels,
                    **identity,
                }
                if is_sweep:
                    event["sweep_parameter"] = request.sweep_parameter
                    event["sweep_values"] = list(request.sweep_values)
                log_dir = getattr(ctx.state, "log_dir", None)
                if log_dir is not None:
                    artifact = NeutralAtomDmResultArtifact(
                        job_id=job_id,
                        device_id=device_id,
                        tool=tool,
                        request_digest=dig,
                        public_device_model_commitment=identity["public_device_model_commitment"],
                        hidden_device_model_commitment=identity.get(
                            "hidden_device_model_commitment"
                        ),
                        request=request,
                        result_status=result.status,
                        result=data,
                    )
                    event["result_artifact"] = persist_result_artifact(
                        log_dir, artifact
                    ).model_dump(mode="json")
                if is_sweep and isinstance(data, JobAtomDmSweepData):
                    raw_data_file = persist_public_raw_record(
                        ctx.state.log_dir, job_id=job_id, record=data.model_dump()
                    )
                    if raw_data_file is not None:
                        result.data = data.model_copy(
                            update={
                                "point_bits_b64": None,
                                "point_executed_b64": None,
                                "point_aborted_b64": None,
                                "raw_data_file": raw_data_file,
                            }
                        )
                        event["raw_data_file"] = raw_data_file
                ctx.state.log(event)
            except Exception:
                return _infrastructure_failure_result(
                    ctx,
                    job_id=job_id,
                    device_id=device_id,
                    tool=tool,
                    digest=dig,
                    identity=identity,
                    stage="evidence_recording",
                )
        return result

    try:
        job_id = ctx.state.jobs.submit(runner)
    except Exception:
        discard = getattr(ctx.state.backend, "discard_program_reservation", None)
        if reservation_token is not None and callable(discard):
            discard(reservation_token)
        raise
    try:
        submit_event: dict[str, Any] = {
            "action": f"submit_{tool}",
            "tool": tool,
            "job_id": job_id,
            "surface": ctx.surface,
            "device_id": device_id,
            "n_ops": len(request.ops),
            "request_digest": dig,
            "layout": request.layout,
            "shots": request.shots,
            "idle_scale": request.idle_scale,
            "noise_scale": request.noise_scale,
            **identity,
        }
        if is_sweep:
            submit_event["sweep_parameter"] = request.sweep_parameter
            submit_event["sweep_values"] = list(request.sweep_values)
        ctx.state.log(submit_event)
    finally:
        submission_logged.set()
    return {"job_id": job_id, "status": "queued"}


def submit_atom_program(
    ctx: ActionContext,
    *,
    device_id: str,
    ops: list[dict],
    shots: int,
    layout: list[int],
    idle_scale: float = 1.0,
    noise_scale: float = 1.0,
) -> dict[str, Any]:
    _require(ctx, device_id)
    request = AtomProgramRequest.model_validate(
        {
            "ops": ops,
            "shots": shots,
            "layout": layout,
            "idle_scale": idle_scale,
            "noise_scale": noise_scale,
        }
    )
    return _submit(ctx, device_id=device_id, request=request, tool="run_atom_program")


def submit_atom_program_sweep(
    ctx: ActionContext,
    *,
    device_id: str,
    ops: list[dict],
    shots: int,
    layout: list[int],
    sweep_parameter: str,
    sweep_values: list[float],
    idle_scale: float = 1.0,
    noise_scale: float = 1.0,
) -> dict[str, Any]:
    _require(ctx, device_id)
    request = AtomProgramSweepRequest.model_validate(
        {
            "ops": ops,
            "shots": shots,
            "layout": layout,
            "idle_scale": idle_scale,
            "noise_scale": noise_scale,
            "sweep_parameter": sweep_parameter,
            "sweep_values": sweep_values,
        }
    )
    return _submit(ctx, device_id=device_id, request=request, tool="run_atom_program_sweep")


def validate_schedule(
    ctx: ActionContext, *, device_id: str, ops: list[dict], layout: list[int]
) -> dict[str, Any]:
    """SYNC + FREE: legality verdict + nominal timing/resource counts.

    Serial nominal estimate at idle_scale = 1 (no shots, no fidelity).
    Conditional (feedforward) ops are checked for range validity and counted in
    the duration/move totals, but site occupancy along conditional paths is
    only enforced at runtime, per executed branch.
    """
    _require(ctx, device_id)
    pub = ctx.state.public
    geom = pub.layout
    dur = pub.durations_us
    reasons: list[str] = []

    if len(ops) > pub.budgets.max_ops:
        reasons.append(f"{len(ops)} ops > max_ops {pub.budgets.max_ops}")
    n = len(layout)
    if n < 1:
        reasons.append("layout must place at least one atom")
    if n > pub.budgets.max_atoms:
        reasons.append(f"layout has {n} atoms > max_atoms {pub.budgets.max_atoms}")
    if len(set(layout)) != n:
        reasons.append("layout assigns two atoms to one site")
    for s in layout:
        if not (isinstance(s, int) and 0 <= s < geom.n_sites):
            reasons.append(f"layout site {s!r} out of range [0,{geom.n_sites})")

    validated: list[AtomOp | None] = []
    for i, raw in enumerate(ops):
        try:
            validated.append(AtomOp.model_validate(raw))
        except Exception as exc:  # noqa: BLE001 - surfaced as a legality reason
            reasons.append(f"op {i}: {exc}")
            validated.append(None)

    sites = list(layout)
    total_us = 0.0
    move_count = 0

    def _atoms_ok(targets: list[int], i: int, what: str) -> bool:
        ok = True
        for a in targets:
            if not (0 <= a < n):
                reasons.append(f"op {i}: {what} atom {a} out of range [0,{n})")
                ok = False
        return ok

    def _charge(op: AtomOp, i: int, conditional: bool) -> None:
        nonlocal total_us, move_count
        if op.type == "gate":
            targets = list(range(n)) if op.scope == "global" else list(op.atoms or [])
            _atoms_ok(targets, i, "gate")
            total_us += dur.rotation_global if op.scope == "global" else dur.rotation_local
        elif op.type == "cz":
            pair = list(op.pair or [])
            if _atoms_ok(pair, i, "cz") and len(pair) == 2 and pair[0] != pair[1]:
                if not conditional:
                    slot_a = geom.slot_of(sites[pair[0]])
                    slot_b = geom.slot_of(sites[pair[1]])
                    if slot_a is None or slot_a != slot_b:
                        reasons.append(
                            f"op {i}: cz pair {pair} not co-located in one gate-zone slot "
                            f"(sites {sites[pair[0]]}, {sites[pair[1]]})"
                        )
            elif len(pair) != 2 or pair[0] == pair[1]:
                reasons.append(f"op {i}: bad cz pair {pair}")
            total_us += dur.cz
        elif op.type == "init":
            targets = [op.atom] if op.atom is not None else list(op.atoms or [])
            if not targets:
                reasons.append(f"op {i}: init requires 'atom' or 'atoms'")
            else:
                _atoms_ok([t for t in targets if isinstance(t, int)], i, "init")
        elif op.type == "move":
            move_count += 1
            k = op.atom
            if k is None or not (0 <= k < n):
                reasons.append(f"op {i}: move atom {k!r} out of range [0,{n})")
            elif op.to_site is None or not (0 <= op.to_site < geom.n_sites):
                reasons.append(
                    f"op {i}: move to_site {op.to_site!r} out of range [0,{geom.n_sites})"
                )
            else:
                total_us += dur.move_per_um * geom.move_distance_um(sites[k], op.to_site)
                total_us += dur.move_settle
                if not conditional:
                    if op.to_site in sites and sites[k] != op.to_site:
                        reasons.append(f"op {i}: move target site {op.to_site} is occupied")
                    sites[k] = op.to_site
        elif op.type == "measure":
            _atoms_ok(list(op.atoms or []), i, "measure")
            total_us += dur.readout
            if op.reset:
                total_us += dur.reset
        elif op.type == "feedforward":
            if op.then is not None:
                _charge(op.then, i, True)

    for i, op in enumerate(validated):
        if op is None:
            continue
        _charge(op, i, False)

    # Branch-cap pre-check: every FREE recorded mid-circuit bit splits the
    # density-matrix evolution, so a trunk with b free bits costs up to 2**b
    # branches at runtime; expected-outcome (heralded-abort) readouts do not
    # split the live tree. Submission rejects the same bound.
    clean_ops = [op for op in validated if op is not None]
    mid_circuit_bits, verified_bits = _mid_circuit_bit_counts(clean_ops)
    if 2**mid_circuit_bits > pub.budgets.max_branches:
        reasons.append(_branch_cap_message(mid_circuit_bits, pub.budgets.max_branches))

    # Working-set pre-check. Submission refuses an over-budget schedule,
    # and a free legality check that cannot say so leaves the agent to discover
    # the limit by being refused. Both sides call the same reason builder, so
    # they cannot drift apart.

    state_matrices = state_bytes = 0.0
    if n >= 1 and len(clean_ops) == len(ops):
        capacity = _state_budget_bytes(ctx)
        state_matrices = program_peak_matrix_copies(clean_ops)
        state_bytes = program_state_bytes(clean_ops, n)
        if state_bytes > capacity:
            reasons.append(memory_budget_reason(clean_ops, n, capacity))

    return {
        "legal": not reasons,
        "reasons": reasons,
        "schedule_duration_us": round(total_us, 3),
        "n_ops": len(ops),
        "atoms_peak": n,
        "move_count": move_count,
        "mid_circuit_bits": mid_circuit_bits,
        "verified_bits": verified_bits,
        "state_matrices": state_matrices,
        "state_bytes": state_bytes,
        "state_budget_bytes": _state_budget_bytes(ctx),
        "note": (
            "duration is a serial nominal estimate at idle_scale=1; it drives "
            "decoherence/loss, not a score. mid_circuit_bits counts FREE recorded "
            "bits (each splits the exact simulation; 2**bits <= max_branches); "
            "verified_bits counts expected-outcome readouts (heralded abort, not "
            "capped). state_bytes is the peak density-matrix working set this schedule "
            "holds at once (state_matrices x 16 x 4**atoms_peak); it must not exceed "
            "state_budget_bytes, and submission refuses it with this same reason if it "
            "does. Conditional-path site occupancy is enforced per branch at runtime."
        ),
    }


__all__ = ["submit_atom_program", "submit_atom_program_sweep", "validate_schedule"]
