"""In-process simulator backend for the bosonic_cavity_qec qtype.

Validates the agent-visible request against the public control envelope (fails
closed before execution) and delegates to the qtype-local runner.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import ValidationError

from qiqcbench.qsim.core.wire import JobResult
from qiqcbench.qsim.qtypes.bosonic_cavity_qec.device import (
    BosonicCavityResourceBudget,
    HiddenBosonicCavityConfig,
    PublicBosonicCavitySpec,
)
from qiqcbench.qsim.qtypes.bosonic_cavity_qec.runner import (
    run_bosonic_program,
    run_bosonic_program_sweep,
)
from qiqcbench.qsim.qtypes.bosonic_cavity_qec.wire import (
    BosonicOp,
    BosonicProgramRequest,
    BosonicProgramSweepRequest,
)

_SWEEPABLE_FIELDS_BY_KIND = {
    "displace": frozenset({"alpha_re", "alpha_im"}),
    "ancilla_rotate": frozenset({"theta", "phi"}),
    "dispersive_wait": frozenset({"duration_ns"}),
    "idle": frozenset({"duration_ns"}),
}

# A deliberately conservative charge for JobManager's retained Python
# ``list[list[int]]`` result representation. It includes each point-shot row,
# the outer-list reference, each element reference, and a non-cached int object.
_RETAINED_RESULT_ROW_BYTES = 96
_RETAINED_RESULT_VALUE_BYTES = 40


@dataclass(frozen=True)
class _CapacityReservation:
    request: BosonicProgramRequest | BosonicProgramSweepRequest
    retained_result_bytes: int
    execution_weighted_branch_ops: int


@dataclass(frozen=True)
class BosonicExecutionCapacity:
    peak_live_branches: int
    weighted_branch_ops: int
    variable_duration_keys: frozenset[float]


def bosonic_execution_capacity(
    programs: Sequence[Sequence[BosonicOp]],
) -> BosonicExecutionCapacity:
    """Compute the published execution allocation over materialized points."""

    peak = 1
    work = 0
    variable_durations: set[float] = set()
    for ops in programs:
        branches = 1
        for op in ops:
            if op.kind == "conditional":
                inner_weight = 2 if op.op.kind == "idle" else 1
                op_weight = 1 + inner_weight
                inner = op.op
            else:
                op_weight = 2 if op.kind in {"idle", "ancilla_measure"} else 1
                inner = op
            work += branches * op_weight
            if inner.kind in {"idle", "dispersive_wait"}:
                variable_durations.add(round(float(inner.duration_ns), 4))
            if op.kind == "ancilla_measure" and op.record:
                branches *= 2
                peak = max(peak, branches)
    return BosonicExecutionCapacity(
        peak_live_branches=peak,
        weighted_branch_ops=work,
        variable_duration_keys=frozenset(variable_durations),
    )


def bosonic_execution_capacity_error(
    capacity: BosonicExecutionCapacity,
    budget: BosonicCavityResourceBudget,
) -> str | None:
    if capacity.peak_live_branches > budget.max_execution_live_branches_per_point:
        return (
            "execution live-branch capacity exceeded: "
            f"{capacity.peak_live_branches} > "
            f"{budget.max_execution_live_branches_per_point}"
        )
    if capacity.weighted_branch_ops > budget.max_execution_weighted_branch_ops_per_call:
        return (
            "execution weighted branch-op capacity exceeded: "
            f"{capacity.weighted_branch_ops} > "
            f"{budget.max_execution_weighted_branch_ops_per_call}"
        )
    if (
        len(capacity.variable_duration_keys)
        > budget.max_execution_distinct_variable_durations_per_call
    ):
        return (
            "execution distinct variable-duration capacity exceeded: "
            f"{len(capacity.variable_duration_keys)} > "
            f"{budget.max_execution_distinct_variable_durations_per_call}"
        )
    return None


class BosonicCavitySimulatorBackend:
    """bosonic_cavity_qec backend backed by the in-process branch-tree simulator."""

    def __init__(self, hidden: HiddenBosonicCavityConfig, public: PublicBosonicCavitySpec) -> None:
        self.hidden = hidden
        self.public = public
        self._budget_lock = threading.Lock()
        self._experiment_calls = 0
        self._recorded_outcomes = 0
        self._outstanding_jobs = 0
        self._retained_result_bytes = 0
        self._execution_weighted_branch_ops = 0
        self._reservations: dict[int, _CapacityReservation] = {}

    def _failed(self, job_id: str, shots: int, error: str) -> JobResult:
        return JobResult(
            job_id=job_id,
            device_id=self.hidden.device_id,
            status="failed",
            shots=shots,
            error=error,
        )

    def _validate_ops(self, ops: list) -> str | None:
        p = self.public
        if len(ops) > p.max_ops_per_program:
            return f"program has {len(ops)} ops, exceeds max {p.max_ops_per_program}"
        max_dur_ns = p.max_duration_us * 1000.0
        n_ancilla = 0  # recorded measures only: unrecorded ones have no syndrome index
        for op in ops:
            err = self._validate_single(op, p, max_dur_ns)
            if err:
                return err
            if op.kind == "ancilla_measure" and op.record:
                n_ancilla += 1
            if op.kind == "conditional":
                if op.on_index >= n_ancilla:
                    return (
                        f"conditional on_index {op.on_index} references a recorded ancilla "
                        f"measurement that has not occurred yet ({n_ancilla} so far; "
                        "record=false measurements carry no syndrome index)"
                    )
        return None

    def _result_capacity_error(self, *, ops: list, shots: int, points: int) -> str | None:
        if points > self.public.max_sweep_points_per_call:
            return (
                f"sweep has {points} points > max_sweep_points_per_call "
                f"{self.public.max_sweep_points_per_call}"
            )
        # Branch capacity is spent by RECORDED ancilla measures only; unrecorded
        # measure-and-reset ops are merged by the engine and return no column.
        ancilla_measurements = sum(op.kind == "ancilla_measure" and op.record for op in ops)
        if ancilla_measurements > self.public.max_ancilla_measurements_per_call:
            return (
                "ancilla-measurement branch capacity exceeded: "
                f"{ancilla_measurements} > {self.public.max_ancilla_measurements_per_call}"
            )
        measurement_columns = sum(
            (op.kind == "ancilla_measure" and op.record) or op.kind == "photon_number_measure"
            for op in ops
        )
        if measurement_columns == 0:
            return (
                "bosonic program must contain at least one measurement that is recorded "
                "(an ancilla_measure with record=true or a photon_number_measure)"
            )
        recorded_outcomes = shots * points * measurement_columns
        if recorded_outcomes > self.public.max_recorded_outcomes_per_call:
            return (
                "raw result exceeds max_recorded_outcomes_per_call: "
                f"{recorded_outcomes} > {self.public.max_recorded_outcomes_per_call}"
            )
        return None

    def _validate_single(self, op, p, max_dur_ns: float) -> str | None:
        if op.kind not in p.allowed_ops:
            return f"op kind {op.kind!r} is not allowed"
        if op.kind == "displace":
            if not all(math.isfinite(value) for value in (op.alpha_re, op.alpha_im)):
                return "displace amplitudes must be finite"
            if max(abs(op.alpha_re), abs(op.alpha_im)) > p.max_alpha + 1e-9:
                return f"|alpha| component exceeds max_alpha {p.max_alpha}"
        elif op.kind == "snap":
            if not all(math.isfinite(value) for value in op.thetas):
                return "snap angles must be finite"
            if len(op.thetas) > min(p.max_snap_len, p.n_max):
                return f"snap length {len(op.thetas)} exceeds max {min(p.max_snap_len, p.n_max)}"
        elif op.kind == "ancilla_rotate":
            if not all(math.isfinite(value) for value in (op.theta, op.phi)):
                return "ancilla rotation angles must be finite"
        elif op.kind in ("idle", "dispersive_wait"):
            if not math.isfinite(op.duration_ns):
                return "duration_ns must be finite"
            if op.duration_ns > max_dur_ns + 1e-6:
                return f"duration {op.duration_ns} ns exceeds max {max_dur_ns} ns"
        elif op.kind == "conditional":
            if op.op.kind in {"ancilla_measure", "photon_number_measure"}:
                return (
                    "conditional measurement is unsupported by the current "
                    "fixed-column result schema"
                )
            inner = self._validate_single(op.op, p, max_dur_ns)
            if inner:
                return f"conditional inner op invalid: {inner}"
        return None

    def _materialize_sweep(
        self, request: BosonicProgramSweepRequest, *, retain_programs: bool = True
    ) -> tuple[list[list[BosonicOp]] | None, str | None]:
        """Rebuild and validate every sweep point before any point executes."""
        if request.sweep_op_index >= len(request.ops):
            return None, "sweep_op_index out of range"

        selected = request.ops[request.sweep_op_index]
        allowed = _SWEEPABLE_FIELDS_BY_KIND.get(selected.kind, frozenset())
        if request.sweep_field not in allowed:
            return (
                None,
                f"field {request.sweep_field!r} is not sweepable for op kind "
                f"{selected.kind!r}; allowed fields are {sorted(allowed)}",
            )

        materialized: list[list[BosonicOp]] | None = [] if retain_programs else None
        base_ops = list(request.ops)
        for point_index, value in enumerate(request.sweep_values):
            if not math.isfinite(value):
                return None, f"sweep value at index {point_index} must be finite"
            payload = selected.model_dump(mode="python")
            payload[request.sweep_field] = value
            try:
                swept = type(selected).model_validate(payload)
            except ValidationError as exc:
                detail = exc.errors(include_url=False)[0].get("msg", str(exc))
                return (
                    None,
                    f"invalid {selected.kind}.{request.sweep_field} at sweep index "
                    f"{point_index}: {detail}",
                )
            ops_point = [
                *base_ops[: request.sweep_op_index],
                swept,
                *base_ops[request.sweep_op_index + 1 :],
            ]
            err = self._validate_ops(ops_point)
            if err:
                return None, f"invalid sweep point {point_index}: {err}"
            if materialized is not None:
                materialized.append(ops_point)
        return materialized, None

    def _preflight(
        self, request: BosonicProgramRequest | BosonicProgramSweepRequest
    ) -> tuple[int, int]:
        if request.shots > self.public.max_shots:
            raise ValueError(f"shots exceeds max_shots {self.public.max_shots}")
        ops = list(request.ops)
        err = self._validate_ops(ops)
        if err:
            raise ValueError(err)
        programs = [ops]
        if isinstance(request, BosonicProgramSweepRequest):
            materialized, err = self._materialize_sweep(request)
            if err:
                raise ValueError(err)
            assert materialized is not None
            programs = materialized
        points = len(programs)
        err = self._result_capacity_error(ops=ops, shots=request.shots, points=points)
        if err:
            raise ValueError(err)
        capacity = bosonic_execution_capacity(programs)
        budget = self.public.budget
        if budget is not None:
            capacity_error = bosonic_execution_capacity_error(capacity, budget)
            if capacity_error is not None:
                raise ValueError(capacity_error)
        return points, capacity.weighted_branch_ops

    def reserve_bosonic_request(
        self, request: BosonicProgramRequest | BosonicProgramSweepRequest
    ) -> int:
        """Validate and atomically charge a request before JobManager enqueue."""
        points, execution_work = self._preflight(request)
        ops = list(request.ops)
        measurement_columns = sum(
            (op.kind == "ancilla_measure" and op.record) or op.kind == "photon_number_measure"
            for op in ops
        )
        recorded_outcomes = request.shots * points * measurement_columns
        point_shots = request.shots * points
        retained_result_bytes = (
            point_shots * _RETAINED_RESULT_ROW_BYTES
            + recorded_outcomes * _RETAINED_RESULT_VALUE_BYTES
        )
        token = id(request)
        with self._budget_lock:
            if token in self._reservations:
                raise RuntimeError("duplicate bosonic request reservation")
            if self._experiment_calls + 1 > self.public.max_experiment_calls:
                raise ValueError(
                    "experiment-call budget exhausted: "
                    f"{self._experiment_calls}+1 > cap {self.public.max_experiment_calls}"
                )
            if (
                self._recorded_outcomes + recorded_outcomes
                > self.public.max_recorded_outcomes_per_run
            ):
                raise ValueError(
                    "run-wide raw-result budget exhausted: "
                    f"{self._recorded_outcomes}+{recorded_outcomes} > cap "
                    f"{self.public.max_recorded_outcomes_per_run}"
                )
            if self._outstanding_jobs + 1 > self.public.max_outstanding_jobs:
                raise ValueError(
                    "outstanding-job budget exhausted: "
                    f"{self._outstanding_jobs}+1 > cap "
                    f"{self.public.max_outstanding_jobs}"
                )
            if (
                self._retained_result_bytes + retained_result_bytes
                > self.public.max_retained_result_bytes_per_run
            ):
                raise ValueError(
                    "retained-result memory budget exhausted: "
                    f"estimated {self._retained_result_bytes}+{retained_result_bytes} "
                    f"> cap {self.public.max_retained_result_bytes_per_run}"
                )
            budget = self.public.budget
            if (
                budget is not None
                and self._execution_weighted_branch_ops + execution_work
                > budget.max_execution_weighted_branch_ops_per_run
            ):
                raise ValueError(
                    "run-wide execution weighted branch-op budget exhausted: "
                    f"{self._execution_weighted_branch_ops}+{execution_work} > cap "
                    f"{budget.max_execution_weighted_branch_ops_per_run}"
                )
            self._experiment_calls += 1
            self._recorded_outcomes += recorded_outcomes
            self._outstanding_jobs += 1
            self._retained_result_bytes += retained_result_bytes
            self._execution_weighted_branch_ops += execution_work
            self._reservations[token] = _CapacityReservation(
                request=request,
                retained_result_bytes=retained_result_bytes,
                execution_weighted_branch_ops=execution_work,
            )
        return token

    def discard_bosonic_reservation(self, token: int) -> None:
        """Drop an enqueue-failed token; charged capacity remains consumed."""
        with self._budget_lock:
            reservation = self._reservations.pop(token, None)
            if reservation is not None:
                self._outstanding_jobs -= 1
                self._retained_result_bytes -= reservation.retained_result_bytes

    def _take_reservation(
        self,
        request: BosonicProgramRequest | BosonicProgramSweepRequest,
        token: int | None,
    ) -> _CapacityReservation:
        if token is None:
            token = self.reserve_bosonic_request(request)
        with self._budget_lock:
            reservation = self._reservations.get(token)
            if reservation is None or reservation.request is not request:
                raise ValueError("bosonic request reservation is missing or does not match")
            del self._reservations[token]
        return reservation

    def _release_outstanding(
        self, reservation: _CapacityReservation, *, result_retained: bool
    ) -> None:
        with self._budget_lock:
            if self._outstanding_jobs <= 0:
                raise RuntimeError("bosonic outstanding-job accounting underflow")
            self._outstanding_jobs -= 1
            if not result_retained:
                self._retained_result_bytes -= reservation.retained_result_bytes
            if self._retained_result_bytes < 0:
                raise RuntimeError("bosonic retained-result accounting underflow")

    def run_bosonic_program(
        self,
        request: BosonicProgramRequest,
        job_id: str,
        salt: int,
        *,
        reservation_token: int | None = None,
    ) -> JobResult:
        try:
            reservation = self._take_reservation(request, reservation_token)
        except (RuntimeError, ValueError) as exc:
            return self._failed(job_id, request.shots, str(exc))
        result_retained = False
        try:
            result = run_bosonic_program(request, self.hidden, job_id, salt)
            result_retained = result.status == "complete"
            return result
        finally:
            self._release_outstanding(reservation, result_retained=result_retained)

    def run_bosonic_program_sweep(
        self,
        request: BosonicProgramSweepRequest,
        job_id: str,
        salt: int,
        *,
        reservation_token: int | None = None,
    ) -> JobResult:
        try:
            reservation = self._take_reservation(request, reservation_token)
        except (RuntimeError, ValueError) as exc:
            return self._failed(job_id, request.shots, str(exc))
        result_retained = False
        try:
            materialized, error = self._materialize_sweep(request)
            if error:
                return self._failed(job_id, request.shots, error)
            assert materialized is not None
            result = run_bosonic_program_sweep(
                request,
                self.hidden,
                job_id,
                salt,
                materialized_programs=materialized,
            )
            result_retained = result.status == "complete"
            return result
        finally:
            self._release_outstanding(reservation, result_retained=result_retained)


def build_bosonic_cavity_simulator_backend(
    hidden: HiddenBosonicCavityConfig, public: PublicBosonicCavitySpec
) -> BosonicCavitySimulatorBackend:
    """Construct the in-process bosonic-cavity QEC simulator backend."""
    return BosonicCavitySimulatorBackend(hidden, public)


__all__ = [
    "BosonicCavitySimulatorBackend",
    "BosonicExecutionCapacity",
    "bosonic_execution_capacity",
    "bosonic_execution_capacity_error",
    "build_bosonic_cavity_simulator_backend",
]
