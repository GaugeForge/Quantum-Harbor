#!/usr/bin/env python3
"""True scoring verifier for ``time_budgeted_hamlearn_10q``.

Self-contained: the Harbor verifier environment does not have the ``qiqcbench``
package installed, so this script reimplements the scoring from public knowledge
(the term ordering) plus the hidden scorer YAML — it does **not** import qiqcbench.

The verifier:

  * resolves the scored answer under the qsim final-answer lifecycle contract
    (``qsim/server.py``): at least one accepted ``submit_final_answer`` event
    for this task, ``final_answer.json`` bound to the **last** accepted event,
    and qsim-recorded final-answer failure markers routed to the reserved
    infrastructure exit rather than a model verdict;
  * reconstructs the **evidence-owned** probe budget from the logged
    ``probe_batch_result`` events (accepted rows + cumulative
    ``budget_used_us``) — never the agent's self-report;
  * binds every probe result to its qsim submission and every retrieved result
    to a completed ``get_job_result`` poll plus a byte-verified archived raw
    result under ``public_job_results``;
  * reports descriptive compatibility diagnostics between retrieved raw
    evidence and the submitted Hamiltonian; this adaptive, same-data statistic
    is not an independent acceptance gate;
  * structurally validates the submitted answer (lengths, plausibility bands,
    enum literals, conventions), including its two-layer activity summary;
    only the structured verifiable layer is compared with qsim evidence, while
    the bounded trajectory narrative is retained for human audit and unscored;
  * loads the hidden scorer YAML and fails closed if the public-material digests
    on disk do not match the bytes the scorer was built from;
  * recomputes the worst-case (L-infinity) and aggregate relative-L2
    coefficient errors against the hidden ``omega_star`` and applies the
    budget/activity rules while reporting support and uncertainty diagnostics,
    producing ``score_report.json``.

Exit ``0`` iff the binary reward is ``1``; exit ``1`` for a model-owned
rejection and reserved exit ``3`` for a verifier/qsim integrity failure.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ruamel.yaml import YAML

try:
    from hamlearn_contract import validate_hidden_scorer_payload
except ModuleNotFoundError:  # repo-local unit/integration tests
    from qiqcbench.qsim.hidden_dynamics.time_budgeted_hamlearn_10q.standalone_contract import (
        validate_hidden_scorer_payload,
    )

TASK_ID = "time_budgeted_hamlearn_10q"
DEVICE_ID = "hamlearn_10q_chain_v1"
SCORE_REPORT_FILENAME = "score_report.json"
SCORE_REPORT_SCHEMA_VERSION = 8
PROBE_BATCH_EVIDENCE_SCHEMA_VERSION = 1
N_TERMS = 111
N_QUBITS = 10
HAMILTONIAN_CONVENTION = "U(t)=exp(-i t/2 sum omega_P P)"
TERM_ORDER_CONVENTION = "one_local_xyz_then_edge_local_xx_xy_xz_yx_yy_yz_zx_zy_zz"
_PROTOCOLS = {
    "fixed_design",
    "adaptive_design",
    "mixed_design",
    "other",
}
_FIT_MODELS = {
    "linear",
    "nonlinear",
    "probabilistic",
    "hybrid",
    "other",
}
_REQUIRED_ANSWER_FIELDS = {
    "hamiltonian_convention",
    "term_order_convention",
    "num_terms",
    "omega_vector_raw_rad_per_us",
    "omega_vector_raw_rad_per_us_1sigma",
    "activity_summary",
}
_OPTIONAL_ANSWER_FIELDS = {
    # The FinalAnswer envelope already carries and validates task_id, so the
    # inner copy is optional-must-match.
    "task_id",
    "omega_vector_refined_rad_per_us",
    "omega_vector_refined_rad_per_us_1sigma",
    "support_terms_detected",
    "linf_self_estimate_rad_per_us",
    "notebook_model_rejected",
    "notebook_rejection_reason",
    "heldout_probe_error_summary",
}
_ACTIVITY_SUMMARY_FIELDS = {"schema_version", "verifiable", "trajectory"}
_VERIFIABLE_ACTIVITY_FIELDS = {
    "accepted_probe_rows",
    "total_accepted_evolution_time_us",
    "min_accepted_evolve_time_us",
    "max_accepted_evolve_time_us",
}
_TRAJECTORY_REVIEW_FIELDS = {"protocol", "fit_model", "trajectory_details"}
_YAML = YAML(typ="safe")
_BUDGET_TOL_US = 1e-6
_REPORT_TOL_US = 1e-6
_MAX_MEAN_BINOMIAL_DEVIANCE = 3.0
_MAX_P90_BINOMIAL_DEVIANCE = 6.0
_VALID_STATES = frozenset({"x+", "x-", "y+", "y-", "z+", "z-"})
_PAULI_2X2 = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
_KET = {
    "z+": np.array([1, 0], dtype=complex),
    "z-": np.array([0, 1], dtype=complex),
    "x+": np.array([1, 1], dtype=complex) / np.sqrt(2),
    "x-": np.array([1, -1], dtype=complex) / np.sqrt(2),
    "y+": np.array([1, 1j], dtype=complex) / np.sqrt(2),
    "y-": np.array([1, -1j], dtype=complex) / np.sqrt(2),
}


class VerifierError(Exception):
    """A model-owned rejection such as a missing probe, poll, or valid answer."""


class InfrastructureError(Exception):
    """A qsim/verifier-owned integrity failure with no model reward."""


class EvidenceIntegrityError(InfrastructureError):
    """A qsim-owned log or archived result failed integrity checks."""


def _term_order() -> list[str]:
    """Rebuild the public 111-term ordering (public knowledge, no hidden data)."""
    terms: list[str] = []
    for i in range(10):
        for axis in ("X", "Y", "Z"):
            terms.append(f"{axis}{i}")
    pairs = [
        ("X", "X"),
        ("X", "Y"),
        ("X", "Z"),
        ("Y", "X"),
        ("Y", "Y"),
        ("Y", "Z"),
        ("Z", "X"),
        ("Z", "Y"),
        ("Z", "Z"),
    ]
    for i in range(9):
        for a, b in pairs:
            terms.append(f"{a}{i}{b}{i + 1}")
    return terms


TERM_ORDER = _term_order()
_TERM_INDEX = {t: i for i, t in enumerate(TERM_ORDER)}


def _term_pauli(term: str) -> str:
    characters = ["I"] * 10
    for index in range(0, len(term), 2):
        characters[int(term[index + 1])] = term[index]
    return "".join(characters)


_TERM_PAULIS = tuple(_term_pauli(term) for term in TERM_ORDER)


# --------------------------------------------------------------------------- IO


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            line = line.strip()
            if line:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError(f"line {line_number} is not an object")
                events.append(event)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvidenceIntegrityError(f"experiment log is unreadable: {exc}") from None
    return events


def _load_answer(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        # Only qsim's atomic submit_final_answer action can create this file,
        # after envelope validation; a torn or unreadable artifact is qsim-owned.
        raise EvidenceIntegrityError(f"malformed final_answer.json: {exc}") from None
    # The FinalAnswer envelope is extra="forbid" with exactly these keys
    # (wire.py); qsim validates it before persisting, so any deviation here is
    # corruption, not a model submission.
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "task_id", "answer"}
        or payload.get("schema_version") != 2
        or payload.get("task_id") != TASK_ID
        or not isinstance(payload.get("answer"), dict)
    ):
        raise EvidenceIntegrityError(
            "final_answer.json envelope is invalid (qsim validates it before persisting)"
        )
    return payload["answer"]


def _load_hidden_scorer(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise InfrastructureError(f"hidden scorer YAML missing at {path}")
    try:
        raw = _YAML.load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise InfrastructureError(f"hidden scorer YAML is unreadable: {exc}") from None
    try:
        return validate_hidden_scorer_payload(
            raw,
            term_order=TERM_ORDER,
            coefficient_bound=0.60,
        )
    except (TypeError, ValueError) as exc:
        raise InfrastructureError(f"hidden scorer contract is invalid: {exc}") from None


def _load_stale_vector(public_dir: Path) -> list[float]:
    """Project the public stale-notebook coefficients onto the 111-term order.

    Used only for the §9g stale-only diagnostic penalty. Reads the public
    material (no hidden data, no qiqcbench import).
    """
    path = public_dir / "stale_notebook.yaml"
    if not path.is_file():
        raise InfrastructureError(f"public stale notebook missing at {path}")
    try:
        raw = _YAML.load(path.read_text(encoding="utf-8"))
        coeffs = dict(raw.get("coefficients_rad_per_us", {}))
        return [float(coeffs.get(term, 0.0)) for term in TERM_ORDER]
    except Exception as exc:
        raise InfrastructureError(f"public stale notebook is invalid: {exc}") from None


def _verify_public_material_digests(scorer: dict[str, Any], public_dir: Path) -> None:
    for filename, expected in scorer["public_material_digests"].items():
        path = public_dir / filename
        if not path.is_file():
            raise InfrastructureError(f"public material {filename!r} missing under {public_dir}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise InfrastructureError(
                f"public material {filename!r} digest mismatch: expected {expected}, got {actual}"
            )


def _contract_real(raw: dict[str, Any], field: str) -> float:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InfrastructureError(f"public execution contract {field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise InfrastructureError(f"public execution contract {field} must be finite")
    return number


def _contract_positive_int(raw: dict[str, Any], field: str) -> int:
    value = raw.get(field)
    if type(value) is not int or value <= 0:
        raise InfrastructureError(f"public execution contract {field} must be a positive integer")
    return value


def _load_execution_contract(
    public_dir: Path, scorer: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Load the digest-bound public constraints needed to replay qsim evidence."""

    path = public_dir / "public_spec.yaml"
    try:
        raw = _YAML.load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise InfrastructureError(f"public execution contract is unreadable: {exc}") from None
    if not isinstance(raw, dict):
        raise InfrastructureError("public execution contract must be a mapping")
    required = {
        "task_id",
        "device_class",
        "n_qubits",
        "dynamics_primitive",
        "hamiltonian_convention",
        "term_order_convention",
        "allowed_initial_states",
        "allowed_observable_weight",
        "evolve_time_range_us",
        "evolve_time_resolution_us",
        "off_grid_time_policy",
        "return_type",
        "num_internal_repetitions",
        "total_evolution_time_budget_us",
        "max_probe_rows",
        "max_rows_per_batch",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise InfrastructureError(f"public execution contract is missing fields: {missing}")
    if (
        raw["task_id"] != TASK_ID
        or raw["device_class"] != "blackbox_analog_dynamics"
        or raw["dynamics_primitive"] != "run_hamiltonian_probe_batch"
        or raw["hamiltonian_convention"] != HAMILTONIAN_CONVENTION
        or raw["term_order_convention"] != TERM_ORDER_CONVENTION
        or raw["off_grid_time_policy"] != "reject"
        or raw["return_type"] != "raw_pauli_outcomes"
    ):
        raise InfrastructureError("public execution contract identifies incompatible semantics")
    n_qubits = _contract_positive_int(raw, "n_qubits")
    if n_qubits != N_QUBITS:
        raise InfrastructureError(f"public execution contract n_qubits must be {N_QUBITS}")

    states = raw["allowed_initial_states"]
    if (
        not isinstance(states, list)
        or not states
        or any(not isinstance(value, str) for value in states)
        or len(states) != len(set(states))
    ):
        raise InfrastructureError("public execution contract allowed_initial_states is invalid")
    if set(states) != _VALID_STATES:
        raise InfrastructureError("public execution contract state vocabulary is unsupported")

    weights = raw["allowed_observable_weight"]
    if (
        not isinstance(weights, list)
        or len(weights) != 2
        or any(type(value) is not int for value in weights)
        or not 1 <= weights[0] <= weights[1] <= n_qubits
    ):
        raise InfrastructureError("public execution contract observable-weight range is invalid")
    time_range = raw["evolve_time_range_us"]
    if not isinstance(time_range, list) or len(time_range) != 2:
        raise InfrastructureError("public execution contract evolve-time range is invalid")
    time_min = _contract_real({"value": time_range[0]}, "value")
    time_max = _contract_real({"value": time_range[1]}, "value")
    resolution = _contract_real(raw, "evolve_time_resolution_us")
    if not 0.0 < time_min <= time_max or resolution <= 0.0:
        raise InfrastructureError("public execution contract evolve-time grid is invalid")

    contract = {
        "n_qubits": n_qubits,
        "allowed_initial_states": frozenset(states),
        "observable_weight_min": weights[0],
        "observable_weight_max": weights[1],
        "time_min_us": time_min,
        "time_max_us": time_max,
        "time_resolution_us": resolution,
        "oracle_noise_shots": _contract_positive_int(raw, "num_internal_repetitions"),
        "budget_us": _contract_real(raw, "total_evolution_time_budget_us"),
        "max_probe_rows": _contract_positive_int(raw, "max_probe_rows"),
        "max_rows_per_batch": _contract_positive_int(raw, "max_rows_per_batch"),
    }
    if contract["max_rows_per_batch"] > contract["max_probe_rows"]:
        raise InfrastructureError("public execution contract batch cap exceeds the run row cap")
    if scorer is not None and (
        abs(contract["budget_us"] - scorer["budget_us"]) > _BUDGET_TOL_US
        or contract["max_probe_rows"] != scorer["max_probe_rows"]
        or contract["oracle_noise_shots"] != scorer["oracle_noise_shots"]
    ):
        raise InfrastructureError("hidden scorer and public execution contract disagree")
    return contract


def _accepted_time_is_valid(value: float, contract: dict[str, Any]) -> bool:
    if not contract["time_min_us"] - 1e-9 <= value <= contract["time_max_us"] + 1e-9:
        return False
    steps = value / contract["time_resolution_us"]
    return abs(steps - round(steps)) <= 1e-6


# ------------------------------------------------------------------- evidence


# qsim logs {"action": "submit_final_answer_failure", "tool":
# "submit_final_answer"} when final-answer persistence or log publication
# breaks internally (actions/common.py). The agent may then be unable to
# complete the submission lifecycle through no fault of its own, so any such
# marker is a qsim-owned infrastructure failure, never a model verdict.
_FINAL_ANSWER_FAILURE_ACTION = "submit_final_answer_failure"


def _require_no_final_answer_failures(events: list[dict[str, Any]]) -> None:
    for event in events:
        if event.get("action") == _FINAL_ANSWER_FAILURE_ACTION:
            raise InfrastructureError(
                "qsim recorded an internal final-answer failure "
                f"(stage {event.get('failure_stage')!r})"
            )


def _select_latest_submission(events: list[dict[str, Any]]) -> tuple[int, dict[str, Any]] | None:
    """Latest accepted ``submit_final_answer`` event for this task, or ``None``.

    Mirrors the in-tree lifecycle (``qsim/final_answer_evidence.py``)
    and the modexp verifier: resubmitting is not a failure — the tool accepts
    every call and atomically overwrites ``final_answer.json``, so the file
    always holds the last accepted submission and that is the answer scored
    . Failure markers carry a different ``action`` and are never
    selected; events bound to another task are excluded.
    """
    submits = [
        (index, event)
        for index, event in enumerate(events)
        if event.get("action", event.get("tool")) == "submit_final_answer"
        and event.get("task_id", TASK_ID) == TASK_ID
    ]
    return submits[-1] if submits else None


def _require_no_post_final_experiments(events: list[dict[str, Any]], final_index: int) -> None:
    experiment_actions = {
        "submit_hamiltonian_probe_batch",
        "probe_batch_result",
        "get_job_result",
    }
    later = [
        event.get("action")
        for event in events[final_index + 1 :]
        if event.get("action") in experiment_actions
    ]
    if later:
        raise VerifierError(
            "experiment activity occurred after submit_final_answer: " + ", ".join(later)
        )


# Sanitized marker qsim logs when a poll's result delivery breaks internally
# (e.g. result-artifact persistence): the poll raises instead of returning, so
# the agent may be unable to reach a terminal state through no fault of its
# own. Any such marker on a probe job is a qsim-owned infrastructure failure,
# never a model verdict. A worker failure inside qsim instead returns a terminal
# poll stamped ``failure_kind=qsim_internal``; that ownership stamp has the same
# disposition even though the poll itself completed.
_JOB_RESULT_FAILURE_ACTION = "get_job_result_failure"


def _require_no_qsim_delivery_failures(events: list[dict[str, Any]]) -> None:
    probe_job_ids = {
        event.get("job_id")
        for event in events
        if event.get("action") == "submit_hamiltonian_probe_batch"
    }
    for event in events:
        if (
            event.get("action") == _JOB_RESULT_FAILURE_ACTION
            and event.get("job_id") in probe_job_ids
        ):
            raise InfrastructureError(
                f"job {event.get('job_id')}: qsim recorded an internal result-delivery "
                f"failure (stage {event.get('failure_stage')!r})"
            )
        if event.get("failure_kind") == "qsim_internal":
            action = event.get("action")
            stage = event.get("failure_stage")
            where = f"{action}" + (f"/{stage}" if stage else "")
            raise InfrastructureError(f"qsim recorded an internal failure ({where})")


# qsim publishes delivery artifacts one level below the nested mount root
# as a separate directory. This scorer runs standalone in the verifier image and deliberately
# does not import qiqcbench, so the level is restated here rather than derived
# from `PUBLISHED_RESULT_SUBDIR`.
_ARTIFACT_RELATIVE_DIR = "public_job_results/results"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _probe_row_digest(initial_state: list[str], evolve_time_us: float, observable: str) -> str:
    encoded = json.dumps(
        {
            "initial_state": initial_state,
            "evolve_time_us": evolve_time_us,
            "observable_pauli": observable,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _verify_result_artifact(
    pointer: dict[str, Any],
    batch_event: dict[str, Any],
    artifacts_dir: Path,
    execution_contract: dict[str, Any],
) -> dict[str, Any]:
    """Verify one archived raw result against its qsim poll pointer."""
    job_id = str(batch_event.get("job_id"))
    if set(pointer) != {"schema_version", "relative_path", "size_bytes", "sha256"}:
        raise EvidenceIntegrityError(f"job {job_id}: artifact pointer fields are invalid")
    if pointer.get("schema_version") != 1:
        raise EvidenceIntegrityError(f"job {job_id}: unsupported artifact pointer schema")
    relative = str(pointer.get("relative_path", ""))
    relative_path = Path(relative)
    parts = relative_path.parts
    if (
        relative_path.is_absolute()
        or len(parts) != 3
        or f"{parts[0]}/{parts[1]}" != _ARTIFACT_RELATIVE_DIR
        or parts[2] != f"{job_id}.json"
    ):
        raise EvidenceIntegrityError(f"job {job_id}: unsafe artifact path {relative!r}")
    result_dir = artifacts_dir / _ARTIFACT_RELATIVE_DIR
    # Both directory levels, not just the innermost: `_ARTIFACT_RELATIVE_DIR`
    # has two components, and guarding only `result_dir` would
    # silently stop covering the mount root.
    if result_dir.parent.is_symlink() or not result_dir.is_dir() or result_dir.is_symlink():
        raise EvidenceIntegrityError(f"job {job_id}: artifact directory missing or invalid")
    path = result_dir / parts[-1]
    if not path.is_file() or path.is_symlink():
        raise EvidenceIntegrityError(f"job {job_id}: archived artifact missing or invalid")
    try:
        raw = path.read_bytes()
        declared_size = pointer.get("size_bytes")
        if isinstance(declared_size, bool) or not isinstance(declared_size, int):
            raise TypeError("size_bytes is not an integer")
        expected_size = declared_size
    except (OSError, TypeError, ValueError) as exc:
        raise EvidenceIntegrityError(f"job {job_id}: artifact cannot be read: {exc}") from None
    if len(raw) != expected_size:
        raise EvidenceIntegrityError(f"job {job_id}: artifact size mismatch for {relative}")
    declared_sha256 = pointer.get("sha256")
    if not _is_sha256(declared_sha256) or hashlib.sha256(raw).hexdigest() != declared_sha256:
        raise EvidenceIntegrityError(f"job {job_id}: artifact sha256 mismatch for {relative}")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise EvidenceIntegrityError(f"job {job_id}: artifact is not valid JSON") from None

    data = payload.get("data") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != 3
        or payload.get("job_id") != job_id
        or payload.get("device_id") != batch_event.get("device_id")
        or batch_event.get("device_id") != DEVICE_ID
        or payload.get("status") != "complete"
        or not isinstance(data, dict)
        or data.get("kind") != "probe_outcome"
    ):
        raise EvidenceIntegrityError(f"job {job_id}: artifact payload or kind mismatch")
    artifact_rows = data.get("rows")
    event_rows = batch_event.get("rows")
    if not isinstance(artifact_rows, list) or not isinstance(event_rows, list):
        raise EvidenceIntegrityError(f"job {job_id}: artifact or event rows are invalid")
    if len(artifact_rows) != len(event_rows):
        raise EvidenceIntegrityError(f"job {job_id}: artifact row count differs from event")
    for artifact_row, event_row in zip(artifact_rows, event_rows, strict=True):
        if not isinstance(artifact_row, dict) or not isinstance(event_row, dict):
            raise EvidenceIntegrityError(f"job {job_id}: artifact/event row is not an object")
        if artifact_row.get("status") not in {"accepted", "rejected"}:
            raise EvidenceIntegrityError(f"job {job_id}: artifact row has invalid status")
        # A rejected row echoes the agent's submitted observable verbatim (the
        # engine rejects invalid observables per-row), so only the nonempty-string
        # transport guarantee applies to it. The full device contract binds
        # accepted rows only: an out-of-contract observable on an accepted row is
        # qsim-owned evidence corruption, never agent input.
        observable = artifact_row.get("observable_pauli")
        if not isinstance(observable, str) or not observable:
            raise EvidenceIntegrityError(f"job {job_id}: artifact row has invalid observable")
        if artifact_row.get("status") == "accepted" and (
            len(observable) != execution_contract["n_qubits"]
            or any(character not in "IXYZ" for character in observable)
            or not execution_contract["observable_weight_min"]
            <= sum(character != "I" for character in observable)
            <= execution_contract["observable_weight_max"]
        ):
            raise EvidenceIntegrityError(f"job {job_id}: accepted row has invalid observable")
        initial_state = event_row.get("initial_state")
        requested_observable = event_row.get("observable_pauli")
        requested_time = event_row.get("evolve_time_us")
        if (
            not isinstance(initial_state, list)
            or not isinstance(requested_observable, str)
            or not requested_observable
        ):
            raise EvidenceIntegrityError(f"job {job_id}: event row lacks probe metadata")
        if isinstance(requested_time, bool) or not isinstance(requested_time, (int, float)):
            raise EvidenceIntegrityError(f"job {job_id}: event row has invalid requested time")
        requested_time_value = float(requested_time)
        if not math.isfinite(requested_time_value):
            raise EvidenceIntegrityError(f"job {job_id}: event row has invalid requested time")
        expected_digest = _probe_row_digest(
            initial_state, requested_time_value, requested_observable
        )
        if event_row.get("digest") != expected_digest:
            raise EvidenceIntegrityError(f"job {job_id}: event row request digest mismatch")
        if artifact_row.get("status") != event_row.get("status"):
            raise EvidenceIntegrityError(f"job {job_id}: artifact/event row status mismatch")
        if observable != requested_observable:
            raise EvidenceIntegrityError(f"job {job_id}: artifact/event observable mismatch")
        if artifact_row.get("reject_reason") != event_row.get("reject_reason"):
            raise EvidenceIntegrityError(f"job {job_id}: artifact/event reject reason mismatch")
        artifact_time = artifact_row.get("accepted_evolve_time_us")
        event_time = event_row.get("accepted_evolve_time_us")
        try:
            times_differ = (artifact_time is None) != (event_time is None) or (
                artifact_time is not None
                and (
                    not math.isfinite(float(artifact_time))
                    or not math.isfinite(float(event_time))
                    or abs(float(artifact_time) - float(event_time)) > 1e-9
                )
            )
        except (TypeError, ValueError):
            times_differ = True
        if times_differ:
            raise EvidenceIntegrityError(f"job {job_id}: artifact/event row time mismatch")
        if artifact_row.get("status") == "accepted":
            if (
                len(initial_state) != execution_contract["n_qubits"]
                or any(
                    label not in execution_contract["allowed_initial_states"]
                    for label in initial_state
                )
                or abs(float(artifact_time) - requested_time_value) > 1e-9
                or not _accepted_time_is_valid(requested_time_value, execution_contract)
            ):
                raise EvidenceIntegrityError(
                    f"job {job_id}: accepted row has invalid probe metadata"
                )
            outcomes = artifact_row.get("raw_pauli_outcomes")
            repetitions = artifact_row.get("num_internal_repetitions")
            if (
                not isinstance(outcomes, list)
                or type(repetitions) is not int
                or repetitions != execution_contract["oracle_noise_shots"]
                or len(outcomes) != repetitions
                or any(type(value) is not int or value not in (-1, 1) for value in outcomes)
            ):
                raise EvidenceIntegrityError(f"job {job_id}: accepted row has invalid raw outcomes")
        elif any(
            artifact_row.get(key) is not None
            for key in (
                "accepted_evolve_time_us",
                "raw_pauli_outcomes",
                "num_internal_repetitions",
            )
        ):
            raise EvidenceIntegrityError(f"job {job_id}: rejected row carries accepted evidence")

    numeric_pairs = (
        ("budget_used_us", 1e-9),
        ("budget_remaining_us", 1e-9),
        ("accepted_row_count", 0.0),
    )
    if (
        type(data.get("accepted_row_count")) is not int
        or type(batch_event.get("accepted_row_count")) is not int
    ):
        raise EvidenceIntegrityError(f"job {job_id}: accepted_row_count must be an integer")
    for key, tolerance in numeric_pairs:
        if isinstance(data.get(key), bool) or isinstance(batch_event.get(key), bool):
            raise EvidenceIntegrityError(f"job {job_id}: invalid {key} evidence")
        try:
            artifact_value = float(data.get(key, -1.0))
            event_value = float(batch_event.get(key, -2.0))
        except (TypeError, ValueError):
            raise EvidenceIntegrityError(f"job {job_id}: invalid {key} evidence") from None
        if not math.isfinite(artifact_value) or not math.isfinite(event_value):
            raise EvidenceIntegrityError(f"job {job_id}: non-finite {key} evidence")
        mismatch = abs(artifact_value - event_value)
        if mismatch > tolerance:
            raise EvidenceIntegrityError(f"job {job_id}: artifact/event {key} mismatch")
    if data.get("max_probe_rows") != execution_contract["max_probe_rows"]:
        raise EvidenceIntegrityError(f"job {job_id}: artifact max_probe_rows is inconsistent")
    return payload


def _reconstruct_evidence(
    events: list[dict[str, Any]],
    artifacts_dir: Path,
    execution_contract: dict[str, Any],
) -> dict[str, Any]:
    """Reconstruct the run without assuming concurrent JSONL append order.

    Jobs are joined by ID. Their cumulative accepted-count snapshots establish
    execution order, while row-level bytes independently reconstruct the final
    budget. This is necessary because a worker can finish and append its result
    before the submitting thread appends the corresponding submission event.
    """
    submissions_by_job: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("action") != "submit_hamiltonian_probe_batch":
            continue
        job_id = event.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise EvidenceIntegrityError("probe submission event lacks a job_id")
        if job_id in submissions_by_job:
            raise EvidenceIntegrityError(f"duplicate probe submission for job {job_id}")
        submissions_by_job[job_id] = event

    batch_by_job: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("action") != "probe_batch_result":
            continue
        job_id = event.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise EvidenceIntegrityError("probe_batch_result event lacks a job_id")
        if job_id in batch_by_job:
            raise EvidenceIntegrityError(f"duplicate probe result for job {job_id}")
        submission = submissions_by_job.get(job_id)
        if submission is None:
            raise EvidenceIntegrityError(f"probe result job {job_id} lacks its submission event")
        if event.get("evidence_schema_version") != PROBE_BATCH_EVIDENCE_SCHEMA_VERSION:
            raise EvidenceIntegrityError(
                f"probe result job {job_id} has unsupported evidence schema"
            )
        rows = event.get("rows")
        submitted_rows = submission.get("n_rows")
        if (
            not isinstance(rows, list)
            or type(submitted_rows) is not int
            or submitted_rows <= 0
            or submitted_rows != len(rows)
        ):
            raise EvidenceIntegrityError(f"probe result job {job_id} has inconsistent row count")
        if submitted_rows > execution_contract["max_rows_per_batch"]:
            raise EvidenceIntegrityError(
                f"probe result job {job_id} completed above the public per-batch cap"
            )
        if submission.get("device_id") != DEVICE_ID or event.get("device_id") != submission.get(
            "device_id"
        ):
            raise EvidenceIntegrityError(f"probe result job {job_id} has a device mismatch")
        batch_by_job[job_id] = event

    terminal_status_by_job: dict[str, str] = {}
    pointers_by_job: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("action") != "get_job_result":
            continue
        job_id = event.get("job_id")
        if job_id not in submissions_by_job or event.get("status") not in {"complete", "failed"}:
            continue
        status = event["status"]
        prior_status = terminal_status_by_job.get(job_id)
        if prior_status is not None and prior_status != status:
            raise EvidenceIntegrityError(f"job {job_id}: conflicting terminal poll statuses")
        terminal_status_by_job[job_id] = status
        if status == "failed":
            continue
        pointer = event.get("offloaded_result")
        if not isinstance(pointer, dict):
            raise EvidenceIntegrityError(
                f"job {job_id}: completed probe poll lacks an offloaded_result pointer"
            )
        prior = pointers_by_job.get(job_id)
        if prior is not None and prior != pointer:
            raise EvidenceIntegrityError(f"job {job_id}: conflicting artifact pointers")
        pointers_by_job[job_id] = pointer
    unresolved_jobs = sorted(set(submissions_by_job) - set(terminal_status_by_job))
    if unresolved_jobs:
        raise VerifierError(
            "probe jobs were not polled to a terminal state before final submission: "
            + ", ".join(unresolved_jobs)
        )
    for job_id, status in terminal_status_by_job.items():
        if status == "failed" and job_id in batch_by_job:
            raise EvidenceIntegrityError(
                f"job {job_id}: failed terminal poll conflicts with completed batch evidence"
            )
        if status == "complete" and job_id not in batch_by_job:
            raise EvidenceIntegrityError(
                f"job {job_id}: completed probe poll lacks probe_batch_result evidence"
            )
        if status == "complete" and job_id not in pointers_by_job:
            raise EvidenceIntegrityError(
                f"job {job_id}: completed probe poll lacks an artifact pointer"
            )
    result_without_terminal = sorted(set(batch_by_job) - set(terminal_status_by_job))
    if result_without_terminal:
        raise VerifierError(
            "completed probe jobs were not polled to a terminal state before final submission: "
            + ", ".join(result_without_terminal)
        )
    if not batch_by_job:
        raise VerifierError("no completed probe_batch_result evidence found in experiment_log")

    verified_accepted_rows: list[dict[str, Any]] = []
    consumed_offloaded_results: list[dict[str, Any]] = []
    batch_records: list[dict[str, Any]] = []
    for job_id, pointer in sorted(pointers_by_job.items()):
        batch = batch_by_job[job_id]
        payload = _verify_result_artifact(
            pointer,
            batch,
            artifacts_dir,
            execution_contract,
        )
        consumed_offloaded_results.append(
            {"role": "hamiltonian_probe_result", "job_id": job_id, **pointer}
        )
        event_rows = batch["rows"]
        accepted_times: list[float] = []
        for row, event_row in zip(payload["data"]["rows"], event_rows, strict=True):
            accepted_time = row.get("accepted_evolve_time_us")
            if row.get("status") == "accepted" and accepted_time is not None:
                accepted_time_value = float(accepted_time)
                accepted_times.append(accepted_time_value)
                verified_accepted_rows.append(
                    {
                        "initial_state": list(event_row["initial_state"]),
                        "evolve_time_us": accepted_time_value,
                        "observable_pauli": row["observable_pauli"],
                        "raw_pauli_outcomes": list(row["raw_pauli_outcomes"]),
                        "num_internal_repetitions": row["num_internal_repetitions"],
                    }
                )
        cumulative_count = batch.get("accepted_row_count")
        if type(cumulative_count) is not int or cumulative_count < 0:
            raise EvidenceIntegrityError(f"job {job_id}: invalid cumulative accepted row count")
        try:
            cumulative_budget = float(batch["budget_used_us"])
            cumulative_remaining = float(batch["budget_remaining_us"])
        except (KeyError, TypeError, ValueError):
            raise EvidenceIntegrityError(
                f"job {job_id}: invalid cumulative budget evidence"
            ) from None
        if (
            not math.isfinite(cumulative_budget)
            or not math.isfinite(cumulative_remaining)
            or cumulative_budget < 0.0
            or cumulative_remaining < 0.0
            or cumulative_count > execution_contract["max_probe_rows"]
            or cumulative_budget > execution_contract["budget_us"] + _BUDGET_TOL_US
        ):
            raise EvidenceIntegrityError(f"job {job_id}: cumulative evidence exceeds its contract")
        batch_records.append(
            {
                "job_id": job_id,
                "accepted_count": len(accepted_times),
                "accepted_time_us": math.fsum(accepted_times),
                "cumulative_count": cumulative_count,
                "cumulative_budget_us": cumulative_budget,
                "cumulative_remaining_us": cumulative_remaining,
            }
        )

    running_count = 0
    accepted_time_parts: list[float] = []
    prefixes: list[tuple[int, float]] = [(0, 0.0)]
    positive_batches = sorted(
        (record for record in batch_records if record["accepted_count"] > 0),
        key=lambda record: (record["cumulative_count"], record["job_id"]),
    )
    for record in positive_batches:
        running_count += record["accepted_count"]
        accepted_time_parts.append(record["accepted_time_us"])
        running_budget = math.fsum(accepted_time_parts)
        if (
            record["cumulative_count"] != running_count
            or abs(record["cumulative_budget_us"] - running_budget) > _BUDGET_TOL_US
        ):
            raise EvidenceIntegrityError(
                f"job {record['job_id']}: cumulative accepted rows/budget do not form a chain"
            )
        expected_remaining = max(0.0, execution_contract["budget_us"] - running_budget)
        if abs(record["cumulative_remaining_us"] - expected_remaining) > _BUDGET_TOL_US:
            raise EvidenceIntegrityError(
                f"job {record['job_id']}: cumulative remaining budget is inconsistent"
            )
        prefixes.append((running_count, running_budget))

    for record in (record for record in batch_records if record["accepted_count"] == 0):
        if not any(
            record["cumulative_count"] == count
            and abs(record["cumulative_budget_us"] - budget) <= _BUDGET_TOL_US
            and abs(
                record["cumulative_remaining_us"]
                - max(0.0, execution_contract["budget_us"] - budget)
            )
            <= _BUDGET_TOL_US
            for count, budget in prefixes
        ):
            raise EvidenceIntegrityError(
                f"job {record['job_id']}: zero-accepted batch does not match any valid prefix"
            )

    accepted = len(verified_accepted_rows)
    budget_used = math.fsum(row["evolve_time_us"] for row in verified_accepted_rows)
    if (
        accepted != running_count
        or abs(budget_used - math.fsum(accepted_time_parts)) > _BUDGET_TOL_US
    ):
        raise EvidenceIntegrityError("artifact rows disagree with cumulative probe evidence")
    if accepted == 0:
        raise VerifierError("no accepted probe rows in evidence")
    ran_nonzero = any(row["evolve_time_us"] > 0.0 for row in verified_accepted_rows)
    if not ran_nonzero:
        raise VerifierError(
            "no completed, artifact-verified poll retrieved an accepted nonzero-time probe"
        )
    return {
        "accepted_row_count": accepted,
        "ran_nonzero_time_probe": ran_nonzero,
        "retrieved_nonzero_time_probe": ran_nonzero,
        "budget_used_us": budget_used,
        "min_accepted_evolve_time_us": min(row["evolve_time_us"] for row in verified_accepted_rows),
        "max_accepted_evolve_time_us": max(row["evolve_time_us"] for row in verified_accepted_rows),
        "accepted_rows": verified_accepted_rows,
        "probe_job_count": len(submissions_by_job),
        "polled_probe_job_count": len(terminal_status_by_job),
        "consumed_offloaded_results": consumed_offloaded_results,
    }


# ------------------------------------------------------------ answer validation


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise VerifierError(f"final answer invalid: {msg}")


def _coeff_vector(answer: dict[str, Any], key: str) -> list[float]:
    v = answer.get(key)
    _require(isinstance(v, list) and len(v) == N_TERMS, f"{key} must be a length-{N_TERMS} list")
    try:
        out = [float(x) for x in v]
    except (TypeError, ValueError):
        raise VerifierError(f"final answer invalid: {key} must contain numbers") from None
    _require(
        all(abs(x) <= 0.75 + 1e-9 for x in out), f"{key} has a coefficient outside [-0.75, 0.75]"
    )
    return out


def _sigma_vector(answer: dict[str, Any], key: str) -> list[float]:
    v = answer.get(key)
    _require(isinstance(v, list) and len(v) == N_TERMS, f"{key} must be a length-{N_TERMS} list")
    try:
        out = [float(x) for x in v]
    except (TypeError, ValueError):
        raise VerifierError(f"final answer invalid: {key} must contain numbers") from None
    _require(all(0.0 < x <= 0.30 + 1e-9 for x in out), f"{key} has a 1-sigma outside (0, 0.30]")
    return out


def _validate_answer(answer: dict[str, Any], execution_contract: dict[str, Any]) -> None:
    unknown_fields = sorted(set(answer) - _REQUIRED_ANSWER_FIELDS - _OPTIONAL_ANSWER_FIELDS)
    _require(not unknown_fields, f"unknown fields: {', '.join(unknown_fields)}")
    # The FinalAnswer envelope's task_id is validated at submission time and
    # again when final_answer.json is read; the inner copy is optional and only
    # checked when present, so an otherwise-correct submission is never
    # rejected over the duplicated field.
    _require(answer.get("task_id") in (None, TASK_ID), "wrong task_id")
    _require(
        answer.get("hamiltonian_convention") == HAMILTONIAN_CONVENTION,
        "wrong hamiltonian_convention",
    )
    _require(
        answer.get("term_order_convention") == TERM_ORDER_CONVENTION, "wrong term_order_convention"
    )
    _require(answer.get("num_terms") == N_TERMS, "num_terms must be 111")
    _coeff_vector(answer, "omega_vector_raw_rad_per_us")
    _sigma_vector(answer, "omega_vector_raw_rad_per_us_1sigma")

    activity = answer.get("activity_summary")
    _require(isinstance(activity, dict), "activity_summary must be an object")
    _require(
        set(activity) == _ACTIVITY_SUMMARY_FIELDS,
        "activity_summary must contain exactly schema_version, verifiable, and trajectory",
    )
    _require(activity.get("schema_version") == 2, "activity_summary.schema_version must be 2")

    verifiable = activity.get("verifiable")
    _require(isinstance(verifiable, dict), "activity_summary.verifiable must be an object")
    _require(
        set(verifiable) == _VERIFIABLE_ACTIVITY_FIELDS,
        "activity_summary.verifiable fields do not match schema version 2",
    )
    n = verifiable.get("accepted_probe_rows")
    _require(
        type(n) is int and 1 <= n <= execution_contract["max_probe_rows"],
        "activity_summary.verifiable.accepted_probe_rows exceeds the public range",
    )
    try:
        total_time = float(verifiable.get("total_accepted_evolution_time_us", -1))
    except (TypeError, ValueError):
        raise VerifierError(
            "final answer invalid: activity_summary.verifiable."
            "total_accepted_evolution_time_us must be numeric"
        ) from None
    _require(
        math.isfinite(total_time) and total_time >= 0.0,
        "activity_summary.verifiable.total_accepted_evolution_time_us must be >= 0",
    )
    reported_times: dict[str, float] = {}
    for key in ("min_accepted_evolve_time_us", "max_accepted_evolve_time_us"):
        try:
            t = float(verifiable.get(key, -1))
        except (TypeError, ValueError):
            raise VerifierError(
                f"final answer invalid: activity_summary.verifiable.{key} must be numeric"
            ) from None
        _require(
            execution_contract["time_min_us"] - 1e-9
            <= t
            <= execution_contract["time_max_us"] + 1e-9,
            f"activity_summary.verifiable.{key} is outside the public time range",
        )
        reported_times[key] = t
    _require(
        reported_times["min_accepted_evolve_time_us"]
        <= reported_times["max_accepted_evolve_time_us"],
        "activity_summary.verifiable minimum evolve time must not exceed maximum",
    )

    trajectory = activity.get("trajectory")
    _require(isinstance(trajectory, dict), "activity_summary.trajectory must be an object")
    _require(
        set(trajectory) == _TRAJECTORY_REVIEW_FIELDS,
        "activity_summary.trajectory fields do not match schema version 2",
    )
    _require(trajectory.get("protocol") in _PROTOCOLS, "unknown trajectory protocol")
    _require(trajectory.get("fit_model") in _FIT_MODELS, "unknown trajectory fit_model")
    details = trajectory.get("trajectory_details")
    _require(
        isinstance(details, str) and bool(details.strip()) and len(details) <= 8192,
        "activity_summary.trajectory.trajectory_details must be 1-8192 nonblank characters",
    )
    # Validate either optional field when present; score the pair only when both
    # are present, matching the canonical scorer's documented raw fallback.
    refined = answer.get("omega_vector_refined_rad_per_us")
    refined_sigma = answer.get("omega_vector_refined_rad_per_us_1sigma")
    if refined is not None:
        _coeff_vector(answer, "omega_vector_refined_rad_per_us")
    if refined_sigma is not None:
        _sigma_vector(answer, "omega_vector_refined_rad_per_us_1sigma")


def _select_scored_vector(answer: dict[str, Any]) -> tuple[list[float], list[float], str]:
    ref = answer.get("omega_vector_refined_rad_per_us")
    ref_sig = answer.get("omega_vector_refined_rad_per_us_1sigma")
    if ref is not None and ref_sig is not None:
        return [float(x) for x in ref], [float(x) for x in ref_sig], "refined"
    return (
        [float(x) for x in answer["omega_vector_raw_rad_per_us"]],
        [float(x) for x in answer["omega_vector_raw_rad_per_us_1sigma"]],
        "raw",
    )


# ------------------------------------------------ scientific evidence compatibility


def _full_pauli_matrix(pauli: str) -> np.ndarray:
    operator = np.array([[1.0 + 0.0j]])
    for character in pauli:
        operator = np.kron(operator, _PAULI_2X2[character])
    return operator


def _state_vector(labels: list[str]) -> np.ndarray:
    state = np.array([1.0 + 0.0j])
    for label in labels:
        state = np.kron(state, _KET[label])
    return state


def _apply_pauli(vector: np.ndarray, pauli: str) -> np.ndarray:
    tensor = vector.reshape((2,) * 10)
    for qubit, character in enumerate(pauli):
        if character == "I":
            continue
        tensor = np.moveaxis(
            np.tensordot(_PAULI_2X2[character], tensor, axes=([1], [qubit])), 0, qubit
        )
    return tensor.reshape(-1)


def _exact_predictions(omega_hat: list[float], rows: list[dict[str, Any]]) -> list[float]:
    hamiltonian = np.zeros((1 << 10, 1 << 10), dtype=complex)
    for coefficient, pauli in zip(omega_hat, _TERM_PAULIS, strict=True):
        if coefficient != 0.0:
            hamiltonian += coefficient * _full_pauli_matrix(pauli)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)

    predictions: list[float] = []
    for row in rows:
        state = _state_vector(row["initial_state"])
        coefficients = eigenvectors.conj().T @ state
        evolved = eigenvectors @ (
            np.exp(-0.5j * eigenvalues * row["evolve_time_us"]) * coefficients
        )
        expectation = float(
            np.real(np.vdot(evolved, _apply_pauli(evolved, row["observable_pauli"])))
        )
        predictions.append(min(max(expectation, -1.0), 1.0))
    return predictions


def _binomial_deviance(outcomes: list[int], expectation: float) -> float:
    repetitions = len(outcomes)
    positive = sum(value == 1 for value in outcomes)
    observed_probability = positive / repetitions
    predicted_probability = min(max((1.0 + expectation) / 2.0, 1e-12), 1.0 - 1e-12)
    deviance = 0.0
    if positive:
        deviance += positive * math.log(observed_probability / predicted_probability)
    negative = repetitions - positive
    if negative:
        deviance += negative * math.log(
            (1.0 - observed_probability) / (1.0 - predicted_probability)
        )
    return 2.0 * deviance


def _certify_scientific_evidence(
    omega_hat: list[float],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not rows:
        return {
            "mean_binomial_deviance": None,
            "p90_binomial_deviance": None,
            "data_compatible": False,
        }
    predictions = _exact_predictions(omega_hat, rows)
    deviances = np.asarray(
        [
            _binomial_deviance(row["raw_pauli_outcomes"], prediction)
            for row, prediction in zip(rows, predictions, strict=True)
        ],
        dtype=float,
    )
    mean_deviance = float(np.mean(deviances))
    p90_deviance = float(np.percentile(deviances, 90.0))
    compatible = bool(
        math.isfinite(mean_deviance)
        and math.isfinite(p90_deviance)
        and mean_deviance <= _MAX_MEAN_BINOMIAL_DEVIANCE
        and p90_deviance <= _MAX_P90_BINOMIAL_DEVIANCE
    )
    return {
        "mean_binomial_deviance": mean_deviance,
        "p90_binomial_deviance": p90_deviance,
        "data_compatible": compatible,
    }


# --------------------------------------------------------------------- scoring


def _linf(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b, strict=True))


def _relative_l2(a: list[float], b: list[float]) -> float:
    denominator = math.sqrt(sum(value * value for value in b))
    if denominator == 0.0:
        raise InfrastructureError("relative L2 error is undefined for an all-zero hidden vector")
    numerator = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))
    return numerator / denominator


def _level(value: float, scorer: dict[str, Any]) -> str:
    if value <= scorer["linf_gold"]:
        return "gold"
    if value <= scorer["linf_silver"]:
        return "silver"
    if value <= scorer["linf_bronze"]:
        return "bronze"
    if value <= scorer["linf_pass"]:
        return "pass"
    return "fail"


def _support_diagnostic(omega_hat: list[float], scorer: dict[str, Any]) -> tuple[int, int]:
    tau = scorer["support_tau"]
    true_support = set(scorer["support_terms"])
    detected = sum(1 for t in true_support if abs(omega_hat[_TERM_INDEX[t]]) > tau)
    false_pos = sum(
        1 for i in range(N_TERMS) if scorer["omega_star"][i] == 0.0 and abs(omega_hat[i]) > tau
    )
    return detected, false_pos


def _uncertainty_covered(
    omega_hat: list[float], sigma: list[float], omega_star: list[float]
) -> int:
    return sum(
        1
        for hat, sig, star in zip(omega_hat, sigma, omega_star, strict=True)
        if abs(hat - star) <= max(2.0 * sig, 0.025)
    )


def score(
    answer: dict[str, Any], scorer: dict[str, Any], evidence: dict[str, Any]
) -> dict[str, Any]:
    omega_star = scorer["omega_star"]
    omega_hat, sigma, kind = _select_scored_vector(answer)

    value = _linf(omega_hat, omega_star)
    relative_l2_value = _relative_l2(omega_hat, omega_star)
    relative_l2_within_threshold = relative_l2_value <= scorer["relative_l2_pass"]
    detected, false_pos = _support_diagnostic(omega_hat, scorer)
    covered = _uncertainty_covered(omega_hat, sigma, omega_star)
    median_sigma = sorted(sigma)[len(sigma) // 2]

    budget_used = evidence["budget_used_us"]
    activity = answer["activity_summary"]["verifiable"]
    self_report = float(activity["total_accepted_evolution_time_us"])
    within_budget = budget_used <= scorer["budget_us"] + _BUDGET_TOL_US
    accepted = evidence["accepted_row_count"]
    ran_probe = evidence["retrieved_nonzero_time_probe"]
    evidence_min_time = evidence["min_accepted_evolve_time_us"]
    evidence_max_time = evidence["max_accepted_evolve_time_us"]
    certification = _certify_scientific_evidence(
        omega_hat,
        evidence["accepted_rows"],
    )

    penalties: list[str] = []
    report_consistent = True
    if abs(budget_used - self_report) > _REPORT_TOL_US:
        report_consistent = False
        penalties.append(
            "activity_summary.verifiable total evolution time "
            f"{self_report:.4f} us != evidence {budget_used:.4f} us"
        )
    reported_rows = int(activity["accepted_probe_rows"])
    if reported_rows != accepted:
        report_consistent = False
        penalties.append(
            "activity_summary.verifiable accepted_probe_rows "
            f"{reported_rows} != evidence {accepted}"
        )
    reported_min = float(activity["min_accepted_evolve_time_us"])
    reported_max = float(activity["max_accepted_evolve_time_us"])
    if (
        evidence_min_time is None
        or evidence_max_time is None
        or abs(reported_min - evidence_min_time) > _REPORT_TOL_US
        or abs(reported_max - evidence_max_time) > _REPORT_TOL_US
    ):
        report_consistent = False
        penalties.append(
            "activity_summary.verifiable probe-time range "
            f"[{reported_min:.4f}, {reported_max:.4f}] us != evidence "
            f"[{evidence_min_time!r}, {evidence_max_time!r}] us"
        )
    if not within_budget:
        penalties.append(f"over budget: used {budget_used:.4f} us > {scorer['budget_us']} us")
    if accepted > scorer["max_probe_rows"]:
        penalties.append(f"too many accepted probe rows: {accepted} > {scorer['max_probe_rows']}")
    if not ran_probe:
        penalties.append("no accepted nonzero-time probe ran")
    # Stale-only diagnostic (the L∞ metric already fails it; flag for clarity).
    # ``stale_vector`` is reconstructed from the public stale notebook (still no
    # qiqcbench import); absent it, the diagnostic is simply skipped.
    stale = scorer.get("stale_vector")
    if stale is not None and _linf(omega_hat, stale) < 0.02 and value > scorer["linf_pass"]:
        penalties.append("submitted vector matches the stale notebook model")

    binary_pass = bool(
        value <= scorer["linf_pass"]
        and relative_l2_within_threshold
        and within_budget
        and accepted <= scorer["max_probe_rows"]
        and ran_probe
        and len(omega_hat) == N_TERMS
        and report_consistent
    )
    return {
        "linf": value,
        "relative_l2": relative_l2_value,
        "relative_l2_within_threshold": relative_l2_within_threshold,
        "level": _level(value, scorer),
        "scored_vector_kind": kind,
        "support_true_total": len(scorer["support_terms"]),
        "support_detected_true": detected,
        "support_false_positives": false_pos,
        "uncertainty_covered": covered,
        "uncertainty_total": N_TERMS,
        "uncertainty_median_1sigma": median_sigma,
        "budget_us": scorer["budget_us"],
        "budget_used_us": budget_used,
        "activity_summary_evolution_time_us": self_report,
        "within_budget": within_budget,
        "accepted_row_count": accepted,
        "ran_nonzero_time_probe": ran_probe,
        "evidence_min_evolve_time_us": evidence_min_time,
        "evidence_max_evolve_time_us": evidence_max_time,
        "verifiable_activity_consistent": report_consistent,
        "evidence_mean_binomial_deviance": certification["mean_binomial_deviance"],
        "evidence_p90_binomial_deviance": certification["p90_binomial_deviance"],
        "evidence_data_compatible": certification["data_compatible"],
        "binary_pass": binary_pass,
        "reward_binary": int(binary_pass),
        "penalties": penalties,
    }


# ----------------------------------------------------------------- locating IO


def _resolve_path(env_var: str, *candidates: Path) -> Path | None:
    """Resolve a private input path with an AUTHORITATIVE environment override.

    When ``env_var`` is set (the production Harbor configuration always sets
    it via ``[verifier.env]``), that exact path is the single source of truth:
    a missing or unusable file is an infrastructure failure, never a cue to
    search fallbacks — a fallback silently substituting different truth is
    exactly the failure mode an execution-scoped scorer must refuse. The
    development fallbacks below apply only when the variable is entirely
    unset (local unit tests running from a repo checkout).
    """
    override = os.environ.get(env_var)
    if override is not None:
        if override and Path(override).exists():
            return Path(override)
        raise InfrastructureError(
            f"{env_var} is set to {override!r} but that exact path does not exist; "
            "the configured private input is authoritative and no fallback is searched"
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _locate_public_dir() -> Path:
    here = Path(__file__).resolve()
    candidates = [Path("/task_materials")]
    if len(here.parents) > 3:
        candidates.append(here.parents[3] / "configs" / "task_materials" / TASK_ID / "public")
    found = _resolve_path("QIQCBENCH_TASK_PUBLIC_DIR", *candidates)
    if found is None or not (found / "hamiltonian_dictionary.json").is_file():
        raise InfrastructureError(
            "could not locate public task materials; set QIQCBENCH_TASK_PUBLIC_DIR or mount /task_materials"
        )
    return found


def _locate_hidden_scorer() -> Path:
    # Production bakes the answer key into the allowlisted, separate verifier
    # image. The configured path is authoritative: missing/unreadable is an
    # infrastructure failure with no fallback search. The repo-relative
    # candidate serves only env-less unit tests in a checkout.
    here = Path(__file__).resolve()
    candidates = []
    if len(here.parents) > 3:
        candidates.append(
            here.parents[3]
            / "configs"
            / "task_materials"
            / TASK_ID
            / "hidden"
            / "hidden_scorer.yaml"
        )
    found = _resolve_path("QIQCBENCH_HIDDEN_SCORER_PATH", *candidates)
    if found is None:
        raise InfrastructureError(
            "could not locate hidden scorer YAML; set QIQCBENCH_HIDDEN_SCORER_PATH"
        )
    return found


def _hidden_instance_commitment(scorer: dict[str, Any]) -> str:
    """Recompute qsim's execution-binding digest from the scoring truth.

    Mirrors ``qsim.qtypes.blackbox_analog_dynamics.device.
    hidden_instance_commitment`` byte-for-byte (this verifier is deliberately
    self-contained and must not import qiqcbench): compact sorted JSON over
    the full coefficient vector in public term order plus the two public
    conventions, hashed with SHA-256.
    """
    payload = json.dumps(
        {
            "scheme": "sha256_omega_vector_v1",
            "qtype": "blackbox_analog_dynamics",
            "hamiltonian_convention": HAMILTONIAN_CONVENTION,
            "term_order_convention": TERM_ORDER_CONVENTION,
            "omega": [float(value) for value in scorer["omega_star"]],
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _verify_hidden_instance_commitment(events: list[dict], scorer: dict[str, Any]) -> None:
    """Refuse to score truth that is not the instance qsim executed.

    qsim logs one ``hidden_instance_commitment`` event at boot, before any
    agent traffic. Whenever the log shows analog probe activity, that event
    must be present and every occurrence must equal the digest recomputed
    from this verifier's scoring truth; any absence or mismatch means the
    executed Hamiltonian and the scoring truth cannot be proven to be the
    same instance, which is an infrastructure failure, never a model verdict.
    A log with no probe activity carries no instance-dependent science, so the
    (truth-independent) no-evidence rejection path may proceed without it.
    """
    commitments = [event for event in events if event.get("action") == "hidden_instance_commitment"]
    probe_activity = any(
        isinstance(event.get("action"), str)
        and event["action"].startswith("run_hamiltonian_probe_batch")
        or event.get("action") in {"probe_batch_result", "get_job_result"}
        for event in events
    )
    if not commitments:
        if probe_activity:
            raise InfrastructureError(
                "qsim logged probe activity but no hidden_instance_commitment event; "
                "cannot bind the executed instance to the scoring truth"
            )
        return
    expected = _hidden_instance_commitment(scorer)
    for event in commitments:
        if event.get("scheme") != "sha256_omega_vector_v1":
            raise InfrastructureError(
                f"hidden_instance_commitment uses an unknown scheme {event.get('scheme')!r}"
            )
        logged = event.get("commitment_sha256")
        if logged != expected:
            raise InfrastructureError(
                "hidden instance mismatch: qsim executed an instance with commitment "
                f"{str(logged)[:16]}..., but the verifier's scoring truth has commitment "
                f"{expected[:16]}...; refusing to score across two different Hamiltonians"
            )


def _write_score_report(report: dict[str, Any], artifacts_dir: Path) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / SCORE_REPORT_FILENAME).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


def _emit_failure_report(artifacts_dir: Path, reason: str) -> None:
    try:
        _write_score_report(
            {
                "schema_version": SCORE_REPORT_SCHEMA_VERSION,
                "task_id": TASK_ID,
                "result": None,
                "penalties": [reason],
                "reward_binary": 0,
            },
            artifacts_dir,
        )
    except OSError:
        pass


def run_verifier(log_path: Path, ans_path: Path, artifacts_dir: Path) -> int:
    if not log_path.is_file():
        raise InfrastructureError(f"missing experiment_log.jsonl at {log_path}")

    events = _load_jsonl(log_path)
    _require_no_final_answer_failures(events)
    _require_no_qsim_delivery_failures(events)
    public_dir = _locate_public_dir()
    scorer = _load_hidden_scorer(_locate_hidden_scorer())
    _verify_public_material_digests(scorer, public_dir)
    execution_contract = _load_execution_contract(public_dir, scorer)
    _verify_hidden_instance_commitment(events, scorer)
    selected = _select_latest_submission(events)
    if not ans_path.is_file():
        if selected is not None:
            raise EvidenceIntegrityError("final_answer.json missing despite an accepted submission")
        # Never submitting at all is the model-owned failure.
        raise VerifierError(f"missing final_answer.json at {ans_path}")
    if selected is None:
        # Only qsim's atomic submit_final_answer action can create this file; an
        # answer with no accepted event means the qsim log channel failed.
        raise EvidenceIntegrityError(
            "final_answer.json exists but no accepted submit event was logged"
        )
    final_index, submission = selected
    # All qsim-owned artifact integrity first; model-owned gates only after.
    answer = _load_answer(ans_path)
    if "answer" not in submission:
        raise EvidenceIntegrityError("latest accepted submission event carries no answer payload")
    if json.dumps(submission["answer"], sort_keys=True) != json.dumps(answer, sort_keys=True):
        raise EvidenceIntegrityError(
            "final_answer.json differs from the latest accepted submission event"
        )
    _require_no_post_final_experiments(events, final_index)
    _validate_answer(answer, execution_contract)

    # Deliberate evidence-window semantics: the window closes at the *last*
    # accepted submission, so probes run between two submissions are real,
    # budget-counted work; experiments after the final submission stay banned.
    evidence = _reconstruct_evidence(events[:final_index], artifacts_dir, execution_contract)
    # stale vector comes from the now-digest-verified public notebook (§9g flag).
    scorer["stale_vector"] = _load_stale_vector(public_dir)

    result = score(answer, scorer, evidence)
    report = {
        "schema_version": SCORE_REPORT_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "result": result,
        "material_digests": dict(scorer["public_material_digests"]),
        "consumed_offloaded_results": list(evidence["consumed_offloaded_results"]),
        "budget_evidence": {
            "budget_us": scorer["budget_us"],
            "budget_used_us": evidence["budget_used_us"],
            "accepted_row_count": evidence["accepted_row_count"],
            "min_accepted_evolve_time_us": evidence["min_accepted_evolve_time_us"],
            "max_accepted_evolve_time_us": evidence["max_accepted_evolve_time_us"],
            "max_probe_rows": scorer["max_probe_rows"],
            "probe_job_count": evidence["probe_job_count"],
            "polled_probe_job_count": evidence["polled_probe_job_count"],
        },
        "penalties": list(result["penalties"]),
        "reward_binary": int(result["reward_binary"]),
    }
    _write_score_report(report, artifacts_dir)
    return int(result["reward_binary"])


def main() -> None:
    if len(sys.argv) < 4:
        print(
            "usage: score_hamlearn.py <experiment_log.jsonl> <final_answer.json> <artifacts_dir>",
            file=sys.stderr,
        )
        sys.exit(1)
    log_path, ans_path, artifacts_dir = (Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    try:
        reward = run_verifier(log_path, ans_path, artifacts_dir)
    except InfrastructureError as exc:
        print(f"VERIFIER INFRASTRUCTURE FAILURE: {exc}", file=sys.stderr)
        sys.exit(3)
    except VerifierError as exc:
        print(f"VERIFIER FAIL: {exc}", file=sys.stderr)
        _emit_failure_report(artifacts_dir, str(exc))
        sys.exit(1)
    except Exception as exc:
        print(
            f"VERIFIER INFRASTRUCTURE FAILURE: unexpected {type(exc).__name__}",
            file=sys.stderr,
        )
        sys.exit(3)

    if reward == 1:
        print("Verifier passed.")
        sys.exit(0)
    print("VERIFIER FAIL: scoring did not pass.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
