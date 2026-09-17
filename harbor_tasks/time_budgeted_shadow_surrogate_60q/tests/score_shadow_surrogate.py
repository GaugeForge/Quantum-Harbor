#!/usr/bin/env python3
"""Harbor verifier entry for time_budgeted_shadow_surrogate_60q.

Thin wrapper over the canonical scorer in
``qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q.scorer``
(this image installs qiqcbench, so no scorer logic is mirrored here).

Exit codes: 0 pass, 1 scientific fail / no submission (reward written),
3 infrastructure failure (missing runtime source, corrupt/empty evidence log,
public-material mismatch, frozen-source mismatch, hidden-model
commitment mismatch, artifact/log binding violation, reward-write failure,
crash; NO reward written).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

INFRA_EXIT = 3

# Runtime files are ferried from qsim. Independent frozen source bytes remain
# in the verifier image; missing ferry inputs never fall back to that image.
PUBLIC_DIR_DEFAULT = "/app/configs/task_materials/time_budgeted_shadow_surrogate_60q/public"
HIDDEN_DEVICE_DEFAULT = "/app/configs/devices/boundedgate_60q_v0.hidden.example.yaml"
PUBLIC_DEVICE_DEFAULT = "/app/configs/devices/boundedgate_60q_v0.public.yaml"
EXPECTED_CONFIGS_DEFAULT = "/app/qiqcbench_configs"
PUBLIC_JOB_RESULTS_SOURCE_DEFAULT = "/qsim_logs/public_job_results"
TASK_ID = "time_budgeted_shadow_surrogate_60q"
EXPECTED_PUBLIC_BUDGET = {
    "total_shot_budget": 315_000,
    "max_jobs": 40,
    "max_blocks_per_job": 64,
    "max_settings_per_job": 192,
    "max_shots_per_setting": 4_000,
    "max_ingress_request_bytes": 8 * 1024 * 1024,
    "max_metadata_calls": 256,
    "max_job_result_polls": 4_096,
    "max_sealed_holdout_reveals": 4,
    "max_final_answer_serialized_bytes": 4 * 1024 * 1024,
    "max_final_answer_submissions": 4,
    "max_answer_string_characters": 16_384,
}


def _reject_json_constant(value: str):
    raise ValueError(f"non-finite JSON constant {value!r} is forbidden")


def _reject_duplicate_object_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r} is forbidden")
        result[key] = value
    return result


def _load_strict_json_object(path: Path, *, owner: str) -> dict:
    """Load one finite JSON object without duplicate-key ambiguity."""

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_object_pairs,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{owner} must be a JSON object")
    return payload


def _hidden_device_source() -> Path:
    return Path(os.environ.get("QIQCBENCH_SHADOW_HIDDEN_DEVICE_FILE", HIDDEN_DEVICE_DEFAULT))


def _verify_current_context() -> None:
    from qiqcbench.eval.contracts.official_runtime import assert_current_repository_runtime_edition
    from qiqcbench.eval.verifier.context import load_verifier_run_context

    context = load_verifier_run_context()
    if context is None:
        return
    if context.task_id != TASK_ID or context.task_edition_runtime_binding is not None:
        raise ValueError("legacy Shadow runtime-manifest contexts are retired for new scoring")
    assert_current_repository_runtime_edition(TASK_ID, context.attempt.task_edition_id)


def _load_baked_hidden_config():
    from ruamel.yaml import YAML

    from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.device import (
        HiddenBoundedgateConfig,
    )

    hidden_path = _hidden_device_source()
    if not hidden_path.is_file():
        raise FileNotFoundError(f"baked hidden device config missing: {hidden_path}")
    with hidden_path.open() as f:
        return HiddenBoundedgateConfig.model_validate(YAML(typ="safe").load(f))


def _load_baked_public_config():
    from ruamel.yaml import YAML

    from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.device import (
        PublicBoundedgateSpec,
    )

    public_path = Path(os.environ.get("QIQCBENCH_SHADOW_PUBLIC_DEVICE_FILE", PUBLIC_DEVICE_DEFAULT))
    if not public_path.is_file():
        raise FileNotFoundError(f"baked public device config missing: {public_path}")
    with public_path.open() as f:
        return PublicBoundedgateSpec.model_validate(YAML(typ="safe").load(f))


def _verify_baked_task_contract(hidden, public) -> str | None:
    """Require the verifier's typed device bundle to match this task revision."""
    from qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q.scorer import (
        DEGREE_BOUND,
        DEVICE_ID,
        INPUT_DIM,
        N_QUBITS,
    )

    if (
        hidden.device_id != DEVICE_ID
        or hidden.input_dimension != INPUT_DIM
        or hidden.n_qubits != N_QUBITS
        or public.schema_version != 3
        or public.device_id != DEVICE_ID
        or public.input_dimension != INPUT_DIM
        or public.n_qubits != N_QUBITS
        or public.max_rotation_gates != 18
        or public.max_harmonic_degree_per_angle != DEGREE_BOUND
        or public.measurement_return != "bitstring"
    ):
        return "baked public/hidden device identity does not match the task contract"
    if public.budget.model_dump() != EXPECTED_PUBLIC_BUDGET:
        return "baked public resource budget does not match the task contract"

    hidden_challenge = hidden.target_challenge
    public_challenge = public.target_challenge
    if hidden_challenge is None or public_challenge is None:
        return "baked public/hidden device lacks the sealed target challenge"
    if (
        public_challenge.challenge_id != hidden_challenge.challenge_id
        or public_challenge.commitment_scheme != hidden_challenge.commitment_scheme
        or public_challenge.target_count != len(hidden_challenge.target_inputs)
        or public_challenge.input_dimension != INPUT_DIM
        or public_challenge.target_commitment_sha256 != hidden_challenge.target_commitment_sha256
    ):
        return "baked public and hidden sealed-target contracts do not match"
    return None


def _verify_hidden_commitment(events: list[dict], hidden) -> str | None:
    """Bind the runtime hidden model to THIS verifier's baked hidden config.

    Execution-bound runs inject a private ``VerifierRunContext``
    (QIQCBENCH_VERIFIER_CONTEXT); the commitment secret is DERIVED from it —
    the same derivation the materializer used for the qsim service — so an
    execution-bound run can never silently skip this check. qsim logged an
    execution-bound HMAC over the hidden config it actually served
    (qsim_bootstrap_context); recomputing it over the baked config catches a
    stale verifier image scoring a different edition than the agent measured.
    Without a verifier context (local/dev), a directly configured
    QIQCBENCH_HIDDEN_COMMITMENT_SECRET still validates; with neither, an
    execution-bound evidence log is itself an infra fault.
    """
    from qiqcbench.eval.verifier.context import load_verifier_run_context
    from qiqcbench.eval.verifier.hidden_commitment import derive_hidden_commitment_secret
    from qiqcbench.qsim.execution_context import (
        EXECUTION_CONTEXT_SCHEMA_VERSION,
        validate_execution_context_id,
        validate_hidden_commitment_secret,
    )
    from qiqcbench.qsim.hidden_commitment import hidden_device_model_commitment

    bootstrap_events = [
        (event_index, event)
        for event_index, event in enumerate(events)
        if event.get("action") == "qsim_bootstrap_context"
    ]
    context = load_verifier_run_context()
    if context is not None:
        expected_context_id = context.execution.qsim_evidence_nonce
        secret = derive_hidden_commitment_secret(context)
        binding_label = "official"
    else:
        secret = os.environ.get("QIQCBENCH_HIDDEN_COMMITMENT_SECRET")
        expected_context_id = os.environ.get("QIQCBENCH_SHADOW_EXPECTED_EXECUTION_CONTEXT_ID")
        if bool(secret) != bool(expected_context_id):
            return (
                "direct verifier execution binding requires both the hidden commitment "
                "secret and expected execution context ID"
            )
        if not secret:
            if bootstrap_events or any("execution_context_id" in event for event in events):
                return (
                    "execution-bound qsim evidence but no verifier context binding is "
                    "configured: refusing to skip the equality proof"
                )
            return None
        try:
            secret = validate_hidden_commitment_secret(secret)
            expected_context_id = validate_execution_context_id(expected_context_id)
        except ValueError as exc:
            return f"invalid direct verifier execution binding: {exc}"
        binding_label = "direct"

    if len(bootstrap_events) != 1 or bootstrap_events[0][0] != 0:
        return f"{binding_label} qsim bootstrap must be the unique first event"
    bootstrap = bootstrap_events[0][1]
    expected_bootstrap_keys = {
        "ts",
        "action",
        "context_schema_version",
        "execution_context_id",
        "public_device_model_commitment",
        "hidden_device_model_commitment",
    }
    if set(bootstrap) != expected_bootstrap_keys:
        return (
            f"{binding_label} qsim bootstrap must contain exactly {sorted(expected_bootstrap_keys)}"
        )
    if not isinstance(bootstrap["ts"], str) or not bootstrap["ts"]:
        return f"{binding_label} qsim bootstrap timestamp must be a non-empty string"
    context_schema_version = bootstrap["context_schema_version"]
    if (
        type(context_schema_version) is not int
        or context_schema_version != EXECUTION_CONTEXT_SCHEMA_VERSION
    ):
        return f"{binding_label} qsim bootstrap has an unsupported context schema version"
    if bootstrap["execution_context_id"] != expected_context_id:
        return "qsim bootstrap execution binding disagrees with the verifier context"
    if any(event.get("execution_context_id") != expected_context_id for event in events[1:]):
        return "qsim event execution binding disagrees with the verifier context"

    logged = bootstrap.get("hidden_device_model_commitment")
    expected = hidden_device_model_commitment(hidden, task_id=TASK_ID, commitment_secret=secret)
    if logged != expected:
        return (
            "runtime hidden-model commitment does not match this verifier's baked "
            "hidden config: the agent measured a DIFFERENT edition than is being scored"
        )
    return None


def _verify_public_commitment(events: list[dict], public) -> str | None:
    """Bind an execution bootstrap to this verifier's baked public model."""
    from qiqcbench.qsim.hidden_commitment import public_device_model_commitment

    bootstraps = [e for e in events if e.get("action") == "qsim_bootstrap_context"]
    if not bootstraps:
        # Local development may run without an execution context. Official
        # evidence is required to carry a bootstrap and is checked below by
        # the hidden-commitment verifier as well.
        return None
    if len(bootstraps) != 1:
        return "qsim evidence requires exactly one qsim_bootstrap_context event"
    logged = bootstraps[0].get("public_device_model_commitment")
    expected = public_device_model_commitment(public, task_id=TASK_ID)
    if logged != expected:
        return (
            "runtime public-device commitment does not match this verifier's baked "
            "public config: the agent saw a DIFFERENT budget or device contract"
        )
    return None


def _verify_public_materials(hidden) -> None:
    from qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q import construction

    public_dir = Path(os.environ.get("QIQCBENCH_SHADOW_PUBLIC_DIR", PUBLIC_DIR_DEFAULT))
    index = _load_strict_json_object(public_dir / "observable_index.json", owner="observable index")
    if index != construction.build_public_observable_index():
        raise ValueError("public observable index does not match the task's semantic contract")
    challenge = _load_strict_json_object(
        public_dir / "target_challenge.json", owner="target challenge"
    )
    expected = construction.build_public_target_challenge_from_metadata(
        hidden.target_challenge.model_dump(mode="json")
    )
    if challenge != expected:
        raise ValueError("public target challenge does not match the runtime device")


def _verify_copied_public_job_results(events: list[dict], report_dir: Path) -> dict[str, dict]:
    """Validate qsim raw evidence, copy its closed manifest, then revalidate."""
    from qiqcbench.qsim.actions.common import PUBLISHED_RESULT_SUBDIR
    from qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q.scorer import (
        MAX_PUBLIC_JOB_RESULT_BYTES,
        verify_public_job_result_artifacts,
    )

    # Both env vars name the ``public_job_results`` tree ROOT, matching the
    # qsim mount point test.sh exports. Delivery artifacts live one level below
    # it, so the read, the write, and the mkdir all derive that level
    # from the producer's constant rather than restating it.
    source_dir = Path(
        os.environ.get(
            "QIQCBENCH_SHADOW_PUBLIC_JOB_RESULTS_SOURCE_DIR",
            PUBLIC_JOB_RESULTS_SOURCE_DEFAULT,
        )
    )
    copied_dir = Path(
        os.environ.get(
            "QIQCBENCH_SHADOW_PUBLIC_JOB_RESULTS_DIR",
            str(report_dir / "public_job_results"),
        )
    )
    source_manifest = verify_public_job_result_artifacts(events, source_dir)
    if copied_dir.exists() or copied_dir.is_symlink():
        raise ValueError("public_job_results destination must be a fresh path")
    copied_dir.mkdir(mode=0o755, parents=False)
    copied_results_dir = copied_dir / PUBLISHED_RESULT_SUBDIR
    copied_results_dir.mkdir(mode=0o755, parents=False)
    for entry in source_manifest.values():
        relative_path = Path(entry["relative_path"])
        source = source_dir / PUBLISHED_RESULT_SUBDIR / relative_path.name
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(source, flags), "rb") as source_file:
            artifact_bytes = source_file.read(MAX_PUBLIC_JOB_RESULT_BYTES + 1)
        if (
            len(artifact_bytes) != entry["size_bytes"]
            or hashlib.sha256(artifact_bytes).hexdigest() != entry["sha256"]
        ):
            raise ValueError("qsim public job-result artifact changed before copy")
        target = copied_results_dir / relative_path.name
        with target.open("xb") as target_file:
            target_file.write(artifact_bytes)
        target.chmod(0o644)
    copied_manifest = verify_public_job_result_artifacts(events, copied_dir)
    missing = sorted(set(source_manifest) - set(copied_manifest))
    undeclared = sorted(set(copied_manifest) - set(source_manifest))
    if missing or undeclared:
        raise ValueError(
            f"public_job_results copy is not closed: missing={missing}, undeclared={undeclared}"
        )
    for name in sorted(source_manifest):
        source = source_manifest[name]
        copied = copied_manifest[name]
        if (
            source["relative_path"] != copied["relative_path"]
            or source["size_bytes"] != copied["size_bytes"]
            or source["sha256"] != copied["sha256"]
        ):
            raise ValueError(
                f"copied public job-result artifact {name!r} differs in confined path, "
                "byte size, or SHA-256 from the qsim source"
            )
    return copied_manifest


def _write_or_infra(path: Path, text: str) -> bool:
    """Reward/report writes must not fail silently: a failed write is infra."""
    try:
        path.write_text(text)
        return True
    except OSError:
        traceback.print_exc()
        return False


def _verify_resource_usage(events: list[dict], public, final_answer: dict | None) -> dict[str, int]:
    """Recompute every accepted action meter against the baked public contract."""
    from qiqcbench.qsim.actions.common import (
        _validate_answer_string_resources,
        compact_final_answer_bytes,
    )

    budget = public.budget
    metadata_actions = {"list_devices", "get_device_spec", "get_lab_notebook"}
    metadata_events = [event for event in events if event.get("action") in metadata_actions]
    metadata_counters: list[int] = []
    for event in metadata_events:
        counter = event.get("metadata_calls_used")
        maximum = event.get("max_metadata_calls")
        if type(counter) is not int or counter <= 0:
            raise ValueError("logged metadata call has an invalid cumulative counter")
        if type(maximum) is not int or maximum != budget.max_metadata_calls:
            raise ValueError("logged metadata-call max disagrees with the baked public contract")
        metadata_counters.append(counter)
    metadata_calls = len(metadata_events)
    if sorted(metadata_counters) != list(range(1, metadata_calls + 1)):
        raise ValueError("logged metadata-call counters do not exactly recount accepted calls")
    if metadata_calls > budget.max_metadata_calls:
        raise ValueError("accepted metadata calls exceed the baked public contract")

    basis_job_ids = {
        event.get("job_id")
        for event in events
        if event.get("action") == "submit_basis_shots" and isinstance(event.get("job_id"), str)
    }
    poll_events = [event for event in events if event.get("action") == "get_job_result"]
    poll_counters: list[int] = []
    for event in poll_events:
        if event.get("job_id") not in basis_job_ids:
            raise ValueError("logged job-result poll is not bound to a basis-shot submission")
        counter = event.get("job_result_polls_used")
        maximum = event.get("max_job_result_polls")
        if type(counter) is not int or counter <= 0:
            raise ValueError("logged job-result poll has an invalid cumulative counter")
        if type(maximum) is not int or maximum != budget.max_job_result_polls:
            raise ValueError("logged job-result poll max disagrees with the baked public contract")
        poll_counters.append(counter)
    polls = len(poll_events)
    # Poll reservations can complete and append out of order across threads,
    # but their run-global counter values must still be exactly 1..N.
    if sorted(poll_counters) != list(range(1, polls + 1)):
        raise ValueError("logged job-result poll counters do not exactly recount accepted polls")

    reveal_events = [event for event in events if event.get("action") == "sealed_holdout_reveal"]
    for expected_counter, event in enumerate(reveal_events, start=1):
        counter = event.get("sealed_holdout_reveals_used")
        maximum = event.get("max_sealed_holdout_reveals")
        if type(counter) is not int or counter != expected_counter:
            raise ValueError("logged sealed-reveal counters do not exactly recount reveals")
        if type(maximum) is not int or maximum != budget.max_sealed_holdout_reveals:
            raise ValueError("logged sealed-reveal max disagrees with the baked public contract")
    reveals = len(reveal_events)
    if reveals > budget.max_sealed_holdout_reveals:
        raise ValueError("accepted sealed reveals exceed the baked public contract")

    final_events = [event for event in events if event.get("action") == "submit_final_answer"]
    final_submissions = len(final_events)
    if polls > budget.max_job_result_polls:
        raise ValueError("accepted job-result polls exceed the baked public contract")
    if final_submissions > budget.max_final_answer_submissions:
        raise ValueError("accepted final submissions exceed the baked public contract")

    for expected_counter, event in enumerate(final_events, start=1):
        counter = event.get("final_answer_submissions_used")
        maximum = event.get("max_final_answer_submissions")
        serialized_bytes = event.get("final_answer_serialized_bytes")
        if type(counter) is not int or counter != expected_counter:
            raise ValueError("logged final-answer counters do not exactly recount submissions")
        if type(maximum) is not int or maximum != budget.max_final_answer_submissions:
            raise ValueError("logged final-answer max disagrees with the baked public contract")
        if event.get("tool") != "submit_final_answer" or event.get("task_id") != TASK_ID:
            raise ValueError("logged final-answer event has an invalid tool/task binding")
        answer = event.get("answer")
        if not isinstance(answer, dict):
            raise ValueError("logged final-answer payload is not an object")
        _validate_answer_string_resources(
            answer,
            max_characters=budget.max_answer_string_characters,
        )
        recomputed_bytes = len(
            compact_final_answer_bytes({"schema_version": 2, "task_id": TASK_ID, "answer": answer})
        )
        if type(serialized_bytes) is not int or serialized_bytes != recomputed_bytes:
            raise ValueError("logged final-answer byte count disagrees with its payload")
        if recomputed_bytes > budget.max_final_answer_serialized_bytes:
            raise ValueError("logged compact final-answer bytes exceed the baked public contract")

    final_bytes = 0
    if final_answer is not None:
        if not isinstance(final_answer, dict):
            raise ValueError("final-answer artifact is not an object")
        answer = final_answer.get("answer")
        if isinstance(answer, dict):
            _validate_answer_string_resources(
                answer,
                max_characters=budget.max_answer_string_characters,
            )
        final_bytes = len(compact_final_answer_bytes(final_answer))
        if final_bytes > budget.max_final_answer_serialized_bytes:
            raise ValueError("compact final-answer bytes exceed the baked public contract")
    return {
        "metadata_calls_used": metadata_calls,
        "metadata_calls_max": budget.max_metadata_calls,
        "job_result_polls_used": polls,
        "job_result_polls_max": budget.max_job_result_polls,
        "sealed_holdout_reveals_used": reveals,
        "sealed_holdout_reveals_max": budget.max_sealed_holdout_reveals,
        "final_answer_submissions_used": final_submissions,
        "final_answer_submissions_max": budget.max_final_answer_submissions,
        "final_answer_compact_bytes": final_bytes,
        "final_answer_compact_bytes_max": budget.max_final_answer_serialized_bytes,
        "mcp_ingress_request_bytes_max": budget.max_ingress_request_bytes,
        "answer_string_characters_max": budget.max_answer_string_characters,
    }


def _score_report(
    *,
    disposition: str,
    passed: bool,
    gates: dict,
    metrics: dict,
    reject_reasons: list[str],
    resource_usage: dict[str, int],
    offload_manifest: dict[str, dict],
) -> dict:
    """Stable task report envelope for model-attributed dispositions."""
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "disposition": disposition,
        "passed": passed,
        "reward_binary": 1 if passed else 0,
        "gates": gates,
        "metrics": metrics,
        "reject_reasons": reject_reasons,
        "resource_usage": resource_usage,
        "public_job_results_manifest": offload_manifest,
    }


def _verify_no_answer_lifecycle(events: list[dict], truth: dict) -> None:
    """Validate trusted experiment evidence before assigning no-answer."""
    from qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q.scorer import (
        verify_lifecycle_evidence,
    )

    verify_lifecycle_evidence(events, truth)


def main() -> int:
    if len(sys.argv) != 5:
        print("usage: score_shadow_surrogate.py LOG ANS REWARD REPORT_DIR", file=sys.stderr)
        return INFRA_EXIT
    log_path, ans_path, reward_path, report_dir = (Path(p) for p in sys.argv[1:5])
    try:
        # Direct invocations must also clear a stale verdict before any failure.
        reward_path.unlink(missing_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        _verify_current_context()
        from qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q.runtime import (
            build_runtime_truth,
            verify_runtime_sources,
        )

        verify_runtime_sources(
            hidden_path=_hidden_device_source(),
            public_path=Path(
                os.environ.get("QIQCBENCH_SHADOW_PUBLIC_DEVICE_FILE", PUBLIC_DEVICE_DEFAULT)
            ),
            public_dir=Path(os.environ.get("QIQCBENCH_SHADOW_PUBLIC_DIR", PUBLIC_DIR_DEFAULT)),
            expected_configs=Path(
                os.environ.get("QIQCBENCH_SHADOW_EXPECTED_CONFIGS", EXPECTED_CONFIGS_DEFAULT)
            ),
        )
    except Exception:
        traceback.print_exc()
        return INFRA_EXIT

    try:
        from qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q.scorer import (
            load_log_events,
            score_submission,
            verify_backend_result_disposition,
        )

        events = load_log_events(log_path)
        verify_backend_result_disposition(events)
    except Exception:
        traceback.print_exc()
        return INFRA_EXIT

    try:
        offload_manifest = _verify_copied_public_job_results(events, report_dir)
    except Exception:
        traceback.print_exc()
        return INFRA_EXIT
    if not _write_or_infra(
        report_dir / "public_job_results_manifest.json",
        json.dumps(offload_manifest, indent=2, sort_keys=True) + "\n",
    ):
        return INFRA_EXIT

    try:
        hidden = _load_baked_hidden_config()
        public = _load_baked_public_config()
        _verify_public_materials(hidden)
        binding_error = (
            _verify_baked_task_contract(hidden, public)
            or _verify_public_commitment(events, public)
            or _verify_hidden_commitment(events, hidden)
        )
    except Exception:
        traceback.print_exc()
        return INFRA_EXIT
    if binding_error is not None:
        print(binding_error, file=sys.stderr)
        return INFRA_EXIT

    try:
        truth = build_runtime_truth(hidden)
        final_answer = (
            _load_strict_json_object(ans_path, owner="final-answer artifact")
            if ans_path.is_file()
            else None
        )
        resource_usage = _verify_resource_usage(events, public, final_answer)
    except Exception:
        traceback.print_exc()
        return INFRA_EXIT

    if not ans_path.is_file():
        # No submission: distinguish a model that ran and never answered
        # (reward 0) from broken bootstrap or trusted lifecycle evidence
        # (infra, no reward).
        if not events:
            print("no submission AND empty evidence log: bootstrap suspect", file=sys.stderr)
            return INFRA_EXIT
        if any(e.get("action") == "submit_final_answer" for e in events):
            # qsim logged a submission whose artifact is missing: the
            # evidence channel itself is broken, never a model outcome.
            print("logged submit_final_answer but no artifact: channel corrupt", file=sys.stderr)
            return INFRA_EXIT
        try:
            _verify_no_answer_lifecycle(events, truth)
        except Exception:
            traceback.print_exc()
            return INFRA_EXIT
        # A LIVE run that never answered is a scientific failure of this
        # time-budgeted task (owner decision after E2E 3, 2026-08-27):
        # delivering calibrated predictions inside the clock is the task, so
        # every gate fails with a typed scientific reason and the attempt is
        # capability evidence. Only a run with no completed measurement at
        # all keeps the non-capability disposition (bootstrap / harness).
        # qsim logs a basis_shots_result event only when a job terminates; a
        # failed job carries status "failed", a completed one carries the
        # block summaries and counters (no status field).
        completed_jobs = sum(
            1
            for e in events
            if e.get("action") == "basis_shots_result" and e.get("status") != "failed"
        )
        sealed = any(e.get("action") == "sealed_holdout_reveal" for e in events)
        if completed_jobs:
            shots_logged = max(
                (
                    int(e.get("shots_used", 0))
                    for e in events
                    if e.get("action") == "basis_shots_result"
                    and isinstance(e.get("shots_used"), int)
                ),
                default=0,
            )
            reason = (
                "G1 accuracy: no prediction was submitted within the wall clock "
                f"({completed_jobs} completed job(s), {shots_logged} shots, "
                f"{'sealed' if sealed else 'not sealed'})"
            )
            report = _score_report(
                disposition="scored",
                passed=False,
                gates={
                    "g1_accuracy": False,
                    "g2_tail": False,
                    "g3_validity": False,
                },
                metrics={},
                reject_reasons=[reason, "no final_answer.json"],
                resource_usage=resource_usage,
                offload_manifest=offload_manifest,
            )
            report["failure_kind"] = "no_submission_within_wall_clock"
            message = "no submission within the wall clock: scored failure; reward 0"
        else:
            report = _score_report(
                disposition="model_failure",
                passed=False,
                gates={},
                metrics={},
                reject_reasons=["no final_answer.json"],
                resource_usage=resource_usage,
                offload_manifest=offload_manifest,
            )
            message = "no final_answer.json; reward 0"
        report_ok = _write_or_infra(
            report_dir / "score_report.json",
            json.dumps(report, indent=2, sort_keys=True) + "\n",
        )
        if not report_ok or not _write_or_infra(reward_path, "0\n"):
            return INFRA_EXIT
        print(message, file=sys.stderr)
        return 1

    try:
        assert final_answer is not None
        # verify_evidence_channel (called inside) RAISES on artifact/log
        # binding violations -> the reserved infra disposition below.
        result = score_submission(final_answer, events, truth)
    except Exception:
        traceback.print_exc()
        return INFRA_EXIT

    report = _score_report(
        disposition="scored",
        passed=result.passed,
        gates=result.gates,
        metrics=result.metrics,
        reject_reasons=result.reject_reasons,
        resource_usage=resource_usage,
        offload_manifest=offload_manifest,
    )
    report_ok = _write_or_infra(
        report_dir / "score_report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    if not report_ok or not _write_or_infra(reward_path, "1\n" if result.passed else "0\n"):
        return INFRA_EXIT
    print(json.dumps({"passed": result.passed, "gates": result.gates}, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
