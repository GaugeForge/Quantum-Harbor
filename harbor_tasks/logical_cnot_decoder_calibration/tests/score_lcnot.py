#!/usr/bin/env python3
"""Replay verifier entry for logical_cnot_decoder_calibration.

Runs inside the separate verifier image (FROM qiqcbench-qsim, + pymatching), so it
imports the in-tree hidden construction + scorer directly — the single source of truth.
It reconstructs the agent's submitted DEMs + decoder configs, replays both decoders on
FRESH hidden-noise shots (independent RNG), applies the two-gate rubric (d=5 memory
floor; CNOT battery average + per-entry cap), derives the ungated Lambda(3->5) and
logical Bell-fidelity witness figures, and emits score_report.json.
Exit 0 == reward 1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _write_report(artifacts: Path, report) -> None:
    artifacts.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.to_dict(), indent=2, allow_nan=False) + "\n"
    (artifacts / "score_report.json").write_text(payload, encoding="utf-8")


def _missing_answer_is_no_submit(log_path: Path) -> bool:
    """Return true only when qsim evidence contains no final submission or corruption."""

    if not log_path.is_file():
        return True
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError("qsim experiment log cannot be read") from exc
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError("qsim experiment log contains invalid JSON") from exc
        if not isinstance(event, dict):
            raise RuntimeError("qsim experiment log contains a non-object record")
        if (
            event.get("action", event.get("tool")) == "submit_final_answer"
            and event.get("task_id") == "logical_cnot_decoder_calibration"
        ):
            return False
    return True


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: score_lcnot.py <experiment_log.jsonl> <final_answer.json> <artifacts_dir>")
        return 2
    log_path, ans_path, artifacts = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "score_report.json").unlink(missing_ok=True)

    from qiqcbench.qsim.hidden_dynamics.logical_cnot_decoder_calibration import scorer as S

    if not ans_path.is_file():
        try:
            no_submit = _missing_answer_is_no_submit(log_path)
        except RuntimeError as exc:
            print(f"verifier evidence integrity failure: {exc}", file=sys.stderr)
            return 2
        if not no_submit:
            print(
                "verifier final-answer integrity failure: logged submission has no artifact",
                file=sys.stderr,
            )
            return 2
        report = S.model_failure_report("final_answer.json is missing")
        _write_report(artifacts, report)
        print("no final answer", file=sys.stderr)
        return 1
    try:
        answer = json.loads(ans_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        print("verifier final-answer integrity failure: invalid UTF-8 JSON", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"verifier final-answer read failure: {type(exc).__name__}", file=sys.stderr)
        return 2
    if not isinstance(answer, dict):
        print(
            "verifier final-answer integrity failure: top-level value is not an object",
            file=sys.stderr,
        )
        return 2
    if (
        answer.get("schema_version") != 2
        or answer.get("task_id") != "logical_cnot_decoder_calibration"
        or not isinstance(answer.get("answer"), dict)
    ):
        print("verifier final-answer integrity failure: invalid envelope", file=sys.stderr)
        return 2

    try:
        evidence = S.recover_experiment_evidence(log_path, answer)
        report = S.score(answer, budget_evidence=evidence)
    except Exception as exc:  # noqa: BLE001 - unexpected failures are verifier infrastructure
        print(f"verifier scoring failure: {type(exc).__name__}", file=sys.stderr)
        return 2

    try:
        _write_report(artifacts, report)
    except (OSError, TypeError, ValueError) as exc:
        print(f"verifier report-write failure: {type(exc).__name__}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                k: report.to_dict().get(k)
                for k in (
                    "reward",
                    "passed",
                    "memory_gate_passed",
                    "cnot_gate_passed",
                    "replayed_memory_rate_per_cycle",
                    "replayed_cnot_error_average",
                    "lambda_replayed",
                    "f_bell_lower_bound",
                )
            }
        )
    )
    return 0 if report.reward == 1 else 1


if __name__ == "__main__":
    try:
        status = main()
    except Exception as exc:  # pragma: no cover - process-level infrastructure boundary
        print(f"verifier infrastructure failure: {type(exc).__name__}", file=sys.stderr)
        status = 2
    sys.exit(status)
