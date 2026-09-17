#!/bin/bash
# Verifier shell for logical_cnot_decoder_calibration.
# Collects qsim evidence + final answer into the artifacts dir and writes a 0/1 reward.
# score_lcnot.py imports the in-tree hidden construction + fresh-shot replay scorer
# (qiqcbench.qsim.hidden_dynamics.logical_cnot_decoder_calibration) — single scorer
# implementation, no vendored copy to drift.
#
# Disposition contract: reward 1 = the science passed;
# reward 0 = a well-formed execution whose science was rejected (a missing or
# malformed final_answer.json is model-owned, since only submit_final_answer
# writes it); reserved exit 2 with NO reward file = infrastructure/verifier
# failure -- a missing qsim mount, a torn evidence transfer, a scorer crash or
# signal, a failed reward write, or a report-validation failure.
set -uo pipefail
# One BLAS/OpenMP thread for the verifier phase: unpinned OpenBLAS can overrun the 300 s cap.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

ARTIFACTS_DIR="${QIQCBENCH_VERIFIER_ARTIFACTS_DIR:-/logs/verifier/artifacts}"
VERIFIER_DIR="${QIQCBENCH_VERIFIER_DIR:-/logs/verifier}"
QSIM_LOGS_DIR="${QIQCBENCH_QSIM_LOGS_DIR:-/qsim_logs}"
mkdir -p "$ARTIFACTS_DIR" "$VERIFIER_DIR"
rm -f "$VERIFIER_DIR/reward.txt" "$ARTIFACTS_DIR/score_report.json"

if [ ! -d "$QSIM_LOGS_DIR" ]; then
    echo "Qsim evidence directory is missing." >&2
    exit 2
fi
if ! cp -r "$QSIM_LOGS_DIR"/. "$ARTIFACTS_DIR/"; then
    echo "Failed to copy qsim evidence into the verifier." >&2
    exit 2
fi
# A qsim source must never supply a stale verifier-owned summary.
rm -f "$ARTIFACTS_DIR/score_report.json"

LOG_FILE="$ARTIFACTS_DIR/experiment_log.jsonl"
ANS_FILE="$ARTIFACTS_DIR/final_answer.json"
PYTHON_BIN="${QIQCBENCH_PYTHON_BIN:-python3}"
SCORER_SCRIPT="$(dirname "$0")/score_lcnot.py"

if [ ! -f "$ANS_FILE" ]; then
    echo "No final_answer.json found." >&2
fi

SCORER_STATUS=0
"$PYTHON_BIN" "$SCORER_SCRIPT" "$LOG_FILE" "$ANS_FILE" "$ARTIFACTS_DIR" || SCORER_STATUS=$?
if [ "$SCORER_STATUS" -ne 0 ] && [ "$SCORER_STATUS" -ne 1 ]; then
    rm -f "$VERIFIER_DIR/reward.txt"
    echo "Verifier infrastructure failure." >&2
    exit 2
fi

REPORT_FILE="$ARTIFACTS_DIR/score_report.json"
EXPECTED_REWARD=0
if [ "$SCORER_STATUS" -eq 0 ]; then EXPECTED_REWARD=1; fi
if ! "$PYTHON_BIN" - "$REPORT_FILE" "$EXPECTED_REWARD" <<'PY'
import json
import sys


def reject_constant(value: str):
    raise ValueError(f"non-finite JSON constant {value}")


with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream, parse_constant=reject_constant)
if not isinstance(report, dict):
    raise SystemExit(1)
if report.get("task_id") != "logical_cnot_decoder_calibration":
    raise SystemExit(1)
if type(report.get("reward")) is not int or report["reward"] != int(sys.argv[2]):
    raise SystemExit(1)
if type(report.get("passed")) is not bool:
    raise SystemExit(1)
if report.get("disposition") not in {"scored", "model_failure"}:
    raise SystemExit(1)
PY
then
    rm -f "$VERIFIER_DIR/reward.txt"
    echo "Verifier report validation failure." >&2
    exit 2
fi

if [ "$SCORER_STATUS" -eq 0 ]; then
    echo "1" > "$VERIFIER_DIR/reward.txt"
    echo "Verifier passed."
else
    echo "0" > "$VERIFIER_DIR/reward.txt"
    echo "Verifier rejected submission." >&2
fi
