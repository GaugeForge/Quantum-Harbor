#!/bin/bash
# Verifier: collect qsim logs + final answer into /logs/artifacts and write a
# 0/1 reward for a true-scored Hamiltonian-learning submission.
#
# score_hamlearn.py reconstructs the evidence-owned probe budget from the
# logged probe_batch_result events, validates the answer, verifies the public
# material digests, recomputes the L-infinity coefficient error against the
# hidden omega_star, and emits score_report.json into the artifacts directory.
#
# Disposition contract: reward 1 = the science passed;
# reward 0 = a well-formed execution whose science was rejected. Never
# submitting a final answer is model-owned; a malformed persisted envelope is
# qsim-owned corruption because only qsim writes that file after validation.
# Reserved exit 2 with NO reward file = infrastructure/verifier failure -- a
# missing qsim mount, torn evidence transfer, scorer crash or signal, or failed
# reward write.
set -uo pipefail
# One BLAS/OpenMP thread for the verifier phase: unpinned OpenBLAS can overrun the 300 s cap.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

ARTIFACTS_DIR="${QIQCBENCH_VERIFIER_ARTIFACTS_DIR:-/logs/artifacts}"
VERIFIER_DIR="${QIQCBENCH_VERIFIER_DIR:-/logs/verifier}"
QSIM_LOGS_DIR="${QIQCBENCH_QSIM_LOGS_DIR:-/qsim_logs}"
REWARD_FILE="$VERIFIER_DIR/reward.txt"

if ! mkdir -p "$ARTIFACTS_DIR" "$VERIFIER_DIR"; then
    echo "Verifier infrastructure failure: output directories are unavailable." >&2
    exit 2
fi
# A stale verdict must never survive into this execution.
if ! rm -f "$REWARD_FILE"; then
    echo "Verifier infrastructure failure: a stale reward could not be removed." >&2
    exit 2
fi
# qsim owns the evidence tree: a missing mount or a torn transfer is
# infrastructure failure, never proof that the model failed scientifically.
if [ ! -d "$QSIM_LOGS_DIR" ] || [ -L "$QSIM_LOGS_DIR" ]; then
    echo "Verifier infrastructure failure: qsim log mount is unavailable or invalid." >&2
    exit 2
fi
if ! cp -R "$QSIM_LOGS_DIR"/. "$ARTIFACTS_DIR"/; then
    echo "Verifier infrastructure failure: qsim evidence transfer failed." >&2
    exit 2
fi
# A qsim source must never supply a stale verifier-owned summary.
if ! rm -f "$ARTIFACTS_DIR/score_report.json"; then
    rm -f "$REWARD_FILE"
    echo "Verifier infrastructure failure: a stale score report could not be removed." >&2
    exit 2
fi

LOG_FILE="$ARTIFACTS_DIR/experiment_log.jsonl"
ANS_FILE="$ARTIFACTS_DIR/final_answer.json"
PYTHON_BIN="${QIQCBENCH_PYTHON_BIN:-python3}"
SCORER_SCRIPT="$(dirname "$0")/score_hamlearn.py"

if [ ! -f "$LOG_FILE" ]; then
    echo "No experiment_log.jsonl found." >&2
fi
if [ ! -f "$ANS_FILE" ]; then
    echo "No final_answer.json found." >&2
fi

SCORER_STATUS=0
"$PYTHON_BIN" "$SCORER_SCRIPT" "$LOG_FILE" "$ANS_FILE" "$ARTIFACTS_DIR" || SCORER_STATUS=$?
if [ "$SCORER_STATUS" -ne 0 ] && [ "$SCORER_STATUS" -ne 1 ]; then
    rm -f "$REWARD_FILE"
    echo "Verifier infrastructure failure (scorer status $SCORER_STATUS)." >&2
    exit 2
fi

# Status alone is not a verdict: Python also exits 1 on an uncaught exception,
# and a broken scorer can exit 0 without publishing an assessment. Every
# completed pass or rejection writes a new regular report first, including a
# healthy no-submission rejection.
REPORT_FILE="$ARTIFACTS_DIR/score_report.json"
if [ ! -f "$REPORT_FILE" ] || [ -L "$REPORT_FILE" ]; then
    rm -f "$REWARD_FILE"
    echo "Verifier infrastructure failure: scorer emitted no regular report." >&2
    exit 2
fi
if ! "$PYTHON_BIN" - "$REPORT_FILE" "$SCORER_STATUS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
status = int(sys.argv[2])
try:
    report = json.loads(path.read_text(encoding="utf-8"))
except (OSError, UnicodeDecodeError, json.JSONDecodeError):
    raise SystemExit(1)
expected_reward = int(status == 0)
result = report.get("result") if isinstance(report, dict) else None
result_matches = isinstance(result, dict) and result.get("binary_pass") is bool(expected_reward)
if (
    not isinstance(report, dict)
    or report.get("task_id") != "time_budgeted_hamlearn_10q"
    or type(report.get("reward_binary")) is not int
    or report.get("reward_binary") != expected_reward
    or (result is None and expected_reward == 1)
    or (result is not None and not result_matches)
):
    raise SystemExit(1)
PY
then
    rm -f "$REWARD_FILE"
    echo "Verifier infrastructure failure: score report contradicts scorer status." >&2
    exit 2
fi

if [ "$SCORER_STATUS" -eq 0 ]; then
    if ! echo "1" > "$REWARD_FILE"; then
        rm -f "$REWARD_FILE"
        echo "Verifier infrastructure failure: the reward could not be written." >&2
        exit 2
    fi
    echo "Verifier passed."
else
    if ! echo "0" > "$REWARD_FILE"; then
        rm -f "$REWARD_FILE"
        echo "Verifier infrastructure failure: the reward could not be written." >&2
        exit 2
    fi
    echo "Verifier rejected submission." >&2
fi
