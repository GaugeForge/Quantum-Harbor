#!/bin/bash
# Verifier shell for time_budgeted_shadow_surrogate_60q (rubric task).
# Collects qsim evidence into the artifacts dir and writes a 0/1 reward from
# the three disclosed gates (accuracy / tail / validity),
# recomputed from the actual bound apparatus, without a cached answer table.
# Infra failures (missing apparatus/evidence, scorer crash) leave NO reward file and exit
# nonzero so they are classified as non-model failures.
#
# Disposition contract: reward 1 = the science passed;
# reward 0 = a well-formed execution whose science was rejected (a missing or
# malformed final_answer.json is model-owned, since only submit_final_answer
# writes it); reserved exit 3 with NO reward file = infrastructure/verifier
# failure -- a missing qsim mount, a torn evidence transfer, a scorer crash or
# signal, a failed reward write, or a report-validation failure.
set -uo pipefail
# One BLAS/OpenMP thread for the verifier phase: unpinned OpenBLAS can overrun the 300 s cap.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

ARTIFACTS_DIR="${QIQCBENCH_VERIFIER_ARTIFACTS_DIR:-/logs/artifacts}"
VERIFIER_DIR="${QIQCBENCH_VERIFIER_DIR:-/logs/verifier}"
QSIM_LOGS_DIR="${QIQCBENCH_QSIM_LOGS_DIR:-/qsim_logs}"
mkdir -p "$ARTIFACTS_DIR" "$VERIFIER_DIR" || exit 3

# A stale pre-existing reward must never survive into this verdict — remove
# it FIRST so every later exit path (including infra) leaves no reward file
# unless this run wrote one.
rm -f "$VERIFIER_DIR/reward.txt" || exit 3

# Evidence collection must not fail silently: a missing evidence log or a
# failed copy is an infrastructure fault (reserved exit >= 3, no reward file),
# never a model-attributed reward 0.
if [ ! -f "$QSIM_LOGS_DIR/experiment_log.jsonl" ]; then
    echo "qsim evidence log missing: $QSIM_LOGS_DIR/experiment_log.jsonl" >&2
    exit 3
fi
cp "$QSIM_LOGS_DIR/experiment_log.jsonl" "$ARTIFACTS_DIR/" || exit 3
if [ -f "$QSIM_LOGS_DIR/final_answer.json" ]; then
    cp "$QSIM_LOGS_DIR/final_answer.json" "$ARTIFACTS_DIR/" || exit 3
fi
if [ ! -d "$QSIM_LOGS_DIR/public_job_results" ] || [ -L "$QSIM_LOGS_DIR/public_job_results" ]; then
    echo "qsim public_job_results evidence directory missing or invalid" >&2
    exit 3
fi
export QIQCBENCH_SHADOW_PUBLIC_JOB_RESULTS_SOURCE_DIR="$QSIM_LOGS_DIR/public_job_results"
export QIQCBENCH_SHADOW_PUBLIC_JOB_RESULTS_DIR="$ARTIFACTS_DIR/public_job_results"

LOG_FILE="$ARTIFACTS_DIR/experiment_log.jsonl"
ANS_FILE="$ARTIFACTS_DIR/final_answer.json"
SCORER_SCRIPT="$(dirname "$0")/score_shadow_surrogate.py"
PYTHON_BIN="${QIQCBENCH_PYTHON_BIN:-python3}"

SCORER_STATUS=0
"$PYTHON_BIN" "$SCORER_SCRIPT" "$LOG_FILE" "$ANS_FILE" \
    "$VERIFIER_DIR/reward.txt" "$ARTIFACTS_DIR" || SCORER_STATUS=$?

if [ "$SCORER_STATUS" -ge 3 ]; then
    # Reserved infra exit: guarantee NO reward file survives.
    rm -f "$VERIFIER_DIR/reward.txt"
    echo "Verifier infrastructure failure (status $SCORER_STATUS)." >&2
    exit "$SCORER_STATUS"
fi

# The scorer owns reward writing (statuses 0/1 always write one). A missing
# or unreadable reward here means the write path is broken: infra, never a
# silently-synthesized 0 and never a clean exit without a verdict.
REWARD_VALUE="$(cat "$VERIFIER_DIR/reward.txt" 2>/dev/null)" || REWARD_VALUE=""
if [ "$REWARD_VALUE" != "0" ] && [ "$REWARD_VALUE" != "1" ]; then
    rm -f "$VERIFIER_DIR/reward.txt"
    echo "Verifier reward file missing or unreadable after scoring: infra." >&2
    exit 3
fi

if [ "$REWARD_VALUE" = "1" ]; then
    echo "Verifier passed (all gates cleared)."
else
    echo "Verifier rejected submission (gate not cleared)." >&2
fi
