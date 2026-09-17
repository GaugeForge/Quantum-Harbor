#!/bin/bash
# Verifier shell for adaptive_clustered_clbcs_h2o (rubric mode, separate verifier).
# Collects qsim evidence into the artifacts dir and writes a 0/1 reward driven by
# five explicit checks: Track-A quality, Track-B quality, locked-request binding,
# energy readback, and cluster-uncertainty readback. The full tiered report lives
# in score_report.json.
#
# Disposition contract: reward 1 = the science passed;
# reward 0 = a well-formed execution whose science was rejected (an absent final
# submission is model-owned); reserved exit 2 with NO reward file =
# infrastructure/verifier failure -- including a malformed existing qsim-owned
# final envelope, a missing qsim mount, a torn evidence transfer, a scorer crash
# or signal, or a failed reward write.
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
rm -f "$ARTIFACTS_DIR/score_report.json"

LOG_FILE="$ARTIFACTS_DIR/experiment_log.jsonl"
ANS_FILE="$ARTIFACTS_DIR/final_answer.json"
PYTHON_BIN="${QIQCBENCH_PYTHON_BIN:-python3}"
SCORER_SCRIPT="$(dirname "$0")/score_clbcs.py"
TRIAL_ROOT="${QIQCBENCH_EVAL_TRIAL_ROOT:-/logs}"

if [ ! -f "$ANS_FILE" ]; then
    echo "No final_answer.json found." >&2
fi

SCORER_STATUS=0
"$PYTHON_BIN" "$SCORER_SCRIPT" "$LOG_FILE" "$ANS_FILE" "$ARTIFACTS_DIR" \
    "$REWARD_FILE" "$TRIAL_ROOT" || SCORER_STATUS=$?
# Only 0=pass and 1=explicit model rejection are score-bearing.  Any other exit
# (an uncaught exception, a signal, an unexpected code) is infrastructure.
if [ "$SCORER_STATUS" -ne 0 ] && [ "$SCORER_STATUS" -ne 1 ]; then
    rm -f "$REWARD_FILE"
    echo "Verifier infrastructure failure (scorer status $SCORER_STATUS)." >&2
    exit 2
fi

# Python also exits 1 on an uncaught exception, so status 1 alone cannot tell a
# crashed verifier from an assessed rejection.  An assessment of a real
# rejection always publishes its report first, including a genuine no-submit
# rejection; a scorer that died on a stale image or a broken mount does not, and
# that must not be recorded as the model's scientific failure.  An absent
# final_answer.json is no longer exempt: an accepted submit event
# with no file is a torn delivery, so absence alone no longer proves the failure
# is the model's.
if [ "$SCORER_STATUS" -eq 1 ] && [ ! -f "$ARTIFACTS_DIR/score_report.json" ]; then
    rm -f "$REWARD_FILE"
    echo "Verifier infrastructure failure: rejecting scorer emitted no report." >&2
    exit 2
fi

# A canonical v2 writer owns immutable reward bytes when it published an
# artifact set; otherwise this shell records the rejection itself.
if [ ! -f "$ARTIFACTS_DIR/verification_artifact_set.json" ] && [ "$SCORER_STATUS" -eq 1 ]; then
    if ! echo "0" > "$REWARD_FILE"; then
        rm -f "$REWARD_FILE"
        echo "Verifier infrastructure failure: the reward could not be written." >&2
        exit 2
    fi
fi

# A score-bearing run must leave exactly one well-formed verdict; a missing or
# unreadable reward means the write path broke, which is infrastructure.
REWARD_VALUE="$(cat "$REWARD_FILE" 2>/dev/null)" || REWARD_VALUE=""
if [ "$REWARD_VALUE" != "0" ] && [ "$REWARD_VALUE" != "1" ]; then
    rm -f "$REWARD_FILE"
    echo "Verifier infrastructure failure: no well-formed reward was produced." >&2
    exit 2
fi

if [ "$SCORER_STATUS" -eq 0 ]; then
    echo "Verifier passed (Pass gate cleared)."
else
    echo "Verifier rejected submission (Pass gate not cleared)." >&2
fi
