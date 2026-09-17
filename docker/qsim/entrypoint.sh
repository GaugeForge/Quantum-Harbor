#!/usr/bin/env bash
set -euo pipefail

# --- qsim log surface reset -----------------------------------
# Harbor reuses ONE Docker volume across every attempt of a trial: the compose
# project name is per-trial (`<trial_name>__env`) and is unchanged on retry,
# and teardown runs `down` WITHOUT `--volumes` unless `environment.delete` is
# set. Meanwhile the host-side trial directory IS deleted
# (`shutil.rmtree` in harbor's retry loop). That asymmetry is the leak: a
# retried agent inherits the previous attempt's delivery surface, including
# truth-labelled job records it was never meant to see, while the evidence that
# the first attempt happened is gone.
#
# Clear the whole surface, not just `public_job_results/`: a retry must not
# inherit evidence, raw results, or delivery spool from a prior attempt. The
# agent's current read-only mount exposes only `public_job_results/`.
#
# Under harbor's own lifecycle this runs once per attempt and cannot destroy
# live evidence: harbor tears the environment down and then runs `down
# --remove-orphans` followed by `up --detach --wait` at the start of every
# attempt, so the entrypoint re-runs; the agent is only started afterwards; and
# no shipped bundle sets a restart policy, so neither a crash nor a healthcheck
# failure re-runs it mid-trial.
#
# It is NOT fail-closed against harbor's own cleanup failing. Harbor swallows
# errors from both the teardown `down` and the startup `down`, and `up` does
# not force recreation, so if BOTH fail compose can reuse a still-running qsim
# and this never runs. That path retains the previous attempt's surface -- the
# previous attempt's surface survives -- rather than doing anything harmful,
# and closing it would need a change in harbor.
#
# A manual `docker compose restart qsim` does re-run this, but such a restart
# already destroys the trial by dropping qsim's in-memory job state, so the
# reset is not what breaks it. Do not hand-restart the device container during
# a scored run.
mkdir -p /qsim_logs
if [ -d /qsim_logs ]; then
    # Content first: FILES and symlinks at any depth. `find` rather than a
    # `<dir>/*` glob, because the glob skips dot-prefixed entries and
    # `.delivery_spool/` holds the previous attempt's materialized-but-unpolled
    # artifacts. Errors are NOT swallowed here -- a surface whose content
    # cannot be cleared must stop the container rather than run blind.
    find /qsim_logs -mindepth 1 \( -type f -o -type l \) -exec rm -f {} +
    # Then the directories the previous attempt created. Their mere existence
    # is a signal: `.delivery_spool/` appears only once qsim has persisted a
    # completed result, and `public_job_results/{results,raw_results}/` only
    # once a terminal poll published one, so leaving them tells a retry what its
    # predecessor got as far as. Best effort, unlike the content pass above:
    # Bundles nest a second volume at /qsim_logs/public_job_results
    # (the poll-gated delivery split); a mount point CANNOT be rmdir'd,
    # and failing on that EBUSY under `set -e` would kill qsim before it serves
    # -- on every attempt, including the first.
    #
    # `-depth` makes this bottom-up: qsim publishes
    # delivery artifacts into `public_job_results/results/` and raw records into
    # `public_job_results/raw_results/`, both ORDINARY directories, so both are
    # emptied by the pass above and unlinked here. Only the mount root survives,
    # and it never holds more than those two entries.
    find /qsim_logs -mindepth 1 -depth -type d -exec rmdir {} + 2>/dev/null || true
    # Finally the timestamps of whatever survived -- in practice the nested
    # mount root, which cannot be removed. Deleting a predecessor's files bumps
    # its parent's mtime, so an unnormalized `stat` distinguishes "the previous
    # attempt published something here" from "it published nothing": the same
    # predecessor-stage signal as the directory names, one field further in.
    # `-t CCYYMMDDhhmm` rather than GNU's `-d @0`: the container is Debian but
    # the test drives these same lines on the host, and BSD touch rejects `-d @0`.
    find /qsim_logs -type d -exec touch -t 200001010000 {} + 2>/dev/null || true
    #
    # ext4 still never shrinks a directory inode when its entries are
    # unlinked, and the mount root still cannot be removed -- but the root no
    # longer accumulates one entry per job, because qsim publishes into
    # `public_job_results/results/` instead of into the root. The root holds at
    # most two subdirectory entries for the life of the volume, so it stays at
    # the single-block minimum and a `stat` of it no longer counts a
    # predecessor's results. The directory that DOES grow is ordinary and is
    # unlinked by the `rmdir` pass above, so a retry inherits a fresh inode.
    #
    # Still true and still deliberately not chased: inode number and birth
    # metadata reveal that the volume was REUSED, which `stats.n_retries`
    # already records. Neither reveals predecessor PROGRESS.
fi
# --- end qsim log surface reset --------------------------------------------

QIQCBENCH_TASK_ID="${QIQCBENCH_TASK_ID:-}"
QIQCBENCH_HIDDEN_PATH="${QIQCBENCH_HIDDEN_PATH:-}"
QSIM_DEVICE_ID="${QSIM_DEVICE_ID:-}"
CONVENTIONAL_HIDDEN_PATH="/app/hidden/hidden.yaml"

RESOLVED_DEVICE_ID="${QSIM_DEVICE_ID}"
if [ -n "${QIQCBENCH_TASK_ID}" ]; then
    RESOLVED_DEVICE_ID="$(python - <<'PY'
import os

from qiqcbench.qsim.tasks import load_task_spec

print(load_task_spec(os.environ["QIQCBENCH_TASK_ID"]).device_id)
PY
)"
fi

# Runtime-constructed hidden apparatus paths, if declared in qsim code, are
# initialized here before resolving the device. No released task declares one.
# This call is inert unless a task is declared; `set -e` preserves fail-closed
# startup on construction errors and missing configured files.
if [ -n "${QIQCBENCH_TASK_ID}" ]; then
    python -m qiqcbench.qsim.hidden_dynamics.runtime_construction \
        --task-id "${QIQCBENCH_TASK_ID}" \
        --hidden-path "${QIQCBENCH_HIDDEN_PATH}"
fi

# Preserve the conventional per-run bind mount before using the local-development
# fallback. An explicitly configured instance path is authoritative and must fail closed.
if [ -z "${QIQCBENCH_HIDDEN_PATH}" ]; then
    if [ -f "${CONVENTIONAL_HIDDEN_PATH}" ]; then
        echo "[qsim] hidden config path unset, using mounted config" >&2
        QIQCBENCH_HIDDEN_PATH="${CONVENTIONAL_HIDDEN_PATH}"
    else
        echo "[qsim] hidden config path unset, using shipped example" >&2
        QIQCBENCH_HIDDEN_PATH="${QIQCBENCH_CONFIGS}/devices/${RESOLVED_DEVICE_ID}.hidden.example.yaml"
    fi
elif [ ! -f "${QIQCBENCH_HIDDEN_PATH}" ]; then
    echo "[qsim] configured hidden config does not exist: ${QIQCBENCH_HIDDEN_PATH}" >&2
    echo "[qsim] this task pins an operator-provisioned instance and fails closed on purpose." >&2
    echo "[qsim] provision the file before launching, e.g. write it to the repo path that the" >&2
    echo "[qsim] bundle mounts, or drop an instance at ${CONVENTIONAL_HIDDEN_PATH} and unset" >&2
    echo "[qsim] QIQCBENCH_HIDDEN_PATH. Do NOT delete the bundle's QIQCBENCH_HIDDEN_PATH line:" >&2
    echo "[qsim] that makes qsim fall back to the shipped development example and score against" >&2
    echo "[qsim] synthetic numbers." >&2
    exit 1
fi

export QIQCBENCH_HIDDEN_PATH

ARGS=(
    qiqcbench qsim serve-mcp
    --host "${QSIM_HOST}"
    --port "${QSIM_PORT}"
    --device-id "${RESOLVED_DEVICE_ID}"
    --hidden-path "${QIQCBENCH_HIDDEN_PATH}"
    --log-dir /qsim_logs
)

if [ -n "${QIQCBENCH_TASK_ID}" ]; then
    ARGS+=(--task-id "${QIQCBENCH_TASK_ID}")
fi

exec "${ARGS[@]}"
