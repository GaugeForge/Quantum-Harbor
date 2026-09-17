"""Liveness heartbeat for the qsim sidecar.

A daemon thread appends one JSON line to ``<log_dir>/heartbeat.jsonl`` every
``HEARTBEAT_INTERVAL_S`` seconds, starting immediately. Host-side ingest reads
the trailing timestamp to distinguish "qsim's process was alive but its HTTP
transport stopped answering" from "qsim's process died": a SIGKILL leaves the
file present but stale, while a live server keeps beating from this plain
thread even when the event loop is starved. The line shape
``{"ts": <unix float>, "action": "qsim_heartbeat"}`` is a shared contract with
the host-side infrastructure classifier -- do not rename its fields.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

HEARTBEAT_FILENAME = "heartbeat.jsonl"
HEARTBEAT_INTERVAL_S = 30.0
HEARTBEAT_ACTION = "qsim_heartbeat"


def write_heartbeat_line(path: Path) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), "action": HEARTBEAT_ACTION}) + "\n")


def start_heartbeat_thread(
    log_dir: Path, interval_s: float = HEARTBEAT_INTERVAL_S
) -> threading.Thread:
    """Start the daemon beat thread and return it.

    The first line is written by the thread immediately, so the file exists as
    soon as qsim is up; the thread is never joined and holds no cleanup hook,
    so an uncontrolled death (SIGKILL, OOM) leaves the stale file behind as
    evidence rather than deleting it.
    """

    path = Path(log_dir) / HEARTBEAT_FILENAME

    def beat() -> None:
        while True:
            try:
                write_heartbeat_line(path)
            except Exception:
                # A failed beat must never take qsim down; the absence of fresh
                # lines is itself the diagnostic signal.
                pass
            time.sleep(interval_s)

    thread = threading.Thread(target=beat, name="qsim-heartbeat", daemon=True)
    thread.start()
    return thread
