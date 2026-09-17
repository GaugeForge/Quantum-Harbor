"""Read qsim's final-answer artifacts under the lifecycle contract qsim writes them by.

``submit_final_answer`` is the only writer of ``final_answer.json``, and it
appends the accepted ``submit_final_answer`` event in the same action, so the
file and the event are evidence about each other. A verifier that scores from
the ferried ``/qsim_logs`` tree must read them by that contract:

* a torn or unreadable log, a symlinked or non-regular answer file, an answer
  file that no accepted event explains, or one that disagrees with the latest
  accepted event is a qsim-owned artifact fault and raises
  :class:`QsimEvidenceFault` (the verifier's reserved non-model exit);
* an absent file with no accepted event is the model's own non-submission and
  resolves to ``None``;
* otherwise the accepted answer body is returned for scientific assessment.

Separate-mode verifier drivers apply these rules identically.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

from qiqcbench.qsim.core.wire import FinalAnswer

EXPERIMENT_LOG_NAME = "experiment_log.jsonl"
FINAL_ANSWER_NAME = "final_answer.json"


class QsimEvidenceFault(RuntimeError):
    """A qsim-owned artifact defect; never a verdict about the model."""


def load_qsim_events(log_path: Path) -> list[dict]:
    """Parse the qsim event log, raising :class:`QsimEvidenceFault` on any damage."""
    if log_path.is_symlink() or not log_path.is_file():
        raise QsimEvidenceFault("qsim experiment log is missing or not a regular file")
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise QsimEvidenceFault("qsim experiment log is unreadable") from exc
    events: list[dict] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            parsed = json.loads(line)
        except (json.JSONDecodeError, ValueError) as exc:
            raise QsimEvidenceFault(f"qsim experiment log line {line_number} is malformed") from exc
        if not isinstance(parsed, dict):
            raise QsimEvidenceFault(f"qsim experiment log line {line_number} is not an object")
        events.append(parsed)
    return events


def resolve_final_answer(
    *,
    log_dir: Path,
    task_id: str,
    events: list[dict],
    max_final_bytes: int | None,
) -> dict | None:
    """Return the accepted answer body, ``None`` for a true no-submission, or raise."""
    final_answer_path = log_dir / FINAL_ANSWER_NAME
    final_events = [
        event
        for event in events
        if event.get("action") == "submit_final_answer" and event.get("task_id") == task_id
    ]
    if final_answer_path.is_symlink():
        raise QsimEvidenceFault("qsim final-answer artifact is not a regular file")
    if not final_answer_path.is_file():
        if final_answer_path.exists() or final_events:
            raise QsimEvidenceFault("qsim final-answer lifecycle is incomplete")
        return None
    try:
        metadata = final_answer_path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise QsimEvidenceFault("qsim final-answer artifact is not a regular file")
        if max_final_bytes is not None:
            if (
                isinstance(max_final_bytes, bool)
                or not isinstance(max_final_bytes, int)
                or max_final_bytes < 1
            ):
                raise QsimEvidenceFault("public final-answer byte policy is invalid")
            if metadata.st_size > max_final_bytes:
                raise QsimEvidenceFault("qsim final-answer artifact exceeds its public byte limit")
            with final_answer_path.open("rb") as handle:
                raw_final = handle.read(max_final_bytes + 1)
            if len(raw_final) != metadata.st_size:
                raise QsimEvidenceFault("qsim final-answer artifact changed while reading")
        else:
            raw_final = final_answer_path.read_bytes()
        payload = json.loads(raw_final)
        final_answer = FinalAnswer.model_validate(payload)
    except Exception as exc:
        # Only qsim's atomic submit_final_answer action can create this file.
        # A torn, unreadable, or invalid envelope is therefore qsim-owned.
        raise QsimEvidenceFault("qsim final-answer artifact is malformed") from exc
    if final_answer.task_id != task_id or not final_events:
        raise QsimEvidenceFault("qsim final-answer artifact is not bound to an accepted task final")
    latest_final = final_events[-1]
    if latest_final.get("answer") != final_answer.answer:
        raise QsimEvidenceFault("qsim final-answer artifact disagrees with its latest logged final")
    return final_answer.answer


def public_final_answer_byte_limit(public: object) -> int | None:
    """The public spec's ``max_final_answer_serialized_bytes``, when it declares one."""
    public_budget = getattr(public, "budget", None) or getattr(public, "budgets", None)
    return getattr(public_budget, "max_final_answer_serialized_bytes", None)


__all__ = [
    "EXPERIMENT_LOG_NAME",
    "FINAL_ANSWER_NAME",
    "QsimEvidenceFault",
    "load_qsim_events",
    "public_final_answer_byte_limit",
    "resolve_final_answer",
]
