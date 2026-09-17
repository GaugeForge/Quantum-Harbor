"""Submission-time answer-shape prechecks for ``submit_final_answer``.

The problem. ``submit_final_answer`` validated only the FinalAnswer *envelope*
(``schema_version``/``task_id``/``answer`` is a dict, finite numbers, public
size and nesting budgets). The object *inside* the envelope was accepted
unread, so an answer that broke the shape the task instructions publish --
an incorrectly shaped field -- returned ``accepted: true``, consumed the
single submission slot, and was rejected later by a verifier the agent never
gets to hear from.

The rule. A task listed here already owns ONE pure function of the submitted
answer that its verifier applies as its own contract gate. Running that same
function at ingress cannot reject an answer the verifier would have accepted
(it is literally the verifier's gate), so this adds feedback without moving any
outcome: a shape-valid wrong answer is accepted exactly as before and scored by
the verifier alone.

Four properties make this safe to run on the agent-facing boundary.

* **Public only.** Every entry is a pure function of the answer dict. It reads
  no hidden truth, no device config, no evidence log, and no scorer thresholds,
  so the refusal text is a restatement of the answer shape the task's
  ``instruction.md`` already publishes. An agent that probes it learns nothing
  it could not compute offline from its own instructions -- it is not an oracle.
* **No binding.** The precheck runs before the submission budget is reserved,
  before the event is logged, and before ``final_answer.json`` is written, so a
  refusal is an ordinary atomic tool rejection: nothing recorded, nothing spent,
  the agent may fix the answer and submit again.
* **No new evidence.** A refusal appends nothing to ``experiment_log.jsonl``,
  like every other rejected tool call. That is deliberate, not an oversight: at
  least one shipped verifier enumerates every logged action against a known set
  and raises on anything else (``time_budgeted_shadow_surrogate_60q``'s
  ``_KNOWN_TASK_QSIM_ACTIONS``), so a new event kind would break scoring across
  the suite. Rejected attempts remain visible to auditors in the trial's agent
  transcript, which carries the tool error verbatim.
* **Fail-open.** If a precheck itself breaks (a bad import, an unexpected raise
  on a hostile answer) the submission proceeds exactly as it did before this
  module existed. A qsim-side bug must never destroy a submission the verifier
  would have scored.

Adding a task is one row plus the tests that bind it. The requirement is not
"this task has a schema" but "this task's verifier already applies THIS function,
on the answer alone, as a gate a failing answer cannot pass".
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Literal

# Bounds on the refusal text. A pydantic error list over a deeply nested answer
# is unbounded; the agent needs the first few problems, not a transcript.
MAX_LISTED_PROBLEMS = 6
MAX_PROBLEM_CHARACTERS = 200

_ContractKind = Literal["model", "reasons"]

# task_id -> (module, attribute, kind)
#
# "model": a pydantic model the task's verifier validates the answer against.
# "reasons": a callable returning a list of contract violations ([] == clean),
#            which the verifier applies as its own gate.
ANSWER_SHAPE_CONTRACTS: dict[str, tuple[str, str, _ContractKind]] = {}


def _bounded(problem: object) -> str:
    text = " ".join(str(problem).split())
    if len(text) > MAX_PROBLEM_CHARACTERS:
        return text[: MAX_PROBLEM_CHARACTERS - 1] + "…"
    return text


def _model_problems(model: Any, answer: dict[str, Any]) -> list[str]:
    """Validation problems as ``field.path: message``, without echoing the input."""

    from pydantic import ValidationError

    try:
        model.model_validate(answer)
    except ValidationError as exc:
        problems = []
        for error in exc.errors(include_url=False):
            location = ".".join(str(part) for part in error.get("loc", ())) or "answer"
            problems.append(f"{location}: {error.get('msg', 'is invalid')}")
        return problems
    return []


def answer_shape_problems(task_id: str, answer: dict[str, Any]) -> list[str] | None:
    """Return public shape violations, ``[]`` when clean, ``None`` when unchecked.

    ``None`` means this task declares no shape contract here, which is the
    default behaviour: the answer object is accepted unread and the verifier
    remains its only judge. Any unexpected failure of a declared contract
    degrades to ``None`` for the same reason.
    """

    entry = ANSWER_SHAPE_CONTRACTS.get(task_id)
    if entry is None:
        return None
    module_name, attribute, kind = entry
    try:
        target = getattr(import_module(module_name), attribute)
        if kind == "model":
            problems = _model_problems(target, answer)
        else:
            problems = list(target(answer))
    except Exception:
        # Fail open: a broken precheck must never cost a submission the
        # verifier would have scored.
        return None
    return [_bounded(problem) for problem in problems[:MAX_LISTED_PROBLEMS]]


def answer_shape_refusal(problems: list[str]) -> str:
    """The agent-facing refusal: public contract violations, no rig detail."""

    return (
        "final answer does not match the answer shape published in the task "
        "instructions: " + "; ".join(problems) + ". Nothing was recorded and no "
        "submission was used -- correct the answer and submit again."
    )


__all__ = [
    "ANSWER_SHAPE_CONTRACTS",
    "MAX_LISTED_PROBLEMS",
    "MAX_PROBLEM_CHARACTERS",
    "answer_shape_problems",
    "answer_shape_refusal",
]
