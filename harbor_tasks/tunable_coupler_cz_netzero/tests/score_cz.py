#!/usr/bin/env python3
"""Verifier driver for ``tunable_coupler_cz_netzero``.

Usage: ``score_cz.py <experiment_log.jsonl> <final_answer.json> <artifacts_dir>``

Replays the agent's PROGRAMMED q1 flux waveform through the control stack to the
realized flux, then through the two-transmon CZ model, and binds every Stage-C
quantity (conditional phase after the reported virtual-Z, leakage, fidelity,
net-zero 1/f advantage, idle ZZ, predistortion residual) plus the Stage-A device
+ settling recovery. Scoring is gated on a minimal execution-evidence floor
(>= min_completed_flux_jobs completed run_flux_pulse jobs in the qsim experiment
log). Writes a 0/1 reward (exit code) and an audit score report.

Exit discipline: exit 1 is a scored model rejection; exit 2
is reserved for infrastructure/verifier failure with no reward file — a
mis-invoked scorer, a qsim experiment log that is missing or does not parse, an
unavailable/invalid hidden scorer, or an unwritable artifacts tree. Answer
content is type-validated before scoring so no agent-controlled payload can
surface as an "unexpected" exception and ride the infrastructure exit
(self-void). ``MemoryError`` is deliberately a MODEL rejection: this task has
no FinalAnswer byte budget, so the only plausible OOM lever is an
agent-inflated payload, and mapping it to exit 2 would hand a losing run a
no-score escape.
"""
# ruff: noqa: E402  (imports follow the sys.path bootstrap below, by design)

from __future__ import annotations

import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
for ancestor in [_THIS, *_THIS.parents]:
    cand = ancestor / "src"
    if (cand / "qiqcbench" / "__init__.py").is_file():
        sys.path.insert(0, str(cand))
        break
else:
    for fb in (Path("/app/src"), Path("/qiqcbench-src")):
        if (fb / "qiqcbench" / "__init__.py").is_file():
            sys.path.insert(0, str(fb))
            break

from qiqcbench.qsim.devices import configs_root, load_hidden_config, load_public_spec
from qiqcbench.qsim.hidden_dynamics.tunable_coupler_cz_netzero import construction as C
from qiqcbench.qsim.hidden_dynamics.tunable_coupler_cz_netzero.scorer import (
    InfrastructureError,
    ScorerError,
    count_flux_evidence,
    load_hidden_scorer,
    score_answer,
)

TASK_ID = "tunable_coupler_cz_netzero"
DEVICE_ID = "tunable_coupler_cz_v0"


def _resolve_scorer():
    mat = configs_root() / "task_materials" / TASK_ID / "hidden" / "scorer.example.yaml"
    if mat.is_file():
        return load_hidden_scorer(mat)
    hidden = load_hidden_config(configs_root() / "devices" / f"{DEVICE_ID}.hidden.example.yaml")
    return C.build_hidden_scorer(load_public_spec(DEVICE_ID), hidden)


def _write_report(art: Path, report: dict) -> None:
    # The report write itself must never surface as a scored rejection: an
    # unwritable artifacts tree is a verifier fault.
    try:
        (art / "score_report.json").write_text(json.dumps(report, indent=2))
    except OSError as exc:
        raise InfrastructureError(f"score report is unwritable: {exc}") from exc


def _run(log_file: Path, ans_file: Path, art: Path) -> int:
    try:
        art.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InfrastructureError(f"artifacts dir is unavailable: {exc}") from exc
    # qsim owns the experiment log: a missing or corrupt one is an infrastructure
    # fault, and counting it as zero completed jobs would charge the evidence
    # floor below to the model for something it cannot touch.
    evidence = count_flux_evidence(log_file)
    report: dict = {"task_id": TASK_ID, "evidence": evidence}

    if not ans_file.is_file():
        report.update(passed=False, reason="no final_answer.json")
        _write_report(art, report)
        print("no final_answer.json", file=sys.stderr)
        return 1

    # The verifier's own configuration (the hidden scorer) is not agent
    # content: failure to load or validate it is infrastructure, never a
    # scored rejection.
    try:
        scorer = _resolve_scorer()
    except Exception as exc:
        raise InfrastructureError(f"hidden scorer is unavailable/invalid: {exc}") from exc

    # Agent-controlled payload: type-validate the containers before reading any
    # field, so no shape of submission can raise past the answer layer.
    payload = json.loads(ans_file.read_text(encoding="utf-8"))
    answer = payload.get("answer") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("task_id") != TASK_ID
        or not isinstance(answer, dict)
    ):
        task = payload.get("task_id") if isinstance(payload, dict) else None
        report.update(passed=False, reason=f"bad task_id/answer ({task!r})")
        _write_report(art, report)
        return 1

    # Execution-evidence floor: the replay is deterministic on the
    # SUBMITTED waveform, so a purely analytic submission would otherwise be
    # scoreable. This is an anti-gaming backstop, not a discriminator -- the
    # physics gates below do the discriminating.
    min_jobs = int(getattr(scorer, "min_completed_flux_jobs", 15))
    done = report["evidence"]["completed_flux_jobs"]
    report["evidence"]["min_completed_flux_jobs"] = min_jobs
    if done < min_jobs:
        report.update(
            passed=False,
            reason=(
                f"insufficient execution evidence: {done} completed run_flux_pulse "
                f"jobs (>= {min_jobs} required)"
            ),
        )
        _write_report(art, report)
        print(report["reason"], file=sys.stderr)
        return 1

    try:
        result = score_answer(answer, scorer)
    except ScorerError as exc:
        report.update(passed=False, reason=f"unscoreable: {exc}")
        _write_report(art, report)
        print(f"unscoreable: {exc}", file=sys.stderr)
        return 1

    report.update(
        passed=bool(result.gate_passed),
        checks={k: bool(v) for k, v in result.checks.items()},
        metrics={k: round(float(v), 6) for k, v in result.metrics.items()},
        penalties=list(result.penalties),
        notes=list(result.notes),
    )
    _write_report(art, report)
    failed = [k for k, v in result.checks.items() if not v]
    print(
        f"tunable_coupler_cz_netzero: gate={'PASS' if result.gate_passed else 'FAIL'} "
        f"phi_2Q_err={result.metrics.get('phi_2Q_err'):.3f} leak={result.metrics.get('leakage'):.2e} "
        f"infid={result.metrics.get('gate_infidelity'):.2e} "
        f"netzero_adv={result.metrics.get('netzero_advantage'):.1f}x "
        f"idle_zz={result.metrics.get('idle_zz_khz'):.1f}kHz failed={failed}"
    )
    return 0 if result.gate_passed else 1


def main(argv: list[str]) -> int:
    if len(argv) < 4:
        print("usage: score_cz.py <log> <final_answer.json> <artifacts_dir>", file=sys.stderr)
        return 2
    log_file, ans_file, art = Path(argv[1]), Path(argv[2]), Path(argv[3])
    # Typed dispositions. Order matters: ``ScorerError`` IS a
    # ``ValueError`` and ``json.JSONDecodeError`` IS a ``ValueError``, so the
    # specific classes are caught before the answer-layer family. The
    # answer-layer family (ValueError/KeyError/TypeError/AttributeError +
    # MemoryError) is reachable through agent-controlled content and maps to a
    # scored rejection; OS/recursion faults and anything genuinely unexpected
    # are verifier-owned and take the reserved exit.
    try:
        return _run(log_file, ans_file, art)
    except InfrastructureError as exc:
        print(f"verifier infrastructure failure: {exc}", file=sys.stderr)
        return 2
    except ScorerError as exc:
        print(f"unscoreable: {exc}", file=sys.stderr)
        return 1
    except MemoryError:
        print("unscoreable: submission exhausted verifier memory", file=sys.stderr)
        return 1
    except (ValueError, KeyError, TypeError, AttributeError, ArithmeticError) as exc:
        # ArithmeticError covers OverflowError (a huge JSON integer is valid
        # agent content) plus ZeroDivision/FloatingPoint faults reachable only
        # through answer-derived values -- scored rejections, never the
        # reserved exit.
        print(f"unscoreable: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except (OSError, RecursionError) as exc:
        print(f"verifier infrastructure failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"verifier infrastructure failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
