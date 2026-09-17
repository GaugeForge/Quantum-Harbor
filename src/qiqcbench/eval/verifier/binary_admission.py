"""Bridge a finalized legacy binary reward into verifier-owned artifacts."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from qiqcbench.eval.contracts.outcome import OutcomeDisposition, VerificationManifest
from qiqcbench.eval.io.canonical import canonical_json_bytes
from qiqcbench.eval.io.config import validate_config_id
from qiqcbench.eval.verifier.bindings import existing_trial_reference, target_trial_reference
from qiqcbench.eval.verifier.context import load_verifier_run_context
from qiqcbench.eval.verifier.legacy_rubric import emit_legacy_rubric_report
from qiqcbench.eval.verifier.writer import VerificationWriteResult

_DISPOSITIONS = frozenset(("scored", "model_failure"))


def _read_strict_reward(path: Path, *, trial_root: Path) -> int:
    _reference, reward_bytes = existing_trial_reference(
        path,
        trial_root=trial_root,
        role="reward",
    )
    if reward_bytes == b"0\n":
        return 0
    if reward_bytes == b"1\n":
        return 1
    raise ValueError("reward must be exactly b'0\\n' or b'1\\n'")


def _resolve_report_target(path: str | os.PathLike[str], *, trial_root: Path) -> Path:
    """Reject lexical and symlink-parent escapes before compatibility output."""

    try:
        root = trial_root.resolve(strict=True)
    except OSError as exc:
        raise ValueError("trial root must be an existing directory") from exc
    if not root.is_dir():
        raise ValueError("trial root must be an existing directory")

    raw = Path(path)
    if ".." in raw.parts:
        raise ValueError("companion report path must not escape the trial root")
    candidate = raw if raw.is_absolute() else root / raw
    try:
        parent = candidate.parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError("companion report parent must exist below the trial root") from exc
    target = parent / candidate.name
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("companion report path must not escape the trial root") from exc
    if os.path.lexists(target) and target.is_symlink():
        raise ValueError("companion report must not be a symlink")
    return target


def _require_disposition(value: str) -> OutcomeDisposition:
    if not isinstance(value, str) or value not in _DISPOSITIONS:
        raise ValueError("disposition must be 'scored' or 'model_failure'")
    return cast(OutcomeDisposition, value)


def emit_binary_admission_report(
    *,
    task_id: str,
    disposition: OutcomeDisposition,
    reward_path: str | os.PathLike[str],
    report_path: str | os.PathLike[str],
    trial_root: str | os.PathLike[str],
    experiment_log_path: str | os.PathLike[str],
    final_answer_path: str | os.PathLike[str] | None = None,
    legacy_report_path: str | os.PathLike[str] | None = None,
    additional_evidence_paths: Mapping[str, str | os.PathLike[str]] | None = None,
    environ: Mapping[str, str] | None = None,
    config_root: str | os.PathLike[str] | None = None,
    predecessor_verification: VerificationManifest | None = None,
) -> VerificationWriteResult | None:
    """Emit a minimal rubric companion for a completed ``reward.txt``.

    The scientific scorer remains the sole authority for the reward.  This
    bridge only validates its exact historical bytes, records that decision in
    a revision-owned JSON adapter, and binds the existing qsim log and final
    answer when canonical verifier context is present.  The caller, not the
    reward bit, owns the distinction between a scored rejection and a model
    failure.
    """

    validated_task_id = validate_config_id(task_id, field="task_id")
    resolved_disposition = _require_disposition(disposition)
    root = Path(trial_root).resolve(strict=True)
    reward = _read_strict_reward(Path(reward_path), trial_root=root)
    if resolved_disposition == "model_failure" and reward != 0:
        raise ValueError("model_failure disposition requires a rejected reward")
    companion_path = _resolve_report_target(report_path, trial_root=root)
    target_trial_reference(companion_path, trial_root=root, role="companion report")
    context = load_verifier_run_context(environ)
    log_role = context.qsim_log_evidence_role if context is not None else "experiment_log"
    evidence_paths: dict[str, Path] = {log_role: Path(experiment_log_path)}
    if final_answer_path is not None:
        if "final_answer" in evidence_paths:
            raise ValueError("evidence roles must not overlap")
        evidence_paths["final_answer"] = Path(final_answer_path)
    if legacy_report_path is not None:
        if "legacy_task_report" in evidence_paths:
            raise ValueError("evidence roles must not overlap")
        evidence_paths["legacy_task_report"] = Path(legacy_report_path)
    for raw_role, path in (additional_evidence_paths or {}).items():
        role = validate_config_id(raw_role, field="evidence_role")
        if role in evidence_paths:
            raise ValueError("evidence roles must not overlap")
        evidence_paths[role] = Path(path)
    for role, path in evidence_paths.items():
        if (
            context is not None
            and context.evidence_binding_mode == "qsim_context_v1"
            and role == log_role
        ):
            # Let bind_verification preserve its EvidenceIntegrityError taxonomy.
            continue
        existing_trial_reference(path, trial_root=root, role=f"evidence {role!r}")

    report_bytes = canonical_json_bytes(
        {
            "schema_version": 1,
            "task_id": validated_task_id,
            "accepted": reward,
            "evidence_roles": sorted(evidence_paths),
        }
    )
    if companion_path.exists() and companion_path.read_bytes() != report_bytes:
        raise ValueError("companion report already contains different bytes")
    return emit_legacy_rubric_report(
        report_bytes,
        reward_binary=reward,
        report_path=companion_path,
        reward_path=reward_path,
        trial_root=root,
        evidence_paths=evidence_paths,
        disposition=resolved_disposition,
        environ=environ,
        verifier_context=context,
        config_root=config_root,
        predecessor_verification=predecessor_verification,
    )


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> None:
        raise ValueError("invalid command arguments")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        description="Write canonical verifier artifacts for an existing binary reward."
    )
    parser.add_argument("task_id")
    parser.add_argument("disposition", choices=sorted(_DISPOSITIONS))
    parser.add_argument("reward_path")
    parser.add_argument("report_path")
    parser.add_argument("trial_root")
    parser.add_argument("experiment_log_path")
    parser.add_argument("--final-answer-path")
    parser.add_argument("--legacy-report-path")
    parser.add_argument(
        "--additional-evidence",
        action="append",
        default=[],
        nargs=2,
        metavar=("ROLE", "PATH"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bridge without exposing verifier paths or context in diagnostics."""

    try:
        args = _parser().parse_args(argv)
        additional_evidence_paths: dict[str, str] = {}
        for role, path in args.additional_evidence:
            if role in additional_evidence_paths:
                raise ValueError("duplicate additional evidence role")
            additional_evidence_paths[role] = path
        emit_binary_admission_report(
            task_id=args.task_id,
            disposition=args.disposition,
            reward_path=args.reward_path,
            report_path=args.report_path,
            trial_root=args.trial_root,
            experiment_log_path=args.experiment_log_path,
            final_answer_path=args.final_answer_path,
            legacy_report_path=args.legacy_report_path,
            additional_evidence_paths=additional_evidence_paths,
        )
    except (OSError, TypeError, ValueError):
        print("binary admission bridge failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["emit_binary_admission_report", "main"]
