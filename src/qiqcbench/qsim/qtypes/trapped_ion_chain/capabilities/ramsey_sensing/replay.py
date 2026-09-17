"""Provider-replay machinery for Ramsey requests.

Request-keyed fixtures: each completed fixture file is named by the SHA256
digest of the canonicalized request bytes. Replay looks up the fixture and
fails closed on miss. Task 9 generates the canonical fixture set by recording a
simulator-mode Codex run.
"""

from __future__ import annotations

import json
from pathlib import Path

from qiqcbench.qsim.backends.provider_artifacts import canonical_request_hash
from qiqcbench.qsim.core.wire import (
    JobResult,
    RamseyExperimentRequest,
    RamseySweepRequest,
)

__all__ = [
    "ramsey_request_digest",
    "ramsey_fixture_path",
    "replay_ramsey_experiment_request",
    "replay_ramsey_sweep_request",
]


def ramsey_request_digest(request: RamseyExperimentRequest | RamseySweepRequest) -> str:
    return canonical_request_hash(request)


def ramsey_fixture_path(
    request: RamseyExperimentRequest | RamseySweepRequest,
    *,
    replay_root: Path,
) -> Path:
    return Path(replay_root) / "ramsey_sensing" / f"{ramsey_request_digest(request)}.json"


def replay_ramsey_experiment_request(
    request: RamseyExperimentRequest,
    *,
    replay_root: Path,
    job_id: str,
    device_id: str,
) -> JobResult:
    path = ramsey_fixture_path(request, replay_root=Path(replay_root))
    if not path.is_file():
        return JobResult(
            job_id=job_id,
            device_id=device_id,
            status="failed",
            shots=request.shots,
            error=(
                f"provider_replay miss: no fixture at {path}. "
                "Replay must fail closed on missing fixtures."
            ),
        )
    # TODO(Task 9): validate fixture shape (request schema match, shots match,
    # data.kind="ramsey_bitstring") before returning. Current Task 5 skeleton only
    # does presence check.
    payload = json.loads(path.read_text(encoding="utf-8"))
    return JobResult.model_validate({**payload, "job_id": job_id})


def replay_ramsey_sweep_request(
    request: RamseySweepRequest,
    *,
    replay_root: Path,
    job_id: str,
    device_id: str,
) -> JobResult:
    path = ramsey_fixture_path(request, replay_root=Path(replay_root))
    if not path.is_file():
        return JobResult(
            job_id=job_id,
            device_id=device_id,
            status="failed",
            shots=request.shots,
            error=f"provider_replay miss: no fixture at {path}",
        )
    # TODO(Task 9): validate fixture shape (request schema match, shots match,
    # data.kind="ramsey_bitstring") before returning. Current Task 5 skeleton only
    # does presence check.
    payload = json.loads(path.read_text(encoding="utf-8"))
    return JobResult.model_validate({**payload, "job_id": job_id})
