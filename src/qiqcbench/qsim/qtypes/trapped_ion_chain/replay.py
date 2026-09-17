"""Provider-replay backend builder for trapped_ion_chain.

Skeleton. Full request-keyed Ramsey fixture machinery lands in
capabilities/ramsey_sensing/replay.py (Task 5) and is wired into the
end-to-end provider_replay path in Task 9. This file exists now so
the registry descriptor (Task 6) has a callable to register.
"""

from __future__ import annotations

from pathlib import Path

from qiqcbench.qsim.qtypes.trapped_ion_chain.device import (
    PublicTrappedIonChainSpec,
)


def build_ion_chain_replay_backend(
    *,
    task_id: str,
    public: PublicTrappedIonChainSpec,
    replay_root: str | Path,
    log_dir: str | Path | None = None,
):
    """Return a replay backend that fails closed on engine construction."""

    class _IonChainReplayBackend:
        def __init__(self) -> None:
            self.task_id = task_id
            self.public = public
            self.replay_root = Path(replay_root)
            self.log_dir = None if log_dir is None else Path(log_dir)

        def new_engine(self, job_salt: int = 0):
            # Replay path executes via capability runtime's replay function;
            # this engine constructor is unused in replay mode but provided
            # to satisfy the duck-typed contract.
            raise RuntimeError(
                "trapped_ion_chain replay mode dispatches via capability "
                "runtime; engine constructor must not be called."
            )

    return _IonChainReplayBackend()


__all__ = ["build_ion_chain_replay_backend"]
