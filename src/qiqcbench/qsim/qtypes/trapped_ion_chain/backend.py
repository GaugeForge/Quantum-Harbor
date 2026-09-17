"""Simulator backend builder for trapped_ion_chain.

Mirrors transmon_pulse/backend.py and digital_gate_model/backend.py shapes.
The qsim/backends/factory.py module dispatches on qtype via the registry and
calls this builder to produce a per-run engine factory.
"""

from __future__ import annotations

import numpy as np

from qiqcbench.qsim.qtypes.trapped_ion_chain.device import (
    HiddenTrappedIonChainConfig,
    PublicTrappedIonChainSpec,
)
from qiqcbench.qsim.qtypes.trapped_ion_chain.engine import IonChainEngine


def build_ion_chain_simulator_backend(
    hidden: HiddenTrappedIonChainConfig,
    public: PublicTrappedIonChainSpec | None = None,
    *,
    salt: int = 0,
):
    """Return an object exposing the per-job engine constructor.

    The top-level backend factory passes ``(hidden, public)`` for every qtype;
    the current Ramsey action path needs only hidden truth and a per-job
    ``new_engine(job_salt)`` hook.
    ``public`` is accepted for factory-signature symmetry; validation lives in
    the Ramsey capability action layer.
    """

    base_seed = (hidden.seed * 1_000_003) ^ salt

    class _IonChainSimulatorBackend:
        def __init__(self) -> None:
            self.hidden = hidden
            self.public = public
            self.replay_root = None

        def new_engine(self, job_salt: int = 0) -> IonChainEngine:
            rng = np.random.default_rng(base_seed ^ job_salt)
            return IonChainEngine(self.hidden, rng)

    return _IonChainSimulatorBackend()


__all__ = ["build_ion_chain_simulator_backend"]
