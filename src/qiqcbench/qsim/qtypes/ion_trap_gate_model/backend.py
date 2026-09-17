"""Simulator backend for the ion-trap gate-model qtype."""

from __future__ import annotations

import secrets

import numpy as np

from qiqcbench.qsim.qtypes.ion_trap_gate_model.device import (
    HiddenIonTrapGateModelConfig,
    PublicIonTrapGateModelSpec,
)
from qiqcbench.qsim.qtypes.ion_trap_gate_model.engine import IonTrapEngine


class IonTrapSimulatorBackend:
    """Own one fresh private entropy root for an entire qsim attempt."""

    def __init__(
        self,
        hidden: HiddenIonTrapGateModelConfig,
        public: PublicIonTrapGateModelSpec | None = None,
        *,
        run_entropy: int | None = None,
        base_salt: int = 0,
    ) -> None:
        if run_entropy is None:
            run_entropy = secrets.randbits(128)
        if (
            isinstance(run_entropy, bool)
            or not isinstance(run_entropy, int)
            or not 0 <= run_entropy < 2**128
        ):
            raise ValueError("run_entropy must be a 128-bit non-negative integer")
        self.hidden = hidden
        self.public = public
        self.replay_root = None
        self._run_entropy = run_entropy
        self._base_salt = int(base_salt)

    def new_rng(self, *, job_salt: int = 0, stream: int = 0) -> np.random.Generator:
        """Derive an independent per-job stream without exposing its entropy."""

        entropy_words = (
            self._run_entropy & ((1 << 64) - 1),
            self._run_entropy >> 64,
        )
        seed = np.random.SeedSequence(
            [
                int(self.hidden.seed) & ((1 << 64) - 1),
                *entropy_words,
                self._base_salt,
                int(job_salt),
                int(stream),
            ]
        )
        return np.random.default_rng(seed)

    def new_engine(self, job_salt: int = 0, *, stream: int = 0) -> IonTrapEngine:
        return IonTrapEngine(
            self.hidden,
            self.new_rng(job_salt=job_salt, stream=stream),
        )


def build_ion_trap_simulator_backend(
    hidden: HiddenIonTrapGateModelConfig,
    public: PublicIonTrapGateModelSpec | None = None,
    *,
    salt: int = 0,
    run_entropy: int | None = None,
) -> IonTrapSimulatorBackend:
    return IonTrapSimulatorBackend(
        hidden,
        public,
        run_entropy=run_entropy,
        base_salt=salt,
    )


__all__ = ["IonTrapSimulatorBackend", "build_ion_trap_simulator_backend"]
