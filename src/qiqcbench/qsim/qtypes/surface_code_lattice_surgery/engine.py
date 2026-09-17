"""Run-long stateful engine for the ``surface_code_lattice_surgery`` qtype.

Holds the hidden circuit-level noise model for the whole run and accumulates a single
shot-budget meter AND a shot-rounds meter across every experiment call (memory + merged +
CNOT — a genuine three-way allocation problem). Per call it builds (cached) the requested
circuit bundle, samples raw detection events + observable flips through the FlipSimulator
pipeline (circuit Pauli noise + classical leakage machine + asymmetric readout), and packs
them losslessly. The engine never decodes and never returns the noise model.
"""

from __future__ import annotations

import base64
import threading

import numpy as np

from qiqcbench.qsim.qtypes.surface_code_lattice_surgery import circuits as CC
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery.device import (
    HiddenLatticeSurgeryConfig,
    PublicLatticeSurgerySpec,
)
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery.noise_model import (
    sampler_noise_from_hidden,
)
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery.sampler import sample_bundle
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery.wire import (
    JobLatticeSurgeryData,
    LsCnotRequest,
    LsMemoryRequest,
    LsMergedRequest,
    _LsBudgetView,
)


def _pack_bits(arr: np.ndarray) -> str:
    packed = np.packbits(np.ascontiguousarray(arr.reshape(-1), dtype=np.uint8))
    return base64.b64encode(packed.tobytes()).decode("ascii")


class LatticeSurgeryEngine:
    def __init__(
        self,
        hidden: HiddenLatticeSurgeryConfig,
        public: PublicLatticeSurgerySpec,
        rng: np.random.Generator,
    ):
        self._rng = rng
        self._lock = threading.Lock()
        self._noise = sampler_noise_from_hidden(hidden)
        self._shots_cap = public.budgets.shot_budget
        self._shot_rounds_cap = public.budgets.shot_rounds_budget
        self._max_shots_per_call = public.budgets.max_shots_per_call
        self._shots_used = 0
        self._shot_rounds_used = 0

    def _budget_view(self) -> _LsBudgetView:
        return _LsBudgetView(
            shots_used=self._shots_used,
            shots_cap=self._shots_cap,
            shot_rounds_used=self._shot_rounds_used,
            shot_rounds_cap=self._shot_rounds_cap,
        )

    def _charge(self, shots: int, rounds: int) -> str | None:
        if shots > self._max_shots_per_call:
            return f"shots {shots} exceeds max_shots_per_call {self._max_shots_per_call}"
        if self._shots_used + shots > self._shots_cap:
            return f"shot budget exhausted: {self._shots_used}+{shots} > cap {self._shots_cap}"
        cost = shots * rounds
        if self._shot_rounds_used + cost > self._shot_rounds_cap:
            return (
                f"shot-rounds budget exhausted: {self._shot_rounds_used}+{cost} > "
                f"cap {self._shot_rounds_cap}"
            )
        self._shots_used += shots
        self._shot_rounds_used += cost
        return None

    def _run(
        self, bundle: CC.CircuitBundle, shots: int, experiment: str, layout: str
    ) -> JobLatticeSurgeryData | str:
        err = self._charge(shots, bundle.rounds)
        if err is not None:
            return err
        res = sample_bundle(bundle, self._noise, shots, self._rng)
        mzz = mxx = mzint = None
        if experiment == "cnot":
            for name, key in (("m_zz", "mzz"), ("m_xx", "mxx"), ("m_z_int", "mzint")):
                recs = list(bundle.named_parities[name])
                bits = np.bitwise_xor.reduce(res.record_values[:, recs], axis=1)
                if key == "mzz":
                    mzz = _pack_bits(bits)
                elif key == "mxx":
                    mxx = _pack_bits(bits)
                else:
                    mzint = _pack_bits(bits)
        return JobLatticeSurgeryData(
            experiment=experiment,
            layout=layout,
            rounds=bundle.rounds,
            shots=shots,
            n_detectors=bundle.n_detectors,
            detection_events_b64=_pack_bits(res.detection_events),
            observable_flips_b64=_pack_bits(res.observable_flips),
            observable_names=[o.name for o in bundle.observables],
            mzz_b64=mzz,
            mxx_b64=mxx,
            mzint_b64=mzint,
            budget=self._budget_view(),
        )

    def run_memory_experiment(self, request: LsMemoryRequest) -> JobLatticeSurgeryData | str:
        with self._lock:
            bundle = CC.memory_bundle(request.layout, request.rounds)
            return self._run(bundle, request.shots, "memory", request.layout)

    def run_merged_memory(self, request: LsMergedRequest) -> JobLatticeSurgeryData | str:
        with self._lock:
            bundle = CC.merged_bundle(request.window, request.rounds, request.include_transitions)
            return self._run(bundle, request.shots, "merged", bundle.name)

    def run_lattice_surgery_cnot(self, request: LsCnotRequest) -> JobLatticeSurgeryData | str:
        with self._lock:
            a, b = int(request.logical_input[0]), int(request.logical_input[1])
            if request.config in ("bell_zz", "bell_xx"):
                a = b = 0  # prep is fixed for the bell reporting configurations
            bundle = CC.cnot_bundle(request.config, a, b)
            return self._run(bundle, request.shots, "cnot", request.config)
