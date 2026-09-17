"""Run-long stateful engine for the ``surface_code_memory`` qtype.

Holds the hidden DEM parameters for the whole run and accumulates a single shot-budget
meter across every ``run_memory_experiment`` call. Per call it builds the full hidden DEM
for the requested number of rounds (cached), samples raw per-shot detection events + the
final logical outcome via the pure-numpy DEM sampler, and packs them losslessly. The engine
never decodes and never returns the noise model, weights, or any logical error rate.
"""

from __future__ import annotations

import base64
import threading

import numpy as np

from qiqcbench.qsim.qtypes.surface_code_memory import surface_code as SC
from qiqcbench.qsim.qtypes.surface_code_memory.device import (
    HiddenSurfaceCodeConfig,
    PublicSurfaceCodeSpec,
)
from qiqcbench.qsim.qtypes.surface_code_memory.wire import (
    JobDetectorData,
    MemoryExperimentRequest,
    _BudgetView,
)


def _pack_bits(arr: np.ndarray) -> str:
    """Pack a 2D/1D 0/1 uint8 array (row-major) into base64 of np.packbits."""
    packed = np.packbits(np.ascontiguousarray(arr.reshape(-1), dtype=np.uint8))
    return base64.b64encode(packed.tobytes()).decode("ascii")


class SurfaceCodeEngine:
    def __init__(
        self,
        hidden: HiddenSurfaceCodeConfig,
        public: PublicSurfaceCodeSpec,
        rng: np.random.Generator,
    ):
        self._rng = rng
        self._lock = threading.Lock()
        self._geo = SC.build_geometry(hidden.distance)
        n = hidden.noise
        self._params = SC.HiddenDemParams(
            p_data=n.p_data,
            p_meas=n.p_meas,
            hook_base=n.hook_base,
            hook_multipliers=list(n.hook_multipliers),
            p_leak=n.p_leak,
            leak_pairs=[(lp.s1, lp.s2, lp.dt, int(lp.flips_observable)) for lp in n.leak_pairs],
        )
        self._shots_cap = public.budgets.shot_budget
        self._max_shots_per_call = public.budgets.max_shots_per_call
        self._shots_used = 0
        self._dem_cache: dict[int, list[tuple[int, int, float, int]]] = {}

    def _budget_view(self) -> _BudgetView:
        return _BudgetView(shots_used=self._shots_used, shots_cap=self._shots_cap)

    def _dem(self, rounds: int) -> list[tuple[int, int, float, int]]:
        if rounds not in self._dem_cache:
            self._dem_cache[rounds] = SC.build_hidden_dem(self._geo, self._params, rounds)
        return self._dem_cache[rounds]

    def run_memory_experiment(self, request: MemoryExperimentRequest) -> JobDetectorData | str:
        """Returns a JobDetectorData, or an error string if the budget is exhausted."""
        with self._lock:
            if request.shots > self._max_shots_per_call:
                return (
                    f"shots {request.shots} exceeds max_shots_per_call {self._max_shots_per_call}"
                )
            if self._shots_used + request.shots > self._shots_cap:
                return (
                    f"shot budget exhausted: {self._shots_used}+{request.shots} "
                    f"> cap {self._shots_cap}"
                )
            rounds = request.rounds
            n_det = self._geo.n_checks * rounds
            dem = self._dem(rounds)
            syn, obs = SC.sample_detectors(dem, request.shots, n_det, self._rng)
            self._shots_used += request.shots
            return JobDetectorData(
                rounds=rounds,
                n_detectors=n_det,
                shots=request.shots,
                detection_events_b64=_pack_bits(syn),
                logical_outcomes_b64=_pack_bits(obs),
                budget=self._budget_view(),
            )
