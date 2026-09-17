"""Tunable-coupler CZ device engine.

Executes a ``FluxPulseRequest``: prepares a state (single-qubit rotations on the
``{0,1}`` subspace), optionally idles (accruing the static ZZ at the coupler
bias), applies the programmed q1 flux pulse THROUGH the control stack (realized =
flux-line distortion of the DAC-quantized program) propagating the 16-dim
two-transmon state at the gate exchange ``J_eff(coupler_flux)``, applies a
tomography basis, and samples level-resolved per-qubit outcomes. Uses the shared
``physics``/``control`` modules so the engine matches the verifier.
"""

from __future__ import annotations

import math
import time

import numpy as np

from qiqcbench.qsim.core.wire import JobResult, JobResultMetadata
from qiqcbench.qsim.qtypes.transmon_pair_tunable_coupler import control as ct
from qiqcbench.qsim.qtypes.transmon_pair_tunable_coupler import physics as ph
from qiqcbench.qsim.qtypes.transmon_pair_tunable_coupler.device import (
    HiddenTunableCouplerConfig,
)
from qiqcbench.qsim.qtypes.transmon_pair_tunable_coupler.wire import (
    FluxPulseRequest,
    JobPairLevelOutcomeData,
    PairRotationOp,
)

_L = 4
_TWO_PI = 2.0 * math.pi
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _rot4(axis: str, angle: float) -> np.ndarray:
    """Single-qubit rotation on the {0,1} subspace, identity on {2,3} (4-level)."""
    pauli = {"x": _X, "y": _Y, "z": _Z}[axis]
    r2 = math.cos(angle / 2) * np.eye(2, dtype=complex) - 1j * math.sin(angle / 2) * pauli
    m = np.eye(_L, dtype=complex)
    m[:2, :2] = r2
    return m


def _embed(m: np.ndarray, qubit: int) -> np.ndarray:
    return np.kron(m, np.eye(_L, dtype=complex)) if qubit == 1 else np.kron(np.eye(_L), m)


class TunableCouplerEngine:
    def __init__(self, hidden: HiddenTunableCouplerConfig, rng: np.random.Generator):
        self.hidden = hidden
        self.rng = rng
        self.p = ph.TransmonPairParams(
            omega_1_max=hidden.omega_1_max_ghz,
            omega_2=hidden.omega_2_ghz,
            alpha_1=hidden.alpha_1_ghz,
            alpha_2=hidden.alpha_2_ghz,
            arc_asym=hidden.arc_asym,
            t1_us=hidden.t1_us,
            t2_us=hidden.t2_us,
        )
        self.cp = ph.CouplerParams(
            omega_c_max=hidden.omega_c_max_ghz,
            g12=hidden.g12_ghz,
            gg=hidden.gg_ghz2,
            omega_1_gate=hidden.omega_2_ghz + hidden.alpha_2_ghz,
            omega_1_idle=hidden.omega_1_max_ghz,
            omega_2=hidden.omega_2_ghz,
            zeta_scale=hidden.zeta_scale,
        )
        self.awg = ct.AwgSpec(
            sample_rate_ghz=2.5, n_bits=16, full_scale_phi0=0.70, analog_bw_ghz=0.70
        )
        self.line = ct.FluxLineModel(
            amplitudes=tuple(hidden.flux_settle_amplitudes),
            taus_ns=tuple(hidden.flux_settle_taus_ns),
        )
        self.readout = [
            (hidden.readout_q1.p_0_to_1, hidden.readout_q1.p_1_to_0),
            (hidden.readout_q2.p_0_to_1, hidden.readout_q2.p_1_to_0),
        ]

    def _apply_ops(self, psi: np.ndarray, ops: list[PairRotationOp]) -> np.ndarray:
        for op in ops:
            psi = _embed(_rot4(op.axis, op.angle_rad), op.qubit) @ psi
        return psi

    def _idle_unitary(self, coupler_flux: float, idle_ns: float) -> np.ndarray:
        """Static-ZZ idle: conditional phase 2*pi*zeta*idle on |11> (others ~free)."""
        zeta = ph.zeta_idle(coupler_flux, self.cp)  # GHz
        u = np.eye(_L * _L, dtype=complex)
        u[ph.IDX[(1, 1)], ph.IDX[(1, 1)]] = np.exp(-1j * _TWO_PI * zeta * idle_ns)
        return u

    def run(self, request: FluxPulseRequest, device_id: str, job_id: str) -> JobResult:
        t0 = time.perf_counter()
        # No blanket except here. The wire model bounds every op field and
        # ``TunableCouplerSimulatorBackend`` checks shots, coupler flux range, DAC
        # range on every programmed sample, and rotation qubits before this call,
        # so anything raised below is a qsim-owned execution fault. Letting it
        # propagate reaches ``JobManager``, which types it as the sanctioned
        # ``JOB_EXECUTION_ERROR`` that ``is_internal_job_failure`` recognizes and
        # ``get_job_result`` stamps ``failure_kind="qsim_internal"``. Returning
        # ``error=str(exc)`` instead replaced that marker with free-form text --
        # empty for ``MemoryError`` -- so a qsim fault was attributed to the
        # model.
        psi = np.zeros(_L * _L, dtype=complex)
        psi[ph.IDX[(0, 0)]] = 1.0
        psi = self._apply_ops(psi, request.prep_ops)
        if request.idle_ns > 0:
            psi = self._idle_unitary(request.coupler_flux, request.idle_ns) @ psi
        if request.programmed_flux_q1:
            programmed = np.asarray(request.programmed_flux_q1, dtype=float)
            realized = ct.realized_flux(programmed, request.sample_dt_ns, self.awg, self.line)
            j = ph.j_eff(request.coupler_flux, self.cp)  # gate exchange
            u = ph.cz_unitary(realized, j, request.sample_dt_ns, self.p)
            psi = u @ psi
        psi = self._apply_ops(psi, request.post_ops)
        outcomes = self._sample(psi, list(request.measure_qubits), request.shots)
        return JobResult(
            job_id=job_id,
            device_id=device_id,
            status="complete",
            shots=request.shots,
            data=JobPairLevelOutcomeData(
                outcomes=[outcomes], n_levels=_L, measure_qubits=list(request.measure_qubits)
            ),
            metadata=JobResultMetadata(wallclock_ms=int((time.perf_counter() - t0) * 1000)),
        )

    def _sample(self, psi: np.ndarray, measure_qubits: list[int], shots: int) -> list[str]:
        pops = np.abs(psi) ** 2
        s = pops.sum()
        pops = pops / s if s > 0 else pops
        idx = self.rng.choice(_L * _L, size=shots, p=pops)
        out: list[str] = []
        for k in idx:
            i, j = divmod(int(k), _L)  # q1=i, q2=j
            levels = {1: i, 2: j}
            chars = []
            for q in measure_qubits:
                lv = levels[q]
                p01, p10 = self.readout[q - 1]
                if lv == 0 and self.rng.random() < p01:
                    lv = 1
                elif lv == 1 and self.rng.random() < p10:
                    lv = 0
                chars.append(str(lv))
            out.append("".join(chars))
        return out


__all__ = ["TunableCouplerEngine"]
