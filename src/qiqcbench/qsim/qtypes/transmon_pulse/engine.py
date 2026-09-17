"""Effective single-transmon engine.

A simplified effective qubit model is used for the T1 MVP, while the
public control surface still looks like pulse-level transmon control.

This engine:
- treats each qubit as a 2-level system (no leakage),
- integrates Lindblad evolution under T1 (relaxation) and T_phi (pure dephasing),
- maps an `amp` value through a hidden `drive_strength_per_amp_rad_per_ns`
  to a Rabi frequency Ω(t),
- evolves the density matrix through the requested PulseOp/DelayOp ops
  using QuTiP's mesolve,
- samples IQ shots from the resulting P(|1>) per the configured readout model.

QuTiP is used for the master-equation integration; per-shot statistics are
generated classically from the final P(|1>). This is correct for the
ensemble-average regime targeted by T1 experiments.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import qutip as qt

from qiqcbench.qsim.core.wire import (
    DelayOp,
    JobIQData,
    JobResult,
    JobResultMetadata,
    MeasureOp,
    PulseOp,
    SequenceOp,
)
from qiqcbench.qsim.jobs import ModelRequestError
from qiqcbench.qsim.qtypes.transmon_pulse.device import (
    HiddenQubit,
    HiddenTransmonConfig,
)

# Hidden coupling for the MVP. Chosen so that amp=0.20 with a 40 ns Gaussian
# (sigma=8 ns) gives a pi-pulse. The stale notebook advertises x180_amp=0.21
# (5% off) so a stale-calibrated X180 still prepares |1> with >99% fidelity
#.
#
# A Gaussian envelope normalized to peak 1 has area = sigma * sqrt(2*pi). The
# pi-pulse condition is integral(Omega(t)) = pi, so Omega_peak = pi / area.
_GAUSS_AREA_NS = 8.0 * math.sqrt(2 * math.pi)
_DRIVE_STRENGTH_PER_AMP_RAD_PER_NS = math.pi / (0.20 * _GAUSS_AREA_NS)


@dataclass
class PulseExecutionParams:
    """Convert a PulseOp + hidden coupling into integrator inputs."""

    duration_ns: float
    samples_n: int
    omega_envelope: np.ndarray  # rad/ns at each timestep
    phase_rad: float

    @classmethod
    def from_pulse_op(cls, op: PulseOp) -> PulseExecutionParams:
        # Resolve duration/shape into an envelope.
        # Discretize at 0.5 ns; minimum 8 samples for tiny pulses.
        dt = 0.5
        n = max(8, int(round(op.duration_ns / dt)))
        ts = np.linspace(0, op.duration_ns, n)
        if op.shape == "gaussian":
            sigma = op.sigma_ns or (op.duration_ns / 5.0)
            t0 = op.duration_ns / 2.0
            envelope = np.exp(-((ts - t0) ** 2) / (2 * sigma**2))
        elif op.shape == "square":
            envelope = np.ones_like(ts)
        else:  # pragma: no cover
            raise ValueError(f"Unsupported shape {op.shape}")
        omega = _DRIVE_STRENGTH_PER_AMP_RAD_PER_NS * op.amp * envelope
        return cls(
            duration_ns=op.duration_ns,
            samples_n=n,
            omega_envelope=omega,
            phase_rad=op.phase_rad,
        )


class TransmonEngine:
    """Stateful per-job engine. One instance per pulse sequence run."""

    def __init__(self, hidden: HiddenTransmonConfig, rng: np.random.Generator):
        self.hidden = hidden
        self.rng = rng
        # Initialize density matrix per qubit (all in |0>).
        self._rho: dict[str, qt.Qobj] = {q.id: qt.ket2dm(qt.basis(2, 0)) for q in hidden.qubits}
        # Cache operators.
        self._sm = qt.destroy(2)
        self._sx = qt.sigmax()
        self._sz = qt.sigmaz()
        self._proj1 = qt.basis(2, 1) * qt.basis(2, 1).dag()

    def _qubit(self, qid: str) -> HiddenQubit:
        for q in self.hidden.qubits:
            if q.id == qid:
                return q
        raise KeyError(f"Unknown qubit {qid!r}")

    def _collapse_ops(self, qid: str) -> list[qt.Qobj]:
        q = self._qubit(qid)
        # Convert s -> ns
        t1_ns = q.t1_s * 1e9
        t2_ns = q.t2_s * 1e9
        # 1/T2 = 1/(2 T1) + 1/T_phi  =>  1/T_phi = 1/T2 - 1/(2 T1)
        inv_tphi = 1.0 / t2_ns - 1.0 / (2 * t1_ns)
        ops = [math.sqrt(1.0 / t1_ns) * self._sm]
        if inv_tphi > 0:
            # Pure dephasing operator: sqrt(1/(2*Tphi)) * sigma_z
            ops.append(math.sqrt(0.5 * inv_tphi) * self._sz)
        return ops

    def _drive_qubit(self, qid: str, op: PulseOp) -> None:
        params = PulseExecutionParams.from_pulse_op(op)
        # H_drive(t) = (Omega(t)/2) [cos(phi) sigma_x + sin(phi) sigma_y]
        # We omit the rotating-frame detuning for the MVP (hidden frequency
        # matches drive frequency by assumption when no `freq_hz` is given,
        # and small detunings are within stale-calibration tolerance).
        ts = np.linspace(0, params.duration_ns, params.samples_n)
        omega = params.omega_envelope
        H_x = qt.sigmax() * 0.5
        H_y = qt.sigmay() * 0.5
        cphi, sphi = math.cos(params.phase_rad), math.sin(params.phase_rad)

        # QuTiP's pythonic coefficient signature f(t, **kwargs); the legacy
        # f(t, args) form is removed in QuTiP 5.5.
        def coeff_x(t, **_kwargs):
            return float(np.interp(t, ts, omega)) * cphi

        def coeff_y(t, **_kwargs):
            return float(np.interp(t, ts, omega)) * sphi

        H = [[H_x, coeff_x], [H_y, coeff_y]]
        c_ops = self._collapse_ops(qid)
        result = qt.mesolve(
            H,
            self._rho[qid],
            ts,
            c_ops=c_ops,
            options={"store_final_state": True, "nsteps": 10_000},
        )
        self._rho[qid] = result.final_state

    def _idle_qubit(self, qid: str, duration_ns: float) -> None:
        if duration_ns <= 0:
            return
        c_ops = self._collapse_ops(qid)
        # Two timepoints suffice for purely-dissipative free evolution.
        ts = np.array([0.0, duration_ns])
        result = qt.mesolve(
            qt.qzero(2),
            self._rho[qid],
            ts,
            c_ops=c_ops,
            options={"store_final_state": True, "nsteps": 10_000},
        )
        self._rho[qid] = result.final_state

    def _measure_iq(self, qid: str, shots: int) -> list[list[float]]:
        rho = self._rho[qid]
        p1 = float(qt.expect(self._proj1, rho).real)
        p1 = max(0.0, min(1.0, p1))
        ro = self.hidden.readout
        # Sample true outcomes
        true_one = self.rng.random(shots) < p1
        # Apply readout flips
        flips_to_one = self.rng.random(shots) < ro.p_0_to_1
        flips_to_zero = self.rng.random(shots) < ro.p_1_to_0
        observed_one = np.where(true_one, ~flips_to_zero, flips_to_one)
        # IQ shots
        i0, q0 = ro.iq_0_mean
        i1, q1 = ro.iq_1_mean
        means_i = np.where(observed_one, i1, i0)
        means_q = np.where(observed_one, q1, q0)
        noise = self.rng.normal(0.0, ro.iq_sigma, size=(shots, 2))
        iq = np.stack([means_i + noise[:, 0], means_q + noise[:, 1]], axis=1)
        # Reset post-measurement: assume projective readout (collapse to outcome).
        if shots > 0:
            avg_post = float(observed_one.mean())
            self._rho[qid] = avg_post * qt.ket2dm(qt.basis(2, 1)) + (1 - avg_post) * qt.ket2dm(
                qt.basis(2, 0)
            )
        return iq.tolist()

    def run_sequence(
        self,
        sequence: list[SequenceOp],
        shots: int,
        device_id: str,
        job_id: str,
    ) -> JobResult:
        """Execute a single pulse sequence and return one JobResult.

        Each MeasureOp produces `shots` IQ samples per listed qubit. If multiple
        MeasureOps appear, only the LAST is returned (matches T1 protocol).
        """
        t_start = time.perf_counter()
        last_iq: dict[str, list[list[float]]] | None = None
        seq_dur_ns = 0.0

        # Only ModelRequestError is caught. The backend bound-checks every pulse
        # field before the engine runs, but it never inspects DelayOp.duration_ns,
        # so a plain sequence carrying an unresolved "$name" placeholder reaches
        # the loop below -- an ordinary model mistake that must stay model-owned
        # (free-form error, no stamp), or it buys a free infrastructure rerun.
        # Everything else raised here is qsim-owned and must propagate to
        # JobManager, which types it as the sanctioned JOB_EXECUTION_ERROR that
        # is_internal_job_failure recognizes and get_job_result stamps
        # failure_kind="qsim_internal". The blanket except this replaces returned
        # str(exc) for BOTH cases -- the empty string for MemoryError -- so a
        # qsim fault was attributed to the model.
        try:
            for op in sequence:
                if isinstance(op, PulseOp):
                    self._drive_qubit(op.channel.split(".")[0], op)
                    seq_dur_ns += op.duration_ns
                elif isinstance(op, DelayOp):
                    if not isinstance(op.duration_ns, (int, float)):
                        raise ModelRequestError(
                            f"Unresolved sweep placeholder in delay: {op.duration_ns!r}"
                        )
                    for q in self.hidden.qubits:
                        self._idle_qubit(q.id, float(op.duration_ns))
                    seq_dur_ns += float(op.duration_ns)
                elif isinstance(op, MeasureOp):
                    last_iq = {qid: self._measure_iq(qid, shots) for qid in op.qubits}
                else:  # pragma: no cover
                    raise ValueError(f"Unknown op kind: {op}")
        except ModelRequestError as exc:
            return JobResult(
                job_id=job_id,
                device_id=device_id,
                status="failed",
                shots=shots,
                error=str(exc),
                metadata=JobResultMetadata(
                    sequence_duration_ns=seq_dur_ns,
                    wallclock_ms=int((time.perf_counter() - t_start) * 1000),
                ),
            )

        if last_iq is None:
            return JobResult(
                job_id=job_id,
                device_id=device_id,
                status="failed",
                shots=shots,
                error="Sequence did not contain a measure op.",
                metadata=JobResultMetadata(sequence_duration_ns=seq_dur_ns),
            )

        return JobResult(
            job_id=job_id,
            device_id=device_id,
            status="complete",
            shots=shots,
            data=JobIQData(iq=last_iq),
            metadata=JobResultMetadata(
                sequence_duration_ns=seq_dur_ns,
                wallclock_ms=int((time.perf_counter() - t_start) * 1000),
            ),
        )
