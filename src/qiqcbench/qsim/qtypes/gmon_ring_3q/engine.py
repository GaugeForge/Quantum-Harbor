"""Joint 3-qutrit gmon-ring engine (open-system, rotating frame).

Unlike ``TransmonEngine`` (which evolves each qubit independently), the tunable
couplers entangle the ring, so this engine carries a single joint density matrix
on the full 3-qutrit space (3**3 = 27 dim). Qutrits capture the finite on-site
interaction U (anharmonicity), so the two-photon "hole" sector is faithful.

Physics
-------
We work in the rotating frame of each bare qubit. The diagonal qubit-frequency
terms vanish; the anharmonic on-site term ``(alpha/2) n(n-1)`` is diagonal in the
number basis and survives (this is the Hubbard U). The hopping term acquires the
frame phase ``exp(i*Delta_jk*t)`` with ``Delta_jk = omega_j - omega_k``:

    H(t) = sum_j (alpha_j/2) n_j (n_j - 1)
         + sum_<jk> A_jk(t) [ e^{i Delta_jk t} a_j^dag a_k + e^{-i Delta_jk t} a_k^dag a_j ]

with the agent-designed coupler modulation
``A_jk(t) = 2 * g0_jk * amp_jk * env_jk(t) * cos(omega_mod_jk t + phi_jk)``.
We integrate this *honestly* (no manual RWA): a resonant slow term survives only
when ``omega_mod ~= |Delta_jk|`` (so the wrong-frequency trap produces no hopping
for free), and the surviving complex coefficient is the Peierls phase that encodes
the synthetic flux. The factor 2 makes the realized RWA hopping coefficient equal
``g0_jk * amp`` (the Stage-A calibration target).

The surviving coefficient is ``exp(-i * sign(Delta_jk) * phi_jk)`` on
``a_j^dag a_k``, which is ``exp(+i*phi_jk)`` on the *downhill* hop (from the
higher-frequency qubit into the lower-frequency one) whichever way the coupler's
qubit pair is written. With the shipped ordering ``omega_Q1 < omega_Q2 < omega_Q3``
the downhill hops are Q2->Q1, Q3->Q2 and Q3->Q1, so CP31's points across the loop
``Q1 -> Q3 -> Q2 -> Q1`` rather than around it and the gauge-invariant flux is the
DIRECTED ring sum ``Phi_B = phi12 + phi23 - phi31``. That directed sum is
the convention used by this engine. Reversing the loop orientation yields
``phi31 - phi12 - phi23`` for the same device state. The directed rule is
derived from ``effective_hamiltonian``.

A hidden per-coupler ``phase_offset_rad`` is added to the agent's commanded phase
(default 0.0). This is the phase analog of the stale-g0 drift: the realized flux
differs from the commanded one by the ring-sum of these offsets, so an agent that
trusts its commanded phases mis-reads the chirality.

Single-qubit XY(Z) rotations are applied as instantaneous unitaries on the {|0>,|1>}
levels (state prep + pre-measure tomography). T1/T2 enter as Lindblad collapse
operators. Measurement folds qutrit |2> into the excited (|1>) readout outcome and
samples raw correlated IQ shots from the joint final state.
"""

from __future__ import annotations

import math
import time

import numpy as np
import qutip as qt

from qiqcbench.qsim.core.angles import combine_angles
from qiqcbench.qsim.core.wire import (
    JobIQData,
    JobResult,
    JobResultMetadata,
)
from qiqcbench.qsim.jobs import ModelRequestError
from qiqcbench.qsim.qtypes.gmon_ring_3q.device import HiddenGmonRingConfig
from qiqcbench.qsim.qtypes.gmon_ring_3q.wire import (
    CouplerSetting,
    GmonDelayOp,
    GmonEvolveOp,
    GmonMeasureOp,
    GmonRotationOp,
    GmonSequenceOp,
)

_TWO_PI = 2.0 * math.pi
_LEVELS = 3  # qutrit per site
_HZ_TO_RAD_PER_NS = _TWO_PI * 1e-9


def _hz_to_rad_per_ns(hz: float) -> float:
    return hz * _HZ_TO_RAD_PER_NS


def _require_float(value: float | str, what: str) -> float:
    if not isinstance(value, (int, float)):
        raise ModelRequestError(f"Unresolved sweep placeholder in {what}: {value!r}")
    return float(value)


def _segment_envelope(envelope: str, ramp_ns: float | None, duration_ns: float):
    """Return env(t_local) for a single evolve segment, t_local in [0, duration]."""
    if envelope == "constant":
        return lambda _t: 1.0
    if envelope == "raised_cosine":
        r = float(ramp_ns or 0.0)
        if r <= 0.0:
            return lambda _t: 1.0
        r = min(r, duration_ns / 2.0)

        def env(t: float) -> float:
            if t < r:
                return 0.5 * (1.0 - math.cos(math.pi * t / r))
            if t > duration_ns - r:
                return 0.5 * (1.0 - math.cos(math.pi * (duration_ns - t) / r))
            return 1.0

        return env
    raise ValueError(f"Unsupported envelope {envelope!r}")


class GmonRingEngine:
    """Stateful per-job engine. One instance per gmon sequence run."""

    def __init__(self, hidden: HiddenGmonRingConfig, rng: np.random.Generator):
        self.hidden = hidden
        self.rng = rng
        self.qids = [q.id for q in hidden.qubits]
        self._site = {qid: i for i, qid in enumerate(self.qids)}
        n = len(self.qids)
        if n != 3:
            raise ValueError("gmon_ring_3q engine expects exactly 3 qubits")

        # Per-site ladder operators on the joint 3-qutrit space.
        ident = qt.qeye(_LEVELS)
        self._a: list[qt.Qobj] = []
        for i in range(n):
            ops = [ident] * n
            ops[i] = qt.destroy(_LEVELS)
            self._a.append(qt.tensor(ops))
        self._num = [a.dag() * a for a in self._a]
        self._ident = qt.tensor([ident] * n)

        # Rotating-frame static term: on-site anharmonicity (Hubbard U).
        self._omega = [_hz_to_rad_per_ns(q.frequency_hz) for q in hidden.qubits]
        self._alpha = [_hz_to_rad_per_ns(q.anharmonicity_hz) for q in hidden.qubits]
        H_static = qt.qzero([_LEVELS] * n)
        for i in range(n):
            ni = self._num[i]
            H_static = H_static + 0.5 * self._alpha[i] * ni * (ni - self._ident)
        self._H_static = H_static

        # Coupler lookup keyed by id, plus the per-bond site indices and detuning.
        self._couplers = {cp.id: cp for cp in hidden.couplers}
        self._g0 = {cp.id: _hz_to_rad_per_ns(cp.g0_hz) for cp in hidden.couplers}
        # Hidden Peierls-phase calibration offset added to the commanded phase
        # (default 0.0 -> a perfectly calibrated device). The realized synthetic
        # flux differs from the commanded one by the ring-sum of these offsets.
        self._phase_offset = {cp.id: float(cp.phase_offset_rad) for cp in hidden.couplers}
        self._bond: dict[str, tuple[int, int, float]] = {}
        for cp in hidden.couplers:
            qa, qb = cp.qubits
            sa, sb = self._site[qa], self._site[qb]
            delta = self._omega[sa] - self._omega[sb]
            self._bond[cp.id] = (sa, sb, delta)

        # Collapse operators (T1 relaxation + pure dephasing).
        self._c_ops: list[qt.Qobj] = []
        for i, q in enumerate(hidden.qubits):
            t1_ns = q.t1_s * 1e9
            t2_ns = q.t2_s * 1e9
            self._c_ops.append(math.sqrt(1.0 / t1_ns) * self._a[i])
            inv_tphi = 1.0 / t2_ns - 1.0 / (2.0 * t1_ns)
            if inv_tphi > 0:
                self._c_ops.append(math.sqrt(2.0 * inv_tphi) * self._num[i])

        # Initial joint state |000>.
        self._rho = qt.ket2dm(qt.tensor([qt.basis(_LEVELS, 0)] * n))
        self._t = 0.0  # absolute clock (ns) for modulation-phase continuity
        self._idle_propagators: dict[float, qt.Qobj] = {}

    # ---- white-box helpers (used by engine tests; not on the agent path) ----

    def set_density_matrix(self, rho: qt.Qobj) -> None:
        self._rho = rho

    @property
    def density_matrix(self) -> qt.Qobj:
        """The current joint state, for verifier-side replay of a submitted job."""
        return self._rho

    def apply_ops(self, sequence: list[GmonSequenceOp]) -> None:
        """Replay control ops without measuring.

        Verifiers use this to reconstruct the state a submitted job actually
        prepared, so a reported observable can be checked against the state the
        agent really made rather than trusted as a scalar. Measurement ops are
        skipped: replay reconstructs the pre-readout state, not the sampling.
        """
        for op in sequence:
            if isinstance(op, GmonRotationOp):
                self._apply_rotation(op)
            elif isinstance(op, GmonEvolveOp):
                self._evolve(op)
            elif isinstance(op, GmonDelayOp):
                self._idle(_require_float(op.duration_ns, "delay duration"))
            elif isinstance(op, GmonMeasureOp):
                continue
            else:
                raise ValueError(f"Unknown op kind: {op!r}")

    def rotation_unitary(self, op: GmonRotationOp) -> qt.Qobj:
        """The joint unitary one ``xy_rot`` applies.

        Public because a verifier reconstructing what a job measured has to pull
        an observable back through the agent's tomography rotations, and a
        second implementation of this matrix would drift from the one the run
        actually used. The ``|2>`` level is untouched: these are qubit-subspace
        rotations.
        """
        theta = float(op.angle_rad)
        c, sn = math.cos(theta / 2.0), math.sin(theta / 2.0)
        if op.axis == "x":
            m = np.array([[c, -1j * sn, 0], [-1j * sn, c, 0], [0, 0, 1]], dtype=complex)
        elif op.axis == "y":
            m = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]], dtype=complex)
        else:  # z
            m = np.array(
                [[np.exp(-1j * theta / 2.0), 0, 0], [0, np.exp(1j * theta / 2.0), 0], [0, 0, 1]],
                dtype=complex,
            )
        ops = [qt.qeye(_LEVELS)] * len(self.qids)
        ops[self._site[op.qubit]] = qt.Qobj(m)
        return qt.tensor(ops)

    def idle_propagator(self, duration_ns: float) -> qt.Qobj:
        """Superoperator of an idle ``delay``, for verifier-side observable pullback.

        Same static Hamiltonian and collapse operators ``_idle`` integrates, so a
        pulled-back observable carries the decoherence the delay actually applied
        instead of an idealized unitary.

        Exponentiated directly rather than integrated: the generator is
        time-independent, so ``exp(L t)`` is exact, and it agrees with
        ``qt.propagator`` to 3e-13 while taking ~0.2 s instead of ~19 s. That
        gap is load-bearing, not cosmetic -- a verifier pulling observables back
        through one delay per cited job would otherwise blow its own timeout on
        an agent that inserted an idle. Results are cached per duration because
        a submission that uses a delay generally uses the same one throughout.
        """
        key = round(float(duration_ns), 9)
        cached = self._idle_propagators.get(key)
        if cached is None:
            cached = (qt.liouvillian(self._H_static, self._c_ops) * key).expm()
            self._idle_propagators[key] = cached
        return cached

    def effective_hamiltonian(self, settings: list[CouplerSetting]) -> qt.Qobj:
        """RWA effective Hamiltonian for resonant settings (test/diagnostic use).

        Mirrors the dynamics' slow-term convention: a coupler modulated at
        ``omega_mod = |Delta|`` contributes ``g0*amp*[e^{-i s phi} a_a^dag a_b + h.c.]``
        where ``s = sign(Delta)``; a DC bond (Delta=0, omega_mod=0) contributes the
        real ``2*g0*amp*cos(phi)`` coupling.
        """
        H = self._H_static
        for s in settings:
            cp = self._couplers[s.coupler]
            sa, sb, delta = self._bond[cp.id]
            amp = _require_float(s.amp, "coupler amp")
            # Reduce at ingestion: cos is 2*pi-periodic in phi, so this is exact,
            # and it is what keeps a huge commanded phase from destroying the
            # drive (see the note in _evolve).
            # Reduce each term before combining, as in _evolve: adding first
            # loses the hidden offset for a large commanded phase.
            phi = combine_angles(
                _require_float(s.phase_rad, "coupler phase"),
                self._phase_offset[cp.id],
            )
            g = self._g0[cp.id] * amp
            op = self._a[sa].dag() * self._a[sb]
            if abs(delta) < 1e-12:
                H = H + 2.0 * g * math.cos(phi) * (op + op.dag())
            else:
                ph = np.exp(-1j * math.copysign(1.0, delta) * phi)
                H = H + g * (ph * op + np.conj(ph) * op.dag())
        return H

    # ---- op handlers ----

    def _apply_rotation(self, op: GmonRotationOp) -> None:
        u = self.rotation_unitary(op)
        self._rho = u * self._rho * u.dag()

    def _evolve(self, op: GmonEvolveOp) -> float:
        duration = _require_float(op.duration_ns, "evolve duration")
        if duration <= 0:
            return 0.0
        env = _segment_envelope(op.envelope, op.ramp_ns, duration)
        t0 = self._t
        H: list = [self._H_static]
        for s in op.couplers:
            if s.coupler not in self._bond:
                raise ValueError(f"Unknown coupler {s.coupler!r}")
            sa, sb, delta = self._bond[s.coupler]
            amp = _require_float(s.amp, "coupler amp")
            # REDUCE THE COMMANDED PHASE AT INGESTION. The coefficient below is
            # cos(omega*t + phi), and at phi = 1e32 the sum absorbs the time term
            # entirely -- phi + omega*t == phi for every t in the window -- so the
            # drive freezes at cos(phi) and stops modulating. Reduced first, the
            # same commanded angle oscillates as it should. cos is 2*pi-periodic
            # in phi, so this is mathematically exact and numerically strictly
            # better; it also makes 'same canonical angle implies same dynamics'
            # true, which is what lets a verifier treat two spellings of one angle
            # as one drive.
            # REDUCE EACH TERM BEFORE COMBINING THEM. The coefficient below is
            # cos(omega*t + phi), and phi is the commanded phase PLUS a hidden
            # per-coupler offset. Adding first and reducing after loses the
            # offset entirely once the commanded phase is large: at 1e32,
            # `phase + offset == phase`, so a hidden offset of -0.65 vanished and
            # the realized angle was wrong by the whole offset. Reducing each
            # term first makes both O(pi), so the addition cannot absorb either.
            # Reduce before combining, so neither angle is absorbed.
            phi = combine_angles(
                _require_float(s.phase_rad, "coupler phase"),
                self._phase_offset[s.coupler],
            )
            freq = _require_float(s.freq_hz, "coupler freq_hz")
            if not (0.0 <= amp <= 1.0):
                # Model-owned: the backend validates a sweep template with
                # allow_placeholders=True, so a swept amp is never bound-checked
                # at admission and only this check sees the resolved value.
                raise ModelRequestError(f"coupler amp out of range [0,1]: {amp}")
            omega_mod = _hz_to_rad_per_ns(freq)
            a_amp = 2.0 * self._g0[s.coupler] * amp
            op_fwd = self._a[sa].dag() * self._a[sb]
            op_bwd = op_fwd.dag()

            def coeff_fwd(t, _a=a_amp, _w=omega_mod, _p=phi, _d=delta, _t0=t0, _env=env):
                tt = t + _t0
                return _a * _env(t) * math.cos(_w * tt + _p) * np.exp(1j * _d * tt)

            def coeff_bwd(t, _a=a_amp, _w=omega_mod, _p=phi, _d=delta, _t0=t0, _env=env):
                tt = t + _t0
                return _a * _env(t) * math.cos(_w * tt + _p) * np.exp(-1j * _d * tt)

            H.append([op_fwd, coeff_fwd])
            H.append([op_bwd, coeff_bwd])

        npts = max(2, int(round(duration / 0.5)) + 1)
        ts = np.linspace(0.0, duration, npts)
        result = qt.mesolve(
            H,
            self._rho,
            ts,
            c_ops=self._c_ops,
            options={"store_final_state": True, "nsteps": 100_000, "max_step": 2.0},
        )
        self._rho = result.final_state
        self._t = t0 + duration
        return duration

    def _idle(self, duration_ns: float) -> float:
        if duration_ns <= 0:
            return 0.0
        ts = np.array([0.0, duration_ns])
        result = qt.mesolve(
            self._H_static,
            self._rho,
            ts,
            c_ops=self._c_ops,
            options={"store_final_state": True, "nsteps": 100_000},
        )
        self._rho = result.final_state
        self._t += duration_ns
        return duration_ns

    def _measure(self, qubits: list[str], shots: int) -> dict[str, list[list[float]]]:
        """Sample raw correlated IQ shots; qutrit |2> folds into the excited outcome."""
        sites = [self._site[q] for q in qubits]
        k = len(sites)
        pops = np.real(self._rho.diag())  # populations of the 27 basis states
        n = len(self.qids)
        # Aggregate populations into the 2^k joint excitation pattern over measured qubits.
        joint = np.zeros(1 << k)
        for idx, p in enumerate(pops):
            occ = []
            rem = idx
            for _ in range(n):
                occ.append(rem % _LEVELS)
                rem //= _LEVELS
            occ.reverse()  # occ[site] = n_site
            pattern = 0
            for bit, site in enumerate(sites):
                if occ[site] >= 1:
                    pattern |= 1 << bit
            joint[pattern] += p
        joint = np.clip(joint, 0.0, None)
        total = joint.sum()
        joint = joint / total if total > 0 else np.full(1 << k, 1.0 / (1 << k))

        draws = self.rng.choice(1 << k, size=shots, p=joint)
        ro_by_id = {r.id: r for r in self.hidden.readout}
        out: dict[str, list[list[float]]] = {}
        for bit, qid in enumerate(qubits):
            ro = ro_by_id[qid]
            true_excited = ((draws >> bit) & 1).astype(bool)
            flip_to_one = self.rng.random(shots) < ro.p_0_to_1
            flip_to_zero = self.rng.random(shots) < ro.p_1_to_0
            observed = np.where(true_excited, ~flip_to_zero, flip_to_one)
            i1, q1 = ro.iq_1_mean
            i0, q0 = ro.iq_0_mean
            means_i = np.where(observed, i1, i0)
            means_q = np.where(observed, q1, q0)
            noise = self.rng.normal(0.0, ro.iq_sigma, size=(shots, 2))
            iq = np.stack([means_i + noise[:, 0], means_q + noise[:, 1]], axis=1)
            out[qid] = iq.tolist()
        return out

    # ---- driver ----

    def run_sequence(
        self,
        sequence: list[GmonSequenceOp],
        shots: int,
        device_id: str,
        job_id: str,
    ) -> JobResult:
        t_start = time.perf_counter()
        last_iq: dict[str, list[list[float]]] | None = None
        seq_dur_ns = 0.0
        # Only ModelRequestError is caught. GmonSimulatorBackend validates a
        # sweep TEMPLATE with allow_placeholders=True, so the value each point
        # resolves to is never bound-checked at admission and reaches the engine
        # -- an ordinary model mistake that must stay model-owned (free-form
        # error, no stamp), or it buys a free infrastructure rerun. Everything
        # else raised here is qsim-owned and must propagate to JobManager, which
        # types it as the sanctioned JOB_EXECUTION_ERROR that
        # is_internal_job_failure recognizes and get_job_result stamps
        # failure_kind="qsim_internal". The blanket except this replaces returned
        # str(exc) for BOTH cases -- the empty string for MemoryError -- so a
        # qsim fault was attributed to the model.
        try:
            for op in sequence:
                if isinstance(op, GmonRotationOp):
                    self._apply_rotation(op)
                elif isinstance(op, GmonEvolveOp):
                    seq_dur_ns += self._evolve(op)
                elif isinstance(op, GmonDelayOp):
                    seq_dur_ns += self._idle(_require_float(op.duration_ns, "delay duration"))
                elif isinstance(op, GmonMeasureOp):
                    last_iq = self._measure(op.qubits, shots)
                else:  # pragma: no cover
                    raise ValueError(f"Unknown op kind: {op!r}")
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
                error="Sequence did not contain a gmon_measure op.",
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


__all__ = ["GmonRingEngine"]
