"""Open-system Lindblad engine for the driven-dissipative transmon array.

Physics
-------
A hard-core (2-level) open chain of ``n`` transmon sites. The agent picks one
adjacent pair and turns on energy-selective local pump/loss reservoirs to
autonomously stabilize the single-excitation Bell singlet
``|-> = (|ge>-|eg>)/sqrt(2)``.

Hamiltonian (rotating frame, angular units rad/us = 2*pi*MHz):

    H = sum_i  w_i  n_i  +  sum_<i,i+1>  g_i ( sig+_i sig-_{i+1} + h.c. )

with ``w_i = 2*pi*Delta_i`` (per-site detuning incl. representative ac-Stark) and
``g_i = 2*pi*J_i`` (per-bond hopping). The single-excitation eigenstates of an
isolated resonant pair are ``|+>`` (energy +J) and ``|->`` (energy -J).

Engineered reservoirs (KEY MODELLING CHOICE — time-independent effective
Lindblad operators; the sideband drives are absorbed into effective rates, so the
Liouvillian is CONSTANT and the steady state is ``null(L)``):

    L_pump,i = sum_{n,m} sqrt( Gamma(E_n-E_m-delta_s; g_s) ) <n|sig+_i|m> |n><m|
    L_loss,i = sum_{n,m} sqrt( Gamma(E_m-E_n-delta_d; g_d) ) <n|sig-_i|m> |n><m|

for each pair site ``i``, where the Lorentzian rate filter

    Gamma(x; g) = rate_scale * (2*pi*g)^2 * kappa / ( x^2 + (kappa/2)^2 )

peaks (width ``kappa``) on the targeted transition. For ``delta_s ~ -J`` (pump
the |gg>->|-> transition) and ``delta_d ~ +J`` (drain |+>->|gg> and |ee>->|->)
the singlet ``|->`` becomes a near-dark steady state. SWAPPING the detunings
stabilizes ``|+>`` instead (the wrong-eigenstate trap).

Intrinsic imperfections (also Lindblad collapse operators):
  * per-site ``T1`` (``sig-_i``) and ``Tphi`` (``n_i``);
  * a COLLECTIVE Purcell channel ``(sig-_a + sig-_b)`` on the pair, which
    annihilates ``|->`` but makes ``|+>`` superradiant (App F: |+> shorter-lived);
  * a broadband thermal pump ``sig+_i`` (rate ~ reservoir thermal population)
    that limits the achievable fidelity, pair-dependently.

Edge-vs-middle gap is EMERGENT: the chain hopping couples each pair site to its
spectator neighbours, so the middle pair (two spectator neighbours) leaks more
than an edge pair (one). Per-bond ``J`` / per-site ``T1`` / thermal asymmetry
breaks the edge-edge degeneracy so exactly one edge pair is best.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence

import numpy as np
import qutip as qt

from qiqcbench.qsim.core.wire import (
    JobBitstringData,
    JobResult,
    JobResultMetadata,
)
from qiqcbench.qsim.qtypes.driven_dissipative_transmon_array.device import HiddenDdtaConfig

_TWO_PI = 2.0 * math.pi
_INV_SQRT2 = 1.0 / math.sqrt(2.0)

# Single-qubit analysis rotations: returned bit carries the named Pauli after a
# computational (Z) measurement.  z -> identity (bare computational readout).
_ROT = {
    "z": np.eye(2, dtype=complex),
    "x": np.array([[_INV_SQRT2, _INV_SQRT2], [_INV_SQRT2, -_INV_SQRT2]], dtype=complex),
    "y": np.array([[_INV_SQRT2, -1j * _INV_SQRT2], [_INV_SQRT2, 1j * _INV_SQRT2]], dtype=complex),
}


def parse_pair(pair: str, qubits: Sequence[str]) -> tuple[int, int]:
    """Map a pair label like "q0,q1" to ordered site indices (a<b)."""
    parts = [p.strip() for p in pair.split(",")]
    if len(parts) != 2:
        raise ValueError(f"pair must be 'qa,qb'; got {pair!r}")
    try:
        a, b = (qubits.index(parts[0]), qubits.index(parts[1]))
    except ValueError as exc:
        raise ValueError(f"pair {pair!r} references unknown qubit(s)") from exc
    if a == b:
        raise ValueError(f"pair {pair!r} must reference two distinct qubits")
    return (a, b) if a < b else (b, a)


class DdtaEngine:
    """Stateless-per-call open-system engine; one instance per device (or job)."""

    def __init__(self, hidden: HiddenDdtaConfig, rng: np.random.Generator) -> None:
        self.hidden = hidden
        self.rng = rng
        self.n = len(hidden.site_t1_us)
        if len(hidden.hopping_mhz) != self.n - 1:
            raise ValueError("hopping_mhz must have length n_sites - 1")
        if len(hidden.site_detuning_mhz) != self.n or len(hidden.site_thermal_pop) != self.n:
            raise ValueError("per-site lists must have length n_sites")
        self.qubits = [r.id for r in hidden.readout]
        if len(self.qubits) != self.n:
            raise ValueError("readout must have one entry per site")
        self._readout = {r.id: (r.p_0_to_1, r.p_1_to_0) for r in hidden.readout}
        self.dim = 2**self.n

        self.kappa_rad = _TWO_PI * hidden.kappa_mhz
        self.rate_scale = hidden.reservoir_rate_scale

        # Per-site embedded ladder/number operators on the full 2**n space.
        ident = qt.qeye(2)
        self._dims = [[2] * self.n, [2] * self.n]
        self._sm: list[qt.Qobj] = []
        self._num: list[qt.Qobj] = []
        for i in range(self.n):
            ops = [ident] * self.n
            ops[i] = qt.destroy(2)
            sm = qt.tensor(ops)
            self._sm.append(sm)
            self._num.append(sm.dag() * sm)

        # Static chain Hamiltonian (rad/us): per-site detuning + nearest-neighbour hopping.
        h = qt.qzero([2] * self.n)
        for i in range(self.n):
            h = h + _TWO_PI * hidden.site_detuning_mhz[i] * self._num[i]
        for bond in range(self.n - 1):
            g = _TWO_PI * hidden.hopping_mhz[bond]
            hop = self._sm[bond].dag() * self._sm[bond + 1]
            h = h + g * (hop + hop.dag())
        self._H = h
        # Cache the bare site raising/lowering matrices (numpy) for the per-call
        # eigenbasis transforms (the energy-selective filters now live in the
        # eigenbasis of the drive-dressed H_eff, which depends on the coupling g).
        self._sp_full = [sm.dag().full() for sm in self._sm]
        self._sm_full = [sm.full() for sm in self._sm]
        self.ac_stark_rad = _TWO_PI * hidden.ac_stark_mhz_per_g2
        self.leak_scale = hidden.leak_scale

        # Intrinsic, settings-independent collapse operators (T1, Tphi on all sites).
        self._intrinsic: list[qt.Qobj] = []
        for i in range(self.n):
            self._intrinsic.append(math.sqrt(1.0 / hidden.site_t1_us[i]) * self._sm[i])
            self._intrinsic.append(math.sqrt(2.0 / hidden.site_tphi_us[i]) * self._num[i])

    # ---- reservoir collapse operators (settings dependent) ----

    def _lorentzian_rate(self, arg_rad: np.ndarray, g_mhz: float) -> np.ndarray:
        g_rad = _TWO_PI * g_mhz
        half = self.kappa_rad / 2.0
        return self.rate_scale * (g_rad**2) * self.kappa_rad / (arg_rad**2 + half**2)

    def _filtered_jump(
        self,
        op_eig: np.ndarray,
        bohr: np.ndarray,
        center_rad: float,
        g_mhz: float,
        vecs: np.ndarray,
    ) -> qt.Qobj:
        """Energy-selective jump operator (computational basis) for one pair site.

        ``bohr[n,m]`` is the system energy injected by ``op`` on the m->n transition
        (pump: E_n-E_m; loss: E_m-E_n) in the eigenbasis ``vecs`` of H_eff. The filter
        peaks where ``bohr == center_rad`` (the static-referenced target E_ref + 2*pi*delta).
        """
        x = bohr - center_rad
        weight = np.sqrt(self._lorentzian_rate(x, g_mhz))
        ltilde = weight * op_eig
        comp = vecs @ ltilde @ vecs.conj().T
        return qt.Qobj(comp, dims=self._dims)

    def pair_reference_energy(self, pair: tuple[int, int]) -> float:
        """Pair single-excitation reference energy (rad/us): mean of the two site detunings.

        Reservoir detunings ``delta_s``/``delta_d`` are advertised relative to the
        selected pair's single-excitation resonance, so the engine references the
        filters here. With a near-resonant pair this is ~(Delta_a+Delta_b)/2 and the
        optimal detunings sit near -/+ J (shifted by intra-pair detuning / dressing).
        """
        a, b = pair
        return _TWO_PI * 0.5 * (self.hidden.site_detuning_mhz[a] + self.hidden.site_detuning_mhz[b])

    def effective_hamiltonian(
        self, pair: tuple[int, int], g_s_mhz: float, g_d_mhz: float
    ) -> qt.Qobj:
        """Drive-dressed Hamiltonian: the reservoir drives ac-Stark-shift the pair sites.

        The shift is ``alpha * (g_s^2 + g_d^2)`` per pair site (proportional to drive
        power). Because the single-excitation states carry this shift but |gg> does
        not, the |gg>-><target> transition moves to ``-J + alpha*g^2``, so the optimal
        pump/loss detunings (referenced to the low-power resonance) drift with g.
        """
        a, b = pair
        shift = self.ac_stark_rad * (g_s_mhz**2 + g_d_mhz**2)
        if shift == 0.0:
            return self._H
        return self._H + shift * (self._num[a] + self._num[b])

    def build_dynamics(
        self,
        pair: tuple[int, int],
        *,
        delta_s_mhz: float,
        delta_d_mhz: float,
        g_s_mhz: float,
        g_d_mhz: float,
    ) -> tuple[qt.Qobj, list[qt.Qobj]]:
        """Return ``(H_eff, c_ops)`` for the given reservoir settings."""
        a, b = pair
        h_eff = self.effective_hamiltonian(pair, g_s_mhz, g_d_mhz)
        evals, vecs = np.linalg.eigh(h_eff.full())
        d_e = evals[:, None] - evals[None, :]  # dE[n,m] = E_n - E_m (H_eff eigenbasis)
        e_ref = self.pair_reference_energy(pair)  # low-power calibration reference
        center_s = e_ref + _TWO_PI * delta_s_mhz
        center_d = e_ref + _TWO_PI * delta_d_mhz
        c_ops = list(self._intrinsic)
        for i in (a, b):
            sp_eig = vecs.conj().T @ self._sp_full[i] @ vecs
            sm_eig = vecs.conj().T @ self._sm_full[i] @ vecs
            # Pump (raising): resonant when E_n - E_m == E_ref + delta_s.
            c_ops.append(self._filtered_jump(sp_eig, d_e, center_s, g_s_mhz, vecs))
            # Loss (lowering): resonant when E_m - E_n == E_ref + delta_d.
            c_ops.append(self._filtered_jump(sm_eig, -d_e, center_d, g_d_mhz, vecs))
            # Broadband thermal pump (~ g^2 * n_th): a ~g-independent infidelity floor.
            n_th = self.hidden.site_thermal_pop[i]
            if n_th > 0:
                gamma_th = self.rate_scale * 4.0 * (_TWO_PI * g_s_mhz) ** 2 / self.kappa_rad * n_th
                c_ops.append(math.sqrt(gamma_th) * self._sm[i].dag())
            # Off-resonant strong-drive leakage (~ g^4): a higher-order drive process,
            # negligible at low g but dominant at high g, so the fidelity has an
            # interior-optimal coupling (max g is NOT best).
            if self.leak_scale > 0:
                gamma_leak = self.leak_scale * (_TWO_PI * g_s_mhz) ** 4 / self.kappa_rad**3
                if gamma_leak > 0:
                    c_ops.append(math.sqrt(gamma_leak) * self._sm[i].dag())
        # Collective Purcell on the pair (dark to |->, superradiant for |+>).
        gamma_coll = self.hidden.collective_purcell_mhz
        if gamma_coll > 0:
            c_ops.append(math.sqrt(gamma_coll) * (self._sm[a] + self._sm[b]))
        return h_eff, c_ops

    # ---- state prep + evolution ----

    def initial_state(self, pair: tuple[int, int], label: str) -> qt.Qobj:
        a, b = pair
        excited = {a: label[0] == "e", b: label[1] == "e"}
        kets = [qt.basis(2, 1 if excited.get(i, False) else 0) for i in range(self.n)]
        return qt.ket2dm(qt.tensor(kets))

    def evolve(
        self,
        pair: tuple[int, int],
        label: str,
        h_eff: qt.Qobj,
        c_ops: list[qt.Qobj],
        times_us: Sequence[float],
    ) -> list[qt.Qobj]:
        """Return the density matrix at each requested duration.

        The Liouvillian is time-independent, so rho(t) = exp(L t) rho0 is exact.
        We build the Lindblad superoperator with ``qutip.liouvillian`` and apply its
        matrix exponential ``(L t).expm()`` to the vectorized state, rather than the
        adaptive ODE in ``mesolve``: at large reservoir coupling the dynamics are
        stiff, so ``mesolve`` takes many tiny steps and a 16-dim problem becomes a
        multi-second one, whereas the (256-dim) superoperator ``expm`` is exact,
        robust to stiffness, and far faster.
        """
        liouv = qt.liouvillian(h_eff, c_ops)
        rho0_vec = qt.operator_to_vector(self.initial_state(pair, label))
        return [qt.vector_to_operator((liouv * float(t)).expm() * rho0_vec) for t in times_us]

    def steady_state(
        self,
        pair: tuple[int, int],
        *,
        delta_s_mhz: float,
        delta_d_mhz: float,
        g_s_mhz: float,
        g_d_mhz: float,
    ) -> qt.Qobj:
        """White-box helper: exact steady state null(L) (documentation / sanity only)."""
        h_eff, c_ops = self.build_dynamics(
            pair,
            delta_s_mhz=delta_s_mhz,
            delta_d_mhz=delta_d_mhz,
            g_s_mhz=g_s_mhz,
            g_d_mhz=g_d_mhz,
        )
        return qt.steadystate(h_eff, c_ops)

    # ---- fidelity / correlator helpers (white-box) ----

    def _pair_bell(self, sign: int) -> qt.Qobj:
        """|±> = (|ge> ± |eg>)/sqrt2 on a 2-qubit space (first qubit = lower-index site)."""
        ge = qt.tensor(qt.basis(2, 0), qt.basis(2, 1))
        eg = qt.tensor(qt.basis(2, 1), qt.basis(2, 0))
        return (ge + sign * eg).unit()

    def fidelity_minus(self, rho: qt.Qobj, pair: tuple[int, int]) -> float:
        """Ideal (no-readout) <-|rho_pair|-> from the reduced pair state."""
        rho_ab = rho.ptrace(list(pair))
        return float(np.real(qt.expect(self._pair_bell(-1).proj(), rho_ab)))

    def fidelity_plus(self, rho: qt.Qobj, pair: tuple[int, int]) -> float:
        rho_ab = rho.ptrace(list(pair))
        return float(np.real(qt.expect(self._pair_bell(+1).proj(), rho_ab)))

    def ideal_correlators(self, rho: qt.Qobj, pair: tuple[int, int]) -> dict[str, float]:
        """Ideal <XX>,<YY>,<ZZ> on the pair (no readout)."""
        rho_ab = rho.ptrace(list(pair))
        return {
            "xx": float(np.real(qt.expect(qt.tensor(qt.sigmax(), qt.sigmax()), rho_ab))),
            "yy": float(np.real(qt.expect(qt.tensor(qt.sigmay(), qt.sigmay()), rho_ab))),
            "zz": float(np.real(qt.expect(qt.tensor(qt.sigmaz(), qt.sigmaz()), rho_ab))),
        }

    def raw_correlators(self, rho: qt.Qobj, pair: tuple[int, int]) -> dict[str, float]:
        """Expected raw <XX>,<YY>,<ZZ> INCLUDING asymmetric readout confusion.

        Deterministic (analytic) counterpart of the sampled measurement: the
        readout maps a true measured bit ``b`` to an observed eigenvalue with
        expectation ``E[o|b=0]=1-2*p01`` and ``E[o|b=1]=-(1-2*p10)`` per site.
        Used to freeze the honest-protocol raw fidelity reference (no shot noise).
        """
        a, b = pair
        p01a, p10a = self._readout[self.qubits[a]]
        p01b, p10b = self._readout[self.qubits[b]]
        ea = (1.0 - 2.0 * p01a, -(1.0 - 2.0 * p10a))
        eb = (1.0 - 2.0 * p01b, -(1.0 - 2.0 * p10b))
        out = {}
        for name, basis in (("xx", "x"), ("yy", "y"), ("zz", "z")):
            probs = self._measure_probs(rho, [(a, basis), (b, basis)])  # idx: a MSB, b LSB
            corr = 0.0
            for idx in range(4):
                corr += probs[idx] * ea[(idx >> 1) & 1] * eb[idx & 1]
            out[name] = float(corr)
        return out

    def raw_fidelity_minus(self, rho: qt.Qobj, pair: tuple[int, int]) -> float:
        """Expected raw <-|rho|-> from readout-degraded correlators (deterministic)."""
        c = self.raw_correlators(rho, pair)
        return (1.0 - c["xx"] - c["yy"] - c["zz"]) / 4.0

    # ---- measurement (raw per-shot bitstrings + readout confusion) ----

    def _measure_probs(self, rho: qt.Qobj, measure: list[tuple[int, str]]) -> np.ndarray:
        """Joint computational-outcome probabilities over the measured sites (post-rotation).

        Outcome index bit ``i`` (most significant first) is ``measure[i]``, which
        is the order ``measured_qubits`` echoes back to the agent.
        """
        sites = [s for s, _ in measure]
        # Apply per-site analysis rotation on the measured sites.
        rot_ops = []
        meas_basis = {s: b for s, b in measure}
        for i in range(self.n):
            if i in meas_basis:
                rot_ops.append(qt.Qobj(_ROT[meas_basis[i]]))
            else:
                rot_ops.append(qt.qeye(2))
        u = qt.tensor(rot_ops)
        rho_rot = u * rho * u.dag()
        reduced = rho_rot.ptrace(sites)
        probs = np.clip(np.real(np.diag(reduced.full())), 0.0, None)
        # ptrace canonicalizes (sorts) the retained subsystems, so the diagonal
        # is indexed in ASCENDING site order no matter what order was requested.
        # Re-index onto the requested order; without this every non-ascending
        # ``measure`` list returns bits whose positions mean the wrong sites.
        ascending = sorted(sites)
        if ascending != sites:
            axes = [ascending.index(site) for site in sites]
            probs = probs.reshape([2] * len(sites)).transpose(axes).reshape(-1)
        total = probs.sum()
        return probs / total if total > 0 else np.full(len(probs), 1.0 / len(probs))

    def sample(
        self,
        rho: qt.Qobj,
        measure: list[tuple[int, str]],
        shots: int,
    ) -> list[str]:
        """Per-shot bitstrings over the measured sites (site i of the string = measure[i])."""
        sites = [s for s, _ in measure]
        k = len(sites)
        probs = self._measure_probs(rho, measure)
        draws = self.rng.choice(2**k, size=shots, p=probs)
        # Decode outcome index -> per-site bits (measure[0] is the most-significant bit).
        bits = np.array(
            [[(o >> (k - 1 - i)) & 1 for i in range(k)] for o in range(2**k)], dtype=int
        )
        out_bits = bits[draws]  # (shots, k)
        # Asymmetric readout confusion per measured site.
        for col, site in enumerate(sites):
            p01, p10 = self._readout[self.qubits[site]]
            col_bits = out_bits[:, col]
            flip0 = (col_bits == 0) & (self.rng.random(shots) < p01)
            flip1 = (col_bits == 1) & (self.rng.random(shots) < p10)
            col_bits[flip0] = 1
            col_bits[flip1] = 0
            out_bits[:, col] = col_bits
        return ["".join(str(b) for b in row) for row in out_bits]

    # ---- driver ----

    def run(
        self,
        *,
        pair_label: str,
        initial_state: str,
        delta_s_mhz: float,
        delta_d_mhz: float,
        g_s_mhz: float,
        g_d_mhz: float,
        durations_us: Sequence[float],
        measure: list[tuple[int, str]],
        shots: int,
        device_id: str,
        job_id: str,
    ) -> JobResult:
        t_start = time.perf_counter()
        # No blanket except here. Public-input rejection belongs to the wire and
        # backend validation layers, which run first, so anything raised below is
        # a qsim-owned execution fault. Letting it propagate reaches JobManager,
        # which types it as the sanctioned JOB_EXECUTION_ERROR that
        # ``is_internal_job_failure`` recognizes and ``get_job_result`` stamps
        # ``failure_kind="qsim_internal"``. Catching it here replaced that marker
        # with ``str(exc)`` -- empty for MemoryError -- so a qsim fault was
        # attributed to the model.
        pair = parse_pair(pair_label, self.qubits)
        h_eff, c_ops = self.build_dynamics(
            pair,
            delta_s_mhz=delta_s_mhz,
            delta_d_mhz=delta_d_mhz,
            g_s_mhz=g_s_mhz,
            g_d_mhz=g_d_mhz,
        )
        states = self.evolve(pair, initial_state, h_eff, c_ops, durations_us)
        per_time = [self.sample(rho, measure, shots) for rho in states]
        return JobResult(
            job_id=job_id,
            device_id=device_id,
            status="complete",
            shots=shots,
            data=JobBitstringData(
                bitstrings=per_time,
                measured_qubits=[s for s, _ in measure],
            ),
            metadata=JobResultMetadata(
                wallclock_ms=int((time.perf_counter() - t_start) * 1000),
                sweep_coords={"duration_us": [float(t) for t in durations_us]},
            ),
        )


__all__ = ["DdtaEngine", "parse_pair"]
