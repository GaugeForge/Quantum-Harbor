"""Hidden-config -> sampler-noise conversion for ``surface_code_lattice_surgery``.

T1/T2 idles are Pauli-twirled per syndrome cycle (standard stabilizer-level
approximation): with ``p1 = 1 - exp(-t/T1)`` and ``pphi = 1 - exp(-t/Tphi)`` where
``1/Tphi = 1/T2 - 1/(2 T1)``, the per-cycle idle channel is
``px = py = p1/4``, ``pz = pphi/2 + p1/4``.

Asymmetric measurement + reset errors and the leakage state machine are carried through
to the sampler; only the Pauli part is compiled into the stim circuit.
"""

from __future__ import annotations

import math

from qiqcbench.qsim.qtypes.surface_code_lattice_surgery.circuits import NoiseParams
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery.device import (
    HiddenLatticeSurgeryConfig,
)
from qiqcbench.qsim.qtypes.surface_code_lattice_surgery.sampler import LeakSpec, SamplerNoise


def idle_paulis(t1_us: float, t2_us: float, cycle_us: float) -> tuple[float, float, float]:
    p1 = 1.0 - math.exp(-cycle_us / t1_us)
    inv_tphi = max(1.0 / t2_us - 0.5 / t1_us, 0.0)
    pphi = 1.0 - math.exp(-cycle_us * inv_tphi)
    px = py = p1 / 4.0
    pz = pphi / 2.0 + p1 / 4.0
    return px, py, pz


def sampler_noise_from_hidden(hidden: HiddenLatticeSurgeryConfig) -> SamplerNoise:
    idle_px: dict[int, float] = {}
    idle_py: dict[int, float] = {}
    idle_pz: dict[int, float] = {}
    p_reset: dict[int, float] = {}
    p_m01: dict[int, float] = {}
    p_m10: dict[int, float] = {}
    for q, n in hidden.qubits.items():
        px, py, pz = idle_paulis(n.t1_us, n.t2_us, hidden.cycle_time_us)
        idle_px[q], idle_py[q], idle_pz[q] = px, py, pz
        p_reset[q] = n.p_reset
        p_m01[q] = n.p_m01
        p_m10[q] = n.p_m10
    p_cz = {(min(e.q1, e.q2), max(e.q1, e.q2)): e.p for e in hidden.cz_errors}
    leak = {
        site.qubit: LeakSpec(
            p_enter=site.p_enter,
            lifetime=site.lifetime,
            p_flip_round=site.p_flip_round,
            mode=site.mode,
            delay=site.delay,
        )
        for site in hidden.leakage
    }
    circuit_noise = NoiseParams(
        idle_px=idle_px,
        idle_py=idle_py,
        idle_pz=idle_pz,
        p_cz=p_cz,
        p_reset=p_reset,
        p_meas={},  # measurement noise is the sampler's asymmetric classification layer
    )
    return SamplerNoise(circuit_noise=circuit_noise, p_m01=p_m01, p_m10=p_m10, leak=leak)


def exact_circuit_noise(hidden: HiddenLatticeSurgeryConfig) -> NoiseParams:
    """The hidden Pauli noise WITH symmetrized measurement flips — the in-circuit model
    used by maintainer-side exact-DEM anchors (leakage excluded by construction)."""
    base = sampler_noise_from_hidden(hidden)
    p_meas = {q: 0.5 * (n.p_m01 + n.p_m10) for q, n in hidden.qubits.items()}
    cn = base.circuit_noise
    return NoiseParams(
        idle_px=cn.idle_px,
        idle_py=cn.idle_py,
        idle_pz=cn.idle_pz,
        p_cz=cn.p_cz,
        p_reset=cn.p_reset,
        p_meas=p_meas,
    )


__all__ = ["exact_circuit_noise", "idle_paulis", "sampler_noise_from_hidden"]
