"""Simulator engine for the Kitaev-chain qtype.

One engine is created per job with fresh private randomness. The three
charge-readout primitives below use the exact free-fermion BdG core in
``physics.py``; pulse batches and Clifford RB use the explicitly effective
three-level model in ``pulse.py``.

* ``run_charge_stability`` -- base-band-pulse a 2-dot window onto one bond and
  scan the two dot detunings. The global quantum-capacitance channel resolves
  parity where the even-branch curvature is strong (near the common-mode
  resonance); the local charge-sensor channel resolves parity with a fidelity
  set by the parity charge-contrast, which vanishes on the indistinguishability
  line ``mu_LD = (Delta-t)/(Delta+t) mu_RD`` and flips sign across it. Returns
  raw per-shot dual-channel parity bits; the agent forms rho_MR and reads the
  line's tilt -> the bond's |t|/|Delta| balance (an amplitude defect tilts it).

* ``run_protection_sweep`` -- global common-mode detuning. The equilibrium
  parity polarization ``P_M = tanh(dE / 2 T_p)`` tracks the edge-mode splitting
  ``dE``; returns raw per-shot parity bits so the agent estimates P_M(mu) and
  fits the protection exponent (degraded only by an amplitude defect).

* ``run_subchain_spectroscopy`` -- measure the bulk excitation gap of a
  contiguous sub-chain (noisy). A phase defect suppresses the gap of windows
  spanning it; an amplitude-balanced bond does not.
"""

from __future__ import annotations

import base64

import numpy as np

from qiqcbench.qsim.qtypes.kitaev_chain import clifford_rb as CRB
from qiqcbench.qsim.qtypes.kitaev_chain import physics as P
from qiqcbench.qsim.qtypes.kitaev_chain import pulse as PU
from qiqcbench.qsim.qtypes.kitaev_chain.device import HiddenKitaevChainConfig
from qiqcbench.qsim.qtypes.kitaev_chain.wire import (
    KitaevChargeStabilityData,
    KitaevCliffordRBData,
    KitaevPolarizationData,
    KitaevPulseBatchData,
    KitaevSubchainGapData,
    MajoranaCliffordRBRequest,
    MajoranaPulseBatchRequest,
)


def _pack_bits(arr: np.ndarray) -> str:
    packed = np.packbits(np.ascontiguousarray(arr.reshape(-1), dtype=np.uint8))
    return base64.b64encode(packed.tobytes()).decode("ascii")


class KitaevChainEngine:
    def __init__(self, hidden: HiddenKitaevChainConfig, rng: np.random.Generator) -> None:
        self.hidden = hidden
        self.rng = rng
        self.n_sites = len(hidden.bond_t_ueV) + 1
        self.t = np.asarray(hidden.bond_t_ueV, dtype=complex)
        self.delta = np.array(
            [
                d * np.exp(1j * phi)
                for d, phi in zip(hidden.bond_delta_ueV, hidden.bond_phase_rad, strict=True)
            ],
            dtype=complex,
        )

    # ---- helpers ----

    def _base_mu(self) -> np.ndarray:
        """Zero on-site potentials plus fresh nuisance charge-noise disorder."""
        mu = np.zeros(self.n_sites)
        sigma = self.hidden.mu_disorder_sigma_ueV
        if sigma > 0:
            mu = mu + self.rng.normal(0.0, sigma, size=self.n_sites)
        return mu

    # ---- charge stability (per-bond 2-dot window) ----

    def run_charge_stability(
        self, bond: int, mu_ld_values: list[float], mu_rd_values: list[float], shots: int
    ) -> KitaevChargeStabilityData:
        t_w = abs(self.t[bond])
        d_w = abs(self.delta[bond])
        n_grid = len(mu_ld_values) * len(mu_rd_values)
        g_bits = np.zeros((n_grid, shots), dtype=np.uint8)
        l_bits = np.zeros((n_grid, shots), dtype=np.uint8)
        g = 0
        for mu_ld in mu_ld_values:
            for mu_rd in mu_rd_values:
                delta_cm = 0.5 * (mu_ld + mu_rd)
                # Odd-branch detuning sign convention (van Loo et al. Methods):
                # the indistinguishability line mu_LD=(Delta-t)/(Delta+t) mu_RD
                # is recovered with eps = (mu_RD - mu_LD)/2.
                eps = 0.5 * (mu_rd - mu_ld)
                # global quantum-capacitance channel: parity-resolving near delta=0
                ratio_g = (d_w**2 / (delta_cm**2 + d_w**2)) ** 1.5 if d_w > 0 else 0.0
                eps_g = P.assignment_error_from_snr(self.hidden.global_snr * ratio_g)
                # local charge-sensor channel: signed parity contrast, zero on the tilt line
                dq = P.charge_sensor_qr_odd(eps, t_w) - P.charge_sensor_qr_even(delta_cm, d_w)
                eps_l = P.assignment_error_from_snr(self.hidden.local_snr * abs(dq))
                parity = self.rng.integers(0, 2, size=shots).astype(np.uint8)
                flip_g = (self.rng.random(shots) < eps_g).astype(np.uint8)
                flip_l = (self.rng.random(shots) < eps_l).astype(np.uint8)
                base_l = parity if dq >= 0 else (1 - parity)
                g_bits[g] = parity ^ flip_g
                l_bits[g] = (base_l.astype(np.uint8)) ^ flip_l
                g += 1
        return KitaevChargeStabilityData(
            bond=bond,
            mu_ld_values=[float(x) for x in mu_ld_values],
            mu_rd_values=[float(x) for x in mu_rd_values],
            shots=shots,
            global_parity_b64=_pack_bits(g_bits),
            local_parity_b64=_pack_bits(l_bits),
        )

    # ---- protection sweep (global common-mode detuning) ----

    def run_protection_sweep(
        self, mu_common_values: list[float], shots: int
    ) -> KitaevPolarizationData:
        t_p = self.hidden.poisoning_temp_ueV
        n_mu = len(mu_common_values)
        bits = np.zeros((n_mu, shots), dtype=np.uint8)
        for k, mu_c in enumerate(mu_common_values):
            mu_vec = self._base_mu() + float(mu_c)
            d_e = P.edge_splitting(mu_vec, self.t, self.delta)
            pol = np.tanh(d_e / (2.0 * t_p))  # P_M in [0,1)
            p_odd = 0.5 * (1.0 - pol)
            bits[k] = (self.rng.random(shots) < p_odd).astype(np.uint8)
        return KitaevPolarizationData(
            mu_common_values=[float(x) for x in mu_common_values],
            shots=shots,
            parity_b64=_pack_bits(bits),
        )

    # ---- sub-chain bulk-gap spectroscopy ----

    def run_subchain_spectroscopy(
        self, site_a: int, site_b: int, shots: int
    ) -> KitaevSubchainGapData:
        mu_vec = self._base_mu()
        gap = P.subchain_bulk_gap(mu_vec, self.t, self.delta, site_a, site_b)
        # Finite-shot spectroscopy: measurement error shrinks as 1/sqrt(shots).
        noise = self.hidden.spectroscopy_noise_ueV / np.sqrt(max(shots, 1))
        measured = float(gap + self.rng.normal(0.0, noise)) if np.isfinite(gap) else gap
        return KitaevSubchainGapData(site_a=site_a, site_b=site_b, measured_gap_ueV=measured)

    # ---- pulse-driven Majorana-qubit gates (calibration + Clifford RB) ----

    def pulse_params(self) -> PU.PulseParams:
        h = self.hidden
        if None in (
            h.pulse_drive_rate_ueV,
            h.pulse_e_gap_ueV,
            h.pulse_leakage_lambda,
            h.pulse_charge_noise_sigma_ueV,
            h.pulse_amp_noise_frac,
            h.pulse_poisoning_per_ns,
        ):
            raise ValueError("device has no pulse-gate parameters configured")
        return PU.PulseParams(
            omega_scale_ueV=h.pulse_drive_rate_ueV,
            e_gap_ueV=h.pulse_e_gap_ueV,
            leakage_lambda=h.pulse_leakage_lambda,
            charge_noise_sigma_ueV=h.pulse_charge_noise_sigma_ueV,
            amp_noise_frac=h.pulse_amp_noise_frac,
            poisoning_per_ns=h.pulse_poisoning_per_ns,
        )

    def run_pulse_batch(self, request: MajoranaPulseBatchRequest) -> KitaevPulseBatchData:
        """Evolve each pulse sequence from |0_L> and read out the charge state per shot."""
        params = self.pulse_params()
        levels = np.empty((len(request.sequences), request.shots), dtype=np.uint8)
        for i, seq in enumerate(request.sequences):
            segs = [
                PU.PulseSegment(
                    amp=s.amp,
                    phase_rad=s.phase_rad,
                    detuning_ueV=s.detuning_ueV,
                    duration_ns=s.duration_ns,
                    drag=s.drag,
                )
                for s in seq
            ]
            levels[i] = PU.sample_outcomes(segs, params, self.rng, request.shots)
        levels_b64 = base64.b64encode(np.ascontiguousarray(levels).tobytes()).decode("ascii")
        return KitaevPulseBatchData(
            n_sequences=len(request.sequences),
            shots=request.shots,
            levels_b64=levels_b64,
        )

    def run_clifford_rb(self, request: MajoranaCliffordRBRequest) -> KitaevCliffordRBData:
        """Run qsim-owned uniform 24-element Clifford randomization."""

        return CRB.run_clifford_rb(request, self.pulse_params(), self.rng)


__all__ = ["KitaevChainEngine"]
