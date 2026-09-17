"""NV sensor-network analytic Ramsey engine (numpy only -- no joint state vector).

Per-node dephasing is independent, so a GHZ over support ``S`` has parity
contrast ``C_S = (prod_{i in S} f_i) * prod_{i in S} exp(-(tau_i/T2*_i)^2)`` and
parity phase ``Phi_S = sum_{i in S} sign_i * theta_i * tau_i``. That lets us
sample raw per-node bitstrings exactly without a 2^k Hilbert space.

Conventions (load-bearing):

* ``theta_i`` (node field rate ``gamma_e B_i``) is in **rad/s**; interrogation
  time ``tau_i`` is in **seconds**; accumulated phase ``phi_i = sign_i theta_i tau_i``.
* Gaussian Ramsey envelope ``C_i(tau) = exp(-(tau/T2*_i)^2)`` (inhomogeneous NV
  dephasing); ``T2*_i`` is stored in microseconds and converted here.
* ``separable``: each node measured independently, ``P(bit=1) = (1 + C_i cos(phi_i + phi_a))/2``.
* ``ghz``: latent parity ``P(even) = (1 + C_S cos(Phi_S + phi_a))/2``; one of the
  ``2^{k-1}`` strings of that parity is drawn uniformly, then per-node readout flips.
* ``link_probe``: node ``i`` entangled with the ideal central station; single-node
  fringe with contrast ``f_i exp(-(tau/T2*_i)^2)`` -- recovers ``f_i`` directly.

Asymmetric readout (``p_0->1``, ``p_1->0``) is applied per node, per shot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NvNodeParams:
    """True per-node parameters, indexed by node id 0..n-1."""

    theta_rad_per_s: tuple[float, ...]
    t2star_s: tuple[float, ...]
    link_fidelity: tuple[float, ...]
    p_0_to_1: tuple[float, ...]
    p_1_to_0: tuple[float, ...]

    @property
    def n_nodes(self) -> int:
        return len(self.theta_rad_per_s)


class NvSensorNetworkEngine:
    """Stateless-per-call analytic Ramsey sampler; one instance per job."""

    def __init__(self, params: NvNodeParams, rng: np.random.Generator) -> None:
        self.p = params
        self.rng = rng

    # ---- contrast / phase ----

    def _envelope(self, node: int, tau_s: float) -> float:
        t2 = self.p.t2star_s[node]
        return float(np.exp(-((tau_s / t2) ** 2))) if t2 > 0 else 0.0

    def _node_phase(self, node: int, sign: int, tau_s: float) -> float:
        return float(sign) * self.p.theta_rad_per_s[node] * tau_s

    # ---- readout ----

    def _apply_readout(self, bits: np.ndarray, node: int) -> np.ndarray:
        """Apply asymmetric 0<->1 readout confusion for ``node`` to a 1-D bit array."""
        p01 = self.p.p_0_to_1[node]
        p10 = self.p.p_1_to_0[node]
        out = bits.copy()
        u = self.rng.random(bits.shape[0])
        flip0 = (bits == 0) & (u < p01)
        flip1 = (bits == 1) & (u < p10)
        out[flip0] = 1
        out[flip1] = 0
        return out

    # ---- sampling ----

    def sample_point(
        self,
        *,
        probe_type: str,
        support: list[int],
        tau_list_s: list[float],
        sign_list: list[int],
        analysis_phase_rad: float,
        shots: int,
    ) -> list[str]:
        """Return ``shots`` raw per-node bitstrings (char j = node support[j])."""
        return _rows_to_strings(
            self.sample_point_bits(
                probe_type=probe_type,
                support=support,
                tau_list_s=tau_list_s,
                sign_list=sign_list,
                analysis_phase_rad=analysis_phase_rad,
                shots=shots,
            )
        )

    def sample_point_bits(
        self,
        *,
        probe_type: str,
        support: list[int],
        tau_list_s: list[float],
        sign_list: list[int],
        analysis_phase_rad: float,
        shots: int,
    ) -> np.ndarray:
        """Return a ``(shots, len(support))`` int array of raw per-node bits."""
        if probe_type == "separable":
            return self._sample_separable(support, tau_list_s, sign_list, analysis_phase_rad, shots)
        if probe_type == "link_probe":
            return self._sample_link_probe(
                support, tau_list_s, sign_list, analysis_phase_rad, shots
            )
        if probe_type == "ghz":
            return self._sample_ghz(support, tau_list_s, sign_list, analysis_phase_rad, shots)
        raise ValueError(f"unknown probe_type {probe_type!r}")

    def _sample_separable(
        self,
        support: list[int],
        tau_list_s: list[float],
        sign_list: list[int],
        phase: float,
        shots: int,
    ) -> np.ndarray:
        k = len(support)
        cols = np.empty((shots, k), dtype=np.int8)
        for j, node in enumerate(support):
            contrast = self._envelope(node, tau_list_s[j])
            phi = self._node_phase(node, sign_list[j], tau_list_s[j])
            p1 = 0.5 * (1.0 + contrast * np.cos(phi + phase))
            latent = (self.rng.random(shots) < p1).astype(np.int8)
            cols[:, j] = self._apply_readout(latent, node)
        return cols

    def _sample_link_probe(
        self,
        support: list[int],
        tau_list_s: list[float],
        sign_list: list[int],
        phase: float,
        shots: int,
    ) -> np.ndarray:
        node = support[0]
        tau = tau_list_s[0]
        contrast = self.p.link_fidelity[node] * self._envelope(node, tau)
        phi = self._node_phase(node, sign_list[0], tau)
        p1 = 0.5 * (1.0 + contrast * np.cos(phi + phase))
        latent = (self.rng.random(shots) < p1).astype(np.int8)
        out = self._apply_readout(latent, node)
        return out.reshape(shots, 1)

    def _sample_ghz(
        self,
        support: list[int],
        tau_list_s: list[float],
        sign_list: list[int],
        phase: float,
        shots: int,
    ) -> np.ndarray:
        k = len(support)
        link = 1.0
        env = 1.0
        phi_tot = 0.0
        for j, node in enumerate(support):
            link *= self.p.link_fidelity[node]
            env *= self._envelope(node, tau_list_s[j])
            phi_tot += self._node_phase(node, sign_list[j], tau_list_s[j])
        contrast = link * env
        p_even = 0.5 * (1.0 + contrast * np.cos(phi_tot + phase))
        # even parity -> XOR of latent bits == 0
        want_even = self.rng.random(shots) < p_even
        free = (self.rng.random((shots, k - 1)) < 0.5).astype(np.int8)
        xor_free = np.bitwise_xor.reduce(free, axis=1) if k > 1 else np.zeros(shots, np.int8)
        last = np.where(want_even, xor_free, xor_free ^ 1).astype(np.int8)
        latent = np.concatenate([free, last.reshape(shots, 1)], axis=1)
        cols = np.empty((shots, k), dtype=np.int8)
        for j, node in enumerate(support):
            cols[:, j] = self._apply_readout(latent[:, j], node)
        return cols


def _rows_to_strings(cols: np.ndarray) -> list[str]:
    return ["".join(str(int(b)) for b in row) for row in cols]


__all__ = ["NvNodeParams", "NvSensorNetworkEngine"]
