"""Bose-Hubbard analog-simulator engine.

Simulates time-domain eigen-energy spectroscopy on a 1D Bose-Hubbard chain. The
engine is handed the *realized* Hamiltonian — the public nominal coefficients
plus the hidden calibration drift, combined by
``runner.realized_hamiltonian`` — and adds the hidden finite-coherence +
readout noise. It never sees the public spec, so it cannot leak which part of
its Hamiltonian was published.

Physics model (numpy only — no Lindblad integration needed):

1. Work in the truncated Fock basis with at most ``max_excitations`` total
   photons and at most ``max_excitations`` per site. For N=5, max_exc=2 this is
   21-dimensional (1 vacuum + 5 single + 15 double-occupancy/two single).
2. Build ``H`` in MHz and eigendecompose it once per device.
3. Prepare the product superposition state, evolve in the eigenbasis, and apply
   **uniform dephasing in the energy eigenbasis** at rate ``1/T2`` — every
   inter-eigenstate coherence (hence every measured spectral beat) is damped by
   ``exp(-t/T2)``, giving each FT peak a Lorentzian of FWHM ``Gamma = 1/(pi*T2)``.
4. Reduce the (dephased) density matrix to the measured sites, apply the X/Y
   analysis rotation on each site's {0,1} subspace, and sample joint per-shot
   bits (occupation 0 -> bit 0, occupation >=1 -> bit 1), folding in asymmetric
   readout confusion.

Energies are MHz (ordinary frequency, E/h); evolution phase is
``exp(-i 2*pi * E[MHz] * t[ns]/1000)``; the dephasing envelope is
``exp(-t[ns]/T2[ns])``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import numpy as np

# Single-qubit analysis rotations on the {0,1} subspace (measure Z afterwards):
#   x: Hadamard            -> returned bit carries sigma^X
#   y: H @ S-dagger        -> returned bit carries sigma^Y
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
_ROT_01 = {
    "x": np.array([[_INV_SQRT2, _INV_SQRT2], [_INV_SQRT2, -_INV_SQRT2]], dtype=complex),
    "y": np.array([[_INV_SQRT2, -1j * _INV_SQRT2], [_INV_SQRT2, 1j * _INV_SQRT2]], dtype=complex),
}


def enumerate_fock_basis(n_sites: int, max_excitations: int) -> list[tuple[int, ...]]:
    """All occupation vectors with total <= max_excitations and per-site <= max_excitations."""
    basis: list[tuple[int, ...]] = []
    for occ in product(range(max_excitations + 1), repeat=n_sites):
        if sum(occ) <= max_excitations:
            basis.append(occ)
    return basis


@dataclass(frozen=True)
class HamiltonianSpec:
    """Realized (nominal + hidden drift) Hamiltonian coefficients in MHz."""

    on_site_mhz: tuple[float, ...]  # mu_n / 2pi per site
    hopping_mhz: tuple[float, ...]  # J per bond (len n_sites - 1)
    interaction_mhz: float  # U / 2pi
    max_excitations: int


def build_hamiltonian(spec: HamiltonianSpec) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    """Return (H in MHz, Fock basis) for the truncated Bose-Hubbard chain."""
    n = len(spec.on_site_mhz)
    basis = enumerate_fock_basis(n, spec.max_excitations)
    index = {occ: i for i, occ in enumerate(basis)}
    dim = len(basis)
    h = np.zeros((dim, dim), dtype=float)
    for occ in basis:
        i = index[occ]
        # Diagonal: on-site potential + interaction.
        diag = sum(spec.on_site_mhz[s] * occ[s] for s in range(n))
        diag += 0.5 * spec.interaction_mhz * sum(o * (o - 1) for o in occ)
        h[i, i] += diag
        # Hopping J (a+_{s+1} a_s + a+_s a_{s+1}) with bosonic factors.
        for bond in range(n - 1):
            j = spec.hopping_mhz[bond]
            for src, dst in ((bond, bond + 1), (bond + 1, bond)):
                if occ[src] > 0:
                    new = list(occ)
                    new[src] -= 1
                    new[dst] += 1
                    if new[dst] > spec.max_excitations:
                        continue
                    new_occ = tuple(new)
                    if new_occ not in index:
                        continue
                    amp = j * math.sqrt(occ[src]) * math.sqrt(new[dst])
                    h[index[new_occ], i] += amp
    return (h + h.T) / 2.0, basis


class BoseHubbardEngine:
    """Stateless-per-call analog engine; one instance per device (or per job)."""

    def __init__(
        self,
        *,
        spec: HamiltonianSpec,
        t2_ns: float,
        readout: dict[int, tuple[float, float]],
        rng: np.random.Generator,
    ) -> None:
        self.spec = spec
        self.t2_ns = float(t2_ns)
        self.readout = readout
        self.rng = rng
        h, self.basis = build_hamiltonian(spec)
        self.n_sites = len(spec.on_site_mhz)
        self.basis_index = {occ: i for i, occ in enumerate(self.basis)}
        # Eigendecomposition (H real symmetric).
        self.eigvals, self.eigvecs = np.linalg.eigh(h)  # eigvals MHz, eigvecs columns
        # Total-excitation number of each eigenstate (H conserves photon number).
        occ_sum = np.array([sum(occ) for occ in self.basis], dtype=float)
        weight = np.abs(self.eigvecs) ** 2
        self.excitation_numbers = np.rint(weight.T @ occ_sum).astype(int)

    def sector_eigenvalues(self, n_excitations: int) -> np.ndarray:
        """Sorted eigen-energies (MHz) of the given total-excitation sector."""
        return np.sort(self.eigvals[self.excitation_numbers == n_excitations])

    def fock_overlap_weights(
        self, occ_vectors: list[tuple[int, ...]], n_excitations: int
    ) -> np.ndarray:
        """Per-eigenstate summed overlap ``sum_v |<phi|v>|^2`` over ``occ_vectors``,
        restricted to the given sector and sorted by energy.

        For the two-photon protocol the relevant ``occ_vectors`` are the
        single-occupancy two-photon Fock states ``|1_n 1_m>`` (the states
        ``chi_2 = 4 sigma+_n sigma+_m`` couples the vacuum to).
        """
        sel = self.excitation_numbers == n_excitations
        order = np.argsort(self.eigvals[sel])
        weights = np.zeros(int(sel.sum()))
        for occ in occ_vectors:
            ket = np.zeros(len(self.basis))
            ket[self.basis_index[occ]] = 1.0
            amp = self.eigvecs[:, sel].conj().T @ ket
            weights += (np.abs(amp) ** 2)[order]
        return weights

    # ---- state preparation + evolution ----

    def _initial_state(self, init_sites: list[int]) -> np.ndarray:
        """|psi0> = product over init_sites of (|0>+|1>)/sqrt2, vacuum elsewhere."""
        index = {occ: i for i, occ in enumerate(self.basis)}
        psi = np.zeros(len(self.basis), dtype=complex)
        amp = 1.0 / (2.0 ** (len(init_sites) / 2.0))
        for r in range(len(init_sites) + 1):
            for subset in _combinations(init_sites, r):
                occ = [0] * self.n_sites
                for s in subset:
                    occ[s] = 1
                psi[index[tuple(occ)]] += amp
        return psi

    def _density_at(self, c_eig: np.ndarray, time_ns: float) -> np.ndarray:
        """Dephased density matrix in the Fock basis at one evolution time."""
        phase = np.exp(-1j * 2 * np.pi * self.eigvals * (time_ns / 1000.0))
        psi_eig = phase * c_eig
        rho_eig = np.outer(psi_eig, np.conj(psi_eig))
        # Uniform dephasing in the energy eigenbasis: damp off-diagonals.
        damp = math.exp(-time_ns / self.t2_ns) if self.t2_ns > 0 else 0.0
        off = ~np.eye(len(self.eigvals), dtype=bool)
        rho_eig[off] *= damp
        # Back to the Fock basis.
        return self.eigvecs @ rho_eig @ self.eigvecs.conj().T

    # ---- measurement ----

    def _site_rotation(self, basis: str) -> np.ndarray:
        d = self.spec.max_excitations + 1
        rot = np.eye(d, dtype=complex)
        rot[:2, :2] = _ROT_01[basis]
        return rot

    def measurement_probs(self, rho_fock: np.ndarray, measure: list[tuple[int, str]]) -> np.ndarray:
        """Joint occupation-outcome probabilities over {0..max_exc}^k.

        Each measured site is read in its analysis basis as an occupation
        outcome 0..max_excitations (occupation-resolving readout). The outcome
        index uses measure[0] as the most-significant digit (base max_exc+1).
        Sigma operators live on the {0,1} subspace; the |2> outcome is leakage
        the agent maps to 0 weight when forming sigma correlators.
        """
        sites = [s for s, _ in measure]
        bases = [b for _, b in measure]
        k = len(sites)
        d = self.spec.max_excitations + 1
        local_dim = d**k
        rest_sites = [s for s in range(self.n_sites) if s not in sites]

        # Partial trace onto the measured sites' local space.
        rho_m = np.zeros((local_dim, local_dim), dtype=complex)
        loc_idx = np.empty(len(self.basis), dtype=int)
        rest_key = []
        for gi, occ in enumerate(self.basis):
            li = 0
            for s in sites:
                li = li * d + occ[s]
            loc_idx[gi] = li
            rest_key.append(tuple(occ[s] for s in rest_sites))
        for i in range(len(self.basis)):
            for j in range(len(self.basis)):
                if rest_key[i] == rest_key[j]:
                    rho_m[loc_idx[i], loc_idx[j]] += rho_fock[i, j]

        # Analysis rotation A = R(sites[0]) ⊗ R(sites[1]) ⊗ ...
        a = self._site_rotation(bases[0])
        for b in bases[1:]:
            a = np.kron(a, self._site_rotation(b))
        rho_rot = a @ rho_m @ a.conj().T

        probs = np.clip(np.real(np.diag(rho_rot)), 0.0, None)
        total = probs.sum()
        if total > 0:
            probs = probs / total
        return probs

    def _apply_readout(self, occ: np.ndarray, sites: list[int]) -> np.ndarray:
        """Apply asymmetric 0<->1 readout confusion per site (|2> leakage unaffected)."""
        out = occ.copy()
        for col, site in enumerate(sites):
            p01, p10 = self.readout.get(site, (0.0, 0.0))
            col_occ = out[:, col]
            flip0 = (col_occ == 0) & (self.rng.random(len(col_occ)) < p01)
            flip1 = (col_occ == 1) & (self.rng.random(len(col_occ)) < p10)
            col_occ[flip0] = 1
            col_occ[flip1] = 0
            out[:, col] = col_occ
        return out

    def probs_time_series(
        self,
        *,
        init_sites: list[int],
        time_grid_ns: list[float],
        measure: list[tuple[int, str]],
    ) -> list[np.ndarray]:
        """Noiseless (pre-shot, pre-readout) joint outcome probabilities per time."""
        psi0 = self._initial_state(init_sites)
        c_eig = self.eigvecs.conj().T @ psi0
        return [
            self.measurement_probs(self._density_at(c_eig, float(t)), measure) for t in time_grid_ns
        ]

    def sample_time_series(
        self,
        *,
        init_sites: list[int],
        time_grid_ns: list[float],
        measure: list[tuple[int, str]],
        shots: int,
    ) -> list[list[str]]:
        """Return per-time lists of per-shot occupation strings over the measured sites.

        Each character is the read-out occupation (0..max_excitations) of the
        corresponding measured site; site i of the string is ``measure[i]``.
        """
        sites = [s for s, _ in measure]
        k = len(sites)
        d = self.spec.max_excitations + 1
        # Decode table: outcome index -> per-site occupations (sites[0] most significant).
        decode = np.array(
            [[(o // d ** (k - 1 - i)) % d for i in range(k)] for o in range(d**k)],
            dtype=int,
        )
        series = self.probs_time_series(
            init_sites=init_sites, time_grid_ns=time_grid_ns, measure=measure
        )
        out: list[list[str]] = []
        for probs in series:
            draws = self.rng.choice(d**k, size=shots, p=probs)
            occ = decode[draws]
            occ = self._apply_readout(occ, sites)
            out.append(["".join(str(o) for o in row) for row in occ])
        return out


def _combinations(items: list[int], r: int):
    from itertools import combinations

    return combinations(items, r)


__all__ = [
    "BoseHubbardEngine",
    "HamiltonianSpec",
    "build_hamiltonian",
    "enumerate_fock_basis",
]
