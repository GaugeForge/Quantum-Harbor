"""Hidden-noise sampler for ``surface_code_lattice_surgery``.

Runs a bundle's noisy circuit through a ``stim.FlipSimulator`` (circuit-level Pauli
noise: T1/T2 idles, inhomogeneous two-qubit depolarizing, reset errors) and layers on the
two noise families that are not expressible as circuit Pauli channels:

- **classical leakage state machine** (Pauli-approximated stuck-ancilla model, McEwen-style): a hidden subset of measure qubits enters a leaked state with a
  per-measured-round probability; while leaked the ancilla's readout misreports,
  flipping its measurement record each lived round with probability ``p_flip_round``.
  Bulk ``stuck`` sites keep a short seed-fixed lifetime (presuming per-cycle leakage
  removal); ``activation`` sites enter once per MID-CIRCUIT measurement run, ``delay``
  rounds after the run start, and may stay leaked until the run ends. A leakage event
  therefore produces long-range spacetime-correlated detection events that no
  public-schedule-derived DEM contains and that must be discovered from data. The
  activation-mode seam excess is visible in single-detector means, but WHICH parity a
  site's excess flips is carried only by detector x observable joint statistics: the
  per-site (delay, lifetime) flavors differ by a detector-silent whole-run flip, so
  flavors sharing a footprint are mean-identical at every window length.
- **asymmetric readout classification noise**: each measurement record is flipped with
  ``p(0->1)`` / ``p(1->0)`` depending on its actual (reference XOR frame) value.

Detection events and observable flips are computed from the total record flips via the
bundle's membership matrices, so both extra families propagate into the returned data
exactly like circuit noise. The sampler never decodes and never exposes the noise model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import stim

from qiqcbench.qsim.qtypes.surface_code_lattice_surgery.circuits import (
    CircuitBundle,
    NoiseParams,
    noisy_segment_circuits,
)


@dataclass(frozen=True)
class LeakSpec:
    p_enter: float  # per measured round (or per activation)
    lifetime: int  # rounds the leaked state persists (may exceed the run: truncated)
    p_flip_round: float  # record-flip probability per lived round
    mode: str = "stuck"  # "stuck" | "activation" (see device.LeakageSite)
    delay: int = 0  # activation mode: rounds after the run start before entry


@dataclass(frozen=True)
class SamplerNoise:
    """Runtime hidden-noise container (built from the hidden device config)."""

    circuit_noise: NoiseParams  # idle/CZ/reset Pauli noise (p_meas empty here)
    p_m01: dict[int, float] = field(default_factory=dict)  # readout 0->1 per qubit
    p_m10: dict[int, float] = field(default_factory=dict)  # readout 1->0 per qubit
    leak: dict[int, LeakSpec] = field(default_factory=dict)  # measure-qubit sites


@dataclass
class SampleResult:
    detection_events: np.ndarray  # [shots, n_detectors] uint8
    observable_flips: np.ndarray  # [shots, n_observables] uint8
    record_values: np.ndarray  # [shots, n_records] uint8 (actual, incl. readout noise)


def _per_record_arrays(bundle: CircuitBundle, noise: SamplerNoise) -> tuple[np.ndarray, np.ndarray]:
    q_of = bundle.meas_qubit_of_record
    p01 = np.array([noise.p_m01.get(q, 0.0) for q in q_of])
    p10 = np.array([noise.p_m10.get(q, 0.0) for q in q_of])
    return p01, p10


def _leak_site_records(
    bundle: CircuitBundle, noise: SamplerNoise
) -> dict[int, tuple[list[int], list[int], list[int]]]:
    """Per leak-prone measure qubit: its 'anc' record indices in measurement order, the
    contiguous-run index of each position, and each position's global round marker."""
    sites: dict[int, tuple[list[int], list[int], list[int]]] = {}
    if not noise.leak:
        return sites
    from qiqcbench.qsim.qtypes.surface_code_lattice_surgery import layout as L

    per_q: dict[int, list[tuple[int, int]]] = {}
    for idx, rec in enumerate(bundle.records):
        if rec.kind != "anc":
            continue
        q = L.qubit_id(rec.qubit)
        if q in noise.leak:
            per_q.setdefault(q, []).append((idx, rec.round))
    for q, seq in per_q.items():
        rec_ids = [i for i, _r in seq]
        rounds = [r for _i, r in seq]
        run_ids: list[int] = []
        run = 0
        for k, (_i, r) in enumerate(seq):
            if k > 0 and r != seq[k - 1][1] + 1:
                run += 1
            run_ids.append(run)
        sites[q] = (rec_ids, run_ids, rounds)
    return sites


def sample_bundle(
    bundle: CircuitBundle,
    noise: SamplerNoise,
    shots: int,
    rng: np.random.Generator,
    *,
    chunk: int = 65_536,
) -> SampleResult:
    segments = noisy_segment_circuits(bundle, noise.circuit_noise)
    full = stim.Circuit()
    for seg in segments:
        full += seg
    ref = bundle.skeleton.reference_sample().astype(np.uint8)  # [n_records]
    m_det = bundle.detector_matrix().astype(np.float32)
    m_obs = bundle.observable_matrix().astype(np.float32)
    p01, p10 = _per_record_arrays(bundle, noise)
    leak_sites = _leak_site_records(bundle, noise)

    n_rec = len(bundle.records)
    dets = np.empty((shots, len(bundle.detectors)), dtype=np.uint8)
    obs = np.empty((shots, len(bundle.observables)), dtype=np.uint8)
    recs = np.empty((shots, n_rec), dtype=np.uint8)

    for start in range(0, shots, chunk):
        m = min(chunk, shots - start)
        sim = stim.FlipSimulator(batch_size=m, seed=int(rng.integers(1 << 62)))
        sim.do(full)
        flips = sim.get_measurement_flips().astype(np.uint8)  # [n_records, m]

        # classical leakage state machine (stuck-ancilla record flips; a leaked state
        # never survives across a measurement-run gap — the ancilla is re-initialized)
        for q, (rec_ids, run_ids, rec_rounds) in leak_sites.items():
            spec = noise.leak[q]
            n_r = len(rec_ids)
            runs = np.asarray(run_ids)
            enter = np.zeros((n_r, m), dtype=bool)
            if spec.mode == "activation":
                # Activation = a parked region coming online MID-CIRCUIT. A run whose
                # first measurement is the circuit's round 0 is a cold global boot, not
                # an activation event, and does not trigger the leak. Entry happens
                # ``delay`` rounds after the run start (0 = the activation transient
                # itself, >0 = a stuck state entered shortly after activation).
                starts = [k for k in range(n_r) if k == 0 or runs[k] != runs[k - 1]]
                entries = []
                for s in starts:
                    if rec_rounds[s] == 0:
                        continue
                    k = s + spec.delay
                    if k < n_r and runs[k] == runs[s]:
                        entries.append(k)
                if entries:
                    enter[entries] = rng.random((len(entries), m)) < spec.p_enter
            else:
                enter[:] = rng.random((n_r, m)) < spec.p_enter
            leak_flips = np.zeros((n_r, m), dtype=np.uint8)
            for off in range(spec.lifetime):
                if off >= n_r:
                    break
                same_run = (runs[off:] == runs[: n_r - off])[:, None]
                lived = (enter[: n_r - off] if off else enter) & same_run
                hit = lived & (rng.random((n_r - off, m)) < spec.p_flip_round)
                leak_flips[off:] ^= hit.astype(np.uint8)
            flips[rec_ids] ^= leak_flips

        # asymmetric readout classification noise
        actual = ref[:, None] ^ flips
        u = rng.random((n_rec, m))
        cls = np.where(actual == 0, u < p01[:, None], u < p10[:, None]).astype(np.uint8)
        total = flips ^ cls

        ft = total.astype(np.float32)
        dets[start : start + m] = (m_det @ ft).astype(np.int64).T.astype(np.uint8) & 1
        obs[start : start + m] = (m_obs @ ft).astype(np.int64).T.astype(np.uint8) & 1
        recs[start : start + m] = (ref[:, None] ^ total).T

    return SampleResult(detection_events=dets, observable_flips=obs, record_values=recs)


__all__ = ["LeakSpec", "SampleResult", "SamplerNoise", "sample_bundle"]
