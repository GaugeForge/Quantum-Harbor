"""g-f erasure-qutrit logical-memory engine (per-shot Monte Carlo).

Simulates a transmon qutrit operated as a g-f erasure qubit
(``|0_L>=|g>``, ``|1_L>=|f>``, ``|e>``=erasure) under repeated mid-circuit
erasure detection. The hidden Markovian rate model is implemented as a
vectorized per-shot trajectory simulation (numpy only - no Lindblad
integration):

Per detection cycle of duration ``t_cycle`` (with XY4 dynamical decoupling on):

1. **Leakage.** The encoded qutrit spends ~half the cycle in ``|f>`` (XY4 swaps
   ``|0_L> <-> |1_L>``), so it leaks to ``|e>`` with probability
   ``eps_leak = 1 - exp(-t_cycle / (2 T1_ef))``. A *fraction* of leak events
   double-decay (``|f> -> |e> -> |g>``) within the same cycle - these are
   **undetectable** and flip the logical value (the fundamental bit-flip limit).
2. **Erasure check.** A leaked qutrit (in ``|e>``) is flagged with probability
   ``1 - eps_fn`` and re-initialized to ``|0_L>``. A *missed* leak (prob
   ``eps_fn``) seeps to ``|g>`` undetected with probability ``fn_seepage_coeff``
   - an undetected bit-flip. A code-space qutrit is *false-positive* flagged with
   probability ``eps_fp``.
3. **Control error.** Imperfect XY4 pulses and cadence-dependent residual-readout
   backaction produce a symmetric logical flip channel.
4. **Dephasing (X basis only).** The logical coherence acquires the envelope
   ``exp(-t_cycle / T_phi)``.

The task-local physical endpoint conditions on **no reported erasure** and a
true final code-space state before the final data-qutrit readout. The engine then
applies the hidden three-level readout-confusion channel and returns only the raw
post-readout assignment together with the complete ancilla syndrome string.
"""

from __future__ import annotations

import math

import numpy as np

from qiqcbench.qsim.qtypes.transmon_erasure_qutrit.model import (
    ErasureRates,
    conditional_xy4_transverse_coherence_lifetime_us,
    control_flip_probability,
    leakage_probabilities,
)

# Physical levels.
_G, _E, _F = 0, 1, 2


class ErasureQutritEngine:
    """Per-shot Monte Carlo of the g-f erasure qubit under mid-circuit detection."""

    def __init__(
        self,
        *,
        rates: ErasureRates,
        readout_confusion: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        self.rates = rates
        # readout_confusion[measured, true]; normalize columns to sum to 1.
        conf = np.asarray(readout_confusion, dtype=float)
        col_sums = conf.sum(axis=0, keepdims=True)
        self.readout = conf / np.where(col_sums > 0, col_sums, 1.0)
        self.rng = rng

    # ---- per-cycle rate helpers ----

    def _cycle_rates(self, t_cycle: float, dd: str) -> dict[str, float | object]:
        r = self.rates
        half_occupied = leakage_probabilities(
            r,
            cycle_time_us=t_cycle,
            f_occupation=0.5,
        )
        full_occupied = leakage_probabilities(
            r,
            cycle_time_us=t_cycle,
            f_occupation=1.0,
        )
        return {
            "half_occupied": half_occupied,
            "full_occupied": full_occupied,
            "t_phi_eff": r.t_phi_us if dd == "xy4" else r.nodd_dephasing_t_phi_us,
            "eps_control_flip": control_flip_probability(r, t_cycle, dd),
            "eps_fn": r.false_negative_rate,
            "eps_fp": r.false_positive_rate,
            "seep": r.fn_seepage_coeff,
        }

    # ---- one (prep, n_rounds) experiment ----

    def _simulate_point(
        self,
        *,
        prep_state: str,
        measure_basis: str,
        n_rounds: int,
        cycle_time_us: float,
        dd: str,
        shots: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return round-resolved syndrome bits and final post-readout assignments."""
        rng = self.rng
        rates = self._cycle_rates(cycle_time_us, dd)

        state = np.zeros(shots, dtype=np.int8)  # 0=code, 1=leaked(|e>)
        logical = np.zeros(shots, dtype=np.int8)  # code-space logical value (Z)
        coherent = np.ones(shots, dtype=bool)  # X-basis phase coherence intact
        syndrome = np.zeros((shots, n_rounds), dtype=np.uint8)

        if prep_state == "1L":
            logical[:] = 1
        # "0L"/"+X"/"-X": logical starts 0; X readout uses prep sign + coherence.

        # Small init (thermal) error: occasional prep flip.
        if rates_thermal := self.rates.thermal_pop:
            init_flip = rng.random(shots) < rates_thermal
            logical[init_flip] ^= 1

        track_phase = measure_basis == "x"

        half_occupied = rates["half_occupied"]
        full_occupied = rates["full_occupied"]
        for round_index in range(n_rounds):
            # 1. leakage from code space
            code = state == _G
            if dd == "xy4" or prep_state in {"+X", "-X"}:
                do_leak = code & (rng.random(shots) < half_occupied.leakage)
                dd_branch = do_leak & (rng.random(shots) < half_occupied.double_decay_given_leakage)
            else:
                # Without toggling, only the physical |f> component can leak.
                f_occupied = code & (logical == 1)
                do_leak = f_occupied & (rng.random(shots) < full_occupied.leakage)
                dd_branch = do_leak & (rng.random(shots) < full_occupied.double_decay_given_leakage)
            leak_branch = do_leak & ~dd_branch
            # Undetected double-decay (|f>->|e>->|g> within a cycle) is a logical
            # bit-flip; XY4 swapping symmetrizes it, so flip the logical value.
            logical[dd_branch] ^= 1
            state[leak_branch] = 1  # leaked to |e>

            # 2. erasure check on all leaked shots
            leaked = state == 1
            detected = leaked & (rng.random(shots) < (1.0 - rates["eps_fn"]))
            round_flag = detected.copy()
            state[detected] = _G
            logical[detected] = 0  # re-initialized to |0_L>
            missed = leaked & ~detected
            seep = missed & (rng.random(shots) < rates["seep"])
            state[seep] = _G
            logical[seep] ^= 1  # undetected seepage: symmetric logical bit-flip

            # 3. code-space pulse error + false positive
            code2 = state == _G
            control_flip = code2 & (rng.random(shots) < rates["eps_control_flip"])
            logical[control_flip] ^= 1
            fp = code2 & (rng.random(shots) < rates["eps_fp"])
            round_flag[fp] = True
            syndrome[:, round_index] = round_flag

            # The legacy no-DD path retains event-resolved coherence loss. Under
            # XY4, final X statistics are sampled from the exact conditional
            # channel endpoint below so the Monte Carlo and deterministic model
            # cannot drift apart.
            if track_phase and dd != "xy4":
                relax = dd_branch | seep | control_flip | detected
                coherent[relax] = False

        # ---- final readout ----
        if measure_basis == "x":
            t_total = float(n_rounds) * float(cycle_time_us)
            if dd == "xy4":
                t2_x = conditional_xy4_transverse_coherence_lifetime_us(
                    self.rates,
                    cycle_time_us,
                )
                p_correct = 0.5 + 0.5 * math.exp(-t_total / t2_x)
            else:
                envelope = math.exp(-t_total / rates["t_phi_eff"])
                p_correct = 0.5 + 0.5 * coherent * envelope
            correct = _G if prep_state == "+X" else _F
            wrong = _F if prep_state == "+X" else _G
            x_level = np.where(rng.random(shots) < p_correct, correct, wrong)
            true_level = np.where(state == 1, _E, x_level).astype(np.int8)
        else:
            true_level = np.where(
                state == 1,
                _E,
                np.where(logical == 1, _F, _G),
            ).astype(np.int8)

        measured = self._apply_readout(true_level)
        return syndrome, measured

    def _apply_readout(self, true_level: np.ndarray) -> np.ndarray:
        """Sample measured qutrit outcome via the 3-state confusion matrix."""
        out = np.empty_like(true_level)
        u = self.rng.random(len(true_level))
        # readout[:, t] is P(measured | true=t)
        cdf = np.cumsum(self.readout, axis=0)  # shape (3,3); cdf[:, t]
        for t in (_G, _E, _F):
            mask = true_level == t
            if not mask.any():
                continue
            col = cdf[:, t]
            uu = u[mask]
            picks = np.searchsorted(col, uu, side="right")
            out[mask] = np.clip(picks, 0, 2)
        return out

    # ---- public driver ----

    def run_memory(
        self,
        *,
        prep_state: str,
        measure_basis: str,
        n_rounds_grid: list[int],
        cycle_time_us: float,
        dd: str,
        shots: int,
    ) -> list[dict]:
        """Run one experiment per ``n_rounds`` value; return per-point records."""
        points: list[dict] = []
        for n in n_rounds_grid:
            syndrome, outcomes = self._simulate_point(
                prep_state=prep_state,
                measure_basis=measure_basis,
                n_rounds=int(n),
                cycle_time_us=cycle_time_us,
                dd=dd,
                shots=shots,
            )
            syndrome_ascii = (syndrome + ord("0")).astype(np.uint8, copy=False)
            syndrome_bitstrings = [bytes(row).decode("ascii") for row in syndrome_ascii]
            flags = syndrome.any(axis=1).astype(np.int8)
            points.append(
                {
                    "n_rounds": int(n),
                    "total_evolution_us": float(n) * float(cycle_time_us),
                    "erasure_flags": [int(x) for x in flags],
                    "mid_circuit_ancilla_post_readout_bitstrings": syndrome_bitstrings,
                    "final_outcomes": [int(x) for x in outcomes],
                    "final_qutrit_post_readout_assignments": [int(x) for x in outcomes],
                }
            )
        return points


__all__ = ["ErasureQutritEngine", "ErasureRates"]
