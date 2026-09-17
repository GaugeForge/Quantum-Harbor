"""Deterministic rate semantics for the transmon erasure-qutrit qtype.

The Monte Carlo engine and hidden task scorers share these functions so the
physical endpoint cannot drift from the shot sampler.  Readout assignment and
finite-shot noise affect observations, but not the hidden pre-final-readout
conditional logical mode computed here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qiqcbench.qsim.qtypes.transmon_erasure_qutrit.device import (
        HiddenErasureQutritConfig,
    )


@dataclass(frozen=True)
class ErasureRates:
    """Hidden error-model rates with microseconds as the time unit."""

    t1_ge_us: float
    t1_ef_us: float
    t_phi_us: float
    thermal_pop: float
    false_positive_rate: float
    false_negative_rate: float
    dd_pulse_error: float
    fn_seepage_coeff: float
    nodd_dephasing_t_phi_us: float
    readout_backaction_error_at_reference: float
    readout_backaction_reference_cycle_us: float
    readout_backaction_decay_time_us: float


@dataclass(frozen=True)
class LeakageProbabilities:
    """Leakage and conditional double-decay probabilities for one cycle."""

    leakage: float
    double_decay_given_leakage: float


def cadence_grid_us(
    *,
    minimum_us: float,
    maximum_us: float,
    resolution_us: float,
) -> tuple[float, ...]:
    """Return the complete advertised cadence grid without float-step drift."""

    minimum = Decimal(str(minimum_us))
    maximum = Decimal(str(maximum_us))
    resolution = Decimal(str(resolution_us))
    if minimum <= 0 or maximum < minimum or resolution <= 0:
        raise ValueError("cadence range and resolution are invalid")
    steps = (maximum - minimum) / resolution
    integral_steps = steps.to_integral_value()
    if steps != integral_steps:
        raise ValueError("cadence range endpoints must align to the resolution")
    return tuple(float(minimum + index * resolution) for index in range(int(integral_steps) + 1))


def erasure_rates_from_hidden_config(
    hidden: HiddenErasureQutritConfig,
) -> ErasureRates:
    """Build the one authoritative rate bundle from hidden device truth."""

    return ErasureRates(
        t1_ge_us=hidden.t1_ge_us,
        t1_ef_us=hidden.t1_ef_us,
        t_phi_us=hidden.t_phi_us,
        thermal_pop=hidden.thermal_pop,
        false_positive_rate=hidden.false_positive_rate,
        false_negative_rate=hidden.false_negative_rate,
        dd_pulse_error=hidden.dd_pulse_error,
        fn_seepage_coeff=hidden.fn_seepage_coeff,
        nodd_dephasing_t_phi_us=hidden.nodd_dephasing_t_phi_us,
        readout_backaction_error_at_reference=(hidden.readout_backaction_error_at_reference),
        readout_backaction_reference_cycle_us=(hidden.readout_backaction_reference_cycle_us),
        readout_backaction_decay_time_us=hidden.readout_backaction_decay_time_us,
    )


def leakage_probabilities(
    rates: ErasureRates,
    *,
    cycle_time_us: float,
    f_occupation: float,
) -> LeakageProbabilities:
    """Return one-cycle cascade probabilities for a stated ``|f>`` duty cycle.

    The quadratic term reduces to the source experiment's
    ``t_cycle**2 / (8 T1_ef T1_ge)`` under XY4, where the encoded state spends
    half of a cycle in ``|f>``.
    """

    duty = min(1.0, max(0.0, float(f_occupation)))
    if duty == 0.0:
        return LeakageProbabilities(0.0, 0.0)
    cycle = float(cycle_time_us)
    leakage = 1.0 - math.exp(-duty * cycle / rates.t1_ef_us)
    double_decay = (duty * cycle) ** 2 / (2.0 * rates.t1_ef_us * rates.t1_ge_us)
    double_decay = min(leakage, max(0.0, double_decay))
    conditional = double_decay / leakage if leakage > 0.0 else 0.0
    return LeakageProbabilities(leakage, conditional)


def readout_backaction_error(rates: ErasureRates, cycle_time_us: float) -> float:
    """Per-cycle logical flip probability from residual readout photons.

    Shorter cycles leave more residual resonator population before the next
    erasure SWAP.  The hidden amplitude and ring-down scale describe that
    task-independent device effect; the probability is capped at 1/2 so it
    remains a valid symmetric logical-error channel outside the shipped range.
    """

    amplitude = rates.readout_backaction_error_at_reference
    if amplitude <= 0.0:
        return 0.0
    exponent = (
        -(float(cycle_time_us) - rates.readout_backaction_reference_cycle_us)
        / rates.readout_backaction_decay_time_us
    )
    return min(0.5, amplitude * math.exp(exponent))


def control_flip_probability(rates: ErasureRates, cycle_time_us: float, dd: str) -> float:
    """Combine independent DD-pulse and readout-backaction flips by parity."""

    pulse = rates.dd_pulse_error if dd == "xy4" else 0.0
    backaction = readout_backaction_error(rates, cycle_time_us)
    return 0.5 * (1.0 - (1.0 - 2.0 * pulse) * (1.0 - 2.0 * backaction))


def conditional_xy4_bit_flip_lifetime_us(
    rates: ErasureRates,
    cycle_time_us: float,
) -> float:
    """Exact pre-final-readout conditional logical-Z lifetime under XY4.

    The no-reported-erasure dynamics is a sub-stochastic channel on
    ``C0,C1,E0,E1`` (code/leakage state crossed with logical label).  Symmetry
    splits it into population and logical-polarization blocks.  Their dominant
    eigenvalue ratio is the per-cycle decay factor after conditioning on
    survival in the true code space.
    """

    cycle = float(cycle_time_us)
    if not math.isfinite(cycle) or cycle <= 0.0:
        raise ValueError("cycle_time_us must be finite and positive")

    leak = leakage_probabilities(rates, cycle_time_us=cycle, f_occupation=0.5)
    flip = control_flip_probability(rates, cycle, "xy4")
    false_positive_survival = 1.0 - rates.false_positive_rate
    missed_and_seep = rates.false_negative_rate * rates.fn_seepage_coeff
    missed_and_stay = rates.false_negative_rate * (1.0 - rates.fn_seepage_coeff)

    # Columns are source states and rows are destination states.
    transition = np.zeros((4, 4), dtype=float)
    for logical in (0, 1):
        code = logical
        leaked = 2 + logical

        def add_code_path(
            probability: float,
            logical_before_control: int,
            source: int,
        ) -> None:
            transition[logical_before_control, source] += (
                probability * (1.0 - flip) * false_positive_survival
            )
            transition[1 - logical_before_control, source] += (
                probability * flip * false_positive_survival
            )

        add_code_path(1.0 - leak.leakage, logical, code)
        add_code_path(
            leak.leakage * leak.double_decay_given_leakage,
            1 - logical,
            code,
        )
        ordinary_leak = leak.leakage * (1.0 - leak.double_decay_given_leakage)
        add_code_path(ordinary_leak * missed_and_seep, 1 - logical, code)
        transition[leaked, code] += ordinary_leak * missed_and_stay

        # A previously missed erasure may be missed again, or seep back into
        # the code space as a logical flip before the control-error channel.
        transition[1 - logical, leaked] += missed_and_seep * (1.0 - flip) * false_positive_survival
        transition[logical, leaked] += missed_and_seep * flip * false_positive_survival
        transition[leaked, leaked] += missed_and_stay

    population_block = np.array(
        [
            [
                transition[0, 0] + transition[1, 0],
                transition[0, 2] + transition[1, 2],
            ],
            [
                transition[2, 0] + transition[3, 0],
                transition[2, 2] + transition[3, 2],
            ],
        ]
    )
    polarization_block = np.array(
        [
            [
                transition[0, 0] - transition[1, 0],
                transition[0, 2] - transition[1, 2],
            ],
            [
                transition[2, 0] - transition[3, 0],
                transition[2, 2] - transition[3, 2],
            ],
        ]
    )
    population_eigenvalue = max(abs(np.linalg.eigvals(population_block)))
    polarization_eigenvalue = max(abs(np.linalg.eigvals(polarization_block)))
    decay_factor = float(polarization_eigenvalue / population_eigenvalue)
    if not 0.0 < decay_factor < 1.0:
        raise ValueError("hidden rates do not define a decaying conditional logical mode")
    return -cycle / math.log(decay_factor)


def conditional_xy4_pure_dephasing_lifetime_us(
    rates: ErasureRates,
    cycle_time_us: float,
) -> float:
    """Pre-final-readout conditional logical pure-dephasing lifetime.

    The shipped Markovian model uses a cadence-independent pure-dephasing rate
    under XY4. ``cycle_time_us`` remains explicit because this endpoint is
    reported together with the two cadence-dependent conditional lifetimes.
    """

    cycle = float(cycle_time_us)
    if not math.isfinite(cycle) or cycle <= 0.0:
        raise ValueError("cycle_time_us must be finite and positive")
    lifetime = float(rates.t_phi_us)
    if not math.isfinite(lifetime) or lifetime <= 0.0:
        raise ValueError("hidden rates do not define a positive pure-dephasing lifetime")
    return lifetime


def conditional_xy4_transverse_coherence_lifetime_us(
    rates: ErasureRates,
    cycle_time_us: float,
) -> float:
    """Pre-final-readout conditional logical-X coherence lifetime under XY4.

    The conditional channel has independent longitudinal-relaxation and pure-
    dephasing components, so ``1/T2_X = 1/(2*T1_Z) + 1/T_phi``. Both components
    are evaluated for the same no-reported-erasure, true-code-space endpoint.
    """

    bit_flip = conditional_xy4_bit_flip_lifetime_us(rates, cycle_time_us)
    pure_dephasing = conditional_xy4_pure_dephasing_lifetime_us(rates, cycle_time_us)
    return 1.0 / (1.0 / (2.0 * bit_flip) + 1.0 / pure_dephasing)


__all__ = [
    "ErasureRates",
    "LeakageProbabilities",
    "cadence_grid_us",
    "conditional_xy4_bit_flip_lifetime_us",
    "conditional_xy4_pure_dephasing_lifetime_us",
    "conditional_xy4_transverse_coherence_lifetime_us",
    "control_flip_probability",
    "erasure_rates_from_hidden_config",
    "leakage_probabilities",
    "readout_backaction_error",
]
