"""Exact angle reduction. Pure math, no task or qtype knowledge.

An angle is physical only mod 2*pi, and every float reduction of one is a MODEL
of reduction against the real 2*pi. Three float reductions worked
until the next binade was found: subtract-then-reduce loses ``ulp(|a|)``,
two-term Cody-Waite loses the quotient itself past 2**53. This is the oracle --
exact rational arithmetic with a single rounding at the end.

It lives in ``qsim/core`` because both sides of the device boundary need it: the
gmon engine reduces a commanded phase at ingestion, and the hidden scorers
reduce reported phases before comparing them. One definition, no copies to drift.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from fractions import Fraction

__all__ = [
    "ANGLE_LINK_RAD",
    "ANGLE_TOL_SLACK_RAD",
    "angle_distance",
    "angles_are_the_same_setting",
    "combine_angles",
    "principal_coordinate",
    "within_angle_tol",
]

# 2*pi to 350 decimal places, held EXACTLY as a rational. Every float is itself
# an exact rational, so reducing one against this constant is exact arithmetic
# with a single rounding at the end -- there is no magnitude at which it
# degrades. Three float reductions worked until
# the next binade was found; any float reduction is a MODEL of reduction against
# the real 2*pi, and this is the oracle. 350 digits is chosen, not arbitrary: the
# largest finite double gives a quotient of ~2.9e307, so the constant's own
# truncation contributes at most ~1.7e-43.
_TAU_EXACT = Fraction(
    628318530717958647692528676655900576839433879875021164194988918461563281257241799725606965068423413596429617302656461329418768921910116446345071881625696223490056820540387704221111928924589790986076392885762195133186689225695129646757356633054240381829129713384692069722090865329642678721452049828254744917401321263117634976304184192565850818343072873,
    10**350,
)

# Every angle tolerance here is a PHYSICAL acceptance band, not an
# exact-arithmetic threshold, so it is compared with a slack that is bounded on
# BOTH sides and is not free to be "tightened":
#
#   LOWER -- it must dominate the reduction's residual, ~1e-15 for every finite
#   double, with no magnitude carve-out.
#   UPPER -- it must sit far below every physical scale the gates resolve. The
#   scan points are pi/2 apart against match windows of 0.1, and 1e-9 rad is
#   5.7e-8 degrees.
ANGLE_TOL_SLACK_RAD = 1e-9


def principal_coordinate(x: float) -> float:
    """The angle ``x`` reduced to (-pi, pi], computed exactly.

    Exact rational arithmetic with ONE rounding, when the result returns to a
    float. ``Fraction(x)`` is the float's exact value, ``//`` is an exact integer
    floor-quotient however many turns it spans, and the remainder is exact, so
    the error is one ulp of the ANSWER and does not grow with the input.

    A value already in the principal range is returned unchanged without building
    a Fraction. That is a short-circuit, not an approximation: for ``|x| <= pi``
    the principal coordinate IS ``x``.
    """
    if -math.pi < x <= math.pi:
        return x
    q = Fraction(x)
    r = q - (q // _TAU_EXACT) * _TAU_EXACT
    if r > _TAU_EXACT / 2:
        r -= _TAU_EXACT
    return float(r)


def combine_angles(*terms: float) -> float:
    """Sum angles after reducing each, so a large term cannot absorb a small one.

    The gauge-field gate checks that a reported triple realizes the commanded
    flux through the directed ring sum ``phi12 + phi23 - phi31``. Summed raw, a
    reported phase of 9.85e15 rad swallows its partners: the triple
    ``(9850368358716648.0, 3.0710290094356782, 9850368358716644.0)`` sums to
    exactly 8.0 and passed the 0.2 tolerance at +pi/2, while its true directed
    sum is 7.071 -- 0.783 rad out, four times the tolerance. Any branch is
    contract-legal, so the physics must survive the addition, not just the
    comparison.
    """
    total = 0.0
    for t in terms:
        total += principal_coordinate(t)
    return principal_coordinate(total)


def angle_distance(a: float, b: float) -> float:
    """Smallest separation between two angles, in radians.

    Each operand is reduced BEFORE they are subtracted, so a large one cannot
    absorb a small one; both are then O(pi) and the subtraction is safe. Fails
    closed (``inf``) on a non-finite input, so every ``<= tol`` test rejects
    rather than raising out of the scorer.
    """
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("inf")
    d = principal_coordinate(a) - principal_coordinate(b)
    return abs(principal_coordinate(d))


def within_angle_tol(a: float, b: float, tol: float) -> bool:
    """Are two angles within ``tol`` of each other, allowing the physical slack?

    The single place the slack is applied, so the gates cannot drift apart.
    """
    return angle_distance(a, b) <= tol + ANGLE_TOL_SLACK_RAD


# Angles are also compared for SAMENESS, not only against a tolerance: the
# circulation reconstruction requires every cited point to share one terminal
# drive. Byte equality is the wrong instrument for that -- physically identical
# drives serialize differently -- but so is a fixed quantization grid, which was
# the first repair: rounding to 1e-12 splits two spellings of one angle whenever
# the value lands within an ulp of a bin boundary. Measured over 400k random
# honest angles spelled two ways one ulp apart, 0.0230% split, concentrated near
# +-pi where the ulp is largest. Fixed-width rounding cannot remove residuals
# without introducing boundaries; it only makes the knife edges narrower and more
# numerous.
#
# So sameness is decided by DISTANCE, chained: two settings belong to one group
# when a chain of pairwise links no longer than ANGLE_LINK_RAD connects them.
# Chaining is safe here because a legitimately distinct setting is separated by
# at least a published tolerance (0.05 rad), 5e7 times the link radius, so a
# chain cannot walk from one honest setting to another.
ANGLE_LINK_RAD = 1e-9


def angles_are_the_same_setting(values: Iterable[float]) -> bool:
    """Do these angles describe ONE physical setting?

    Single-linkage over circular distance, which measures sameness directly
    instead of modelling it with a lattice. Non-finite values are never the same
    setting as anything, including each other.
    """
    remaining = [float(v) for v in values]
    if not remaining:
        return True
    if any(not math.isfinite(v) for v in remaining):
        return False
    group = [remaining.pop()]
    changed = True
    while changed:
        changed = False
        for v in list(remaining):
            if any(angle_distance(v, g) <= ANGLE_LINK_RAD for g in group):
                group.append(v)
                remaining.remove(v)
                changed = True
    return not remaining
