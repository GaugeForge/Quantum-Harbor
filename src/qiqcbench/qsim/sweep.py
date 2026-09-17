"""Sweep helpers shared across qsim qtypes."""

from __future__ import annotations

import itertools
from typing import Literal

SweepMode = Literal["product", "zip"]


def expand_sweep_points(
    keys: list[str],
    sweep: dict[str, list[float]],
    mode: SweepMode,
) -> list[dict[str, float]]:
    """Expand sweep arrays into ordered binding dictionaries."""
    if mode == "zip":
        lengths = {len(sweep[key]) for key in keys}
        if len(lengths) != 1:
            raise ValueError("zip mode requires equal-length sweep arrays")
        return [{key: sweep[key][i] for key in keys} for i in range(next(iter(lengths), 0))]

    if mode == "product":
        return [
            dict(zip(keys, combo, strict=True))
            for combo in itertools.product(*(sweep[key] for key in keys))
        ]

    raise ValueError(f"unsupported sweep mode: {mode}")
