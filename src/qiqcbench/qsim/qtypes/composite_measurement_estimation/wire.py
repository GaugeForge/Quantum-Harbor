"""Wire schemas for the ``composite_measurement_estimation`` qtype.

Two async experiment tools return raw bitstring counts:

* ``run_pilot_measurements`` -> ``JobPilotCountsData``
* ``execute_locked_composite_scheme`` -> ``JobProductionCountsData``

The synchronous ``get_observable_spec`` / ``evaluate_composite_scheme`` calculators
return plain dicts (no job). Result-data models carry a ``kind`` discriminator so
the registry-derived ``JobData`` union is well-formed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _StrictRequest(_Strict):
    """_Strict with NaN/Infinity refused on every float field."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


# --------------------------------------------------------------------------- #
# Requests (validated in the action layer)
# --------------------------------------------------------------------------- #
class PilotSetting(_StrictRequest):
    basis: str
    shots: int


class PilotBatchRequest(_StrictRequest):
    rows: list[PilotSetting]


class ControlVariateEntry(_StrictRequest):
    term_index: int
    # A control variate is a pinned Pauli mean: physically bounded to [-1, 1].
    mean: float = Field(ge=-1.0, le=1.0)


class LockedProductionRequest(_StrictRequest):
    mixture_weights: list[float]
    local_basis_probabilities_xyz: list[list[list[float]]]
    control_variate_entries: list[ControlVariateEntry] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
class _PilotBudgetView(_Strict):
    pilot_settings_used: int
    pilot_settings_cap: int
    pilot_shots_used: int
    pilot_shots_cap: int


class PilotRowResult(_Strict):
    basis: str
    shots: int
    status: Literal["accepted", "rejected"]
    counts: dict[str, int] | None = None
    reject_reason: str | None = None


class JobPilotCountsData(_Strict):
    kind: Literal["cme_pilot_counts"] = "cme_pilot_counts"
    rows: list[PilotRowResult]
    budget: _PilotBudgetView


class ProductionRowResult(_Strict):
    """Agent-facing production row. The sampled component ID is NOT returned
    (verifier-side only); exposing it would invite a component-conditional estimator."""

    basis: str
    counts: dict[str, int]


class JobProductionCountsData(_Strict):
    kind: Literal["cme_production_counts"] = "cme_production_counts"
    rows: list[ProductionRowResult]
    request_digest: str
    production_settings: int
    production_shots_per_setting: int
