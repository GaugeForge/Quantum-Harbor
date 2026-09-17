"""Hidden scorer schema for ``adaptive_clustered_clbcs_h2o``.

Compact, curated reference data the verifier needs: tier thresholds, protocol
sizes, the public-Hamiltonian digest, and scoring constants. The hidden state
and per-term means are NOT stored here — the verifier reconstructs them from
:mod:`construction` (independent recompute).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TierThresholds(BaseModel):
    """Lower-is-better tier cut-offs (pass is the binary-reward gate)."""

    model_config = ConfigDict(extra="forbid")

    pass_: float = Field(..., alias="pass")
    bronze: float
    silver: float
    gold: float


class AdaptiveClbcsHiddenScorer(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_version: Literal[3] = 3
    task_id: str = "adaptive_clustered_clbcs_h2o"
    public_hamiltonian_sha256: str

    num_components: int = 16
    coverage_floor: float = 1e-9
    production_settings: int = 192
    production_shots: int = 256
    max_control_variates: int = 192
    pilot_max_settings: int = 96
    pilot_max_shots: int = 24576

    track_a_thresholds: TierThresholds
    track_b_thresholds: TierThresholds

    # Gate/tier comparison precision: the recomputed metric is
    # ROUNDED to this many decimals before the `<=` tier/pass comparison, so the
    # boundary is documented and reproducible rather than resting on raw float bits.
    track_a_comparison_decimals: int = 2
    track_b_comparison_decimals: int = 4

    # Submitted energy must reproduce the verifier's pinned estimator on the
    # same raw counts. Hidden exact energy is diagnostic, not a per-draw gate.
    energy_report_abs_tolerance_hartree: float = 1e-4

    # Submitted uncertainty must reproduce the canonical setting-cluster SE on
    # the same production counts.
    uncertainty_report_abs_tolerance_hartree: float = 1e-4

    points_validity: int = 10
    points_track_a: int = 20
    points_track_b: int = 50
    points_energy: int = 10
    points_uncertainty: int = 10

    best_known_track_a_v_haar: float = 570.28886401966963
    best_known_track_b_sigma: float = 0.0367446746005135
