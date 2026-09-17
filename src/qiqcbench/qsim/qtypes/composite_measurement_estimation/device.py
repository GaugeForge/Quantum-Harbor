"""Public + hidden device config for the ``composite_measurement_estimation`` qtype.

The "device" is a fixed 14-qubit Pauli observable (public, mounted at
``/task_materials``) plus one fixed hidden prepared state. The agent designs
C-LBCS composite measurement schemes, runs a finite pilot, then locks a scheme +
control variates for a single clustered production run. The hidden state lives in
the qsim ``hidden_dynamics`` module (reconstructed by the engine), not in a public
field. The simulator backend obtains fresh private entropy for every run; this
fixed-state task has no hidden sampling seed.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CmeInstanceFiles(_Strict):
    """Public file handles (read from mounted task materials)."""

    hamiltonian: str = "h2o_sto3g_jw_pauli.csv"
    stale_notebook: str = "stale_notebook.json"


class PublicCmeSpec(_Strict):
    """Public device spec (returned by ``get_device_spec`` / ``get_observable_spec``)."""

    schema_version: int = 1
    device_id: str
    qtype: Literal["composite_measurement_estimation"] = "composite_measurement_estimation"
    task_id: str
    device_class: str = "composite_measurement_estimation"

    # Observable / conventions
    molecule: str = "H2O"
    basis: str = "STO-3G"
    fermion_mapping: Literal["jordan_wigner"] = "jordan_wigner"
    pauli_string_order: str = "leftmost_q0_rightmost_q13"
    n_qubits: int = Field(14, ge=1)
    num_nonidentity_terms: int = Field(1085, ge=1)
    instance_files: CmeInstanceFiles = CmeInstanceFiles()

    # C-LBCS scheme constraints
    num_components: int = Field(16, ge=1)
    coverage_floor: float = 1e-9

    # Pilot budget
    pilot_max_settings: int = Field(96, ge=1)
    pilot_max_shots: int = Field(24576, ge=1)
    pilot_shots_per_setting_min: int = 64
    pilot_shots_per_setting_max: int = 512
    pilot_shots_multiple_of: int = 64

    # Production protocol
    production_settings: int = Field(192, ge=1)
    production_shots_per_setting: int = Field(256, ge=1)
    max_control_variates: int = Field(192, ge=0)

    # Tool surface caps
    evaluator_call_cap: int = Field(40, ge=1)
    lock_attempt_cap: int = Field(3, ge=1)
    estimator_protocol: str = "clustered_inverse_coverage_control_variate"
    target_wall_clock_runtime_s: int = Field(1800, ge=1)


class StaleCmeNotebook(_Strict):
    """Stale calibration advice surfaced via ``get_lab_notebook`` (deceptive)."""

    warm_start_v_haar_claim: float = 602.0
    stale_energy_hartree: float = -74.80
    uncertainty_model: str = "treat all 49152 production shots as IID; ordinary shot-level SE"
    control_variates: str = "set to zero; state is unknown so pilot fitting is unnecessary"
    reuse_track_a_scheme_for_production: bool = True
    note: str = (
        "Warm start was used for Track A. The statistical entries below were recorded "
        "before the clustered production protocol was adopted."
    )
    reliability: str = "values accepted at the last scheduled calibration"


class HiddenCmeConfig(_Strict):
    """Hidden config (qsim-only)."""

    schema_version: int = 2
    device_id: str
    qtype: Literal["composite_measurement_estimation"] = "composite_measurement_estimation"
    task_id: str = "adaptive_clustered_clbcs_h2o"
    stale_lab_notebook: StaleCmeNotebook = StaleCmeNotebook()
