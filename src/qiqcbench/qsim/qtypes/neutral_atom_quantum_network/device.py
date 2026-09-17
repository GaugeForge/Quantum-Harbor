"""Public + hidden device config for the ``neutral_atom_quantum_network`` qtype.

Two neutral-atom processing nodes joined by a photonic interconnect (a low-rate, lossy,
error-detected **quantum channel** + a high-bandwidth **classical channel**); all
communication crossing the link is metered against one budget. The agent estimates a single
cross-node association between two binary cohorts (one on each node) within that budget.

Architecture invariant 2: the public/hidden split is structural — both classes use
``extra="forbid"``. The public spec exposes the node/channel/budget facts and the selectable
estimation primitives + their **communication costs** (the cost model is public). It never
exposes the hidden data, the true association, or the effective noise floor. No tell of the
inner-product reduction or the 1/eps lever appears here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------- public ----------------


class ChannelSpec(_Strict):
    """A photonic-link channel. Costs are in communication units against ``comm_budget``."""

    name: Literal["quantum", "classical"]
    description: str
    # Public, advisory hardware regime (rate/fidelity/heralding). These set wall-clock + the
    # *raw* link quality; the EFFECTIVE post-heralding error floor that governs achievable
    # accuracy is hidden and must be calibrated. They do NOT change the communication unit count.
    rate_per_s: float | None = None
    raw_fidelity: float | None = None
    heralding_success: float | None = None


class EstimationPrimitiveSpec(_Strict):
    """A selectable distributed-estimation primitive + its communication cost (public).

    ``cost_formula`` is the deterministic per-call communication cost (the device also returns
    the exact charged units per call). The *accuracy* each primitive achieves at a given
    precision is NOT published — it depends on the hidden effective noise floor. The device
    does not rank the primitives.
    """

    estimator: Literal["estimator_a", "estimator_b", "estimator_c"]
    channel: Literal["quantum", "classical"]
    precision_param: str
    cost_formula: str


class NetworkBudgets(_Strict):
    comm_budget: int = Field(..., ge=1)  # total communication units (quantum + classical)
    max_index_bits: int = Field(18, ge=1)


class PublicNetworkSpec(_Strict):
    """Agent-visible device specification (no hidden truth)."""

    schema_version: int = 2
    device_id: str
    qtype: Literal["neutral_atom_quantum_network"] = "neutral_atom_quantum_network"
    task_id: str | None = None

    architecture: str = (
        "two neutral-atom processing nodes (A = exposure node, B = outcome node) joined by a "
        "photonic interconnect carrying a quantum channel and a classical channel"
    )
    # Data partition: same N patients, the exposure indicator on A, the outcome indicator on B.
    n_records: int  # N
    n_index_bits: int  # n = ceil(log2 N)
    data_encoding: str = "binary indicator strings loaded via a per-node data-loading oracle"
    partition: Literal["vertical"] = "vertical"

    channels: list[ChannelSpec]
    estimation_primitives: list[EstimationPrimitiveSpec]
    budgets: NetworkBudgets

    # The accuracy target the association estimate must reach (additive error).
    target_correlation_error: float = 1e-3
    # Nominal service value; the true effective floor remains hidden.
    nominal_link_error_floor: float = 2.5e-3
    notes: str = ""


# ---------------- hidden ----------------


class HiddenInstance(_Strict):
    """The synthetic instance (hidden truth)."""

    # True PRODUCTION cross-node association c = (1/N) sum_i s_i u_i in [-1, 1] (the only
    # quantity that couples the two nodes), returned at the full index register. A genuine ±1
    # sample moment on the 1/N grid.
    correlation: float = Field(..., ge=-1.0, le=1.0)
    # Separate CALIBRATION sub-instance association, returned when the agent probes at a reduced
    # index register (n_index_bits < n_index_bits_full); same floor + estimator scaling but a
    # DIFFERENT value, so a cheap small-register probe cannot steal the production answer.
    calibration_correlation: float = Field(..., ge=-1.0, le=1.0)
    # Effective per-correlation noise floor of the error-detected link + oracle/readout (hidden;
    # the public/stale nominal floor is larger and misleading).
    noise_floor_true: float = Field(..., gt=0.0, lt=1.0)


class StaleNetworkNotebook(_Strict):
    model_config = ConfigDict(extra="forbid")

    framing: str = (
        "This is federated health analytics — focus on privacy: secure aggregation / "
        "homomorphic encryption across the nodes."
    )
    table_advice: str = "assemble the full contingency table jointly across both nodes."
    correlation_advice: str = "estimate the association by repeated overlap measurement."
    sampling_advice: str = (
        "if unsure, just estimate the association classically by sampling records."
    )
    data_advice: str = "ship a record vector to one node when convenient."
    link_advice: str = (
        "the quantum link is slow and noisy (raw fidelity ~0.95, heralding ~0.2); prefer "
        "classical messaging and avoid the quantum channel."
    )
    note: str = ""


class HiddenNetworkConfig(_Strict):
    schema_version: int = 1
    device_id: str
    qtype: Literal["neutral_atom_quantum_network"] = "neutral_atom_quantum_network"
    seed: int

    n_records: int
    n_index_bits: int
    comm_budget: int

    instance: HiddenInstance
    stale_lab_notebook: StaleNetworkNotebook = StaleNetworkNotebook()
