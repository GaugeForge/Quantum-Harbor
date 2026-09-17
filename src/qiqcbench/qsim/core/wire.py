"""MCP wire schemas. Request/result transports carry schema_version=3.

These are the on-the-wire shapes the agent sees. Keep them small and concrete.
"""

from __future__ import annotations

import math
from functools import reduce
from operator import or_
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    create_model,
    model_serializer,
    model_validator,
)

# SCHEMA_VERSION applies to MCP experiment request/result transports
# (RamseyExperimentRequest, RamseySweepRequest, ObservableBatchRequest,
# MultilevelPulseRequest, etc.). FinalAnswer is a separately-versioned surface
# and remains at Literal[2]. Kept at 3 for the transmon_multilevel_pulse
# additions: the multilevel drive segments + JobLevelOutcomeData are *additive*
# new message types (simulator-only, no replay fixtures), and existing
# provider-replay fixtures are content-hashed including schema_version, so a bump
# would invalidate every committed fixture for unrelated tasks. The new types
# join the current wire generation rather than starting a new one.
SCHEMA_VERSION = 3


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _StrictRequest(_Strict):
    """Base for agent-supplied request payloads and the models nested inside them.

    ``allow_inf_nan=False`` refuses NaN and Infinity on every float field of
    every request, on every path, so the guarantee no longer rests on JSON
    having no non-finite literal. A refusal is a pydantic
    ``ValidationError`` raised at the action boundary, which is a model-owned
    error -- the point of the bound is that a model-supplied value must never
    reach the engine and earn an infrastructure ``failure_kind`` stamp.

    Result models keep the permissive ``_Strict`` base on purpose: a non-finite
    value there is qsim's own computed output, not model input. ``kitaev_chain``
    ``subchain_bulk_gap`` legitimately returns ``+inf`` for a one-site window.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


# ---------- Pulse sequence ops ----------


class PulseOp(_StrictRequest):
    kind: Literal["pulse"] = "pulse"
    channel: str
    shape: Literal["gaussian", "square"]
    amp: float = Field(..., description="Drive amplitude (dimensionless, 0-1).")
    duration_ns: float = Field(..., gt=0)
    sigma_ns: float | None = Field(
        None, gt=0, description="Required for gaussian; ignored for square."
    )
    phase_rad: float = 0.0
    freq_hz: float | None = Field(
        None, description="If absent, drive at the qubit's nominal frequency."
    )


class DelayOp(_StrictRequest):
    kind: Literal["delay"] = "delay"
    duration_ns: float | str = Field(
        ...,
        description='Duration in ns. Strings like "$delay_ns" are sweep placeholders.',
    )


class MeasureOp(_StrictRequest):
    kind: Literal["measure"] = "measure"
    qubits: list[str] = Field(..., min_length=1)


SequenceOp = Annotated[PulseOp | DelayOp | MeasureOp, Field(discriminator="kind")]


# ---------- run_pulse_sequence request ----------


class PulseSequenceRequest(_StrictRequest):
    schema_version: int = SCHEMA_VERSION
    shots: int = Field(..., gt=0, le=1_000_000)
    sequence: list[SequenceOp]


# ---------- run_sweep request ----------


class SweepRequest(_StrictRequest):
    schema_version: int = SCHEMA_VERSION
    shots: int = Field(..., gt=0, le=1_000_000)
    template_sequence: list[SequenceOp]
    sweep: dict[str, list[float]] = Field(
        ...,
        description=(
            "Map of placeholder name (e.g. delay_ns) to list of values. "
            "Placeholders appear in the template as the string '$<name>'."
        ),
    )
    mode: Literal["product", "zip"] = "product"


# ---------- Circuit (digital) ops ----------


class GateOp(_StrictRequest):
    kind: Literal["gate"] = "gate"
    name: Literal[
        "i",
        "x",
        "y",
        "z",
        "h",
        "s",
        "sdg",
        "t",
        "tdg",
        "rx",
        "ry",
        "rz",
        "cx",
        "cz",
        "swap",
        "ms",
    ]
    qubits: list[int] = Field(..., min_length=1)
    params: list[float | str] = Field(
        default_factory=list,
        description=(
            "Floats are concrete angles in radians. Strings name parameters "
            "declared in CircuitRequest.parameters and bound at sweep time."
        ),
    )


class CircuitMeasureOp(_StrictRequest):
    kind: Literal["circuit_measure"] = "circuit_measure"
    qubits: list[int] = Field(..., min_length=1)
    classical: list[int] = Field(..., min_length=1)


CircuitOp = Annotated[GateOp | CircuitMeasureOp, Field(discriminator="kind")]


# ---------- run_circuit / run_circuit_sweep requests ----------


class CircuitRequest(_StrictRequest):
    schema_version: int = SCHEMA_VERSION
    shots: int = Field(..., gt=0, le=1_000_000)
    circuit: list[CircuitOp]
    parameters: list[str] = Field(
        default_factory=list,
        description=(
            "Declared parameter names. Concrete (non-sweep) requests must "
            "leave this empty; CircuitSweepRequest binds these names."
        ),
    )


class CircuitSweepRequest(_StrictRequest):
    schema_version: int = SCHEMA_VERSION
    shots: int = Field(..., gt=0, le=1_000_000)
    template_circuit: list[CircuitOp]
    parameters: list[str] = Field(..., min_length=1)
    sweep: dict[str, list[float]] = Field(
        ...,
        description=("Map parameter name -> list of values. Keys must equal `parameters`."),
    )
    mode: Literal["product", "zip"] = "product"


# ---------- Observable batch (digital VQE) ----------


class ObservableBatchPoint(_StrictRequest):
    """One parameter point in an observable-batch request.

    Transport-only: the parameter vector length is not constrained here.
    Qtype/capability material logic (e.g. the digital VQE validator) is
    responsible for enforcing task-specific parameter counts.
    """

    point_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
    )
    values: list[float] = Field(..., min_length=1, max_length=256)


class ObservableBatchRequest(_StrictRequest):
    """Batched per-Pauli measurement request.

    Transport-only constraints: positive shots, nonempty profile string,
    nonempty parameter_convention string, at least one point. The set of
    supported profile names and parameter_convention names is qtype/capability
    material-specific and is validated by the owning qtype's material logic.
    """

    schema_version: int = SCHEMA_VERSION
    shots_per_setting: int = Field(..., ge=1, le=100_000)
    profile: str = Field(..., min_length=1, max_length=128)
    parameter_convention: str = Field(..., min_length=1, max_length=128)
    points: list[ObservableBatchPoint] = Field(..., min_length=1, max_length=256)


# ---------- Hamiltonian probe batch (blackbox analog dynamics) ----------


class ProbeRowOp(_StrictRequest):
    """One requested probe in a Hamiltonian probe batch.

    Transport-only constraints: a nonempty product-state label list and a
    nonempty observable Pauli string. Grid/range/weight enforcement is
    qtype-owned: the analog engine rejects off-grid or out-of-range times,
    invalid observables, and malformed states per-row (rejected rows consume
    zero budget).
    """

    initial_state: list[str] = Field(..., min_length=1)
    evolve_time_us: float
    observable_pauli: str = Field(..., min_length=1)


class HamiltonianProbeBatchRequest(_StrictRequest):
    """Batched black-box probe request for the analog-dynamics qtype.

    The internal repetition count and the total-evolution-time budget are
    enforced by qsim and are not set by the agent. An oversized batch (more
    rows than the device's ``max_rows_per_batch``) is a batch-level failure.
    """

    schema_version: int = SCHEMA_VERSION
    # ``max_length`` is a transport-level sanity ceiling against a pathological
    # request, not the business limit: the device enforces the real per-batch cap
    # (``max_rows_per_batch``), and any batch above it fails as a whole at the
    # backend with an informative error.
    rows: list[ProbeRowOp] = Field(..., min_length=1, max_length=4096)


# ---------- Ramsey (trapped-ion chain) ----------


class RamseyExperimentRequest(_StrictRequest):
    """Single-point Ramsey experiment job request.

    The prepare_circuit and analysis_circuit are agent-built from the qtype's
    native gate set. free_evolution_us is in MICROSECONDS. The phase source
    selects whether the engine applies the agent-supplied
    `reference_detuning_hz` (calibration mode) or the hidden device detuning
    (estimation mode).

    Unit convention: `reference_detuning_hz` is a frequency (Hz), not an
    angular frequency. The engine converts to rad/us internally.
    """

    schema_version: int = SCHEMA_VERSION
    shots: int = Field(..., gt=0, le=100_000)
    prepare_circuit: list[CircuitOp]
    free_evolution_us: float = Field(..., gt=0)
    phase_source: Literal["reference", "target"]
    reference_detuning_hz: float | None = Field(
        None,
        description=(
            "Required when phase_source=='reference'; ignored when 'target'. "
            "Frequency in Hz; engine converts to rad/us internally."
        ),
    )
    analysis_circuit: list[CircuitOp]
    measured_ions: list[int] = Field(..., min_length=1)
    profile: Literal["default"] = "default"

    @model_validator(mode="after")
    def _check_phase_source_reference_coupling(self) -> RamseyExperimentRequest:
        if self.phase_source == "reference" and self.reference_detuning_hz is None:
            raise ValueError("reference_detuning_hz is required when phase_source == 'reference'")
        if self.phase_source == "target" and self.reference_detuning_hz is not None:
            raise ValueError(
                "reference_detuning_hz must be None when phase_source == 'target' "
                "(otherwise replay fixture hashes would diverge for semantically "
                "equivalent requests)"
            )
        return self


class RamseySweepRequest(_StrictRequest):
    """Multi-point Ramsey sweep job request.

    Sweeps over interrogation time at a single (probe, analysis) configuration.
    Used for Track A coherence-decay sweeps. Track C, which sweeps over N, must
    issue multiple RamseyExperimentRequest jobs (one per N) because the probe
    and analysis circuits differ per N.
    """

    schema_version: int = SCHEMA_VERSION
    shots: int = Field(..., gt=0, le=100_000)
    prepare_circuit: list[CircuitOp]
    free_evolution_us_values: list[Annotated[float, Field(gt=0)]] = Field(
        ...,
        min_length=1,
        description="Per-point free-evolution time in MICROSECONDS; each value > 0.",
    )
    phase_source: Literal["reference", "target"]
    reference_detuning_hz: float | None = None
    analysis_circuit: list[CircuitOp]
    measured_ions: list[int] = Field(..., min_length=1)
    profile: Literal["default"] = "default"

    @model_validator(mode="after")
    def _check_phase_source_reference_coupling(self) -> RamseySweepRequest:
        if self.phase_source == "reference" and self.reference_detuning_hz is None:
            raise ValueError("reference_detuning_hz is required when phase_source == 'reference'")
        if self.phase_source == "target" and self.reference_detuning_hz is not None:
            raise ValueError(
                "reference_detuning_hz must be None when phase_source == 'target' "
                "(otherwise replay fixture hashes would diverge for semantically "
                "equivalent requests)"
            )
        return self


# ---------- Bose-Hubbard analog (evolve under fixed H + tomographic readout) ----------


class MeasureSetting(_StrictRequest):
    """One site to read out, with the single-qubit analysis basis to apply.

    `basis="x"` rotates by a Hadamard before the computational measurement so the
    returned bit carries ``sigma^X``; `basis="y"` applies S-dagger then Hadamard
    so the bit carries ``sigma^Y``. Outcomes are per-shot 0/1.
    """

    site: int = Field(..., ge=0)
    basis: Literal["x", "y"]


class BoseHubbardEvolveRequest(_StrictRequest):
    """Prepare a product superposition, evolve under the fixed device Hamiltonian
    for one time, and tomographically read out the requested sites."""

    schema_version: int = SCHEMA_VERSION
    shots: int = Field(..., gt=0, le=1_000_000)
    init_excited_sites: list[int] = Field(
        ...,
        min_length=1,
        description="Sites prepared in (|0>+|1>)/sqrt(2); the rest start in |0>.",
    )
    evolution_time_ns: float = Field(..., ge=0)
    measure: list[MeasureSetting] = Field(..., min_length=1)


class BoseHubbardEvolveSweepRequest(_StrictRequest):
    """Same as BoseHubbardEvolveRequest but over a grid of evolution times.

    `time_grid_ns` is the dedicated sweep axis (no placeholder substitution); one
    measurement record is produced per time, in order.
    """

    schema_version: int = SCHEMA_VERSION
    shots: int = Field(..., gt=0, le=1_000_000)
    init_excited_sites: list[int] = Field(..., min_length=1)
    time_grid_ns: list[float] = Field(..., min_length=1)
    measure: list[MeasureSetting] = Field(..., min_length=1)


# ---------- NV sensor-network sensing ops ----------


def _check_nv_protocol(probe_type: str, support: list[int], echo_sign: list[int]) -> None:
    """Shared structural validation for NV sensing requests."""
    if len(support) != len(set(support)):
        raise ValueError("support node indices must be distinct")
    if any(s < 0 for s in support):
        raise ValueError("support node indices must be >= 0")
    if len(echo_sign) != len(support):
        raise ValueError("echo_sign must have one entry per support node")
    if any(s not in (-1, 1) for s in echo_sign):
        raise ValueError("echo_sign entries must be -1 or +1")
    if probe_type == "link_probe" and len(support) != 1:
        raise ValueError("link_probe acts on exactly one node")
    if probe_type == "ghz" and len(support) < 2:
        raise ValueError("ghz requires a support of at least two nodes")


class NvSensingProbeRequest(_StrictRequest):
    """One NV sensor-network probe: a separable, GHZ, or link-calibration shot batch.

    Each of ``shots`` repetitions interrogates every node in ``support`` for its
    per-node ``interrogation_time_s`` with the chosen ``echo_sign`` (a mid-window
    echo pi-pulse realizes a signed effective sensing time). ``separable`` measures
    each node independently; ``ghz`` distributes a GHZ across ``support`` (consuming
    ``len(support)-1`` heralded Bell pairs) and the returned per-node bits carry the
    joint parity; ``link_probe`` entangles one node with the ideal central station to
    calibrate that node's Bell-pair fidelity. Interrogation times lie on the device
    10 ns grid. Cumulative interrogation time charged = ``shots * sum(interrogation_time_s)``.
    """

    schema_version: int = SCHEMA_VERSION
    probe_type: Literal["separable", "ghz", "link_probe"]
    support: list[int] = Field(..., min_length=1)
    interrogation_time_s: list[float] = Field(..., min_length=1)
    echo_sign: list[int] = Field(..., min_length=1)
    analysis_phase_rad: float = 0.0
    shots: int = Field(..., gt=0, le=1_000_000)

    @model_validator(mode="after")
    def _validate(self) -> NvSensingProbeRequest:
        _check_nv_protocol(self.probe_type, self.support, self.echo_sign)
        if len(self.interrogation_time_s) != len(self.support):
            raise ValueError("interrogation_time_s must have one entry per support node")
        if any(t <= 0 for t in self.interrogation_time_s):
            raise ValueError("interrogation_time_s entries must be > 0")
        return self


class NvSensingSweepRequest(_StrictRequest):
    """Sweep a single interrogation time (applied to every support node) over a grid.

    One measurement record per grid value, in order. Used for Track-A characterization
    (per-node Ramsey envelope/fringe for T2* and field, link-fidelity extrapolation) and
    for fringe scans. ``echo_sign`` and ``analysis_phase_rad`` are shared across the grid.
    """

    schema_version: int = SCHEMA_VERSION
    probe_type: Literal["separable", "ghz", "link_probe"]
    support: list[int] = Field(..., min_length=1)
    interrogation_time_grid_s: list[float] = Field(..., min_length=1)
    echo_sign: list[int] = Field(..., min_length=1)
    analysis_phase_rad: float = 0.0
    shots: int = Field(..., gt=0, le=1_000_000)

    @model_validator(mode="after")
    def _validate(self) -> NvSensingSweepRequest:
        _check_nv_protocol(self.probe_type, self.support, self.echo_sign)
        if any(t <= 0 for t in self.interrogation_time_grid_s):
            raise ValueError("interrogation_time_grid_s entries must be > 0")
        return self


# ---------- Multilevel transmon pulse (4-level Duffing, two-quadrature + detuning) ----------


class AnalyticDriveSegment(_StrictRequest):
    """An analytic gaussian/square drive segment (single quadrature via phase).

    Scalar fields may be sweep placeholders of the form ``"$name"`` in a
    ``MultilevelPulseSweepRequest``; concrete requests must use floats.
    Amplitudes are DAC units; ``carrier_detuning_hz`` is the cyclic detuning
    ``delta/2pi = omega_01 - omega_drive``.
    """

    kind: Literal["analytic"] = "analytic"
    shape: Literal["gaussian", "square"]
    amp_dac: float | str
    duration_ns: float | str
    sigma_ns: float | str | None = None
    phase_rad: float | str = 0.0
    carrier_detuning_hz: float | str = 0.0


class SampledDriveSegment(_StrictRequest):
    """A sampled two-quadrature drive segment with per-sample detuning.

    Equal-length DAC arrays for the two quadratures plus a per-sample cyclic
    detuning (Hz). This is how the agent submits a designed DRAG pulse.
    """

    kind: Literal["sampled"] = "sampled"
    omega_x_dac: list[float] = Field(..., min_length=1)
    omega_y_dac: list[float] = Field(..., min_length=1)
    detuning_hz: list[float] = Field(..., min_length=1)
    sample_dt_ns: float = Field(..., gt=0)


DriveSegment = Annotated[AnalyticDriveSegment | SampledDriveSegment, Field(discriminator="kind")]


class MultilevelPulseRequest(_StrictRequest):
    """Run a concatenated drive sequence from |0> and read out level-resolved."""

    schema_version: int = SCHEMA_VERSION
    shots: int = Field(..., gt=0, le=1_000_000)
    segments: list[DriveSegment] = Field(..., min_length=1)


class MultilevelPulseSweepRequest(_StrictRequest):
    """Sweep one or more named scalar placeholders across analytic segments."""

    schema_version: int = SCHEMA_VERSION
    shots: int = Field(..., gt=0, le=1_000_000)
    template_segments: list[DriveSegment] = Field(..., min_length=1)
    sweep: dict[str, list[float]] = Field(
        ...,
        description="Map placeholder name (referenced as '$name') to list of values.",
    )
    mode: Literal["product", "zip"] = "product"


# ---------- Result ----------


class JobIQData(_Strict):
    """Per-qubit IQ shots. For run_pulse_sequence: list[ [I, Q] ] of length shots.
    For run_sweep: list of per-point lists.
    """

    kind: Literal["iq"] = "iq"
    iq: dict[str, list]


class JobBitstringData(_Strict):
    """Per-shot bitstrings from a digital circuit run.

    `bitstrings` outer index is the sweep point (length 1 for non-sweep);
    inner index is the shot. Each string has length `len(measured_qubits)`,
    with `bitstring[i]` giving the value of `measured_qubits[i]`.
    """

    kind: Literal["bitstring"] = "bitstring"
    bitstrings: list[list[str]]
    measured_qubits: list[int]


class ObservableSettingBitstrings(_Strict):
    """Per-shot bitstrings for a single (point, Pauli-setting) cell."""

    point_id: str
    setting_id: str
    pauli: str
    coefficient: float
    measured_qubits: list[int]
    bitstrings: list[str]


class JobObservableBitstringData(_Strict):
    """Raw bitstrings grouped by (parameter point, Pauli measurement setting)."""

    kind: Literal["observable_bitstring"] = "observable_bitstring"
    results: list[ObservableSettingBitstrings]


class ProbeOutcomeRow(_Strict):
    """One probe outcome row returned to the agent.

    Accepted rows carry ``num_internal_repetitions`` raw ±1 Pauli outcomes and
    the ``accepted_evolve_time_us`` charged to the budget. Rejected rows carry a
    ``reject_reason`` and consume zero budget.
    """

    observable_pauli: str
    status: Literal["accepted", "rejected"]
    accepted_evolve_time_us: float | None = None
    raw_pauli_outcomes: list[int] | None = None
    num_internal_repetitions: int | None = None
    reject_reason: str | None = None


class JobProbeOutcomeData(_Strict):
    """Raw ±1 probe outcomes plus evidence-owned budget accounting.

    ``budget_used_us`` / ``accepted_row_count`` are cumulative across every
    probe batch in the run: the scarce resource is total accepted evolution
    time, so the budget persists between ``run_hamiltonian_probe_batch`` calls.
    """

    kind: Literal["probe_outcome"] = "probe_outcome"
    rows: list[ProbeOutcomeRow]
    budget_used_us: float
    budget_remaining_us: float
    accepted_row_count: int
    max_probe_rows: int


class JobRamseyPointBitstrings(_Strict):
    """Per-shot bitstrings for one (free_evolution_us) sweep point."""

    free_evolution_us: float
    bitstrings: list[str] = Field(..., min_length=1)
    measured_ions: list[int] = Field(..., min_length=1)


class JobRamseyBitstringData(_Strict):
    """Raw bitstrings from a Ramsey experiment or sweep, grouped by sweep point."""

    kind: Literal["ramsey_bitstring"] = "ramsey_bitstring"
    points: list[JobRamseyPointBitstrings] = Field(..., min_length=1)
    phase_source: Literal["reference", "target"]
    reference_detuning_hz: float | None


class JobNvSensingPoint(_Strict):
    """Per-shot raw bitstrings for one NV sensing point (one probe or sweep value).

    ``interrogation_time_s[j]`` is the free-evolution time of node ``support[j]`` at
    this point; ``bitstrings[shot]`` is a string whose character ``j`` is the read-out
    bit of node ``support[j]``. The agent forms node populations / GHZ parity itself.
    """

    interrogation_time_s: list[float] = Field(..., min_length=1)
    bitstrings: list[str] = Field(..., min_length=1)


class JobNvSensingData(_Strict):
    """Raw bitstrings from an NV sensor-network probe or interrogation-time sweep."""

    kind: Literal["nv_sensing"] = "nv_sensing"
    probe_type: Literal["separable", "ghz", "link_probe"]
    support: list[int]
    echo_sign: list[int]
    analysis_phase_rad: float
    bell_pairs_consumed: int
    points: list[JobNvSensingPoint] = Field(..., min_length=1)


class JobLevelOutcomeData(_Strict):
    """Per-shot level-resolved readout outcomes for the multilevel transmon.

    ``outcomes`` outer index is the sweep point (length 1 for a single run);
    inner index is the shot. Each character is the reported level (0..n_levels-1)
    after the hidden readout-assignment confusion. The agent aggregates these
    into P0..P{n-1} and chooses any readout mitigation.
    """

    kind: Literal["level_outcome"] = "level_outcome"
    outcomes: list[list[str]]
    n_levels: int


class JobResultArtifact(_Strict):
    """Machine-readable locator for a completed result published as a file.

    Set only when ``data`` was too large to return inline. The named file holds
    the complete unmodified ``JobResult``, including the ``data`` this response
    omits, and is readable at ``/qsim_logs/<relative_path>``.

    ``result_kind`` is the ``data.kind`` of the offloaded payload. It lives here
    rather than in the qsim-owned ``offloaded_result`` evidence pointer, whose
    exact field set is a verifier-facing contract.
    """

    schema_version: Literal[1] = 1
    relative_path: str
    size_bytes: int = Field(..., ge=0)
    sha256: str
    result_kind: str


class JobResultMetadata(_Strict):
    public_warnings: list[str] = []
    sequence_duration_ns: float | None = None
    wallclock_ms: int | None = None
    sweep_coords: dict[str, list[float]] | None = None
    result_artifact: JobResultArtifact | None = None

    @model_serializer(mode="wrap")
    def _omit_absent_result_artifact(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        """Serialize the locator only when there is one.

        The other optional fields here predate this one and serialize as explicit
        nulls, so an agent's payload already carries them. ``result_artifact`` is
        new, and emitting ``"result_artifact": null`` on every poll would change
        the delivered bytes of every result on every task -- including the ones
        that never approach the size bound, and including the metadata record
        that `bosonic_cavity_qec` writes into `experiment_log.jsonl`. Omitting it
        when unset keeps this change confined to the results it is about.
        """
        dumped = dict(handler(self))
        if dumped.get("result_artifact") is None:
            dumped.pop("result_artifact", None)
        return dumped


# ---------- JobData / JobResult (built dynamically from the qtype registry) ----------
#
# The set of result-data shapes is contributed by qtypes via
# ``QtypeDescriptor.result_data_models`` (each a ``_Strict`` model with a
# ``kind`` Literal discriminator). A new qtype that introduces a new result kind
# defines it in its own ``qtypes/<name>/wire.py`` and lists it there — JobData
# picks it up with no edit to this file.
#
# These are resolved LAZILY (via module ``__getattr__``), never at import time:
# each qtype's ``qtype.py`` imports concrete result classes from this module
# DURING registry discovery, so calling ``all_descriptors()`` at this module's
# top level would hit a half-built registry. Backends/runners that need
# ``JobResult`` are imported only after discovery completes, so the lazy build
# always sees a fully-populated registry.

_JOB_DATA_CACHE: object | None = None
_JOB_RESULT_CACHE: type | None = None


def _build_job_data() -> object:
    from qiqcbench.qsim.qtypes.registry import all_descriptors

    by_kind: dict[str, type] = {}
    for descriptor in all_descriptors():
        for model in descriptor.result_data_models:
            kind = model.model_fields["kind"].default
            existing = by_kind.get(kind)
            if existing is not None and existing is not model:
                raise ValueError(
                    f"Conflicting JobData models for kind {kind!r}: "
                    f"{existing.__name__} vs {model.__name__}"
                )
            by_kind[kind] = model
    if not by_kind:
        raise RuntimeError("no JobData result-data models registered")
    models = tuple(by_kind[kind] for kind in sorted(by_kind))
    if len(models) == 1:
        return models[0]
    return Annotated[reduce(or_, models), Field(discriminator="kind")]


def _build_job_result() -> type:
    job_data = _build_job_data()
    job_result = create_model(
        "JobResult",
        __base__=_Strict,
        schema_version=(int, SCHEMA_VERSION),
        job_id=(str, ...),
        device_id=(str, ...),
        status=(Literal["queued", "running", "complete", "failed"], ...),
        shots=(int | None, None),
        data=(job_data | None, None),
        error=(str | None, None),
        metadata=(JobResultMetadata, Field(default_factory=JobResultMetadata)),
    )
    job_result.__module__ = __name__
    return job_result


def __getattr__(name: str) -> object:
    """Lazily expose registry-derived ``JobData`` / ``JobResult`` (PEP 562)."""
    global _JOB_DATA_CACHE, _JOB_RESULT_CACHE
    if name == "JobData":
        if _JOB_DATA_CACHE is None:
            _JOB_DATA_CACHE = _build_job_data()
        return _JOB_DATA_CACHE
    if name == "JobResult":
        if _JOB_RESULT_CACHE is None:
            _JOB_RESULT_CACHE = _build_job_result()
        return _JOB_RESULT_CACHE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------- Final answer ----------


class FinalAnswer(_Strict):
    schema_version: Literal[2] = 2
    task_id: str
    answer: dict

    @model_validator(mode="after")
    def _answer_numbers_are_finite(self) -> FinalAnswer:
        def visit(value: object, path: str) -> None:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{path} must contain only finite numbers")
            if isinstance(value, dict):
                for key, nested in value.items():
                    visit(nested, f"{path}.{key}")
            elif isinstance(value, (list, tuple)):
                for index, nested in enumerate(value):
                    visit(nested, f"{path}[{index}]")

        visit(self.answer, "answer")
        return self
