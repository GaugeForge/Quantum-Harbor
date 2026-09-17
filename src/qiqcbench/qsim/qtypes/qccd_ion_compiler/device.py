"""Public + hidden device config for the ``qccd_ion_compiler`` qtype.

This is a **device-less / static-instance** qtype (documented
exception to the async-job invariant): there is no quantum dynamics, no shots,
and no async job model. The "device" is a zoned QCCD trapped-ion *cost model* — a
deterministic resource/error/timing model used to score a compiled schedule, NOT
a quantum simulator. The agent compiles a QAOA circuit onto the zoned trap and is
scored by the total estimated circuit infidelity of the compiled schedule.

The public spec carries the **full, authoritative cost model** (architecture,
the MS-gate error coefficients, the transport-heating table, recooling, idle /
dynamical-decoupling, readout) plus *metadata* about the target circuit — but NOT
the per-instance QAOA graph + angles, which are served by
``get_compilation_instance`` (a deterministic function of
``QIQCBENCH_INSTANCE_SEED``). The hidden config carries only the stale lab
notebook (the trap recommendations); the scoring anchors live verifier-side.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- Cost-model sub-structures (all public, authoritative) ----------


class QccdTopology(_Strict):
    """Zoned-trap topology + capacities (§4). Adjacency is explicit so QCCD
    legality (split/merge adjacency, junction exclusion) is decidable."""

    gate_zones: list[str] = Field(default_factory=lambda: ["G_A", "G_B", "G_C"])
    storage_segments: list[str] = Field(default_factory=lambda: ["S_A", "S_B", "S_C"])
    junction: str = "J"
    readout_zone: str = "readout"
    # S_X is adjacent to its gate zone G_X (intra-shuttle / split / merge); the
    # three storage segments connect ONLY via the central junction J.
    segment_gate_adjacency: dict[str, str] = Field(
        default_factory=lambda: {"S_A": "G_A", "S_B": "G_B", "S_C": "G_C"}
    )
    gate_zone_capacity: int = Field(4, ge=2)
    storage_capacity: int = Field(10, ge=1)
    junction_capacity: int = Field(1, ge=1)


class ModeCost(_Strict):
    """Per-normal-mode MS-error coefficients."""

    floor: float = Field(..., ge=0)
    kappa: float = Field(..., ge=0)


class MsErrorModel(_Strict):
    """``ε_MS = floor(mode) + s_angle·(|θ|/(π/2)) + kappa(mode)·(2·n̄_mode+1)``.

    ``s_angle`` is mode-independent (cancels in the COM↔stretch comparison).
    Gate time ``t_g(θ) = gate_time_us_pi_half · (|θ|/(π/2))``.
    """

    com: ModeCost = Field(default_factory=lambda: ModeCost(floor=4e-4, kappa=5e-4))
    stretch: ModeCost = Field(default_factory=lambda: ModeCost(floor=2.0e-3, kappa=1.0e-4))
    s_angle: float = Field(1.5e-3, ge=0)
    gate_time_us_pi_half: float = Field(30.0, gt=0)


class TransportSpeedCost(_Strict):
    """``(time, Δn̄_COM)`` for one transport primitive at one speed."""

    time_us: float = Field(..., gt=0)
    dnbar_com: float = Field(..., ge=0)


class TransportPrimitiveCost(_Strict):
    """A transport primitive's ``optimized`` (low-heating, slow) and ``fast``
    (diabatic, hot) cost pair."""

    optimized: TransportSpeedCost
    fast: TransportSpeedCost


class TransportTable(_Strict):
    """Per-primitive transport-heating table (§4). Each move adds ``Δn̄`` to the
    COM mode and ``stretch_heating_ratio·Δn̄`` to the stretch mode."""

    shuttle_intra: TransportPrimitiveCost = Field(
        default_factory=lambda: TransportPrimitiveCost(
            optimized=TransportSpeedCost(time_us=60, dnbar_com=0.8),
            fast=TransportSpeedCost(time_us=20, dnbar_com=2.5),
        )
    )
    shuttle_inter: TransportPrimitiveCost = Field(
        default_factory=lambda: TransportPrimitiveCost(
            optimized=TransportSpeedCost(time_us=280, dnbar_com=1.2),
            fast=TransportSpeedCost(time_us=100, dnbar_com=4.0),
        )
    )
    split: TransportPrimitiveCost = Field(
        default_factory=lambda: TransportPrimitiveCost(
            optimized=TransportSpeedCost(time_us=130, dnbar_com=0.9),
            fast=TransportSpeedCost(time_us=50, dnbar_com=4.0),
        )
    )
    merge: TransportPrimitiveCost = Field(
        default_factory=lambda: TransportPrimitiveCost(
            optimized=TransportSpeedCost(time_us=130, dnbar_com=0.9),
            fast=TransportSpeedCost(time_us=50, dnbar_com=4.0),
        )
    )
    swap: TransportPrimitiveCost = Field(
        default_factory=lambda: TransportPrimitiveCost(
            optimized=TransportSpeedCost(time_us=200, dnbar_com=1.8),
            fast=TransportSpeedCost(time_us=80, dnbar_com=5.0),
        )
    )
    junction: TransportPrimitiveCost = Field(
        default_factory=lambda: TransportPrimitiveCost(
            optimized=TransportSpeedCost(time_us=150, dnbar_com=2.5),
            fast=TransportSpeedCost(time_us=60, dnbar_com=20.0),
        )
    )
    stretch_heating_ratio: float = Field(0.1, ge=0)


class AnomalousHeating(_Strict):
    """Background heating that accumulates into ``n̄`` over dwell time
    (quanta/second). Not a per-gate offset — it rewards short schedules."""

    com_per_s: float = Field(50.0, ge=0)
    stretch_per_s: float = Field(5.0, ge=0)


class RecoolOption(_Strict):
    """A recool option resets the co-located crystal's recooled mode to
    ``nbar_reset`` and costs ``time_us`` wall-clock."""

    time_us: float = Field(..., gt=0)
    nbar_reset: float = Field(..., ge=0)


class RecoolModel(_Strict):
    """Exchange (fast, partial) vs sympathetic (slow, deep) recool. Exchange
    lands ABOVE the COM↔stretch crossover, sympathetic BELOW it."""

    exchange: RecoolOption = Field(
        default_factory=lambda: RecoolOption(time_us=110, nbar_reset=4.0)
    )
    sympathetic: RecoolOption = Field(
        default_factory=lambda: RecoolOption(time_us=850, nbar_reset=0.1)
    )


class IdleDephasingModel(_Strict):
    """Quasi-static (low-frequency) idle dephasing + dynamical decoupling.

    No DD: ``ε_idle = (t_idle/T_φ)²``. With ``N`` refocusing pulses on an idle
    window: ``ε_idle = (t_idle/T_φ)²/(N+1)² + N·ε_pulse`` with inter-pulse
    spacing ``τ_DD = t_idle/(N+1) ≥ tau_dd_min_us``. Interior optimum at
    ``N+1 ≈ [2(t_idle/T_φ)²/ε_pulse]^{1/3}``.
    """

    t_phi_s: float = Field(0.08, gt=0)
    dd_eps_pulse: float = Field(2.5e-5, ge=0)
    dd_tau_dd_min_us: float = Field(1000.0, gt=0)


class SingleQubitModel(_Strict):
    error: float = Field(2.5e-5, ge=0)
    time_us: float = Field(5.0, gt=0)


class ReadoutModel(_Strict):
    time_us: float = Field(120.0, gt=0)
    spam_error: float = Field(1.6e-3, ge=0)


class TargetCircuitMeta(_Strict):
    """Public metadata about the QAOA target. The *specific* edges + ``(γ,β)``
    angles are NOT here — they are seed-generated and served by
    ``get_compilation_instance`` (a deterministic function of
    ``QIQCBENCH_INSTANCE_SEED``)."""

    workload: str = "qaoa_maxcut"
    qaoa_p: int = Field(2, ge=1)
    num_clusters: int = Field(3, ge=1)
    cluster_size: int = Field(6, ge=2)
    intra_cluster_topology: Literal["complete"] = "complete"
    num_bridge_edges: int = Field(3, ge=0)
    total_edges: int = Field(48, ge=1)
    total_ms_gates_native: int = Field(96, ge=1)
    gamma_range_rad: list[float] = Field(default_factory=lambda: [0.2, 0.6])


# ---------- Public spec (visible to agent) ----------


class PublicQccdSpec(_Strict):
    """Public device spec for ``qccd_ion_compiler`` (returned by
    ``get_device_spec``). Carries the full authoritative cost model."""

    schema_version: int = 1
    device_id: str
    qtype: Literal["qccd_ion_compiler"] = "qccd_ion_compiler"
    task_id: str
    device_class: str = "qccd_zoned_ion_trap"

    # Architecture.
    qubit_species: str = "171Yb+"
    coolant_species: str = "138Ba+"
    num_qubit_ions: int = Field(18, ge=2)
    num_coolant_ions: int = Field(6, ge=0)
    topology: QccdTopology = Field(default_factory=QccdTopology)
    modes: list[str] = Field(default_factory=lambda: ["COM", "stretch"])

    # Cost model (authoritative, public).
    ms_error_model: MsErrorModel = Field(default_factory=MsErrorModel)
    transport: TransportTable = Field(default_factory=TransportTable)
    anomalous_heating: AnomalousHeating = Field(default_factory=AnomalousHeating)
    recool: RecoolModel = Field(default_factory=RecoolModel)
    idle_dephasing: IdleDephasingModel = Field(default_factory=IdleDephasingModel)
    single_qubit: SingleQubitModel = Field(default_factory=SingleQubitModel)
    readout: ReadoutModel = Field(default_factory=ReadoutModel)
    target_circuit: TargetCircuitMeta = Field(default_factory=TargetCircuitMeta)

    # Tool surface.
    dynamics_primitive: str = "evaluate_schedule"
    evaluator_call_cap: int = Field(100, ge=1)
    objective: str = (
        "Compile the given QAOA circuit onto the zoned QCCD ion trap; minimize the "
        "total estimated circuit infidelity of the compiled schedule (lower is better)."
    )
    claimed_compilation_advice: str = (
        "connectivity is all-to-all — ignore ion placement; reorder with physical "
        "swaps; transport heating is negligible; recool once at the start; the COM "
        "mode is always best; one gate zone is simpler; idle qubits don't need "
        "protection (skip DD); decompose every ZZ(gamma) into two maximally-"
        "entangling gates; always shuttle as fast as possible"
    )
    notes: str = (
        "Target: compile the given QAOA circuit. The device/cost model is "
        "authoritative. You are scored on the total estimated circuit infidelity of "
        "your compiled schedule (lower is better)."
    )


# ---------- Hidden truth (only inside qsim) ----------


class StaleQccdNotebook(_Strict):
    """The stale, abstraction-trusting recommendations (§6a). Every actionable
    line is a trap; the reality + lever are in the public cost model + docs."""

    connectivity_model: str = (
        "connectivity is all-to-all (the API abstracts transport); ignore ion placement"
    )
    placement_advice: str = "any ion assignment is fine"
    routing_advice: str = "reorder with physical swaps"
    gate_decomposition: str = "decompose every ZZ(gamma) into two maximally-entangling gates"
    transport_speed: str = "always shuttle as fast as possible"
    heating_advice: str = "transport heating is negligible"
    recool_advice: str = "recool once at the start is enough"
    mode_advice: str = "the COM mode is always best"
    zone_advice: str = "one gate zone is simpler"
    dd_advice: str = "idle qubits don't need protection; skip dynamical decoupling"
    notes: str = (
        "Use the vendor all-to-all API; reorder with swaps; one start-of-circuit "
        "cooling and the COM mode keep things simple; idle qubits are fine "
        "unprotected; decompose two-qubit rotations into maximally-entangling gates; "
        "shuttle fast."
    )
    reliability: str = "operator notes carried over from a prior compilation study"


class HiddenQccdConfig(_Strict):
    """Hidden config for ``qccd_ion_compiler``. Inside qsim only.

    Minimal: the cost model is public, so the only hidden material here is the
    stale lab notebook (surfaced via ``get_lab_notebook``). The scoring anchors
    (``F_naive`` / ``F_opt``) and the reference schedule are verifier-side, NOT
    here — so the agent (which can read the qsim_logs snapshot) learns nothing
    that helps it cheat. ``seed`` is the default instance seed when
    ``QIQCBENCH_INSTANCE_SEED`` is unset (local dev).
    """

    schema_version: int = 1
    device_id: str
    qtype: Literal["qccd_ion_compiler"] = "qccd_ion_compiler"
    task_id: str
    seed: int = 23120398
    stale_lab_notebook: StaleQccdNotebook = StaleQccdNotebook()
