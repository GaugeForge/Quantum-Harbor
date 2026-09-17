"""Strict public/hidden device models for site-resolved Rydberg arrays."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ManyBodyBudgets(_Strict):
    shot_budget: int = Field(600_000, ge=1)
    max_shots_per_call: int = Field(50_000, ge=1)


class PublicRydbergProtocol(_Strict):
    protocol_id: Literal["depth6_rydberg_ising_v2"] = "depth6_rydberg_ising_v2"
    depth: Literal[6] = 6
    rotation_angles_rad: list[float] = Field(
        default_factory=lambda: [1.814, 1.217, 1.527, 1.093, 1.767, -0.709],
        min_length=6,
        max_length=6,
    )
    zz_angles_rad: list[float] = Field(
        default_factory=lambda: [1.229, 0.707, 0.757, -1.079, -0.245, -0.611],
        min_length=6,
        max_length=6,
    )
    # Non-destructive occupancy imaging interleaved into the sequence. An atom appears in
    # this image when it survived layers 0..1, so the image separates atoms lost early from
    # atoms lost late -- which is the point of erasure imaging on this hardware.
    midsequence_image_after_layer: Literal[2] = 2


class PublicNeutralAtomManyBodySpec(_Strict):
    # v2 removed the per-layer ``loss_fault_radii``: the loss-influence reach is hidden truth.
    # v3 removes the corresponding hidden knob entirely -- loss now acts by removing the atom
    # from the circuit, so the reach follows from the published protocol -- and replaces the
    # single post-sequence image with the three-image erasure record the readout actually
    # produces. Bumped deliberately rather than reshaping the agent-facing spec silently.
    schema_version: int = 3
    device_id: str
    qtype: Literal["neutral_atom_manybody_simulator"] = "neutral_atom_manybody_simulator"
    task_id: str | None = None
    supported_system_sizes: list[int] = Field(default_factory=lambda: [25, 49, 81])
    protocol: PublicRydbergProtocol = PublicRydbergProtocol()
    budgets: ManyBodyBudgets = ManyBodyBudgets()
    measurement_return: Literal["three_occupancy_images_with_target_z"] = (
        "three_occupancy_images_with_target_z"
    )
    notes: str = (
        "All sites start in |0>. Each protocol layer applies a global Ry rotation and then an "
        "alternating nearest-neighbour Rydberg-Ising ZZ matching. An atom that leaves its trap "
        "takes no further part: it receives no rotation and its two bonds carry no ZZ gate for "
        "the remaining layers, so the surviving atoms evolve under a different circuit. The "
        "array is imaged after loading, once mid-sequence, and again at the end; the state of "
        "the two central target atoms is read out at the end."
    )


class HiddenRydbergManyBodyNoise(_Strict):
    # Trap-loss channel. ``preparation_loss`` is a site that never loaded, so it is absent for
    # the whole sequence; ``layer_loss`` is the per-layer probability of losing an atom that is
    # still trapped. Bursts model correlated loading/vacuum events. Nothing here encodes how far
    # a loss propagates: that follows from the published protocol.
    preparation_loss: float = Field(0.021, ge=0.0, le=1.0)
    layer_loss: float = Field(0.049, ge=0.0, le=1.0)
    burst_probability: float = Field(0.018, ge=0.0, le=1.0)
    burst_multiplier: float = Field(3.0, ge=1.0)
    readout_p0_to_1: float = Field(0.018, ge=0.0, le=1.0)
    readout_p1_to_0: float = Field(0.041, ge=0.0, le=1.0)


class StaleRydbergManyBodyNotebook(_Strict):
    last_calibrated: str = "2026-06-12"
    loss_note: str = "Discard any shot with a visibly missing atom; this is safest."
    correlation_note: str = "Only the two measured sites need to be occupied."
    readout_note: str = "Fluorescence errors are approximately symmetric and negligible."


class HiddenNeutralAtomManyBodyConfig(_Strict):
    schema_version: int = 1
    device_id: str
    qtype: Literal["neutral_atom_manybody_simulator"] = "neutral_atom_manybody_simulator"
    seed: int
    noise: HiddenRydbergManyBodyNoise = HiddenRydbergManyBodyNoise()
    stale_lab_notebook: StaleRydbergManyBodyNotebook = StaleRydbergManyBodyNotebook()


__all__ = [
    "HiddenNeutralAtomManyBodyConfig",
    "HiddenRydbergManyBodyNoise",
    "ManyBodyBudgets",
    "PublicNeutralAtomManyBodySpec",
    "PublicRydbergProtocol",
    "StaleRydbergManyBodyNotebook",
]
