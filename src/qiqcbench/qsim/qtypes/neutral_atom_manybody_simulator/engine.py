"""Loss-resolved trajectory engine for the fixed public Rydberg protocol.

An atom that leaves its trap stops taking part in the sequence: from the layer it is
lost it receives no rotation and its two bonds carry no ZZ gate, so the surviving
atoms genuinely evolve under a different circuit. That makes the reach of a loss a
consequence of the published protocol rather than a free parameter, and it makes the
reach shrink the later the loss happens.

The scored observable is local, so only a ten-site block around the target pair can
affect it (see :mod:`.influence`). The exact outcome distribution of that block is
enumerated once per protocol over the finite set of loss configurations, after which
every shot is a table lookup -- which is what keeps a six-hundred-thousand-shot run
cheap while still simulating the loss-modified dynamics exactly.
"""

from __future__ import annotations

import numpy as np

from qiqcbench.qsim.qtypes.neutral_atom_manybody_simulator import influence
from qiqcbench.qsim.qtypes.neutral_atom_manybody_simulator.device import (
    HiddenNeutralAtomManyBodyConfig,
    PublicNeutralAtomManyBodySpec,
)
from qiqcbench.qsim.qtypes.neutral_atom_manybody_simulator.wire import (
    JobRydbergTrajectoryData,
    RydbergTrajectoryRequest,
    pack_binary,
)

#: Absence code for an atom that is still trapped at the final image.
SURVIVES = influence.DEPTH + 1
#: Smallest absence code still visible in the load image.
LOAD_IMAGE_MIN = 1


def midsequence_image_min(public: PublicNeutralAtomManyBodySpec) -> int:
    """Smallest absence code still visible in the mid-sequence erasure image."""
    return public.protocol.midsequence_image_after_layer + 1


_PAIR_TABLE_CACHE: dict[tuple[tuple[float, ...], tuple[float, ...]], np.ndarray] = {}


def pair_distribution_table(
    rotation_angles: list[float] | tuple[float, ...],
    zz_angles: list[float] | tuple[float, ...],
) -> np.ndarray:
    """Exact target-pair outcome distributions for every loss configuration."""
    key = (tuple(rotation_angles), tuple(zz_angles))
    if key not in _PAIR_TABLE_CACHE:
        _PAIR_TABLE_CACHE[key] = influence.build_pair_distributions(*key)
    return _PAIR_TABLE_CACHE[key]


def target_left_site(system_size: int) -> int:
    """Absolute index of the left target atom."""
    return system_size // 2


def supports_system_size(system_size: int) -> bool:
    """Whether the influence block fits, on the brickwork parity the table was built on."""
    centre = target_left_site(system_size)
    return (
        centre % 2 == 0
        and centre + min(influence.INFLUENCE_OFFSETS) >= 0
        and centre + max(influence.INFLUENCE_OFFSETS) < system_size
    )


def dressed_correlation(joint: np.ndarray, p0_to_1: float, p1_to_0: float) -> np.ndarray:
    """Displayed ``<Z Z>`` after the asymmetric readout channel."""
    joint = np.asarray(joint)
    zz = joint[..., 0] - joint[..., 1] - joint[..., 2] + joint[..., 3]
    left = joint[..., 0] + joint[..., 1] - joint[..., 2] - joint[..., 3]
    right = joint[..., 0] - joint[..., 1] + joint[..., 2] - joint[..., 3]
    scale, offset = 1.0 - p0_to_1 - p1_to_0, p1_to_0 - p0_to_1
    return scale * scale * zz + scale * offset * (left + right) + offset * offset


class NeutralAtomManyBodyEngine:
    def __init__(
        self,
        hidden: HiddenNeutralAtomManyBodyConfig,
        public: PublicNeutralAtomManyBodySpec,
        rng: np.random.Generator,
    ) -> None:
        self.hidden = hidden
        self.public = public
        self.rng = rng
        self.shots_used = 0

    def _table(self) -> np.ndarray:
        protocol = self.public.protocol
        return pair_distribution_table(protocol.rotation_angles_rad, protocol.zz_angles_rad)

    def _absence_layers(self, shots: int, n: int) -> np.ndarray:
        """Per-shot, per-site absence code: ``0`` never loaded, ``k`` lost during layer
        ``k-1``, :data:`SURVIVES` still trapped at the end."""
        noise = self.hidden.noise
        burst = self.rng.random(shots) < noise.burst_probability
        multiplier = np.where(burst, noise.burst_multiplier, 1.0)
        preparation = np.minimum(1.0, noise.preparation_loss * multiplier)[:, None]
        per_layer = np.minimum(1.0, noise.layer_loss * multiplier)[:, None]
        absent = np.full((shots, n), SURVIVES, dtype=np.int8)
        trapped = self.rng.random((shots, n)) >= preparation
        absent[~trapped] = 0
        for layer in range(influence.DEPTH):
            lost = trapped & (self.rng.random((shots, n)) < per_layer)
            absent[lost] = layer + 1
            trapped &= ~lost
        return absent

    def _target_readout(self, absent: np.ndarray, n: int) -> np.ndarray:
        """Raw fluorescence bits of the two target atoms, ``(shots, 2)``."""
        centre = target_left_site(n)
        columns = [centre + offset for offset in influence.FREE_OFFSETS]
        classes = influence.absence_to_class(absent[:, columns])
        probabilities = self._table()[influence.configuration_index(classes)]
        draw = self.rng.random(len(absent))[:, None]
        outcome = (np.cumsum(probabilities, axis=1) < draw).sum(axis=1).clip(0, 3)
        bits = np.stack([outcome >> 1, outcome & 1], axis=1).astype(bool)
        noise = self.hidden.noise
        flip = np.where(bits, noise.readout_p1_to_0, noise.readout_p0_to_1)
        bits ^= self.rng.random(bits.shape) < flip
        # An atom that is not in the final image emits nothing. Both target bits are
        # reported dark whenever either target atom is missing: the pair correlation has
        # no meaning in such a shot, and no admissible selection rule can retain one.
        both_present = (absent[:, centre] == SURVIVES) & (absent[:, centre + 1] == SURVIVES)
        return bits & both_present[:, None]

    def run(self, request: RydbergTrajectoryRequest) -> JobRydbergTrajectoryData | str:
        if request.protocol_id != self.public.protocol.protocol_id:
            return f"unsupported protocol {request.protocol_id!r}"
        if request.system_size not in self.public.supported_system_sizes:
            return f"system_size {request.system_size} is not publicly available"
        if not supports_system_size(request.system_size):
            return f"system_size {request.system_size} is not a supported array geometry"
        if request.shots > self.public.budgets.max_shots_per_call:
            return "shots exceeds max_shots_per_call"
        if self.shots_used + request.shots > self.public.budgets.shot_budget:
            return "run-long trajectory shot budget exceeded"
        n, shots = request.system_size, request.shots
        absent = self._absence_layers(shots, n)
        readout = self._target_readout(absent, n)
        self.shots_used += shots
        centre = target_left_site(n)
        return JobRydbergTrajectoryData(
            protocol_id=request.protocol_id,
            system_size=n,
            shots=shots,
            target_sites=[centre, centre + 1],
            load_occupancy_b64=pack_binary(absent >= LOAD_IMAGE_MIN),
            midsequence_occupancy_b64=pack_binary(absent >= midsequence_image_min(self.public)),
            final_occupancy_b64=pack_binary(absent >= SURVIVES),
            target_z_readout_b64=pack_binary(readout),
        )

    def no_loss_raw_zz_reference(self) -> float:
        """Displayed central correlation with atom loss switched off.

        Independent of the system size: the observable's backward support is ten sites
        wide, so every supported array is bulk for this quantity.
        """
        noise = self.hidden.noise
        joint = self._table()[influence.configuration_index(influence.all_present_classes())]
        return float(dressed_correlation(joint, noise.readout_p0_to_1, noise.readout_p1_to_0))


__all__ = [
    "LOAD_IMAGE_MIN",
    "SURVIVES",
    "NeutralAtomManyBodyEngine",
    "dressed_correlation",
    "midsequence_image_min",
    "pair_distribution_table",
    "supports_system_size",
    "target_left_site",
]
