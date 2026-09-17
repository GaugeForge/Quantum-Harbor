"""Firmware-owned matching decoder for syndrome-feedback experiments.

The agent cannot configure this object.  Qsim constructs it once from the
device's fixed nominal DEM and applies it to every raw control-epoch detector
record.  Keeping the decoder beside the capability makes the feedback
instrument's physical chain explicit without turning decoder configuration
into an agent-facing surface.
"""

from __future__ import annotations

import numpy as np
import pymatching

from qiqcbench.qsim.qtypes.surface_code_memory import surface_code as SC


def _edge_weight(probability: float) -> float:
    probability = min(max(float(probability), 1.0e-9), 0.4)
    return float(np.log((1.0 - probability) / probability))


class FixedFirmwareDecoder:
    """A fixed graphlike MWPM decoder returning one logical-frame bit."""

    def __init__(
        self,
        dem: list[tuple[int, int, float, int]],
        *,
        n_detectors: int,
    ) -> None:
        if n_detectors <= 0:
            raise ValueError("n_detectors must be positive")
        matching = pymatching.Matching()
        for detector_a, detector_b, probability, flips_observable in dem:
            if detector_a < 0 or detector_a >= n_detectors:
                raise ValueError("fixed decoder DEM references an invalid detector")
            fault_ids = {0} if flips_observable else set()
            weight = _edge_weight(probability)
            if detector_b == SC.BOUNDARY:
                matching.add_boundary_edge(
                    detector_a,
                    fault_ids=fault_ids,
                    weight=weight,
                    merge_strategy="independent",
                )
            else:
                if detector_b < 0 or detector_b >= n_detectors:
                    raise ValueError("fixed decoder DEM references an invalid detector")
                if detector_a == detector_b:
                    continue
                matching.add_edge(
                    detector_a,
                    detector_b,
                    fault_ids=fault_ids,
                    weight=weight,
                    merge_strategy="independent",
                )
        matching.set_boundary_nodes(set())
        self._matching = matching
        self.n_detectors = n_detectors

    def decode(self, detector_records: np.ndarray) -> np.ndarray:
        records = np.asarray(detector_records, dtype=np.uint8)
        if records.ndim != 2 or records.shape[1] != self.n_detectors:
            raise ValueError(f"detector_records must have shape [shots, {self.n_detectors}]")
        predictions = np.asarray(self._matching.decode_batch(records), dtype=np.uint8)
        if predictions.size == 0:
            return np.zeros(records.shape[0], dtype=np.uint8)
        return predictions.reshape(records.shape[0], -1)[:, 0]


__all__ = ["FixedFirmwareDecoder"]
