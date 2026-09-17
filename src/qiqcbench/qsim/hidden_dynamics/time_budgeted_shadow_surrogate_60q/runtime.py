"""One fixed apparatus, with reference values derived from its physical model.

Neither runtime nor grading needs a cached answer table, a study opening, or a
tier-specific scientific package. The device remains private to qsim/verifier.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q import construction
from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.device import HiddenBoundedgateConfig

TASK_ID = construction.TASK_ID
HIDDEN_DEVICE_RELATIVE = Path("devices/boundedgate_60q_v0.hidden.example.yaml")
PUBLIC_DEVICE_RELATIVE = Path("devices/boundedgate_60q_v0.public.yaml")
PUBLIC_RELATIVE = Path("task_materials") / TASK_ID / "public"
PUBLIC_FILES = frozenset({"observable_index.json", "target_challenge.json"})
RUNTIME_FILES = (
    HIDDEN_DEVICE_RELATIVE,
    PUBLIC_DEVICE_RELATIVE,
    *(PUBLIC_RELATIVE / name for name in sorted(PUBLIC_FILES)),
)


def verify_runtime_sources(
    *, hidden_path: Path, public_path: Path, public_dir: Path, expected_configs: Path
) -> None:
    """Bind ferried inputs to independently baked, versioned source bytes.

    The expected root is verifier-owned. It must never alias Harbor's upload
    tree: otherwise uploading a substituted device would overwrite both sides
    of the comparison. Missing ferry files have no baked-source fallback.
    """
    if public_dir.is_symlink() or not public_dir.is_dir():
        raise ValueError("public material directory must be a real directory")
    if {path.name for path in public_dir.iterdir()} != PUBLIC_FILES:
        raise ValueError("public material inventory is not closed")
    actual_paths = (
        hidden_path,
        public_path,
        *(public_dir / name for name in sorted(PUBLIC_FILES)),
    )
    for relative, actual in zip(RUNTIME_FILES, actual_paths, strict=True):
        expected = expected_configs / relative
        for path in (actual, expected):
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"runtime source must be a regular non-symlink file: {path}")
        if actual.samefile(expected):
            raise ValueError("ferried runtime source aliases the verifier's expected source")
        if (
            hashlib.sha256(actual.read_bytes()).digest()
            != hashlib.sha256(expected.read_bytes()).digest()
        ):
            raise ValueError(f"runtime source differs from the frozen task: {relative}")


def build_runtime_truth(hidden: HiddenBoundedgateConfig) -> dict:
    """Compute the as-measured pair responses of the device actually served.

    The raw-bitstring sampler and this Pauli back-propagation calculation are
    independently checked in scientific tests. No reference learner is used.
    """
    if (
        hidden.device_id != construction.DEVICE_ID
        or hidden.n_qubits != construction.N_QUBITS
        or hidden.input_dimension != construction.INPUT_DIM
        or hidden.target_challenge is None
        or len(hidden.target_challenge.target_inputs) != construction.N_TARGETS
    ):
        raise ValueError("runtime device does not define this task's sealed physical instance")
    if any(hidden.readout_drift_per_qubit or []):
        raise ValueError("the scored apparatus must have stationary readout")
    if any(not 0.01 <= flip <= 0.05 for flip in hidden.depolarizing_flip_prob):
        raise ValueError("runtime readout noise lies outside the task's physical family")

    polys = construction.build_pair_polys(hidden.circuit, hidden.depolarizing_flip_prob)
    for pair_index, poly in enumerate(polys):
        coefficients = construction.canonical_fourier_coefficients(poly)
        if len(coefficients) > construction.MAX_FOURIER_MODES_PER_RESPONSE:
            raise ValueError(f"pair response {pair_index} exceeds the public Fourier mode cap")
        if any(
            abs(harmonic) > construction.DEGREE_BOUND
            for frequency in coefficients
            for harmonic in frequency
        ):
            raise ValueError(f"pair response {pair_index} exceeds the public harmonic bound")
    diagnostics = construction.normal_form_support_diagnostics(polys)
    if (
        diagnostics.over_cap_pairs
        or diagnostics.max_support > construction.ADMISSION_MAX_NORMAL_FORM_SUPPORT
        or diagnostics.max_terms_per_pair > construction.MAX_RAW_TERMS_PER_PAIR
    ):
        raise ValueError("runtime pair responses violate the fixed-instance normal-form contract")

    challenge = hidden.target_challenge.model_dump(mode="json")
    targets = np.asarray(challenge["target_inputs"], dtype=float)
    labels = construction.evaluate_pair_polys(polys, targets)
    return {
        "task_id": TASK_ID,
        "device_id": hidden.device_id,
        "n_qubits": hidden.n_qubits,
        "input_dimension": hidden.input_dimension,
        "target_challenge": challenge,
        "target_inputs": targets.tolist(),
        "labels": labels.tolist(),
    }
