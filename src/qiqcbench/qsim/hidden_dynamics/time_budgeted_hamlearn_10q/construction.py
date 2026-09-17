"""Fixed hidden instance for ``time_budgeted_hamlearn_10q``.

The instance is **fixed**, not seed-generated (``2026061719`` is only a private
RNG salt; production mixes it with fresh per-run entropy, and feasibility tests
may inject explicit generators into :class:`Oracle`). This module
owns:

* the true 24-term hidden coefficient vector and its self-consistency
  invariants (Patch 4: no ``|omega|`` in ``[0.085, 0.130]``);
* the stale lab notebook;
* the agent-facing public materials (dictionary, public spec, stale notebook);
* the hidden scorer payload + the sha256 digests binding it to the public bytes;
* a ``write_materials`` helper that emits the ``public/`` + ``hidden/`` files.

Operating point:
``BUDGET_US = 220`` and ``ORACLE_NOISE_SHOTS = 1024`` — the original
``120 µs / 512`` shot values fail the gate, and σ=0.045 cannot pass at any
budget rung, so the ladder's noise-reduction step is mandatory.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ruamel.yaml import YAML

from qiqcbench.qsim.hidden_dynamics.time_budgeted_hamlearn_10q.dictionary import (
    N_QUBITS,
    N_TERMS,
    TERM_ORDER,
    TERM_ORDER_CONVENTION,
    pauli_string,
    term_weight,
)
from qiqcbench.qsim.hidden_dynamics.time_budgeted_hamlearn_10q.schema import (
    HAMILTONIAN_CONVENTION,
    HamLearnHiddenScorer,
)

__all__ = [
    "BUDGET_US",
    "COEFFICIENT_BOUND",
    "FORBIDDEN_GAP",
    "HIDDEN_OMEGA",
    "MAX_PROBE_ROWS",
    "MAX_ROWS_PER_BATCH",
    "ORACLE_NOISE_SHOTS",
    "STALE_NOTEBOOK",
    "SUPPORT_TAU",
    "HamLearnInstance",
    "build_instance",
    "build_omega_star",
    "build_stale_vector",
    "support_terms",
    "write_materials",
]

# --- locked operating point ------------------------------------------------
BUDGET_US = 220.0
ORACLE_NOISE_SHOTS = 1024
MAX_PROBE_ROWS = 600
MAX_ROWS_PER_BATCH = 64

# --- public physics constants -----------------------------------
TIME_MIN_US = 0.010
TIME_MAX_US = 2.500
TIME_RESOLUTION_US = 0.002
OBS_MIN_WEIGHT = 1
OBS_MAX_WEIGHT = 4
COEFFICIENT_BOUND = 0.60
SUPPORT_TAU = 0.10
FORBIDDEN_GAP = (0.085, 0.130)  # no hidden nonzero coefficient may live here
STATE_LABELS = ("z+", "z-", "x+", "x-", "y+", "y-")

# --- hidden truth -------------------------------------
HIDDEN_OMEGA: dict[str, float] = {
    "X0": +0.31,
    "Z0": -0.18,
    "Y1": +0.27,
    "Z2": +0.22,
    "X3": -0.34,
    "Y4": +0.19,
    "Z4": +0.075,
    "Z5": -0.29,
    "X6": +0.24,
    "Y6": +0.155,
    "Y7": -0.21,
    "X9": -0.26,
    "Z0Z1": +0.42,
    "X1Z2": -0.33,
    "Z1Y2": +0.17,
    "Y2Y3": +0.28,
    "X2X3": -0.165,
    "Z3X4": -0.37,
    "X4X5": +0.25,
    "Y5Z6": -0.31,
    "X5X6": +0.155,
    "Z6Z7": +0.45,
    "X7Y8": +0.23,
    "Y8X9": -0.27,
}

# --- stale lab notebook ----------------------------------------
STALE_NOTEBOOK: dict[str, float] = {
    "X0": +0.29,
    "Z0": -0.10,
    "Y1": +0.25,
    "Z2": +0.13,
    "X3": -0.32,
    "Y4": +0.11,
    "Z5": -0.16,
    "X6": +0.18,
    "Y7": -0.16,
    "Z9": -0.22,
    "Z0Z1": +0.20,
    "X1Z2": -0.15,
    "Y2Y3": +0.10,
    "Z3X4": -0.36,
    "X4X5": +0.21,
    "Z6Z7": +0.12,
    "X7Y8": +0.18,
    "X0Y1": +0.14,
    "Z7Z8": -0.19,
    "X8X9": -0.15,
}

STALE_NOTE = (
    "Historical, unverified coefficient estimates from an earlier device "
    "characterization. Entries may be biased, missing, or mislabeled. Current-device "
    "qsim evidence is authoritative."
)
PUBLIC_NOTE = (
    "The Hamiltonian dictionary is public, but the coefficient vector and "
    "support are hidden. The lab notebook may be stale. Passing requires both "
    "worst-coefficient and relative full-vector accuracy. The evolve-time range "
    "and resolution are device constraints, not protocol recommendations."
)


def build_omega_star() -> np.ndarray:
    """The hidden coefficients projected onto the 111-term ordering."""
    return np.array([HIDDEN_OMEGA.get(t, 0.0) for t in TERM_ORDER], dtype=float)


def build_stale_vector() -> np.ndarray:
    """The stale notebook coefficients projected onto the 111-term ordering."""
    return np.array([STALE_NOTEBOOK.get(t, 0.0) for t in TERM_ORDER], dtype=float)


def support_terms(tau: float = SUPPORT_TAU) -> list[str]:
    """Terms with ``|omega*| > tau`` (the §9b above-threshold support set)."""
    return [t for t in TERM_ORDER if abs(HIDDEN_OMEGA.get(t, 0.0)) > tau]


def _assert_self_consistent() -> None:
    """Patch-4 invariants the instance must satisfy (validated by tests too)."""
    nonzero = [v for v in HIDDEN_OMEGA.values() if v != 0.0]
    assert len(nonzero) == 24, f"expected 24 nonzero terms, got {len(nonzero)}"
    lo, hi = FORBIDDEN_GAP
    bad = [t for t, v in HIDDEN_OMEGA.items() if lo <= abs(v) <= hi]
    assert not bad, f"coefficients inside forbidden gap {FORBIDDEN_GAP}: {bad}"
    assert all(abs(v) <= COEFFICIENT_BOUND for v in HIDDEN_OMEGA.values())
    above = support_terms()
    assert len(above) == 23, f"expected 23 above-tau terms, got {len(above)}"
    # every hidden term is a valid dictionary term
    assert set(HIDDEN_OMEGA) <= set(TERM_ORDER)
    assert set(STALE_NOTEBOOK) <= set(TERM_ORDER)


# --- serialization helpers (mirror the VQE digest convention) --------------
_yaml = YAML()
_yaml.default_flow_style = False
_yaml.width = 4096


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2) + "\n").encode("utf-8")


def _yaml_bytes(payload: dict[str, Any]) -> bytes:
    buf = io.BytesIO()
    _yaml.dump(payload, buf)
    return buf.getvalue()


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def build_hamiltonian_dictionary() -> dict[str, Any]:
    """Public 111-term dictionary exposed via ``get_device_spec``.

    Carries every term's name, length-10 Pauli string, and weight so the agent
    can build its own design matrix without any hidden knowledge.
    """
    return {
        "term_order_convention": TERM_ORDER_CONVENTION,
        "num_terms": N_TERMS,
        "n_qubits": N_QUBITS,
        "hamiltonian_convention": HAMILTONIAN_CONVENTION,
        "terms": [
            {"name": t, "pauli": pauli_string(t), "weight": term_weight(t)} for t in TERM_ORDER
        ],
    }


def build_public_spec() -> dict[str, Any]:
    """Agent-facing public device spec."""
    return {
        "task_id": "time_budgeted_hamlearn_10q",
        "device_class": "blackbox_analog_dynamics",
        "n_qubits": N_QUBITS,
        "qubit_labels": [f"q{i}" for i in range(N_QUBITS)],
        "dynamics_primitive": "run_hamiltonian_probe_batch",
        "hamiltonian_convention": HAMILTONIAN_CONVENTION,
        "coefficient_unit": "rad_per_us",
        "locality_promise": "1-local terms plus nearest-neighbor 2-local terms",
        "sparsity_promise_max_nonzero": 30,
        "coefficient_bound_rad_per_us": [-COEFFICIENT_BOUND, COEFFICIENT_BOUND],
        "allowed_initial_states": list(STATE_LABELS),
        "allowed_observable_weight": [OBS_MIN_WEIGHT, OBS_MAX_WEIGHT],
        "evolve_time_range_us": [TIME_MIN_US, TIME_MAX_US],
        "evolve_time_resolution_us": TIME_RESOLUTION_US,
        "off_grid_time_policy": "reject",
        "return_type": "raw_pauli_outcomes",
        "num_internal_repetitions": ORACLE_NOISE_SHOTS,
        "nominal_single_probe_noise_1sigma": float(round(1.0 / np.sqrt(ORACLE_NOISE_SHOTS), 4)),
        "total_evolution_time_budget_us": BUDGET_US,
        "max_probe_rows": MAX_PROBE_ROWS,
        "max_rows_per_batch": MAX_ROWS_PER_BATCH,
        "target_wall_clock_runtime_s": 1200,
        "term_order_convention": TERM_ORDER_CONVENTION,
        # Finite transport/action caps (DoS hardening, not scientific budgets).
        # Enforced by shared MCP/action seams; every value sits far above
        # legitimate agent behavior — the reference-solver answer envelope is
        # ~6 KB, all 600 probe rows serialize well under the aggregate ingress
        # allowance, and one submission suffices.
        "budget": {
            "max_ingress_request_bytes": 1_000_000,
            "max_final_answer_serialized_bytes": 262_144,
            "max_final_answer_submissions": 5,
            "max_answer_string_characters": 20_000,
            "max_final_answer_nesting_depth": 16,
            "max_job_result_polls": 20_000,
            "max_probe_jobs": 640,
        },
        "notes": PUBLIC_NOTE,
    }


def build_stale_notebook() -> dict[str, Any]:
    """Agent-facing stale lab notebook."""
    return {
        "task_id": "time_budgeted_hamlearn_10q",
        "coefficients_rad_per_us": dict(STALE_NOTEBOOK),
        "note": STALE_NOTE,
        "reliability": "stale; use only as a weak prior",
    }


@dataclass(frozen=True)
class HamLearnInstance:
    """Frozen fixed instance: public materials + hidden scorer + digests.

    Mirrors ``ScrambledSpinGlassInstance``. Holds the public materials in the
    JSON/YAML structures they are written as, the hidden ``omega_star`` vector,
    and the sha256 digests binding the hidden scorer to the public bytes. The
    instance does **not** know oracle-noise seeds.
    """

    omega_star: np.ndarray
    hamiltonian_dictionary: dict[str, Any]
    public_spec: dict[str, Any]
    stale_notebook: dict[str, Any]
    hidden_scorer: HamLearnHiddenScorer
    public_material_digests: dict[str, str]

    def hamiltonian_dictionary_bytes(self) -> bytes:
        return _json_bytes(self.hamiltonian_dictionary)

    def public_spec_bytes(self) -> bytes:
        return _yaml_bytes(self.public_spec)

    def stale_notebook_bytes(self) -> bytes:
        return _yaml_bytes(self.stale_notebook)

    def hidden_scorer_bytes(self) -> bytes:
        return _yaml_bytes(self.hidden_scorer.model_dump(mode="json"))


# filenames are part of the public/hidden contract
HAMILTONIAN_DICTIONARY_FILE = "hamiltonian_dictionary.json"
PUBLIC_SPEC_FILE = "public_spec.yaml"
STALE_NOTEBOOK_FILE = "stale_notebook.yaml"
HIDDEN_SCORER_FILE = "hidden_scorer.yaml"


def build_instance() -> HamLearnInstance:
    """Build the fixed ``sparse_10q_chain_linf_v1`` instance with invariants checked."""
    _assert_self_consistent()

    hamiltonian_dictionary = build_hamiltonian_dictionary()
    public_spec = build_public_spec()
    stale_notebook = build_stale_notebook()

    public_material_digests = {
        HAMILTONIAN_DICTIONARY_FILE: _digest(_json_bytes(hamiltonian_dictionary)),
        PUBLIC_SPEC_FILE: _digest(_yaml_bytes(public_spec)),
        STALE_NOTEBOOK_FILE: _digest(_yaml_bytes(stale_notebook)),
    }

    omega_star = build_omega_star()
    hidden_scorer = HamLearnHiddenScorer(
        schema_version=3,
        public_material_digests=public_material_digests,
        omega_star=omega_star.tolist(),
        linf_pass=0.085,
        linf_bronze=0.075,
        linf_silver=0.055,
        linf_gold=0.035,
        relative_l2_pass=0.12,
        support_tau=SUPPORT_TAU,
        support_terms=support_terms(),
        budget_us=BUDGET_US,
        max_probe_rows=MAX_PROBE_ROWS,
        oracle_noise_shots=ORACLE_NOISE_SHOTS,
    )

    return HamLearnInstance(
        omega_star=omega_star,
        hamiltonian_dictionary=hamiltonian_dictionary,
        public_spec=public_spec,
        stale_notebook=stale_notebook,
        hidden_scorer=hidden_scorer,
        public_material_digests=public_material_digests,
    )


def write_materials(dest_dir: Path) -> dict[str, Path]:
    """Write ``public/`` + ``hidden/`` material files; return the written paths.

    Used with a ``tmp_path`` during the feasibility gate; the real
    ``configs/task_materials/...`` is committed only after the gate is green.
    """
    dest_dir = Path(dest_dir)
    public = dest_dir / "public"
    hidden = dest_dir / "hidden"
    public.mkdir(parents=True, exist_ok=True)
    hidden.mkdir(parents=True, exist_ok=True)

    instance = build_instance()
    paths = {
        HAMILTONIAN_DICTIONARY_FILE: public / HAMILTONIAN_DICTIONARY_FILE,
        PUBLIC_SPEC_FILE: public / PUBLIC_SPEC_FILE,
        STALE_NOTEBOOK_FILE: public / STALE_NOTEBOOK_FILE,
        HIDDEN_SCORER_FILE: hidden / HIDDEN_SCORER_FILE,
    }
    paths[HAMILTONIAN_DICTIONARY_FILE].write_bytes(instance.hamiltonian_dictionary_bytes())
    paths[PUBLIC_SPEC_FILE].write_bytes(instance.public_spec_bytes())
    paths[STALE_NOTEBOOK_FILE].write_bytes(instance.stale_notebook_bytes())
    paths[HIDDEN_SCORER_FILE].write_bytes(instance.hidden_scorer_bytes())
    return paths
