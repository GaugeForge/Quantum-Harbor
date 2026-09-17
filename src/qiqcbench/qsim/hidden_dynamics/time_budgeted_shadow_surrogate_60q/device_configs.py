"""Materialize ``time_budgeted_shadow_surrogate_60q`` onto the bounded-gate qtype.

Bridges the task's fixed hidden instance (``construction.py``) to the reusable
``blackbox_boundedgate_circuit`` device-config shapes.

* ``build_public_device_spec()`` -> dict validating as ``PublicBoundedgateSpec``
  (qubit count, input dimension, disclosed bounds, budgets; no circuit).
* ``build_hidden_device_config()`` -> dict validating as
  ``HiddenBoundedgateConfig`` (the full hidden circuit, per-qubit depolarizing
  rates, shot-RNG seed, stale notebook). Hidden truth: committed only as
  ``*.hidden.example.yaml`` and never mounted into the agent container.

Task code depending on the reusable qtype models is the intended direction;
the qtype never imports this module.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import TYPE_CHECKING

from ruamel.yaml import YAML

from qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q.construction import (
    DEGREE_BOUND,
    INPUT_DIM,
    INSTANCE_SEED,
    MAX_BLOCKS_PER_JOB,
    MAX_BOND,
    MAX_JOBS,
    MAX_SETTINGS_PER_JOB,
    MAX_SHOTS_PER_SETTING,
    N_QUBITS,
    RETIRED_DEVELOPMENT_INSTANCE_SEEDS,
    ROTATION_GATES,
    TASK_ID,
    TOTAL_SHOT_BUDGET,
    AdmissionReport,
    ShadowSurrogateInstance,
    build_instance,
    challenge_metadata,
)
from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.device import (
    HiddenBoundedgateConfig,
    PublicBoundedgateSpec,
)

if TYPE_CHECKING:
    from qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q.study import (
        SuccessorStudyFixtureProof,
    )

__all__ = [
    "DEVICE_ID",
    "boundedgate_configs_from_instance",
    "build_hidden_device_config",
    "build_public_device_spec",
    "derived_shot_seed",
    "emit_edition",
    "emit_official_edition",
    "write_device_configs",
]

DEVICE_ID = "boundedgate_60q_v0"
_DEVELOPMENT_FIXTURE_HEADER = (
    b"# qiqcbench:development-fixture-only\n"
    b"# This tracked example supports self-contained development checks only.\n"
)


def derived_shot_seed(instance_seed: int) -> int:
    """256-bit shot-RNG root, domain-separated from the edition root.

    A public fixed shot seed is an actionable oracle: an agent that can
    reproduce the sampling uniforms of its own jobs extracts probability
    intervals far tighter than shot noise. Deriving the shot root from the
    edition root keeps it private exactly when the edition root is private
    (``new_private_seed()``); the truth polynomials and labels do not depend
    on it. SHA-256 domain derivation (the same KDF construction as
    ``construction._domain_rng``, and exactly what the edition manifest
    declares) — not SeedSequence, whose mixing is statistical, not hiding.
    """
    digest = hashlib.sha256(f"qiqcbench/{TASK_ID}/v2/shot/{instance_seed}".encode()).digest()
    return int.from_bytes(digest, "big")


_STALE_NOTEBOOK = {
    # Neutral, in-world characterization memo for the stationary edition. It
    # records only operational facts (readout stability, raw bitstrings, no reusable fit)
    # and makes NO claim about response structure -- not even "all pairs
    # respond", which is both a hidden structural fact and false for the
    # roughly 4 % of pairs admission allows to be constant.
    "previous_fit_summary": (
        "Acceptance characterization: raw outcome bitstrings were recorded at a "
        "handful of inputs to confirm the device operates; no response model "
        "was fitted and no earlier fit is reusable for this run."
    ),
    "suggested_degrees": None,
    "suggested_protocol": None,
    "last_characterized": "2026-08-20",
    "notes": (
        "Per-qubit readout error was stable for the duration of the acceptance "
        "session; the device is operated at its start-of-run condition and "
        "reports raw outcome bitstrings only."
    ),
}


def boundedgate_configs_from_instance(
    instance: ShadowSurrogateInstance,
    *,
    device_id: str = DEVICE_ID,
    seed: int | None = None,
) -> tuple[PublicBoundedgateSpec, HiddenBoundedgateConfig]:
    if seed is None:
        seed = derived_shot_seed(instance.seed)
    challenge = challenge_metadata(instance)
    public = PublicBoundedgateSpec.model_validate(
        {
            "schema_version": 3,
            "device_id": device_id,
            "qtype": "blackbox_boundedgate_circuit",
            "n_qubits": N_QUBITS,
            "input_dimension": INPUT_DIM,
            "max_rotation_gates": ROTATION_GATES,
            "max_harmonic_degree_per_angle": DEGREE_BOUND,
            "budget": {
                "total_shot_budget": TOTAL_SHOT_BUDGET,
                "max_jobs": MAX_JOBS,
                "max_blocks_per_job": MAX_BLOCKS_PER_JOB,
                "max_settings_per_job": MAX_SETTINGS_PER_JOB,
                "max_shots_per_setting": MAX_SHOTS_PER_SETTING,
                "max_ingress_request_bytes": 8 * 1024 * 1024,
                "max_metadata_calls": 256,
                "max_job_result_polls": 4096,
                "max_sealed_holdout_reveals": 4,
                "max_final_answer_serialized_bytes": 4 * 1024 * 1024,
                "max_final_answer_submissions": 4,
                "max_answer_string_characters": 16_384,
            },
            "target_challenge": {
                "schema_version": 1,
                "challenge_id": challenge["challenge_id"],
                "commitment_scheme": "qiqcbench_sealed_target_sha256_v1",
                "target_count": len(challenge["target_inputs"]),
                "input_dimension": INPUT_DIM,
                "target_commitment_sha256": challenge["target_commitment_sha256"],
            },
        }
    )
    from qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q.construction import (
        instance_digest,
    )
    from qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q.scorer import (
        scorer_policy_sha256,
    )

    hidden = HiddenBoundedgateConfig.model_validate(
        {
            "schema_version": 2,
            "device_id": device_id,
            "qtype": "blackbox_boundedgate_circuit",
            "seed": seed,
            "input_dimension": INPUT_DIM,
            "circuit": [op.model_dump(exclude_none=True) for op in instance.circuit],
            "depolarizing_flip_prob": instance.flip_prob,
            "readout_drift_per_qubit": instance.drift,
            "max_bond": MAX_BOND,
            "target_challenge": {
                "schema_version": 1,
                "challenge_id": challenge["challenge_id"],
                "commitment_scheme": "qiqcbench_sealed_target_sha256_v1",
                "target_inputs": challenge["target_inputs"],
                "target_inputs_sha256": challenge["target_inputs_sha256"],
                "target_commitment_sha256": challenge["target_commitment_sha256"],
                "commitment_nonce": challenge["commitment_nonce"],
            },
            "stale_lab_notebook": _STALE_NOTEBOOK,
            # The runtime hidden-model commitment (HMAC over this whole config)
            # therefore transitively binds the task edition and the scorer
            # policy revision, not just the executable circuit/noise.
            "edition_binding": {
                "instance_sha256": instance_digest(instance),
                "scorer_policy_sha256": scorer_policy_sha256(),
            },
        }
    )
    return public, hidden


def build_public_device_spec(instance: ShadowSurrogateInstance | None = None) -> dict:
    public, _ = boundedgate_configs_from_instance(instance or build_instance())
    return public.model_dump()


def build_hidden_device_config(instance: ShadowSurrogateInstance | None = None) -> dict:
    _, hidden = boundedgate_configs_from_instance(instance or build_instance())
    return hidden.model_dump(exclude_none=True)


def _yaml_bytes(payload: dict) -> bytes:
    yaml = YAML()
    yaml.default_flow_style = False
    yaml.width = 4096
    buf = io.BytesIO()
    yaml.dump(payload, buf)
    return buf.getvalue()


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_device_configs(
    devices_dir: Path, instance: ShadowSurrogateInstance | None = None
) -> dict[str, Path]:
    """Write the public + hidden device YAMLs for ``instance``.

    Passing ``instance`` is REQUIRED for a fresh edition: without it this
    rebuilds the pinned default, silently pairing an old circuit/noise with
    whatever materials the caller wrote elsewhere. Prefer ``emit_edition``,
    which materializes everything from one instance atomically.
    """
    devices_dir = Path(devices_dir)
    devices_dir.mkdir(parents=True, exist_ok=True)
    public_path = devices_dir / f"{DEVICE_ID}.public.yaml"
    hidden_path = devices_dir / f"{DEVICE_ID}.hidden.example.yaml"
    _write_atomic(public_path, _yaml_bytes(build_public_device_spec(instance)))
    _write_atomic(
        hidden_path,
        _DEVELOPMENT_FIXTURE_HEADER + _yaml_bytes(build_hidden_device_config(instance)),
    )
    return {"public": public_path, "hidden": hidden_path}


def _validate_certificate(
    certificate: object,
    digest: str,
    admission: AdmissionReport,
    *,
    expected_seed: int,
    signing_key: str | None = None,
) -> None:
    """Authenticate + re-validate the certificate's COMPLETE content.

    A certificate is evidence, not a flag: a hand-written
    ``{"passed": True, "instance_sha256": ...}`` must be rejected. Two layers:

    1. AUTHENTICATION: the HMAC signature (minted by ``certify_edition`` under
       the operator signing key) must verify under the key in force, so a
       fabricated-but-plausible stats payload minted outside a real
       certification run is rejected outright.
    2. CONTENT: every design seed's stats are re-checked against the current
       thresholds and policy-minimum margins, every declared empirical
       ablation's kill is re-derived from its stats, and every exact structural
       diagnostic must equal the target-independent admission evidence bound
       into the instance digest. The thresholds recorded in the certificate
       must equal the scorer policy in force.
    """
    import hmac as hmac_mod
    import math

    from qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q.construction import (
        CERT_ABLATION_DESIGN_SEED,
        CERT_ABLATION_EXPECTED_GATE,
        CERT_DESIGN_SEEDS,
        CERT_MIN_ACCURACY_MARGIN,
        CERT_MIN_TAIL_MARGIN,
        CERT_MIN_WORST_TARGET_MARGIN,
        CERT_STRUCTURAL_DIAGNOSTIC_POLICY,
        CERTIFICATE_SCHEMA_VERSION,
        REFERENCE_POLICY_ID,
        TASK_ID,
        certificate_expected_budget,
        certificate_signature,
        certificate_structural_diagnostics,
        load_cert_signing_key,
    )
    from qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q.scorer import (
        MAX_CELL_ERROR_THRESHOLD,
        RMSE_THRESHOLD,
        WORST_TARGET_RMSE_THRESHOLD,
        scorer_policy,
    )

    def fail(message: str) -> None:
        raise ValueError(f"invalid certificate: {message}")

    def finite_leq(value: object, bound: float) -> bool:
        return type(value) in (int, float) and math.isfinite(value) and value <= bound

    verification_key = load_cert_signing_key() if signing_key is None else signing_key
    if not isinstance(certificate, dict):
        fail("not a certificate object")
    signature = certificate.get("signature")
    if not isinstance(signature, str) or not hmac_mod.compare_digest(
        signature, certificate_signature(certificate, verification_key)
    ):
        fail("signature missing or does not verify under the signing key in force")
    expected_certificate_keys = {
        "certificate_schema_version",
        "task_id",
        "reference_policy_id",
        "ablation_design_seed",
        "instance_seed",
        "instance_sha256",
        "design_seed_stats",
        "empirical_ablation_stats",
        "structural_diagnostics",
        "margins",
        "thresholds",
        "passed",
        "signature",
    }
    if set(certificate) != expected_certificate_keys:
        fail("certificate fields do not match the schema exactly")
    if certificate.get("passed") is not True:
        fail("not a passing certificate")
    if certificate.get("certificate_schema_version") != CERTIFICATE_SCHEMA_VERSION:
        fail(
            "certificate_schema_version "
            f"{certificate.get('certificate_schema_version')!r} != "
            f"{CERTIFICATE_SCHEMA_VERSION}"
        )
    if certificate.get("task_id") != TASK_ID:
        fail(f"task_id {certificate.get('task_id')!r} != {TASK_ID!r}")
    if certificate.get("reference_policy_id") != REFERENCE_POLICY_ID:
        fail(
            "reference_policy_id "
            f"{certificate.get('reference_policy_id')!r} != {REFERENCE_POLICY_ID!r}"
        )
    if (
        type(certificate.get("ablation_design_seed")) is not int
        or certificate.get("ablation_design_seed") != CERT_ABLATION_DESIGN_SEED
    ):
        fail(
            "ablation_design_seed "
            f"{certificate.get('ablation_design_seed')!r} != "
            f"{CERT_ABLATION_DESIGN_SEED!r}"
        )
    if (
        type(certificate.get("instance_seed")) is not int
        or certificate.get("instance_seed") != expected_seed
    ):
        fail(f"instance_seed {certificate.get('instance_seed')!r} != expected {expected_seed!r}")
    if certificate.get("instance_sha256") != digest:
        fail(f"not bound to this instance ({certificate.get('instance_sha256')!r} != {digest!r})")
    if certificate.get("thresholds") != scorer_policy():
        fail("recorded thresholds do not match the scorer policy in force")

    margins = certificate.get("margins")
    expected_margins = {
        "accuracy": CERT_MIN_ACCURACY_MARGIN,
        "worst_target": CERT_MIN_WORST_TARGET_MARGIN,
        "tail": CERT_MIN_TAIL_MARGIN,
    }
    if not isinstance(margins, dict) or set(margins) != set(expected_margins):
        fail("margin fields do not match the schema exactly")
    for margin_name, minimum in expected_margins.items():
        value = margins.get(margin_name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
            fail(f"{margin_name} margin {value!r} below policy minimum")
    accuracy_margin = margins["accuracy"]
    worst_target_margin = margins["worst_target"]
    tail_margin = margins["tail"]

    stat_names = (
        "rmse",
        "worst_target_rmse",
        "max_cell_error",
    )
    diagnostic_stat_names = (
        "max_pair_rmse",
        "p99_pair_rmse",
        "nll",
        "nll_pair_p95",
        "nll_cell_p95",
        "coverage_1sigma",
        "coverage_2sigma",
        "sharpness_ratio",
        "validation_rmse",
    )
    expected_stat_names = (
        set(stat_names)
        | set(diagnostic_stat_names)
        | {
            "shots_used",
            "jobs_used",
        }
    )
    seed_stats = certificate.get("design_seed_stats")
    if not isinstance(seed_stats, dict) or set(seed_stats) != {str(s) for s in CERT_DESIGN_SEEDS}:
        fail(f"design seed coverage must be exactly {list(CERT_DESIGN_SEEDS)}")
    for design_seed, stats in seed_stats.items():
        if not isinstance(stats, dict):
            fail(f"design seed {design_seed}: stats missing")
        if set(stats) != expected_stat_names:
            fail(f"design seed {design_seed}: stat fields do not match the schema exactly")
        expected_shots, expected_jobs = certificate_expected_budget("global")
        for field_name, expected in (
            ("shots_used", expected_shots),
            ("jobs_used", expected_jobs),
        ):
            value = stats.get(field_name)
            if type(value) is not int or value != expected:
                fail(f"design seed {design_seed}: {field_name} {value!r} != expected {expected}")
        for name, bound in (
            ("rmse", RMSE_THRESHOLD / accuracy_margin),
            ("worst_target_rmse", WORST_TARGET_RMSE_THRESHOLD / worst_target_margin),
            ("max_cell_error", MAX_CELL_ERROR_THRESHOLD / tail_margin),
        ):
            if not finite_leq(stats.get(name), bound):
                fail(f"design seed {design_seed}: {name} {stats.get(name)!r} !<= {bound:.4f}")
        for name in diagnostic_stat_names:
            value = stats.get(name)
            if type(value) not in (int, float) or not math.isfinite(value):
                fail(f"design seed {design_seed}: {name} not finite")

    structural_diagnostics = certificate.get("structural_diagnostics")
    expected_structural_diagnostics = certificate_structural_diagnostics(admission)
    if not isinstance(structural_diagnostics, dict) or set(structural_diagnostics) != set(
        CERT_STRUCTURAL_DIAGNOSTIC_POLICY
    ):
        fail(
            "structural diagnostic coverage must be exactly "
            f"{sorted(CERT_STRUCTURAL_DIAGNOSTIC_POLICY)}"
        )
    if structural_diagnostics != expected_structural_diagnostics:
        fail("structural diagnostics do not match exact target-independent admission evidence")
    if not all(entry["passed"] is True for entry in structural_diagnostics.values()):
        fail("a structural diagnostic does not demonstrate its required obstruction")

    ablations = certificate.get("empirical_ablation_stats")
    if not isinstance(ablations, dict) or set(ablations) != set(CERT_ABLATION_EXPECTED_GATE):
        fail(f"empirical ablation coverage must be exactly {sorted(CERT_ABLATION_EXPECTED_GATE)}")
    # Each gate is the conjunction of its component checks; killed == any
    # component exceeded. Mirrors scorer.score_submission's gate composition.
    gate_components = {
        "g1_accuracy": (
            ("rmse", RMSE_THRESHOLD),
            ("worst_target_rmse", WORST_TARGET_RMSE_THRESHOLD),
        ),
        "g2_tail": (("max_cell_error", MAX_CELL_ERROR_THRESHOLD),),
    }
    for protocol, expected_gate in CERT_ABLATION_EXPECTED_GATE.items():
        entry = ablations[protocol]
        if not isinstance(entry, dict) or set(entry) != {"stats", "gates", "expected_gate"}:
            fail(f"ablation {protocol}: fields do not match the schema exactly")
        if entry.get("expected_gate") != expected_gate:
            fail(
                f"ablation {protocol}: expected_gate {entry.get('expected_gate')!r} "
                f"!= policy {expected_gate!r}"
            )
        stats = entry.get("stats") if isinstance(entry, dict) else None
        if not isinstance(stats, dict):
            fail(f"ablation {protocol}: stats missing")
        if set(stats) != expected_stat_names:
            fail(f"ablation {protocol}: stat fields do not match the schema exactly")
        expected_shots, expected_jobs = certificate_expected_budget(protocol)
        for field_name, expected in (
            ("shots_used", expected_shots),
            ("jobs_used", expected_jobs),
        ):
            value = stats.get(field_name)
            if type(value) is not int or value != expected:
                fail(f"ablation {protocol}: {field_name} {value!r} != expected {expected}")
        for name in stat_names:
            value = stats.get(name)
            if type(value) not in (int, float) or not math.isfinite(value):
                fail(f"ablation {protocol}: {name} not finite")
        for name in diagnostic_stat_names:
            value = stats.get(name)
            if type(value) not in (int, float) or not math.isfinite(value):
                fail(f"ablation {protocol}: {name} not finite")
        declared_gates = entry.get("gates")
        if (
            not isinstance(declared_gates, dict)
            or set(declared_gates) != set(gate_components)
            or any(type(value) is not bool for value in declared_gates.values())
        ):
            fail(f"ablation {protocol}: gate fields do not match the schema exactly")
        recomputed_gates = {
            gate_name: all(finite_leq(stats[name], bound) for name, bound in components)
            for gate_name, components in gate_components.items()
        }
        if declared_gates != recomputed_gates:
            fail(f"ablation {protocol}: gates do not match recomputed scorer policy")
        # Re-derive the declared kill from the stats, never trust stored flags.
        killed = not recomputed_gates[expected_gate]
        if not killed:
            fail(f"ablation {protocol}: declared kill gate {expected_gate} not demonstrated")


def emit_edition(
    materials_dir: Path,
    devices_dir: Path,
    *,
    certificate: dict,
    instance: ShadowSurrogateInstance | None = None,
    seed: int | None = None,
    study_proof: SuccessorStudyFixtureProof | None = None,
) -> dict[str, Path]:
    """Materialize ONE edition end-to-end, REBUILT from its declared seed.

    The sealed rule: what gets emitted is ``build_instance(seed)`` recomputed
    HERE, never caller-supplied content. A caller-supplied ``instance`` is
    only an expectation to compare against — its cached admission report,
    pair polynomials, labels, and seed field are all untrusted (verified
    attack: relabeling a stale public instance with a private-looking seed
    passed every previous check because nothing proved the content was
    derived from the declared seed). Requires a SIGNED
    ``reference.certify_edition`` certificate digest-bound to the rebuilt
    instance (fresh-shot multi-design-seed pass + every empirical ablation
    kill + exact target-independent structural diagnostics). Emits
    the public task materials, hidden truth, and public + hidden device YAMLs
    all from the rebuilt instance via temp-file+rename, then writes the
    binding manifest (``hidden/edition_manifest.json``, embedding the
    certificate) last, so a mixed edition (fresh labels + stale circuit, or
    vice versa) is provable rather than silent. Manifest-last consistency,
    not multi-file atomicity: candidate distribution must ship the whole
    directory as one staged bundle. For a scored private edition pass
    ``seed=new_private_seed()``.
    """
    import hashlib
    import json

    from qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q.construction import (
        TASK_ID,
        instance_digest,
        verify_instance_consistency,
        write_materials,
    )

    if instance is not None and seed is not None and seed != instance.seed:
        raise ValueError(f"seed {seed} does not match instance.seed {instance.seed}")
    root_seed = seed if seed is not None else (instance.seed if instance is not None else None)
    effective_seed = INSTANCE_SEED if root_seed is None else root_seed
    if effective_seed in RETIRED_DEVELOPMENT_INSTANCE_SEEDS:
        raise ValueError(f"refusing to emit retired development root {effective_seed}")
    # Cheap authentication + content checks FIRST (against the supplied
    # content's digest when there is one): a forged/weak certificate is
    # rejected before the expensive rebuild.
    expected_digest = instance_digest(instance) if instance is not None else None
    if expected_digest is not None:
        _validate_certificate(
            certificate,
            expected_digest,
            instance.admission,
            expected_seed=instance.seed,
        )

    # Proof preflight is still before build_instance: a missing, directly
    # constructed, nonpassing, or wrong-root proof cannot materialize targets.
    if study_proof is None:
        raise ValueError("successor edition emission requires a loaded passing study proof")
    from qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q.study import (
        preflight_fixture_emission_proof,
    )

    preflight_fixture_emission_proof(study_proof, expected_seed=effective_seed)

    # Rebuild from the seed: admission, circuit, noise, targets, and labels
    # are all recomputed — nothing cached in a supplied instance is trusted.
    rebuilt = build_instance(root_seed) if root_seed is not None else build_instance()
    if not rebuilt.admission.passed:  # pragma: no cover - build_instance already raises
        raise ValueError(f"refusing to emit an inadmissible instance: {rebuilt.admission.failures}")
    digest = instance_digest(rebuilt)
    if expected_digest is None:
        _validate_certificate(
            certificate,
            digest,
            rebuilt.admission,
            expected_seed=rebuilt.seed,
        )
    elif digest != expected_digest:
        raise ValueError(
            "supplied instance content is not build_instance(seed) of its declared "
            "seed: refusing to emit relabeled or stale content"
        )
    # The rebuilt labels were computed from the rebuilt circuit inside
    # build_instance, so digest equality above IS the physics consistency
    # proof for supplied content; the explicit recompute stays as a
    # last-line guard on the emitted object.
    verify_instance_consistency(rebuilt)

    # Every nonretired successor is released only after the exact fixture-0
    # root completes the precommitted 3/3 prospective study.  Render the five
    # deterministic pre-manifest roles in an isolated temporary directory and
    # validate them before touching either caller destination.  The final
    # manifest is deliberately excluded from the study outcome to avoid a
    # recursive digest; it binds the precommit/outcome digests below.
    from qiqcbench.eval.io.canonical import sha256_bytes
    from qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q.study import (
        PRE_MANIFEST_FILE_ROLES,
        build_pre_manifest_files,
        validate_fixture_emission_proof,
    )

    rendered = build_pre_manifest_files(rebuilt)
    pre_manifest_files_sha256 = {
        role: sha256_bytes(rendered[role]) for role in PRE_MANIFEST_FILE_ROLES
    }
    study_binding = validate_fixture_emission_proof(
        proof=study_proof,
        instance_seed=rebuilt.seed,
        certificate=certificate,
        pre_manifest_files_sha256=pre_manifest_files_sha256,
    )

    paths = dict(write_materials(Path(materials_dir), rebuilt))
    paths.update(write_device_configs(Path(devices_dir), rebuilt))

    # Carry the two safe, canonical public proof artifacts with the hidden
    # development bundle so bootstrap/verifier code can recompute the study
    # digests instead of trusting only the manifest's claimed hashes.  Private
    # root openings, candidate indices, and the master key are never copied.
    from qiqcbench.eval.io.canonical import canonical_json_bytes

    proof_artifact_paths = {
        "successor_study_precommit": (
            Path(materials_dir) / "hidden" / "successor_study_precommit.json",
            canonical_json_bytes(study_proof.precommit),
        ),
        "successor_study_outcome": (
            Path(materials_dir) / "hidden" / "successor_study_outcome.json",
            canonical_json_bytes(study_proof.outcome),
        ),
    }
    for role, (path, payload) in proof_artifact_paths.items():
        _write_atomic(path, payload)
        paths[role] = path

    actual_pre_manifest_paths = {
        "public_target_challenge": paths["target_challenge"],
        "public_observable_index": paths["observable_index"],
        "hidden_truth": paths["hidden_truth"],
        "public_device_spec": paths["public"],
        "hidden_device_config": paths["hidden"],
    }
    actual_pre_manifest_files_sha256 = {
        role: hashlib.sha256(actual_pre_manifest_paths[role].read_bytes()).hexdigest()
        for role in PRE_MANIFEST_FILE_ROLES
    }
    if actual_pre_manifest_files_sha256 != pre_manifest_files_sha256:
        raise RuntimeError("emitted pre-manifest files differ from the study-validated rendering")

    manifest = {
        "manifest_schema_version": 2,
        # Fixture 0 is intentionally publishable only as a development
        # fixture. Scored admission must reject this tier rather than
        # treating the public opening as scored evidence.
        "edition_tier": "development",
        "task_id": TASK_ID,
        "device_id": DEVICE_ID,
        "seed": rebuilt.seed,
        "derivation": {
            "edition_rng_scheme": "sha256(qiqcbench/<task>/v2/<domain>/<seed>)",
            "edition_rng_domains": [
                "circuit",
                "noise",
                "drift",
                "targets",
                "admission",
                "mps-admission",
                "shot",
            ],
            "challenge_id_scheme": "sha256(qiqcbench/<task>/challenge-id/v1/<seed>)",
            "challenge_nonce_scheme": (
                "sha256(qiqcbench/<task>/challenge-nonce/v1/<seed>) as full lowercase hex"
            ),
            "job_shot_scheme": (
                "sha256(qiqcbench/blackbox_boundedgate_circuit/v2/shot-job/<shot_root>/<salt>)"
            ),
        },
        "instance_sha256": digest,
        "certificate": certificate,
        "successor_study": study_binding.model_dump(mode="json"),
        "pre_manifest_files_sha256": pre_manifest_files_sha256,
        "files_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sorted(paths.items())
        },
    }
    manifest_path = Path(materials_dir) / "hidden" / "edition_manifest.json"
    _write_atomic(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    paths["edition_manifest"] = manifest_path
    return paths


def emit_official_edition(
    materials_dir: Path,
    devices_dir: Path,
    *,
    certificate: dict,
    study_proof: SuccessorStudyFixtureProof,
    root_slot_index: int,
    task_edition_id: str,
    instance_id: str,
) -> dict[str, Path]:
    """Materialize ONE owner-private edition from its study slot.

    The slot-to-tier assignment is frozen in ``study.OFFICIAL_SLOT_TIERS``
    (slot 1 -> candidate, slot 2 -> canonical); slot 0 is the public
    development fixture and is refused here.  The content rule is the same as
    ``emit_edition``: what gets emitted is ``build_instance(seed)`` recomputed
    HERE from the slot's precommitted private opening, and every rendered byte
    must equal the digests the study certified for that slot, so emission must
    run inside the study's exact runtime image.  Destinations are a private
    materialization tree; such editions are never committed to a public
    repository.
    """

    import hashlib
    import hmac
    import json
    import re

    from qiqcbench.eval.io.canonical import sha256_bytes
    from qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q.construction import (
        instance_digest,
        verify_instance_consistency,
        write_materials,
    )
    from qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q.study import (
        OFFICIAL_SLOT_TIERS,
        PRE_MANIFEST_FILE_ROLES,
        StudyError,
        build_pre_manifest_files,
        preflight_slot_emission_proof,
        validate_slot_emission_proof,
    )

    if root_slot_index not in OFFICIAL_SLOT_TIERS:
        raise ValueError("official emission draws only an owner-private study slot")
    edition_tier = OFFICIAL_SLOT_TIERS[root_slot_index]
    if re.fullmatch(r"task_edition_[0-9a-f]{64}", task_edition_id) is None:
        raise ValueError("official emission requires the exact catalog task_edition_id")
    if re.fullmatch(r"instance_[0-9a-f]{64}", instance_id) is None:
        raise ValueError("official emission requires the exact catalog instance_id")

    # Proof preflight is before build_instance: a missing, directly
    # constructed, nonpassing, or wrong-slot proof cannot materialize targets.
    try:
        opening = study_proof.private_plan.root_openings[root_slot_index]
    except (AttributeError, IndexError, TypeError) as exc:
        raise ValueError("official emission requires a loaded passing study proof") from exc
    root_seed = int(opening.root_seed_hex, 16)
    if root_seed in RETIRED_DEVELOPMENT_INSTANCE_SEEDS:
        raise ValueError(f"refusing to emit retired development root {root_seed}")
    preflight_slot_emission_proof(
        study_proof,
        root_slot_index=root_slot_index,
        expected_seed=root_seed,
    )

    # Rebuild from the seed: admission, circuit, noise, targets, and labels
    # are all recomputed — nothing outside the precommitted opening is trusted.
    rebuilt = build_instance(root_seed)
    if not rebuilt.admission.passed:  # pragma: no cover - build_instance already raises
        raise ValueError(f"refusing to emit an inadmissible instance: {rebuilt.admission.failures}")
    digest = instance_digest(rebuilt)
    _validate_certificate(
        certificate,
        digest,
        rebuilt.admission,
        expected_seed=rebuilt.seed,
    )
    verify_instance_consistency(rebuilt)

    rendered = build_pre_manifest_files(rebuilt)
    pre_manifest_files_sha256 = {
        role: sha256_bytes(rendered[role]) for role in PRE_MANIFEST_FILE_ROLES
    }
    try:
        study_binding = validate_slot_emission_proof(
            proof=study_proof,
            root_slot_index=root_slot_index,
            instance_seed=rebuilt.seed,
            certificate=certificate,
            pre_manifest_files_sha256=pre_manifest_files_sha256,
        )
    except StudyError as exc:
        raise ValueError(f"official emission proof validation failed: {exc}") from exc

    instance_source_revision_value = hmac.new(
        bytes.fromhex(study_proof.private_plan.master_key_hex),
        (
            f"qiqcbench/{TASK_ID}/official-instance-source-revision/v1/"
            f"{study_binding.study_id}/slot/{root_slot_index}"
        ).encode(),
        hashlib.sha256,
    ).hexdigest()

    paths = dict(write_materials(Path(materials_dir), rebuilt))
    paths.update(write_device_configs(Path(devices_dir), rebuilt))

    # Carry the two safe, canonical public proof artifacts with the official
    # bundle so the verifier can recompute the study digests instead of
    # trusting only the manifest's claimed hashes.  Private root openings,
    # candidate indices, and the master key are never copied.
    from qiqcbench.eval.io.canonical import canonical_json_bytes

    proof_artifact_paths = {
        "successor_study_precommit": (
            Path(materials_dir) / "hidden" / "successor_study_precommit.json",
            canonical_json_bytes(study_proof.precommit),
        ),
        "successor_study_outcome": (
            Path(materials_dir) / "hidden" / "successor_study_outcome.json",
            canonical_json_bytes(study_proof.outcome),
        ),
    }
    for role, (path, payload) in proof_artifact_paths.items():
        _write_atomic(path, payload)
        paths[role] = path

    actual_pre_manifest_paths = {
        "public_target_challenge": paths["target_challenge"],
        "public_observable_index": paths["observable_index"],
        "hidden_truth": paths["hidden_truth"],
        "public_device_spec": paths["public"],
        "hidden_device_config": paths["hidden"],
    }
    actual_pre_manifest_files_sha256 = {
        role: hashlib.sha256(actual_pre_manifest_paths[role].read_bytes()).hexdigest()
        for role in PRE_MANIFEST_FILE_ROLES
    }
    if actual_pre_manifest_files_sha256 != pre_manifest_files_sha256:
        raise RuntimeError("emitted pre-manifest files differ from the study-validated rendering")

    manifest = {
        "manifest_schema_version": 2,
        # An owner-private slot is admissible only at its frozen official
        # tier; development admission must reject it rather than treating a
        # sealed instance as a public fixture.
        "edition_tier": edition_tier,
        "task_id": TASK_ID,
        "device_id": DEVICE_ID,
        "task_edition_id": task_edition_id,
        "instance_id": instance_id,
        "instance_source_revision": {
            "scheme": "owner_hmac_sha256",
            "value": instance_source_revision_value,
        },
        "seed": rebuilt.seed,
        "derivation": {
            "edition_rng_scheme": "sha256(qiqcbench/<task>/v2/<domain>/<seed>)",
            "edition_rng_domains": [
                "circuit",
                "noise",
                "drift",
                "targets",
                "admission",
                "mps-admission",
                "shot",
            ],
            "challenge_id_scheme": "sha256(qiqcbench/<task>/challenge-id/v1/<seed>)",
            "challenge_nonce_scheme": (
                "sha256(qiqcbench/<task>/challenge-nonce/v1/<seed>) as full lowercase hex"
            ),
            "job_shot_scheme": (
                "sha256(qiqcbench/blackbox_boundedgate_circuit/v2/shot-job/<shot_root>/<salt>)"
            ),
        },
        "instance_sha256": digest,
        "certificate": certificate,
        "successor_study": study_binding.model_dump(mode="json"),
        "pre_manifest_files_sha256": pre_manifest_files_sha256,
        "files_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sorted(paths.items())
        },
    }
    manifest_path = Path(materials_dir) / "hidden" / "edition_manifest.json"
    _write_atomic(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    paths["edition_manifest"] = manifest_path
    return paths
