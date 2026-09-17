"""Public-class reference solver and certification for the stationary edition.

The reference knows ONLY the public contract: every scored pair correlator is
a trigonometric polynomial in six angles with per-angle harmonic degree at
most three and at most the disclosed number of complex Fourier modes.  It is a
solvability check for certification, never a performance
baseline, and it obtains the target coordinates only through the same sealed
reveal an agent uses.

Pipeline (policy ``orthogonal_fourier_omp_pilot_axis_v2``):

1. Pilot job: eight uniform inputs measured in the three global settings
   (500 shots each) to check which Pauli sectors carry input dependence.
2. Design: select the global axis with the largest pilot input variance, then
   measure 1,000 uniform inputs on the torus in that axis (303 shots each) in
   16 jobs of at most 64 blocks.  Together with the pilot this is exactly
   315,000 shots in 17 jobs.
3. Estimation: per pair, orthogonal matching pursuit over the orthogonal real
   Fourier basis with harmonics |f_k| <= 3 (117,649 modes, the same span as
   the normal-form class) with a noise-level stopping rule, then an OLS refit.
4. Internal diagnostic scale: sigma^2 = design variance + excess residual
   variance, floored at 0.001.  This diagnostic does not enter task scoring.

Predeclared weak baselines (each must fail its declared gate on its own shot
stream): even X/Y/Z allocation, single-pass Fourier thresholding, additive
(single-angle) model, nearest measured input, per-pair constant mean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q.construction import (
    CERT_ABLATION_DESIGN_SEED,
    CERT_ABLATION_EXPECTED_GATE,
    CERT_DESIGN_SEEDS,
    CERT_MIN_ACCURACY_MARGIN,
    CERT_MIN_TAIL_MARGIN,
    CERT_MIN_WORST_TARGET_MARGIN,
    CERTIFICATE_SCHEMA_VERSION,
    DEGREE_BOUND,
    INPUT_DIM,
    INSTANCE_SEED,
    MAX_BLOCKS_PER_JOB,
    N_QUBITS,
    N_TARGETS,
    REFERENCE_POLICY_ID,
    RETIRED_DEVELOPMENT_INSTANCE_SEEDS,
    TOTAL_SHOT_BUDGET,
    ShadowSurrogateInstance,
    build_instance,
    certificate_expected_budget,
    certificate_structural_diagnostics,
    pair_order,
)
from qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q.device_configs import (
    boundedgate_configs_from_instance,
)
from qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q.scorer import (
    MAX_CELL_ERROR_THRESHOLD,
    RMSE_THRESHOLD,
    WORST_TARGET_RMSE_THRESHOLD,
)
from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.backend import (
    BoundedgateSimulatorBackend,
)
from qiqcbench.qsim.qtypes.blackbox_boundedgate_circuit.wire import BasisShotRequest

__all__ = [
    "CertificateRejectedError",
    "CollectedData",
    "PipelineResult",
    "ablation_gates",
    "certify_edition",
    "collect",
    "evaluate",
    "fourier_design",
    "run_pipeline",
]

_PAIRS = pair_order()
_IU = np.triu_indices(N_QUBITS, 1)

# ---------- frozen design constants ----------
PILOT_INPUTS = 8
PILOT_SHOTS_PER_SETTING = 500
DESIGN_INPUTS = 1_000
PILOT_SHOTS = PILOT_INPUTS * 3 * PILOT_SHOTS_PER_SETTING  # 12,000
DESIGN_SHOTS_PER_INPUT = (TOTAL_SHOT_BUDGET - PILOT_SHOTS) // DESIGN_INPUTS  # 303
EVEN_XYZ_SHOTS_PER_SETTING = TOTAL_SHOT_BUDGET // (3 * DESIGN_INPUTS)  # 105
assert PILOT_SHOTS + DESIGN_INPUTS * DESIGN_SHOTS_PER_INPUT == TOTAL_SHOT_BUDGET
assert 3 * DESIGN_INPUTS * EVEN_XYZ_SHOTS_PER_SETTING == TOTAL_SHOT_BUDGET
OMP_MAX_TERMS = 120
OMP_STOP_Z = 5.0  # ~ sqrt(2 ln 117,649)
REFERENCE_SIGMA_FLOOR = 1e-3

_DRIFT_AWARE = frozenset({"global"})
_BASELINES = frozenset(CERT_ABLATION_EXPECTED_GATE)


class CertificateRejectedError(RuntimeError):
    """A committed edition failed fresh-shot reference certification (terminal)."""


# ---------- data collection through the real backend ----------


@dataclass
class CollectedData:
    inputs: np.ndarray  # (n, 6)
    c_hat: np.ndarray  # (n, 1770) estimated C_ij
    shots_per_input: int
    shots_used: int
    jobs_used: int
    pilot_axis_variance: np.ndarray | None = None  # (3,) input variance per sector
    selected_axis: str | None = None
    pilot_inputs: np.ndarray | None = None


def _bits_array(bitstrings: list[str]) -> np.ndarray:
    return np.frombuffer("".join(bitstrings).encode(), dtype=np.uint8).reshape(
        len(bitstrings), N_QUBITS
    ) - ord("0")


def _pair_means_from_bits(bits: np.ndarray) -> np.ndarray:
    z = 1.0 - 2.0 * bits.astype(float)
    return (z.T @ z / len(bits))[_IU]


def _submit(
    backend: BoundedgateSimulatorBackend,
    blocks: list[dict[str, Any]],
    *,
    job_counter: int,
    salt_base: int,
):
    request = BasisShotRequest.model_validate({"blocks": blocks})
    usage_before = backend.reserve_basis_shots(request)
    result = backend.run_basis_shots(
        request, f"ref_job_{job_counter}", salt=salt_base + job_counter, usage_before=usage_before
    )
    if result.status != "complete" or result.data is None:
        raise RuntimeError(f"reference job failed: {result.error}")
    return result


def measure_pilot_axis_variance(
    backend: BoundedgateSimulatorBackend,
    pilot_inputs: np.ndarray,
    *,
    salt_base: int = 0,
) -> np.ndarray:
    """Measure the pilot and return input variance for global X, Y, and Z."""
    if len(pilot_inputs) != PILOT_INPUTS:
        raise ValueError(f"pilot_inputs must contain exactly {PILOT_INPUTS} rows")
    settings = [
        {"basis": axis * N_QUBITS, "shots": PILOT_SHOTS_PER_SETTING} for axis in ("x", "y", "z")
    ]
    blocks = [{"x": [float(v) for v in x], "settings": settings} for x in pilot_inputs]
    result = _submit(backend, blocks, job_counter=1, salt_base=salt_base)
    axis_means = np.zeros((PILOT_INPUTS, 3, len(_PAIRS)))
    for block_index, block in enumerate(result.data.blocks):
        for axis_index, setting in enumerate(block.settings):
            axis_means[block_index, axis_index] = _pair_means_from_bits(
                _bits_array(setting.bitstrings)
            )
    return axis_means.var(axis=0).mean(axis=1)


def collect(
    backend: BoundedgateSimulatorBackend,
    inputs: np.ndarray,
    *,
    protocol: str,
    pilot_inputs: np.ndarray | None,
    salt_base: int = 0,
) -> CollectedData:
    """Drive the backend exactly like an agent: batched block jobs, raw bitstrings.

    ``protocol="global"`` spends the pilot in X/Y/Z and measures the design in
    whichever global axis has the largest input-dependent pilot variance;
    ``protocol="even_xyz"`` spends the whole budget evenly over X/Y/Z with no
    pilot and keeps only the Z third for estimation (the other two thirds are
    the price of not discovering the empty sectors).
    """
    job_counter = 0
    shots_used = 0
    n_inputs = len(inputs)
    c_hat = np.zeros((n_inputs, len(_PAIRS)))
    pilot_axis_variance = None
    selected_axis = None
    if protocol == "global":
        assert pilot_inputs is not None and len(pilot_inputs) == PILOT_INPUTS
        job_counter += 1
        pilot_axis_variance = measure_pilot_axis_variance(
            backend,
            pilot_inputs,
            salt_base=salt_base,
        )
        shots_used += PILOT_SHOTS
        selected_axis = ("x", "y", "z")[int(np.argmax(pilot_axis_variance))]
        per_input_settings = [{"basis": selected_axis * N_QUBITS, "shots": DESIGN_SHOTS_PER_INPUT}]
        shots_per_input = DESIGN_SHOTS_PER_INPUT
    elif protocol == "even_xyz":
        selected_axis = "z"
        per_input_settings = [
            {"basis": axis * N_QUBITS, "shots": EVEN_XYZ_SHOTS_PER_SETTING}
            for axis in ("x", "y", "z")
        ]
        shots_per_input = EVEN_XYZ_SHOTS_PER_SETTING
    else:
        raise ValueError(f"unknown collection protocol {protocol!r}")
    for start in range(0, n_inputs, MAX_BLOCKS_PER_JOB):
        idxs = list(range(start, min(start + MAX_BLOCKS_PER_JOB, n_inputs)))
        blocks = [
            {"x": [float(v) for v in inputs[idx]], "settings": per_input_settings} for idx in idxs
        ]
        job_counter += 1
        result = _submit(backend, blocks, job_counter=job_counter, salt_base=salt_base)
        for idx, block in zip(idxs, result.data.blocks, strict=True):
            selected_setting = next(s for s in block.settings if s.basis[0] == selected_axis)
            c_hat[idx] = _pair_means_from_bits(_bits_array(selected_setting.bitstrings)) / 3.0
            shots_used += sum(s.shots for s in block.settings)
    return CollectedData(
        inputs=inputs,
        c_hat=c_hat,
        shots_per_input=shots_per_input,
        shots_used=shots_used,
        jobs_used=job_counter,
        pilot_axis_variance=pilot_axis_variance,
        selected_axis=selected_axis,
        pilot_inputs=pilot_inputs,
    )


# ---------- orthogonal Fourier design and batched OMP ----------


def _angle_table(x: np.ndarray) -> np.ndarray:
    cols = [np.ones_like(x)]
    for h in range(1, DEGREE_BOUND + 1):
        cols.append(np.cos(h * x) * np.sqrt(2.0))
        cols.append(np.sin(h * x) * np.sqrt(2.0))
    return np.stack(cols, axis=1)


def fourier_design(xs: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Orthonormal (on the torus) real Fourier features, 7^6 columns."""
    xs = np.atleast_2d(np.asarray(xs, dtype=float))
    tables = [_angle_table(xs[:, k]) for k in range(INPUT_DIM)]
    cols = tables[0]
    for k in range(1, INPUT_DIM):
        cols = (cols[:, :, None] * tables[k][:, None, :]).reshape(len(xs), -1)
    return cols.astype(dtype)


def batched_omp(
    design: np.ndarray,
    targets: np.ndarray,
    sigma_noise: float,
    *,
    kmax: int = OMP_MAX_TERMS,
    z_stop: float = OMP_STOP_Z,
    chunk: int = 200,
) -> tuple[list[list[int]], list[np.ndarray], np.ndarray]:
    """OMP for every column of ``targets`` (n x m) over the shared design."""
    n, m = targets.shape
    scale = math.sqrt(n)
    supports: list[list[int]] = [[] for _ in range(m)]
    coefs: list[np.ndarray] = [np.zeros(0)] * m
    active = np.ones(m, dtype=bool)
    residual = targets.astype(np.float32).copy()
    for _ in range(kmax):
        idx_active = np.where(active)[0]
        if len(idx_active) == 0:
            break
        for c0 in range(0, len(idx_active), chunk):
            cols = idx_active[c0 : c0 + chunk]
            corr = (design.T @ residual[:, cols]) / scale
            for j, p in enumerate(cols):
                if supports[p]:
                    corr[supports[p], j] = 0.0
            best = np.argmax(np.abs(corr), axis=0)
            zval = np.abs(corr[best, np.arange(len(cols))]) / sigma_noise
            for j, p in enumerate(cols):
                if zval[j] < z_stop:
                    active[p] = False
                    continue
                supports[p].append(int(best[j]))
                columns = design[:, supports[p]].astype(np.float64)
                coef, *_ = np.linalg.lstsq(columns, targets[:, p], rcond=None)
                coefs[p] = coef
                residual[:, p] = (targets[:, p] - columns @ coef).astype(np.float32)
    return supports, coefs, residual.astype(np.float64)


# ---------- estimators ----------


@dataclass
class PipelineResult:
    predictions: np.ndarray  # (40, 1770)
    sigmas: np.ndarray  # (40, 1770)
    shots_used: int
    jobs_used: int
    validation_rmse: float
    protocol: str


def _noise_variance(c_hat: np.ndarray, shots_per_input: int) -> float:
    return float(np.mean((1.0 - (3.0 * c_hat) ** 2).clip(0.0, 1.0)) / (9.0 * shots_per_input))


def _predict_ols(
    design: np.ndarray,
    design_targets: np.ndarray,
    y: np.ndarray,
    support: list[int],
    noise_var: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS prediction on a support with design + excess-residual variance."""
    n = len(y)
    if not support:
        mu = np.full(len(design_targets), float(y.mean()))
        design_var = np.full(len(design_targets), noise_var / n)
        resid_var = float(np.var(y))
        return mu, np.sqrt(design_var + max(resid_var - noise_var, 0.0)), math.sqrt(resid_var)
    columns = design[:, support].astype(np.float64)
    coef, *_ = np.linalg.lstsq(columns, y, rcond=None)
    target_columns = design_targets[:, support].astype(np.float64)
    mu = target_columns @ coef
    gram_inv = np.linalg.pinv(columns.T @ columns)
    design_var = np.einsum("ij,jk,ik->i", target_columns, gram_inv, target_columns) * noise_var
    resid = y - columns @ coef
    resid_var = float(np.sum(resid**2) / max(n - len(support), 1))
    sigma = np.sqrt(design_var + max(resid_var - noise_var, 0.0))
    return mu, sigma, math.sqrt(resid_var)


def fit_fourier_omp(data: CollectedData, targets: np.ndarray) -> PipelineResult:
    noise_var = _noise_variance(data.c_hat, data.shots_per_input)
    design = fourier_design(data.inputs)
    design_targets = fourier_design(targets, dtype=np.float64)
    supports, _coefs, _resid = batched_omp(design, data.c_hat, math.sqrt(noise_var))
    predictions = np.zeros((len(targets), len(_PAIRS)))
    sigmas = np.zeros_like(predictions)
    cv = np.zeros(len(_PAIRS))
    for p in range(len(_PAIRS)):
        mu, sigma, resid_rms = _predict_ols(
            design, design_targets, data.c_hat[:, p], supports[p], noise_var
        )
        predictions[:, p] = mu
        sigmas[:, p] = np.maximum(sigma, REFERENCE_SIGMA_FLOOR)
        cv[p] = resid_rms
    return PipelineResult(
        predictions=predictions,
        sigmas=sigmas,
        shots_used=data.shots_used,
        jobs_used=data.jobs_used,
        validation_rmse=float(np.sqrt(np.mean(cv**2))),
        protocol="global",
    )


def fit_fourier_single(data: CollectedData, targets: np.ndarray) -> PipelineResult:
    """Weak baseline: one projection pass with hard thresholding, no iteration."""
    noise_var = _noise_variance(data.c_hat, data.shots_per_input)
    n = len(data.inputs)
    design = fourier_design(data.inputs)
    design_targets = fourier_design(targets, dtype=np.float64)
    proj = (design.T @ data.c_hat) / n
    tau = OMP_STOP_Z * math.sqrt(noise_var / n)
    predictions = np.zeros((len(targets), len(_PAIRS)))
    sigmas = np.zeros_like(predictions)
    cv = np.zeros(len(_PAIRS))
    for p in range(len(_PAIRS)):
        support = np.where(np.abs(proj[:, p]) > tau)[0]
        if len(support) > n // 3:
            support = support[np.argsort(-np.abs(proj[support, p]))[: n // 3]]
        support = sorted(set(support.tolist()) | {0})
        mu, sigma, resid_rms = _predict_ols(
            design, design_targets, data.c_hat[:, p], support, noise_var
        )
        predictions[:, p] = mu
        sigmas[:, p] = np.maximum(sigma, REFERENCE_SIGMA_FLOOR)
        cv[p] = resid_rms
    return PipelineResult(
        predictions,
        sigmas,
        data.shots_used,
        data.jobs_used,
        float(np.sqrt(np.mean(cv**2))),
        "fourier_single",
    )


def _additive_design(xs: np.ndarray) -> np.ndarray:
    cols = [np.ones(len(xs))]
    for k in range(INPUT_DIM):
        for h in range(1, DEGREE_BOUND + 1):
            cols.append(np.cos(h * xs[:, k]))
            cols.append(np.sin(h * xs[:, k]))
    return np.stack(cols, axis=1)


def fit_additive(data: CollectedData, targets: np.ndarray) -> PipelineResult:
    """Weak baseline: single-angle harmonics only (no cross-angle terms)."""
    noise_var = _noise_variance(data.c_hat, data.shots_per_input)
    design = _additive_design(data.inputs)
    design_targets = _additive_design(targets)
    support = list(range(design.shape[1]))
    predictions = np.zeros((len(targets), len(_PAIRS)))
    sigmas = np.zeros_like(predictions)
    cv = np.zeros(len(_PAIRS))
    for p in range(len(_PAIRS)):
        mu, sigma, resid_rms = _predict_ols(
            design, design_targets, data.c_hat[:, p], support, noise_var
        )
        predictions[:, p] = mu
        sigmas[:, p] = np.maximum(sigma, REFERENCE_SIGMA_FLOOR)
        cv[p] = resid_rms
    return PipelineResult(
        predictions,
        sigmas,
        data.shots_used,
        data.jobs_used,
        float(np.sqrt(np.mean(cv**2))),
        "additive",
    )


def fit_nearest_input(data: CollectedData, targets: np.ndarray) -> PipelineResult:
    delta = (targets[:, None, :] - data.inputs[None, :, :] + np.pi) % (2 * np.pi) - np.pi
    nearest = np.argmin(np.sum(delta**2, axis=2), axis=1)
    predictions = data.c_hat[nearest]
    sigma = max(math.sqrt(_noise_variance(data.c_hat, data.shots_per_input)), REFERENCE_SIGMA_FLOOR)
    return PipelineResult(
        predictions,
        np.full_like(predictions, sigma),
        data.shots_used,
        data.jobs_used,
        sigma,
        "nearest_input",
    )


def fit_constant_mean(data: CollectedData, targets: np.ndarray) -> PipelineResult:
    mean = data.c_hat.mean(axis=0)
    std = np.maximum(data.c_hat.std(axis=0), REFERENCE_SIGMA_FLOOR)
    predictions = np.tile(mean, (len(targets), 1))
    return PipelineResult(
        predictions,
        np.tile(std, (len(targets), 1)),
        data.shots_used,
        data.jobs_used,
        float(np.sqrt(np.mean(std**2))),
        "constant_mean",
    )


# ---------- pipeline ----------


def build_reference_design(design_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Target-blind design: uniform pilot inputs and uniform design inputs."""
    rng = np.random.default_rng(design_seed)
    pilot = rng.uniform(-np.pi, np.pi, size=(PILOT_INPUTS, INPUT_DIM))
    design = rng.uniform(-np.pi, np.pi, size=(DESIGN_INPUTS, INPUT_DIM))
    return pilot, design


def run_pipeline(
    instance: ShadowSurrogateInstance,
    *,
    protocol: str = "global",
    design_seed: int,
    shot_salt_base: int = 0,
) -> PipelineResult:
    """Collect data through the real backend, seal, reveal, then predict."""
    if protocol not in _DRIFT_AWARE | _BASELINES:
        raise ValueError(f"unknown protocol {protocol!r}")
    public, hidden = boundedgate_configs_from_instance(instance)
    # Certification replays exact streams: inject the edition's derived shot
    # seed as the attempt entropy.  Production qsim backends draw fresh
    # private entropy per attempt instead.
    backend = BoundedgateSimulatorBackend(hidden, public, run_entropy=hidden.seed)
    pilot, design = build_reference_design(design_seed)
    collection_protocol = "even_xyz" if protocol == "even_xyz" else "global"
    data = collect(
        backend,
        design,
        protocol=collection_protocol,
        pilot_inputs=None if collection_protocol == "even_xyz" else pilot,
        salt_base=shot_salt_base,
    )
    # The reference never sees a scored coordinate before the seal.
    reveal = backend.lock_measurements_and_reveal_challenge()
    targets = np.asarray(reveal.target_inputs, dtype=float)
    if targets.shape != (N_TARGETS, INPUT_DIM):
        raise RuntimeError("reveal returned an unexpected target matrix")
    measured = np.concatenate([design, pilot]) if data.pilot_inputs is not None else design
    if any(np.any(np.all(np.isclose(measured, t, atol=0.0), axis=1)) for t in targets):
        raise RuntimeError("reference design collided with a revealed target coordinate")
    if protocol in ("global", "even_xyz"):
        result = fit_fourier_omp(data, targets)
        result.protocol = protocol
    elif protocol == "fourier_single":
        result = fit_fourier_single(data, targets)
    elif protocol == "additive":
        result = fit_additive(data, targets)
    elif protocol == "nearest_input":
        result = fit_nearest_input(data, targets)
    else:
        result = fit_constant_mean(data, targets)
    return result


# ---------- evaluation and certification ----------


def evaluate(result: PipelineResult, labels: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=float)
    resid = result.predictions - labels
    sigmas = result.sigmas
    pair_rmse = np.sqrt(np.mean(resid**2, axis=0))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        cell_nll = resid**2 / (2.0 * sigmas**2) + np.log(sigmas)
    rmse = float(np.sqrt(np.mean(resid**2)))
    return {
        "rmse": rmse,
        "worst_target_rmse": float(np.max(np.sqrt(np.mean(resid**2, axis=1)))),
        "max_pair_rmse": float(pair_rmse.max()),
        "p99_pair_rmse": float(np.quantile(pair_rmse, 0.99, method="linear")),
        "max_cell_error": float(np.max(np.abs(resid))),
        "nll": float(np.mean(cell_nll)),
        "nll_pair_p95": float(np.quantile(np.mean(cell_nll, axis=0), 0.95, method="linear")),
        "nll_cell_p95": float(np.quantile(cell_nll, 0.95, method="linear")),
        "coverage_1sigma": float(np.mean(np.abs(resid) <= sigmas)),
        "coverage_2sigma": float(np.mean(np.abs(resid) <= 2.0 * sigmas)),
        "sharpness_ratio": float(np.sqrt(np.mean(sigmas**2)) / max(rmse, 1e-12)),
        "validation_rmse": float(result.validation_rmse),
        "shots_used": int(result.shots_used),
        "jobs_used": int(result.jobs_used),
    }


def _finite_leq(value: float, bound: float) -> bool:
    return math.isfinite(value) and value <= bound


def ablation_gates(stats: dict[str, Any]) -> dict[str, bool]:
    """Mirror of ``scorer.score_submission``'s scientific gate composition."""
    return {
        "g1_accuracy": _finite_leq(stats["rmse"], RMSE_THRESHOLD)
        and _finite_leq(stats["worst_target_rmse"], WORST_TARGET_RMSE_THRESHOLD),
        "g2_tail": _finite_leq(stats["max_cell_error"], MAX_CELL_ERROR_THRESHOLD),
    }


def _certificate_budget_failures(protocol: str, stats: dict[str, Any]) -> list[str]:
    expected_shots, expected_jobs = certificate_expected_budget(protocol)
    failures = []
    if stats["shots_used"] != expected_shots:
        failures.append(f"{protocol}: shots_used {stats['shots_used']} != {expected_shots}")
    if stats["jobs_used"] != expected_jobs:
        failures.append(f"{protocol}: jobs_used {stats['jobs_used']} != {expected_jobs}")
    return failures


def certify_edition(
    instance: ShadowSurrogateInstance,
    *,
    accuracy_margin: float = CERT_MIN_ACCURACY_MARGIN,
    worst_target_margin: float = CERT_MIN_WORST_TARGET_MARGIN,
    tail_margin: float = CERT_MIN_TAIL_MARGIN,
) -> dict[str, Any]:
    """Fresh-shot certification for a committed edition.

    Runs the reference at every design seed in ``CERT_DESIGN_SEEDS`` on an
    independent shot stream (each must clear every gate with margin; NaN
    fails closed) and every predeclared weak baseline on the separate
    ablation stream (finite statistics; each must fail its declared gate).
    Structural diagnostics come from target-independent admission.  Once this
    function evaluates realized labels, a failure is terminal for the
    committed edition.  Returns a certificate HMAC-signed under the operator
    key; ``device_configs.emit_edition`` re-validates signature and content.
    """
    from qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q.construction import (
        TASK_ID,
        certificate_signature,
        instance_digest,
        load_cert_signing_key,
    )
    from qiqcbench.qsim.hidden_dynamics.time_budgeted_shadow_surrogate_60q.scorer import (
        scorer_policy,
    )

    if instance.seed in RETIRED_DEVELOPMENT_INSTANCE_SEEDS:
        raise CertificateRejectedError(
            f"edition seed {instance.seed} is retired and cannot be certified"
        )
    if (
        accuracy_margin < CERT_MIN_ACCURACY_MARGIN
        or worst_target_margin < CERT_MIN_WORST_TARGET_MARGIN
        or tail_margin < CERT_MIN_TAIL_MARGIN
    ):
        raise ValueError(
            "certification margins below policy minimums "
            f"(accuracy >= {CERT_MIN_ACCURACY_MARGIN}, worst_target >= "
            f"{CERT_MIN_WORST_TARGET_MARGIN}, tail >= {CERT_MIN_TAIL_MARGIN})"
        )
    signing_key = load_cert_signing_key()  # fail BEFORE the expensive runs

    # Fail fast: a structural failure rejects before any measurement stream is
    # consumed, and the first failing global stream rejects before later
    # streams are spent (burned streams are terminal evidence).
    structural_diagnostics = certificate_structural_diagnostics(instance.admission)
    for name, diagnostic in structural_diagnostics.items():
        if diagnostic["passed"] is not True:
            raise CertificateRejectedError(
                f"edition seed {instance.seed} fails reference certification: structural "
                f"diagnostic {name} does not demonstrate its obstruction"
            )
    labels = instance.labels
    per_seed: dict[str, dict[str, Any]] = {}
    bounds = (
        ("rmse", RMSE_THRESHOLD / accuracy_margin),
        ("worst_target_rmse", WORST_TARGET_RMSE_THRESHOLD / worst_target_margin),
        ("max_cell_error", MAX_CELL_ERROR_THRESHOLD / tail_margin),
    )
    for run_index, design_seed in enumerate(CERT_DESIGN_SEEDS):
        result = run_pipeline(
            instance, design_seed=design_seed, shot_salt_base=10_000 * (run_index + 1)
        )
        stats = evaluate(result, labels)
        failures = _certificate_budget_failures("global", stats)
        for name, bound in bounds:
            if not _finite_leq(stats[name], bound):
                failures.append(
                    f"design seed {design_seed}: {name} {stats[name]:.5f} !<= {bound:.5f}"
                )
        if failures:
            raise CertificateRejectedError(
                f"edition seed {instance.seed} fails reference certification: "
                + "; ".join(failures)
            )
        per_seed[str(design_seed)] = stats

    empirical_ablation_stats: dict[str, dict[str, Any]] = {}
    for ablation_index, protocol in enumerate(sorted(CERT_ABLATION_EXPECTED_GATE)):
        expected_gate = CERT_ABLATION_EXPECTED_GATE[protocol]
        result = run_pipeline(
            instance,
            protocol=protocol,
            design_seed=CERT_ABLATION_DESIGN_SEED,
            shot_salt_base=100_000 * (ablation_index + 1),
        )
        stats = evaluate(result, labels)
        budget_failures = _certificate_budget_failures(protocol, stats)
        if budget_failures:
            raise CertificateRejectedError(
                f"edition seed {instance.seed} fails reference certification: "
                + "; ".join(budget_failures)
            )
        if not all(
            math.isfinite(stats[k]) for k in ("rmse", "worst_target_rmse", "max_cell_error", "nll")
        ):
            raise CertificateRejectedError(
                f"edition seed {instance.seed} fails reference certification: "
                f"ablation {protocol}: non-finite stats (kill unverifiable)"
            )
        gates = ablation_gates(stats)
        if gates[expected_gate]:
            raise CertificateRejectedError(
                f"edition seed {instance.seed} fails reference certification: "
                f"ablation {protocol} passes its declared kill gate {expected_gate}"
            )
        empirical_ablation_stats[protocol] = {
            "stats": stats,
            "gates": gates,
            "expected_gate": expected_gate,
        }

    certificate = {
        "certificate_schema_version": CERTIFICATE_SCHEMA_VERSION,
        "reference_policy_id": REFERENCE_POLICY_ID,
        "ablation_design_seed": CERT_ABLATION_DESIGN_SEED,
        "task_id": TASK_ID,
        "instance_seed": instance.seed,
        "instance_sha256": instance_digest(instance),
        "design_seed_stats": per_seed,
        "empirical_ablation_stats": empirical_ablation_stats,
        "structural_diagnostics": structural_diagnostics,
        "margins": {
            "accuracy": accuracy_margin,
            "worst_target": worst_target_margin,
            "tail": tail_margin,
        },
        "thresholds": scorer_policy(),
        "passed": True,
    }
    certificate["signature"] = certificate_signature(certificate, signing_key)
    return certificate


def _print_stats(label: str, stats: dict[str, float]) -> None:
    print(
        f"{label:>22}: rmse={stats['rmse']:.4f} wt_rmse={stats['worst_target_rmse']:.4f} "
        f"max_pair={stats['max_pair_rmse']:.4f} max_cell={stats['max_cell_error']:.4f} "
        f"nll={stats['nll']:.3f} coverage={stats['coverage_1sigma']:.3f} "
        f"sharp={stats['sharpness_ratio']:.2f} shots={stats['shots_used']} "
        f"jobs={stats['jobs_used']} val_rmse={stats['validation_rmse']:.4f}"
    )


def main(seed: int | None = None) -> None:
    root = INSTANCE_SEED if seed is None else seed
    if root in RETIRED_DEVELOPMENT_INSTANCE_SEEDS:
        raise SystemExit(f"refusing to run the reference panel for retired development root {root}")
    instance = build_instance(root)
    labels = instance.labels
    print(f"instance seed {instance.seed}: admission {instance.admission}")
    for run_index, design_seed in enumerate(CERT_DESIGN_SEEDS):
        result = run_pipeline(
            instance, design_seed=design_seed, shot_salt_base=10_000 * (run_index + 1)
        )
        _print_stats(f"global[{design_seed}]", evaluate(result, labels))
    for ablation_index, protocol in enumerate(sorted(CERT_ABLATION_EXPECTED_GATE)):
        result = run_pipeline(
            instance,
            protocol=protocol,
            design_seed=CERT_ABLATION_DESIGN_SEED,
            shot_salt_base=100_000 * (ablation_index + 1),
        )
        _print_stats(protocol, evaluate(result, labels))


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
