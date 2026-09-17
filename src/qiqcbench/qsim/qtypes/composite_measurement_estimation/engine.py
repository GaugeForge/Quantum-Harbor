"""Run-long stateful engine for ``composite_measurement_estimation``.

Holds the hidden 14-qubit statevector and three run-long meters: the pilot budget
(accumulates across ``run_pilot_measurements`` calls), the evaluator call counter
(``evaluate_composite_scheme``), and the one-shot production guard
(``execute_locked_composite_scheme`` succeeds at most once). Sampling rotates the
statevector into each requested product-Pauli basis and draws multinomial raw
counts. Pilot outcomes, production bases, and production outcomes use separate
injected generators and are never reseeded.
"""

from __future__ import annotations

import threading

import numpy as np

from qiqcbench.qsim.hidden_dynamics.adaptive_clustered_clbcs_h2o import construction as C
from qiqcbench.qsim.qtypes.composite_measurement_estimation.device import PublicCmeSpec

_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_SDG = np.array([[1, 0], [0, -1j]], dtype=complex)
_ROT = {"Z": np.eye(2, dtype=complex), "X": _H, "Y": _H @ _SDG}
_AXES = ("X", "Y", "Z")


class CompositeMeasurementError(ValueError):
    """Lock-time / validation failure surfaced to the agent (generic, no diagnostics)."""


class CompositeMeasurementEngine:
    def __init__(
        self,
        hidden,
        public: PublicCmeSpec,
        *,
        pilot_rng: np.random.Generator,
        production_basis_rng: np.random.Generator,
        production_outcome_rng: np.random.Generator,
    ):
        self._pilot_rng = pilot_rng
        self._production_basis_rng = production_basis_rng
        self._production_outcome_rng = production_outcome_rng
        self._public = public
        self._lock = threading.Lock()
        self._nq = public.n_qubits
        self._state = C.build_hidden_state()  # cached 2^14 statevector
        self._terms, self._coeffs, self._term_indices, _ = C.load_hamiltonian(public.task_id)

        # run-long meters
        self._pilot_settings_used = 0
        self._pilot_shots_used = 0
        self._evaluate_calls = 0
        self._lock_attempts = 0
        self._production_executed = False

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #
    def _sample_counts(
        self, basis: str, shots: int, *, rng: np.random.Generator
    ) -> dict[str, int]:
        t = self._state.reshape([2] * self._nq)
        for q in range(self._nq):
            u = _ROT[basis[q]]
            if basis[q] != "Z":
                t = np.moveaxis(np.tensordot(u, t, axes=([1], [q])), 0, q)
        probs = np.abs(t.reshape(-1)) ** 2
        probs = probs / probs.sum()
        draws = rng.choice(probs.size, size=shots, p=probs)
        vals, cnts = np.unique(draws, return_counts=True)
        return {f"{int(v):0{self._nq}b}": int(c) for v, c in zip(vals, cnts, strict=True)}

    def _sample_basis_from_scheme(self, weights: np.ndarray, beta: np.ndarray) -> tuple[str, int]:
        k = int(self._production_basis_rng.choice(len(weights), p=weights))
        chars = []
        for i in range(self._nq):
            b = beta[k, i]
            chars.append(_AXES[int(self._production_basis_rng.choice(3, p=b))])
        return "".join(chars), k

    # ------------------------------------------------------------------ #
    # Synchronous: evaluate_composite_scheme (call-capped, public computation)
    # ------------------------------------------------------------------ #
    def evaluate_scheme(self, weights: list, beta: list) -> dict:
        with self._lock:
            if self._evaluate_calls >= self._public.evaluator_call_cap:
                return {
                    "accepted": False,
                    "reason": "evaluator_call_cap_exhausted",
                    "evaluator_calls_used": self._evaluate_calls,
                    "evaluator_call_cap": self._public.evaluator_call_cap,
                }
            self._evaluate_calls += 1
            calls_used = self._evaluate_calls
        try:
            w, b = C.normalize_scheme(
                np.asarray(weights, dtype=float),
                np.asarray(beta, dtype=float),
                num_components=self._public.num_components,
            )
        except (ValueError, TypeError) as exc:
            return {
                "accepted": False,
                "reason": f"invalid_scheme: {exc}",
                "evaluator_calls_used": calls_used,
                "evaluator_call_cap": self._public.evaluator_call_cap,
            }
        h, hmin = C.coverage_floor(self._terms, w, b)
        valid = hmin >= self._public.coverage_floor
        result = {
            "accepted": True,
            "valid": valid,
            "min_coverage_probability": float(hmin),
            "coverage_quantiles": {
                "p01": float(np.quantile(h, 0.01)),
                "p10": float(np.quantile(h, 0.10)),
                "p50": float(np.quantile(h, 0.50)),
            },
            "evaluator_calls_used": calls_used,
            "evaluator_calls_remaining": self._public.evaluator_call_cap - calls_used,
        }
        if valid:
            result["average_one_shot_variance"] = C.track_a_variance(
                self._terms, self._coeffs, w, b
            )
        return result

    # ------------------------------------------------------------------ #
    # Async: run_pilot_measurements
    # ------------------------------------------------------------------ #
    def _validate_basis(self, basis: str) -> str | None:
        if len(basis) != self._nq or any(ch not in _AXES for ch in basis):
            return f"basis must be {self._nq} chars over X/Y/Z"
        return None

    def _validate_shots(self, shots: int) -> str | None:
        p = self._public
        if shots < p.pilot_shots_per_setting_min or shots > p.pilot_shots_per_setting_max:
            return f"shots must be in [{p.pilot_shots_per_setting_min}, {p.pilot_shots_per_setting_max}]"
        if shots % p.pilot_shots_multiple_of != 0:
            return f"shots must be a multiple of {p.pilot_shots_multiple_of}"
        return None

    def run_pilot_batch(self, rows: list[tuple[str, int]]) -> dict:
        out_rows = []
        with self._lock:
            p = self._public
            for basis, shots in rows:
                reason = self._validate_basis(basis) or self._validate_shots(shots)
                if reason is None:
                    if self._pilot_settings_used + 1 > p.pilot_max_settings:
                        reason = "pilot_settings_budget_exhausted"
                    elif self._pilot_shots_used + shots > p.pilot_max_shots:
                        reason = "pilot_shots_budget_exhausted"
                if reason is not None:
                    out_rows.append(
                        {
                            "basis": basis,
                            "shots": shots,
                            "status": "rejected",
                            "reject_reason": reason,
                        }
                    )
                    continue
                counts = self._sample_counts(basis, shots, rng=self._pilot_rng)
                self._pilot_settings_used += 1
                self._pilot_shots_used += shots
                out_rows.append(
                    {"basis": basis, "shots": shots, "status": "accepted", "counts": counts}
                )
            budget = {
                "pilot_settings_used": self._pilot_settings_used,
                "pilot_settings_cap": p.pilot_max_settings,
                "pilot_shots_used": self._pilot_shots_used,
                "pilot_shots_cap": p.pilot_max_shots,
            }
        return {"rows": out_rows, "budget": budget}

    # ------------------------------------------------------------------ #
    # Async: execute_locked_composite_scheme (one-shot)
    # ------------------------------------------------------------------ #
    def run_locked_production(self, weights: list, beta: list, control_entries: list[dict]) -> dict:
        with self._lock:
            if self._production_executed:
                raise CompositeMeasurementError(
                    "production already executed once; only one successful run is allowed"
                )
            if self._lock_attempts >= self._public.lock_attempt_cap:
                raise CompositeMeasurementError("lock-attempt budget exhausted")
            try:
                raw_weights = np.asarray(weights, dtype=float)
                raw_beta = np.asarray(beta, dtype=float)
                C.validate_scheme(
                    raw_weights,
                    raw_beta,
                    num_components=self._public.num_components,
                )
                valid_indices = set(self._term_indices)
                seen: set[int] = set()
                for e in control_entries:
                    ti = int(e["term_index"])
                    if ti == 0 or ti not in valid_indices or ti in seen:
                        raise ValueError("invalid control-variate entry")
                    mean = float(e["mean"])
                    if not np.isfinite(mean) or abs(mean) > 1.0:
                        raise ValueError("control-variate mean out of [-1, 1]")
                    seen.add(ti)
                if len(control_entries) > self._public.max_control_variates:
                    raise ValueError("too many control variates")

                canonical = C.canonical_request(
                    raw_weights,
                    raw_beta,
                    control_entries,
                    self._public.task_id,
                    C.hamiltonian_sha256(self._public.task_id),
                )
                w, b, locked_controls = C.locked_request_from_canonical(
                    canonical, num_components=self._public.num_components
                )
                _, hmin = C.coverage_floor(self._terms, w, b)
                if hmin < self._public.coverage_floor:
                    raise ValueError("coverage floor violated")
            except (ValueError, TypeError, KeyError):
                self._lock_attempts += 1
                raise CompositeMeasurementError(
                    "invalid scheme or control variates (lock rejected)"
                ) from None

            # Sample the canonical digest-bearing law (still under lock: one-shot).
            digest = C.request_digest(canonical)
            rows = []
            for _ in range(self._public.production_settings):
                basis, k = self._sample_basis_from_scheme(w, b)
                counts = self._sample_counts(
                    basis,
                    self._public.production_shots_per_setting,
                    rng=self._production_outcome_rng,
                )
                rows.append({"basis": basis, "component_id": k, "counts": counts})
            self._production_executed = True

        return {
            "rows": rows,
            "request_digest": digest,
            "canonical": canonical,
            "locked_weights": w.tolist(),
            "locked_beta": b.tolist(),
            "locked_control_entries": locked_controls,
            "production_settings": self._public.production_settings,
            "production_shots_per_setting": self._public.production_shots_per_setting,
        }
