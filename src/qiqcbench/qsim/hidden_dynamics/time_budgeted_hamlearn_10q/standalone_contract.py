"""Dependency-light contracts shared with the standalone HamLearn verifier.

This module intentionally imports only the standard library.  The verifier
image stages these exact bytes as ``hamlearn_contract.py`` so the in-package
Pydantic model and the isolated scorer cannot drift into accepting different
hidden-scorer documents.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

HIDDEN_SCORER_SCHEMA_VERSION = 3
PUBLIC_MATERIAL_FILENAMES = frozenset(
    {
        "hamiltonian_dictionary.json",
        "public_spec.yaml",
        "stale_notebook.yaml",
    }
)

_FIELDS = frozenset(
    {
        "schema_version",
        "public_material_digests",
        "omega_star",
        "linf_pass",
        "linf_bronze",
        "linf_silver",
        "linf_gold",
        "relative_l2_pass",
        "support_tau",
        "support_terms",
        "budget_us",
        "max_probe_rows",
        "oracle_noise_shots",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _real(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def validate_hidden_scorer_payload(
    raw: Any,
    *,
    term_order: Sequence[str],
    coefficient_bound: float,
) -> dict[str, Any]:
    """Validate and canonicalize the complete hidden scorer document.

    No field is optional.  Defaults belong in the instance constructor, not at
    the trust boundary where a truncated file must fail closed.
    """

    if not isinstance(raw, Mapping):
        raise ValueError("hidden scorer must be a mapping")
    keys = set(raw)
    missing = sorted(_FIELDS - keys)
    unknown = sorted(keys - _FIELDS)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing fields: {missing}")
        if unknown:
            details.append(f"unknown fields: {unknown}")
        raise ValueError("; ".join(details))
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 3:
        raise ValueError(
            f"schema_version must be {HIDDEN_SCORER_SCHEMA_VERSION}, got {raw['schema_version']!r}"
        )

    digests = raw["public_material_digests"]
    if not isinstance(digests, Mapping) or set(digests) != PUBLIC_MATERIAL_FILENAMES:
        raise ValueError(
            f"public_material_digests must name exactly {sorted(PUBLIC_MATERIAL_FILENAMES)}"
        )
    canonical_digests: dict[str, str] = {}
    for filename in sorted(PUBLIC_MATERIAL_FILENAMES):
        digest = digests[filename]
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"public material {filename!r} has no lowercase sha256")
        canonical_digests[filename] = digest

    if not isinstance(raw["omega_star"], list) or len(raw["omega_star"]) != len(term_order):
        raise ValueError(f"omega_star must be a length-{len(term_order)} list")
    omega = [_real(value, f"omega_star[{index}]") for index, value in enumerate(raw["omega_star"])]
    if any(abs(value) > coefficient_bound for value in omega):
        raise ValueError(
            f"omega_star coefficient outside public bound [-{coefficient_bound}, {coefficient_bound}]"
        )

    thresholds = {
        field: _real(raw[field], field)
        for field in (
            "linf_pass",
            "linf_bronze",
            "linf_silver",
            "linf_gold",
            "relative_l2_pass",
            "support_tau",
            "budget_us",
        )
    }
    if not (
        0.0
        < thresholds["linf_gold"]
        < thresholds["linf_silver"]
        < thresholds["linf_bronze"]
        < thresholds["linf_pass"]
    ):
        raise ValueError("L-infinity thresholds must satisfy 0 < gold < silver < bronze < pass")
    for field in ("relative_l2_pass", "support_tau", "budget_us"):
        if thresholds[field] <= 0.0:
            raise ValueError(f"{field} must be positive")
    if thresholds["support_tau"] > coefficient_bound:
        raise ValueError("support_tau exceeds the public coefficient bound")

    support = raw["support_terms"]
    if not isinstance(support, list) or any(not isinstance(term, str) for term in support):
        raise ValueError("support_terms must be a list of term names")
    if len(set(support)) != len(support):
        raise ValueError("support_terms contains duplicates")
    known_terms = set(term_order)
    unknown_terms = sorted(set(support) - known_terms)
    if unknown_terms:
        raise ValueError(f"support_terms contains unknown terms: {unknown_terms}")
    expected_support = [
        term
        for term, coefficient in zip(term_order, omega, strict=True)
        if abs(coefficient) > thresholds["support_tau"]
    ]
    if support != expected_support:
        raise ValueError("support_terms does not match omega_star at support_tau in term order")

    return {
        "schema_version": HIDDEN_SCORER_SCHEMA_VERSION,
        "public_material_digests": canonical_digests,
        "omega_star": omega,
        **thresholds,
        "support_terms": list(support),
        "max_probe_rows": _positive_int(raw["max_probe_rows"], "max_probe_rows"),
        "oracle_noise_shots": _positive_int(raw["oracle_noise_shots"], "oracle_noise_shots"),
    }
