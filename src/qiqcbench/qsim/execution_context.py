"""Opaque execution context accepted by qsim at process bootstrap."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

EXECUTION_CONTEXT_ENV = "QIQCBENCH_EXECUTION_CONTEXT_ID"
HIDDEN_COMMITMENT_SECRET_ENV = "QIQCBENCH_HIDDEN_COMMITMENT_SECRET"
EXECUTION_CONTEXT_SCHEMA_VERSION = 1
_OPAQUE_CONTEXT_ID = re.compile(r"[0-9a-f]{64}\Z")


def validate_execution_context_id(value: str) -> str:
    """Validate one opaque 256-bit context ID without interpreting its owner."""

    if not isinstance(value, str) or _OPAQUE_CONTEXT_ID.fullmatch(value) is None:
        raise ValueError("execution context ID must be 32 bytes encoded as 64 lowercase hex")
    return value


def load_execution_context_id(
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Read the optional qsim-only execution context environment binding."""

    source = os.environ if environ is None else environ
    if EXECUTION_CONTEXT_ENV not in source:
        return None
    return validate_execution_context_id(source[EXECUTION_CONTEXT_ENV])


def load_hidden_commitment_secret(
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Read the private equality-proof key injected into an official qsim.

    Development qsim processes remain usable without a key.  An execution-bound
    process fails at bootstrap when the materializer omitted it.
    """

    source = os.environ if environ is None else environ
    if HIDDEN_COMMITMENT_SECRET_ENV not in source:
        if EXECUTION_CONTEXT_ENV in source:
            raise ValueError(
                f"{HIDDEN_COMMITMENT_SECRET_ENV} is required when {EXECUTION_CONTEXT_ENV} is set"
            )
        return None
    if EXECUTION_CONTEXT_ENV not in source:
        raise ValueError(
            f"{HIDDEN_COMMITMENT_SECRET_ENV} is invalid without {EXECUTION_CONTEXT_ENV}"
        )
    return validate_hidden_commitment_secret(source[HIDDEN_COMMITMENT_SECRET_ENV])


def validate_hidden_commitment_secret(value: str) -> str:
    """Validate one materializer-derived private HMAC key."""

    if not isinstance(value, str) or _OPAQUE_CONTEXT_ID.fullmatch(value) is None:
        raise ValueError("hidden commitment secret must be 32 bytes encoded as lowercase hex")
    return value


__all__ = [
    "EXECUTION_CONTEXT_ENV",
    "EXECUTION_CONTEXT_SCHEMA_VERSION",
    "HIDDEN_COMMITMENT_SECRET_ENV",
    "load_execution_context_id",
    "load_hidden_commitment_secret",
    "validate_execution_context_id",
    "validate_hidden_commitment_secret",
]
