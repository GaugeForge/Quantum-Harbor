"""Versioned private key derivation for hidden-device evidence commitments."""

from __future__ import annotations

import hashlib
import hmac

from qiqcbench.eval.verifier.context import VerifierRunContext


def derive_hidden_commitment_secret(context: VerifierRunContext) -> str:
    """Derive qsim/verifier equality-proof material in an isolated key domain."""

    message = (
        "qiqcbench-hidden-device-commitment-key-v1\0"
        f"{context.task_id}\0{context.execution.execution_id}"
    ).encode()
    return hmac.new(
        bytes.fromhex(context.verifier_replay_secret),
        message,
        hashlib.sha256,
    ).hexdigest()


__all__ = ["derive_hidden_commitment_secret"]
