"""Host-side evaluation layer.

This package preserves the shipped EloScoreReport v1 compatibility surface and
contains the versioned contracts and host-side machinery for evaluation
framework v2. Evaluation consumes verifier-owned facts after runs complete;
it never becomes a second task-physics or hidden-truth authority.

The package lives outside ``qsim/qtypes/`` so qtype discovery never treats it
as a device abstraction. The legacy report writer remains dependency-light for
baked separate-mode verifiers while v2 host contracts use Pydantic.
"""
