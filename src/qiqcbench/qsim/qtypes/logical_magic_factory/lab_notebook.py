"""Lab-notebook model and builder for the logical_magic_factory qtype.

The notebook is intentionally stale: it quotes a fidelity ~2.7x better than
truth via a "single-copy Pauli characterization" protocol that is Omega(1/eps^2)
in magic-state cost and therefore budget-infeasible. Copying the notebook, or
following its recommended protocol, is the primary trap the
``logical_magic_bell_benchmarking`` task probes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.logical_magic_factory.device import HiddenMagicFactoryConfig


class MagicFactoryLabNotebook(BaseModel):
    """Stale lab-notebook view for the logical_magic_factory qtype."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["logical_magic_factory"] = "logical_magic_factory"
    device_id: str
    last_updated: str = "stale"
    logical_fidelity_claim: float | None = None
    recommended_protocol: str | None = None
    recommended_shots: int | None = None
    syndrome_acceptance_claim: float | None = None
    readout_fidelity_claim: float | None = None
    noise_model_claim: str | None = None
    recommended_verification: str | None = None
    escape_default_n: int | None = None
    cultivation_rounds_default: int | None = None
    retention_claim: float | None = None
    notes: str = ""


def build_magic_factory_lab_notebook(
    hidden: HiddenMagicFactoryConfig,
) -> MagicFactoryLabNotebook:
    """Build the agent-facing lab notebook from the hidden stale entry."""
    nb = hidden.stale_lab_notebook
    return MagicFactoryLabNotebook(
        device_id=hidden.device_id,
        last_updated=nb.last_calibrated or "stale",
        logical_fidelity_claim=nb.logical_fidelity_claim,
        recommended_protocol=nb.recommended_protocol,
        recommended_shots=nb.recommended_shots,
        syndrome_acceptance_claim=nb.syndrome_acceptance_claim,
        readout_fidelity_claim=nb.readout_fidelity_claim,
        noise_model_claim=nb.noise_model_claim,
        recommended_verification=nb.recommended_verification,
        escape_default_n=nb.escape_default_n,
        cultivation_rounds_default=nb.cultivation_rounds_default,
        retention_claim=nb.retention_claim,
        notes=nb.notes,
    )


__all__ = ["MagicFactoryLabNotebook", "build_magic_factory_lab_notebook"]
