"""Lab-notebook model and builder for the chain pulse-compiler qtype.

Surfaces the *stale* textbook compilation advice and the optimistic gate-fidelity
claim to the agent: decompose each controlled-phase into 2 CNOTs (the single
biggest unforced error), implement Rz as physical gates, use a physical SWAP
network for the bit reversal, and trust the shipped (mis-calibrated) entangling
pulse. The true device noise floor, the shipped over-rotation, the asymmetric
readout, and the RNG seed stay hidden.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.transmon_chain_pulse_compiler.device import (
    HiddenChainCompilerConfig,
)


class ChainCompilerLabNotebook(BaseModel):
    """Stale lab-notebook entry for the chain pulse-compiler qtype.

    Contents are intentionally optimistic/out of date. An agent that follows this
    builds the naive textbook QFT (CNOT-decomposed controlled-phases, physical Rz
    and SWAPs, a serial schedule, the shipped mis-calibrated pulse) and scores the
    low anchor F_naive.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["transmon_chain_pulse_compiler"] = "transmon_chain_pulse_compiler"
    device_id: str
    last_updated: str = "earlier cooldown"
    two_qubit_fidelity_claim: float | None = None
    controlled_phase_recipe: str | None = None
    rz_recipe: str | None = None
    swap_recipe: str | None = None
    approximation_advice: str | None = None
    entangling_pulse_claim: str | None = None
    readout_claim: str | None = None
    notes: str = ""


def build_chain_compiler_lab_notebook(
    hidden: HiddenChainCompilerConfig,
) -> ChainCompilerLabNotebook:
    """Build the agent-facing stale lab notebook from the hidden stale entry."""
    nb = hidden.stale_lab_notebook
    return ChainCompilerLabNotebook(
        device_id=hidden.device_id,
        last_updated=nb.last_calibrated or "earlier cooldown",
        two_qubit_fidelity_claim=nb.two_qubit_fidelity_claim,
        controlled_phase_recipe=nb.controlled_phase_recipe,
        rz_recipe=nb.rz_recipe,
        swap_recipe=nb.swap_recipe,
        approximation_advice=nb.approximation_advice,
        entangling_pulse_claim=nb.entangling_pulse_claim,
        readout_claim=nb.readout_claim,
        notes=(
            nb.notes
            or (
                "Use the standard textbook QFT compilation; the shipped two-qubit gate "
                "is well calibrated, so you should not need to retune it."
            )
        ),
    )


__all__ = ["ChainCompilerLabNotebook", "build_chain_compiler_lab_notebook"]
