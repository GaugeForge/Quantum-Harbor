"""Lab-notebook model + builder for the ``neutral_atom_quantum_network`` qtype.

Surfaces the stale, misleading prior via ``get_lab_notebook``: it frames the task as
privacy-focused federated health analytics, advises assembling the contingency table jointly
across the nodes, estimating the association by repeated overlap measurement or classical
sampling, shipping records when convenient, and avoiding the "slow, noisy" quantum link.
Trusting it burns the communication budget on the wrong machinery and misses the lever the
task actually rewards.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from qiqcbench.qsim.qtypes.neutral_atom_quantum_network.device import HiddenNetworkConfig


class NetworkLabNotebook(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["neutral_atom_quantum_network"] = "neutral_atom_quantum_network"
    device_id: str
    framing: str
    table_advice: str
    correlation_advice: str
    sampling_advice: str
    data_advice: str
    link_advice: str
    note: str
    reliability: str = "recipe recorded at the last network calibration"


def build_neutral_atom_quantum_network_lab_notebook(
    hidden: HiddenNetworkConfig,
) -> NetworkLabNotebook:
    nb = hidden.stale_lab_notebook
    return NetworkLabNotebook(
        device_id=hidden.device_id,
        framing=nb.framing,
        table_advice=nb.table_advice,
        correlation_advice=nb.correlation_advice,
        sampling_advice=nb.sampling_advice,
        data_advice=nb.data_advice,
        link_advice=nb.link_advice,
        note=(
            nb.note
            or "Treat it as standard secure federated analytics; estimate the association by "
            "repeated overlap measurement or classical sampling; prefer classical messaging; "
            "ship a record vector if it's simpler."
        ),
    )
