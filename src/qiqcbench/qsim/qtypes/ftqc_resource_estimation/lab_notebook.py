"""Lab-notebook model + builder for the ``ftqc_resource_estimation`` qtype."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from qiqcbench.qsim.qtypes.ftqc_resource_estimation.device import HiddenFtqcConfig


class FtqcLabNotebook(BaseModel):
    """Stale lab notebook for the FTQC resource-estimation qtype."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    qtype: Literal["ftqc_resource_estimation"] = "ftqc_resource_estimation"
    device_id: str
    mu1_hartree: float | None = None
    mu2_hartree: float | None = None
    scalar_lambda_hartree: float | None = None
    recommendations: list[str] = Field(default_factory=list)
    note: str = ""
    reliability: str = "operator notes carried over from a prior compilation study"


def build_ftqc_lab_notebook(hidden: HiddenFtqcConfig) -> FtqcLabNotebook:
    """Build the FTQC lab-notebook view from a hidden config's stale prior."""
    nb = hidden.stale_lab_notebook
    return FtqcLabNotebook(
        device_id=hidden.device_id,
        mu1_hartree=nb.mu1_hartree,
        mu2_hartree=nb.mu2_hartree,
        scalar_lambda_hartree=nb.scalar_lambda_hartree,
        recommendations=list(nb.recommendations),
        note=nb.note or "These notes are stale and may be wrong; verify before relying on them.",
        reliability=nb.reliability,
    )
