"""surface_code_lattice_surgery control + result wire schemas (qtype-local).

Three experiment primitives on the async job model: ``run_memory_experiment`` (single
patch), ``run_merged_memory`` (merged configuration, optionally with the full
split->merge->hold->split transition cycle), and ``run_lattice_surgery_cnot`` (the fixed
public CNOT schedule). All return **raw per-shot detection events** (packed bit arrays)
plus raw readout evidence; never the noise model, weights, decoded results, or logical
error rates. One shot budget AND one shot-rounds budget accumulate across every call.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from qiqcbench.qsim.core.wire import SCHEMA_VERSION, _Strict, _StrictRequest

MemoryLayout = Literal["d3_control", "d3_intermediate", "d3_target", "d5"]
MergeWindow = Literal["zz", "xx"]
CnotConfig = Literal["z", "x", "bell_zz", "bell_xx"]
CnotInput = Literal["00", "01", "10", "11"]


class LsMemoryRequest(_StrictRequest):
    schema_version: int = SCHEMA_VERSION
    layout: MemoryLayout
    rounds: int = Field(..., ge=1, le=32)
    shots: int = Field(..., ge=1, le=100_000)
    # maintainer-controlled in scoring; the device prepares logical |0> and measures Z̄.
    logical_input: Literal[0] = 0


class LsMergedRequest(_StrictRequest):
    schema_version: int = SCHEMA_VERSION
    window: MergeWindow
    rounds: int = Field(..., ge=1, le=16)  # merged-hold rounds
    shots: int = Field(..., ge=1, le=100_000)
    include_transitions: bool = False


class LsCnotRequest(_StrictRequest):
    schema_version: int = SCHEMA_VERSION
    config: CnotConfig
    logical_input: CnotInput = "00"  # ignored for the bell configurations (prep is fixed)
    shots: int = Field(..., ge=1, le=100_000)


class _LsBudgetView(_Strict):
    shots_used: int
    shots_cap: int
    shot_rounds_used: int
    shot_rounds_cap: int


class JobLatticeSurgeryData(_Strict):
    kind: Literal["lattice_surgery_detectors"] = "lattice_surgery_detectors"
    experiment: Literal["memory", "merged", "cnot"]
    layout: str  # layout / window / reporting-configuration name
    rounds: int  # syndrome-extraction rounds executed
    shots: int
    n_detectors: int
    # Raw per-shot detection events, row-major [shots][n_detectors], packed via
    # base64(np.packbits(bits)). Column order per the public detector metadata.
    detection_events_b64: str
    # Per-shot flips of the configuration's public observables relative to the noiseless
    # value, row-major [shots][n_observables] (memory: the single obs_logical; for the
    # |0>-prepared Z memory this IS the raw logical outcome).
    observable_flips_b64: str
    observable_names: list[str]
    # CNOT runs only: raw per-shot values of the named joint parities (packed [shots]).
    mzz_b64: str | None = None
    mxx_b64: str | None = None
    mzint_b64: str | None = None
    budget: _LsBudgetView


__all__ = [
    "CnotConfig",
    "CnotInput",
    "JobLatticeSurgeryData",
    "LsCnotRequest",
    "LsMemoryRequest",
    "LsMergedRequest",
    "MemoryLayout",
    "MergeWindow",
    "_LsBudgetView",
]
