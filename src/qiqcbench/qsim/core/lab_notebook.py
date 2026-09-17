"""Qtype-agnostic lab-notebook union surface.

Concrete per-qtype notebook models live in the qtype packages and are
registered via ``qtypes/registry.py``. The cross-qtype ``AnyLabNotebook`` union
is built dynamically from the registry, so adding a qtype never edits this file.

``LabNotebook`` and ``DigitalLabNotebook`` are re-exported for the handful of
callers (e.g. ``qsim/devices.py``) that reference those concrete notebook types
by name.
"""

from __future__ import annotations

from functools import reduce
from operator import or_

from qiqcbench.qsim.qtypes.digital_gate_model.lab_notebook import DigitalLabNotebook
from qiqcbench.qsim.qtypes.registry import all_descriptors
from qiqcbench.qsim.qtypes.transmon_pulse.lab_notebook import LabNotebook

AnyLabNotebook = reduce(or_, (d.notebook_model for d in all_descriptors()))

__all__ = ["AnyLabNotebook", "DigitalLabNotebook", "LabNotebook"]
