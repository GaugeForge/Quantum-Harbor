"""Qiskit-free constants for live-provider plumbing.

Lives in its own module so that `factory.py` and other call sites can reference
the IBM channel string and the disabled-mode error text without transitively
importing `qiskit` / `qiskit-ibm-runtime`. The live backend module
(`live_ibm.py`) re-exports both names so existing imports there keep working.
"""

from __future__ import annotations

IBM_QUANTUM_PLATFORM_CHANNEL = "ibm_quantum_platform"
LIVE_PROVIDER_DISABLED_ERROR = "live provider mode is not enabled"
