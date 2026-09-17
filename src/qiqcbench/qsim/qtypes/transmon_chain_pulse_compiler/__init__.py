"""transmon_chain_pulse_compiler qtype.

A 5-qubit linear-chain transmon with a native continuous controlled-phase
``CPhase(theta)`` on each edge, virtual-Z single-qubit phases, finite-duration
``Rx``/``Ry``, multilevel leakage to ``|2>``, and a *pulse-level calibratable*
entangling gate. The agent submits a scheduled gate program (a compilation
surface) and is scored continuously by the SPAM-inclusive average-gate fidelity
of the implemented operation against an ideal target unitary (e.g. ``QFT_5``).

Simulator-only, MCP-only.
"""
