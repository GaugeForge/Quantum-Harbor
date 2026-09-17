"""Per-Qtype simulator implementations.

Each qtype-owned package holds concrete device config, lab-notebook builders,
engine/backend/runner logic, and MCP tool registration for one hardware family
/ abstraction (e.g. `digital_gate_model`). Core modules in `qsim/core/` stay
Qtype-agnostic.
"""
