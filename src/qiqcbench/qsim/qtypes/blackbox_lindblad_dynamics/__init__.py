"""Blackbox Lindblad-dynamics qtype: learn an unknown open-system generator.

Public/hidden device split, a stateful run-long simulator engine (budget
accumulates across batches), and the ``run_lindblad_probe_batch`` MCP tool.
Task-specific construction and scoring are separate from the qtype engine.

The qtype descriptor lives in ``qtype.py`` (discovered by
``qtypes/registry.py``); import concrete classes from the submodules
(``device``, ``engine``, ``backend``) directly. Kept import-light so descriptor
discovery does not pull in engines/backends.
"""
