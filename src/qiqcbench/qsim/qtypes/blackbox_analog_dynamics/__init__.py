"""Blackbox analog-dynamics qtype: learn an unknown Hamiltonian via probe batches.

Public/hidden device split, a stateful run-long simulator engine (budget
accumulates across batches), and the ``run_hamiltonian_probe_batch`` MCP tool.
Task-specific construction/scoring for ``time_budgeted_hamlearn_10q`` lives under
``qsim.hidden_dynamics.time_budgeted_hamlearn_10q``, not here.

The qtype descriptor lives in ``qtype.py`` (discovered by
``qtypes/registry.py``); import concrete classes from the submodules
(``device``, ``engine``, ``backend``) directly. Kept import-light so descriptor
discovery does not pull in engines/backends.
"""
