"""Task-specific hidden-dynamics modules.

Each subpackage owns the hidden-truth generation/scoring substrate for a single
benchmark task. Generic VQE/digital substrate stays in
``qiqcbench.qsim.qtypes.digital_gate_model``; long-tail task logic lives here
so it cannot contaminate the reusable framework.
"""
