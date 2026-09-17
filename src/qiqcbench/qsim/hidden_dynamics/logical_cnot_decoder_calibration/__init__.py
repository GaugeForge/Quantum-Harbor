"""Hidden construction + scoring for ``logical_cnot_decoder_calibration``.

The seeded circuit-level hidden noise instance (per-qubit T1/T2, inhomogeneous CZ errors,
asymmetric measurement/reset, leakage topology, routing-region degradation), the replay
decoders, the two-gate scorer (d=5 memory + lattice-surgery CNOT battery), the
agent-strategy-ladder anchors, and the data-driven reference solver.
"""
