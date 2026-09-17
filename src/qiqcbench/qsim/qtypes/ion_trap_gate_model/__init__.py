"""ion_trap_gate_model qtype.

A 16-qubit all-to-all trapped-ion gate-model device. Native single-qubit
rotations (rx/ry/rz; gpi/gpi2 equivalents) plus an arbitrary-angle two-qubit
Molmer-Sorensen entangler ms(theta) = exp(-i (theta/2) X_i X_j) on a chosen
pair. Pure-numpy state-vector trajectory sampling (per-native-gate depolarizing
Pauli noise + asymmetric readout), NEVER a density matrix.

The ``randomized_measurement`` capability adds a randomized-measurement
batch tool for estimating the
second Renyi entropy of a bulk subsystem from CUE-random local-basis bitstrings.
Simulator-only, MCP-only.
"""
