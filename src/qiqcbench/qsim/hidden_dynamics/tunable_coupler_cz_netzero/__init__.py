"""Hidden construction + scoring for the tunable_coupler_cz_netzero task.

Task-specific (not reusable qtype) logic: the Stage-A device+settling recovery
checks, the Stage-B/C replay of the submitted PROGRAMMED waveform through the
control stack + the two-transmon CZ model (conditional phase after virtual-Z,
leakage, fidelity, net-zero 1/f advantage, idle ZZ, predistortion residual), the
reference solver (net-zero design + predistortion that passes), and the
materializer that freezes the hidden truth + thresholds.
"""
