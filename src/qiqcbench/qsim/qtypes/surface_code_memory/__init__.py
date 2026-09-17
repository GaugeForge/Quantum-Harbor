"""surface_code_memory qtype: rotated distance-d surface-code memory experiment.

Detector/DEM level (no pulse, no full stabilizer-circuit simulation): a lightweight
numpy detector-error-model sampler returns raw per-shot detection events under a hidden,
spacetime-correlated noise model. The replay decoder + logical-error scoring live in the
verifier, not in qsim.
"""
