"""bosonic_cavity_qec qtype package.

A storage cavity (bosonic mode, Fock cutoff n_max) dispersively coupled to a
transmon ancilla, with SNAP/displacement universal control, a dispersive
photon-number-parity measurement, and mid-circuit measurement + classical
feedback (``bosonic_qec_control`` capability). The descriptor lives in
``qtype.py`` (discovered by ``qtypes/registry.py``); import concrete classes from
the submodules (``device``, ``physics``, ``engine``, ``backend``, ``runner``,
``wire``) directly. Kept import-light so descriptor discovery does not pull in
engines/backends. Simulator-only, MCP-only.
"""
