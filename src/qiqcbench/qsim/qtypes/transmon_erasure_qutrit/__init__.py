"""transmon_erasure_qutrit qtype package.

A transmon qutrit (g/e/f) operated as a g-f erasure qubit (|0_L>=|g>, |1_L>=|f>,
|e>=erasure) with a neighbouring ancilla for mid-circuit erasure detection
(``mid_circuit_erasure_detection`` capability). The qtype descriptor lives in
``qtype.py`` (discovered by ``qtypes/registry.py``); import concrete classes from
the submodules (``device``, ``engine``, ``backend``, ``runner``, ``wire``)
directly. Kept import-light so descriptor discovery does not pull in
engines/backends. Simulator-only, MCP-only.
"""
