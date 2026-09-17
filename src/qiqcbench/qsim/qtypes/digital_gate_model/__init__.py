"""digital_gate_model qtype package.

Hardware-agnostic digital/gate-model device (GHZ/Bell + the ``digital_vqe``
capability). The qtype descriptor lives in ``qtype.py`` (discovered by
``qtypes/registry.py``); import concrete classes from the submodules
(``device``, ``engine``, ``replay``, ``capabilities.*``) directly. Kept
import-light so descriptor discovery does not pull in engines/backends.
"""
