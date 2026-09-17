"""bose_hubbard_chain qtype package.

Programmable Bose-Hubbard photon chain (the ``bose_hubbard_spectroscopy``
capability). The qtype descriptor lives in ``qtype.py`` (discovered by
``qtypes/registry.py``); import concrete classes from the submodules
(``device``, ``engine``, ``backend``) directly. Kept import-light so descriptor
discovery does not pull in engines/backends.
"""
