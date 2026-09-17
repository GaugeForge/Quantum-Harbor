"""gmon_ring_3q qtype package.

Analog 3-qubit gmon ring with tunable couplers (the ``ring_modulation``
capability). The qtype descriptor lives in ``qtype.py`` (discovered by
``qtypes/registry.py``); import concrete classes from the submodules
(``device``, ``engine``, ``backend``, ``wire``) directly. Kept import-light so
descriptor discovery does not pull in engines/backends.
"""
