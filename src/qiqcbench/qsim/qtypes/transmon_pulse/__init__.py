"""transmon_pulse qtype package.

Single-transmon pulse-level device with T1/T2/readout noise. The qtype
descriptor lives in ``qtype.py`` (discovered by ``qtypes/registry.py``); import
concrete classes from the submodules (``device``, ``engine``, ``backend``,
``replay``) directly. Kept import-light so descriptor discovery does not pull in
engines/backends.
"""
