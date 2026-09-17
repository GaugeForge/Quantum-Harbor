"""driven_dissipative_transmon_array qtype package.

Analog open-system 4-site hard-core transmon chain with local energy-selective
pump/loss reservoirs (the ``local_reservoir_stabilization`` capability). The qtype
descriptor lives in ``qtype.py`` (discovered by ``qtypes/registry.py``); import
concrete classes from the submodules (``device``, ``engine``, ``backend``,
``wire``, ``lab_notebook``, ``runner``) directly. Kept import-light so descriptor
discovery does not pull in engines/backends.
"""
