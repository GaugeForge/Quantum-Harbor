"""dipolar_spin_ensemble qtype package.

Dipolar-coupled spin ensemble with global toggling-frame pulse control (the
``toggling_frame_control`` capability). The qtype descriptor lives in
``qtype.py`` (discovered by ``qtypes/registry.py``); import concrete classes
from the submodules (``device``, ``engine``, ``backend``, ``wire``) directly.
Kept import-light so descriptor discovery does not pull in engines/backends.
"""
