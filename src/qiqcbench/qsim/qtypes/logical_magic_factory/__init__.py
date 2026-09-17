"""logical_magic_factory qtype package.

A black-box fault-tolerant Steane-code ([[7,1,3]]) magic-state factory (Goto,
Sci. Rep. 6, 19578, 2016): the agent draws noisy logical |H> copies from the
factory (charged against a run-long magic-state budget), applies transversal
logical Clifford twirls and an optional two-copy joint (Bell/SWAP-test)
measurement (``logical_magic_benchmarking`` capability), and must recover the
factory's logical infidelity epsilon to multiplicative precision. The qtype
descriptor lives in ``qtype.py`` (discovered by ``qtypes/registry.py``); import
concrete classes from the submodules (``device``, ``engine``, ``backend``,
``wire``) directly. Kept import-light so descriptor discovery does not pull in
engines/backends. Simulator-only, MCP-only.
"""
