"""neutral_atom_quantum_network qtype: a two-node neutral-atom quantum network.

Protocol/algorithm level (no atomic-physics pulse design): two processing nodes joined by a
photonic interconnect carrying a budgeted quantum channel + classical channel. The engine
executes selectable distributed-estimation primitives over the link (paying a deterministic
communication cost) to estimate a single cross-node association; the hidden instance,
re-execution scorer, and accuracy/budget/binding gates belong to the verifier.
"""
