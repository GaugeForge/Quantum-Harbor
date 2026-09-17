"""kitaev_chain qtype package.

A semiconductor Kitaev chain (gate-defined quantum dots coupled via
superconductor hybrid segments) operated through programmable gates and charge
readout. Charge-readout characterization uses exact free-fermion / matchgate
physics; pulse control uses an explicitly effective three-level model.
Simulator-only, MCP-only. Import-light by design: the DESCRIPTOR in ``qtype.py``
keeps schema imports eager and backend/mcp lazy so registry discovery stays
cycle-free.
"""
