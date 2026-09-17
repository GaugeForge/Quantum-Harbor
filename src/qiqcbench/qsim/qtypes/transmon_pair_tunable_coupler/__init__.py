"""transmon_pair_tunable_coupler qtype.

Two transmons (q1 flux-tunable, q2 fixed) coupled through a frequency-tunable
coupler, operated through a realistic control stack (finite-bit-depth AWG + a
distorting flux line). The agent builds a high-fidelity controlled-Z via a
net-zero adiabatic flux pulse driving the ``|11>-|02>`` avoided crossing, working
*through* the flux-line distortion (program DAC samples -> predistort -> realized
flux). Hosts the ``tunable_coupler_cz_netzero`` task. Simulator-only, MCP-only.
"""
