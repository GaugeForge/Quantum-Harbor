"""``trapped_ion_state_copy_randomized_measurement`` qtype: a six-ion trapped-ion state-copy device.

Protocol/algorithm level: a long-range XX simulator that prepares independent copies of an
unknown mixed state and exposes single-copy randomized measurements + collective copy-block
(cyclic-shift) measurements for estimating nonlinear functionals Tr(Z0Z1 rho^k)/Tr(rho^k). The
engine samples raw evidence from a cached 64x64 spectrum under a run-long state-copy budget; the
hidden state, the exact moment anchors, and scoring belong to the verifier.
"""
