# Qtypes

Browse the 36 qtypes and the five that back a released task.

A qtype describes a device or control abstraction, including public and hidden schemas and its enabled runtime tools. The [registry](../../src/qiqcbench/qsim/qtypes/registry.py) discovers each non-underscore package under `src/qiqcbench/qsim/qtypes/` and loads its `qtype.py` descriptor. A registered qtype does not by itself provide a released task. **31 of the 36 qtypes ship without a released task** in this repository. Only the five qtypes used by released tasks have device files in `configs/devices/`, so the sidecar has no device to serve for the other 31.

| Qtype | Platform family | Physics or control description | Released task |
| --- | --- | --- | --- |
| `blackbox_analog_dynamics` | Analog spin dynamics | Product-state probes and Pauli measurements of an unknown Hamiltonian. | [Hamiltonian learning](../tasks/released-tasks.md#time-budgeted-hamiltonian-learning) |
| `blackbox_boundedgate_circuit` | Digital gate circuit | Pauli-basis shot sampling of a hidden parameterized circuit. | [Surrogate modeling](../tasks/released-tasks.md#surrogate-modeling-of-a-60-qubit-bounded-gate-circuit) |
| `blackbox_lindblad_dynamics` | Open-system spin dynamics | Product-state probes of a hidden coherent and dissipative Lindblad generator. | No |
| `bose_hubbard_chain` | Analog bosonic lattice | Bose-Hubbard evolution and site-resolved quadrature readout. | No |
| `bosonic_cavity_qec` | Bosonic cavity | Storage cavity and transmon ancilla with displacement, SNAP, and parity measurement. | No |
| `composite_measurement_estimation` | Pauli measurement | Mixtures of local measurement bases for estimating a hidden-state observable. | [Composite measurement](../tasks/released-tasks.md#adaptive-composite-measurement-design-for-h2o) |
| `cycle_error_recon` | Superconducting CZ cycle | Folded error reconstruction from repeated five-qubit CZ-cycle measurements. | No |
| `digital_gate_model` | Digital gate model | Circuit-gate execution and per-shot bitstring readout. | No |
| `dipolar_spin_ensemble` | NV spin ensemble | Global rotations and collective magnetization of interacting diamond spins. | No |
| `driven_dissipative_transmon_array` | Transmon array | Pump and loss controls on a driven open-system chain. | No |
| `ftqc_resource_estimation` | Fault-tolerant cost model | Static circuit-resource estimation without device shots. | No |
| `gmon_ring_3q` | Superconducting gmon | Pulse-level modulation of couplers in a three-qubit ring. | No |
| `ion_trap_gate_model` | Trapped ions | All-to-all ion gate circuits with Mølmer–Sørensen entanglers. | No |
| `kitaev_chain` | Semiconductor quantum dots | Kitaev-chain gate control and charge readout with superconducting links. | No |
| `logical_circuit_optimizer` | Logical circuit | Static Clifford+T circuit equivalence and T-count optimization. | No |
| `logical_magic_factory` | Logical magic-state factory | Steane-code factory at the logical-effective level. | No |
| `neutral_atom_dm_processor` | Neutral atoms | Rydberg gates and tweezer schedules with exact density-matrix noise. | No |
| `neutral_atom_ftqc_compiler` | Neutral-atom architecture | Static compiler cost model for zoned atom arrays and surface-code tiles. | No |
| `neutral_atom_logical_processor` | Neutral atoms | Rydberg gates, atom transport, fluorescence readout, and feedforward. | No |
| `neutral_atom_manybody_simulator` | Neutral atoms | Site-resolved Rydberg-array many-body evolution. | No |
| `neutral_atom_quantum_network` | Neutral-atom network | Distributed estimation over two nodes and a communication budget. | No |
| `nv_sensor_network` | NV-center sensors | Distributed Ramsey magnetometry with optional entangled sensor nodes. | No |
| `qccd_ion_compiler` | Trapped-ion architecture | Static zoned-trap schedule evaluation with transport, gates, and recooling. | No |
| `rydberg_multitarget_surface_code` | Neutral-atom surface code | Memory experiments with simultaneous multi-target Rydberg CZ gates. | No |
| `scheduled_transmon_gate_model` | Transmon gate schedule | Native-gate schedules tested through mirror and readout-reference measurements. | No |
| `spin_chain_control` | Coherent spin chain | Three-spin controller transfer and raw computational-basis readout. | No |
| `surface_code_lattice_surgery` | Transmon surface code | Rotated surface-code patches, detector events, and lattice-surgery CNOT. | [Decoder calibration](../tasks/released-tasks.md#logical-cnot-decoder-calibration) |
| `surface_code_memory` | Surface-code processor | Surface-code memory shots and detector records under hidden noise. | No |
| `transmon_chain_pulse_compiler` | Transmon chain | Scheduled phase gates and microwave rotations with leakage. | No |
| `transmon_erasure_qutrit` | Transmon qutrit | Erasure detection and dynamical decoupling of a g–f encoded qubit. | No |
| `transmon_multilevel_pulse` | Transmon pulse control | Four-level Duffing transmon driven by two quadratures and detuning. | No |
| `transmon_pair_tunable_coupler` | Transmon pair | Flux-line-distorted pulses and tunable-coupler dynamics for a CZ gate. | [Net-zero CZ](../tasks/released-tasks.md#tunable-coupler-net-zero-cz) |
| `transmon_pulse` | Transmon pulse control | Single-transmon pulse sequences and readout. | No |
| `transmon_repeater_array` | Transmon repeater | Noisy Bell-pair transport and local purification across array endpoints. | No |
| `trapped_ion_chain` | Trapped ions | Linear ion chain with global Mølmer–Sørensen gates and Ramsey evolution. | No |
| `trapped_ion_state_copy_randomized_measurement` | Trapped ions | Six-ion state copies and randomized local measurement batches. | No |
