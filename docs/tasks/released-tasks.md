# Released tasks

Compare the objectives, tools, budgets, and submissions of the five released tasks.

The full scientific objective and answer schema for each task are in its `instruction.md`. Budgets below come from the public device specifications. Agent and verifier timeouts come from each bundle's `task.toml`. Pass thresholds appear in each task's `instruction.md`, as the agent sees them, and are not repeated here.

## Time-budgeted Hamiltonian learning

The agent probes a 10-qubit analog device and estimates the coefficients of a hidden Hamiltonian over a public 111-term Pauli dictionary. It chooses product states, evolution times, and observables, then fits a coefficient vector from raw measurement outcomes. Grading checks full-vector and worst-coefficient accuracy against verified probe evidence.

- [Bundle](../../harbor_tasks/time_budgeted_hamlearn_10q/) · [Instructions](../../harbor_tasks/time_budgeted_hamlearn_10q/instruction.md)
- Device `hamlearn_10q_chain_v1`, qtype `blackbox_analog_dynamics`.
- Experiment tool `run_hamiltonian_probe_batch`. The always-on tools supply device information, job polling, and final-answer submission.
- Public budget: 220 µs cumulative accepted evolution time, up to 600 accepted probe rows and 640 probe jobs, with at most 64 rows per batch.
- Submit a 111-coefficient vector with uncertainties using `submit_final_answer`, inline or from a JSON file in `/submission` with its SHA-256 digest.
- Timeouts: agent 3,600 seconds, verifier 300 seconds.

## Surrogate modeling of a 60-qubit bounded-gate circuit

The agent measures a fixed but hidden 60-qubit circuit and learns its pair-correlation responses to six input angles. After locking measurement admission, the agent receives precommitted challenge inputs and predicts the as-measured correlations. Grading compares the predictions with the hidden device values and checks the measurement and reveal evidence.

- [Bundle](../../harbor_tasks/time_budgeted_shadow_surrogate_60q/) · [Instructions](../../harbor_tasks/time_budgeted_shadow_surrogate_60q/instruction.md)
- Device `boundedgate_60q_v0`, qtype `blackbox_boundedgate_circuit`.
- Experiment tools `run_basis_shots` and `lock_measurements_and_reveal_challenge`, plus the always-on tools.
- Public budget: 315,000 measured shots, 40 experiment jobs, up to 64 blocks and 192 measurement settings per job, and 4 final-answer submissions.
- Submit the prediction matrix and challenge receipt using `submit_final_answer`. A JSON answer file in `/submission` with its SHA-256 digest avoids putting a large matrix in a tool call. Inline submission is also supported.
- Timeouts: agent 1,800 seconds, verifier 300 seconds.

## Tunable-coupler net-zero CZ

The agent characterizes two transmons and the settling of a flux control line, then programs a net-zero flux pulse and corrections for a controlled-Z gate. The lab notebook holds values from a previous cooldown, so the agent has to measure the device as it is now. Grading replays the submitted pulse and coupler settings against the hidden device and checks completed experiment evidence.

- [Bundle](../../harbor_tasks/tunable_coupler_cz_netzero/) · [Instructions](../../harbor_tasks/tunable_coupler_cz_netzero/instruction.md)
- Device `tunable_coupler_cz_v0`, qtype `transmon_pair_tunable_coupler`.
- Experiment tool `run_flux_pulse`, plus the always-on tools.
- Public limits: at most 100,000 shots per measurement, a 0.4 ns DAC sample grid, and a 36–100 ns gate window. The task also requires at least 15 completed flux-pulse jobs.
- Submit characterization values, intended and programmed pulse samples, and gate settings as an inline `submit_final_answer` payload.
- Timeouts: agent 9,000 seconds, verifier 600 seconds.

## Logical CNOT decoder calibration

The agent gathers detector events on a surface-code processor, estimates detector error models, and configures decoders for both memory and lattice-surgery CNOT. It can run single-patch, merged-patch, and CNOT experiments before submitting the decoder models. Grading replays the models on fresh hidden-noise shots and evaluates both memory and CNOT behavior.

- [Bundle](../../harbor_tasks/logical_cnot_decoder_calibration/) · [Instructions](../../harbor_tasks/logical_cnot_decoder_calibration/instruction.md)
- Device `sc_lattice_surgery_114q_v0`, qtype `surface_code_lattice_surgery`.
- Experiment tools `run_memory_experiment`, `run_merged_memory`, and `run_lattice_surgery_cnot`, plus the always-on tools.
- Public budget: 2,000,000 shots and 28,000,000 shot-rounds shared across experiments, with at most 100,000 shots per call.
- Submit memory and CNOT detector error models, decoder configurations, and activity fields as an inline `submit_final_answer` payload.
- Timeouts: agent 10,800 seconds, verifier 1,200 seconds.

## Adaptive composite measurement design for H2O

The agent designs a 14-qubit composite measurement scheme for a published Pauli Hamiltonian and a hidden prepared state. It may collect pilot data, lock one production design, then estimate energy and setting-cluster uncertainty from the resulting counts. Grading recomputes the measurement variance of the submitted scheme from the public Hamiltonian and the hidden state, and checks the reported energy and uncertainty against the collected counts.

- [Bundle](../../harbor_tasks/adaptive_clustered_clbcs_h2o/) · [Instructions](../../harbor_tasks/adaptive_clustered_clbcs_h2o/instruction.md)
- Device `clbcs_h2o_14q_v0`, qtype `composite_measurement_estimation`.
- Task tools `get_observable_spec`, `evaluate_composite_scheme`, `run_pilot_measurements`, and `execute_locked_composite_scheme`, plus the always-on tools.
- Public budget: up to 96 pilot settings and 24,576 pilot shots, 40 scheme evaluations, and one locked production run of 192 bases with 256 shots per base.
- Submit the measurement scheme, the locked production request digest, and the energy and cluster uncertainty as an inline `submit_final_answer` payload.
- Timeouts: agent 1,800 seconds, verifier 600 seconds.
