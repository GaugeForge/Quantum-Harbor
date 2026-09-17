# Calibrate the decoders for a lattice-surgery logical CNOT on a surface-code processor

You operate a **~114-qubit transmon-style 2D grid** hosting rotated surface-code patches,
with a **HIDDEN circuit-level noise model**. Fault-tolerant algorithms (the Clifford
scaffolding of a QFT or Shor's algorithm) are built from logical memory and
lattice-surgery Cliffords; your job is the **decoder-calibration layer** that makes both
work on THIS device: estimate detector error models (DEMs) from raw detection-event data,
configure matching decoders, and drive **both** replayed error rates below their floors:

1. **d=5 memory gate** — replayed logical error rate per cycle `< floor_mem = 1.0e-3`.
2. **CNOT gate** — replayed logical-CNOT error over the fixed 8-entry two-basis battery
   `< floor_cnot = 0.175` on average, AND every battery entry `< 0.185`.

The lab notebook records the previous calibration campaign; nominal numbers are stale —
verify before relying on them. Do not assume either an i.i.d. model or a particular correlated
model: determine which mechanisms are supported by the current detector data.

## Surfaces (MCP server `qsim`; device_id `sc_lattice_surgery_114q_v0`)

- `get_device_spec(device_id=...)` — chip map (qubit id -> [x,y]), the six hosted layouts
  (data qubits + every plaquette's ancilla/basis/support), schedules, budgets, floors,
  and the **byproduct-frame convention** (which named parities fold into each output
  observable).
- `get_lab_notebook(device_id=...)` — the stale calibration prior (do not trust it).
- `run_memory_experiment(device_id, layout, rounds, shots)` — single-patch Z-basis
  memory; `layout` in `{d3_control, d3_intermediate, d3_target, d5}`, `rounds` 1-32,
  `shots` <= 100000/call.
- `run_merged_memory(device_id, window, rounds, shots, include_transitions=False)` — the
  merged configuration (two patches + routing region as ONE code); `window` in
  `{zz, xx}`; with `include_transitions=true` the run is a full split->merge->hold->split
  cycle (2 separate rounds, `rounds` merged rounds, split, 2 separate rounds) so the
  merge/split transition detectors appear.
- `run_lattice_surgery_cnot(device_id, config, shots, logical_input="00")` — the fixed
  public CNOT schedule (prep -> 1 separate round -> rough-merge C-INT for 3 rounds ->
  split -> 1 round -> smooth-merge INT-T for 3 rounds -> split -> 1 round -> M_Z(INT) +
  transversal readout). `config` in `{z, x, bell_zz, bell_xx}`; `logical_input` in
  `{"00","01","10","11"}` selects the battery entry for z/x (ignored for bell configs).
- All experiment tools are **async**: they return `{job_id, status}`; poll
  `get_job_result(job_id)` until `complete`.
- `submit_final_answer(answer={...})` — submit the schema below.

All experiment calls share **one shot budget (2e6)** AND **one shot-rounds budget
(2.8e7)**; both meters are echoed in every result. You must run experiments: a
submission with no logged shots is rejected.

## Task material files (read-only at `/task_materials`)

- `public_spec.yaml`, `stale_notebook.yaml` — same content as the MCP tools.
- `memory_<layout>_scoring.json` — for each memory layout at the fixed scoring layout
  (`rounds = 12`): the ordered detector table (`id`, `ancilla [x,y]`, `round`, `kind`,
  `basis`), the detector-ordering rule for any `rounds`, and the **nominal
  matching-graph adjacency** (`matching_graph_nominal`: edges with 1-2 detector ids,
  `p_nominal` under the nominal uniform noise, and `flips_observables`). The topology is
  authoritative; the RATES are the last recorded nominal model.
- `merged_<window>_transitions.json` — the same, for the merged windows with transitions
  (`rounds = 3`).
- `cnot_superset.json` — the CNOT spacetime detector **superset**: every detector across
  the four reporting configurations gets one global `id`; per config you get the active
  `detector_columns` (the exact column order of the returned bit arrays), the nominal
  adjacency in superset ids, the named parity definitions (`m_zz`, `m_xx`, `m_z_int`,
  `split_*` as measurement-record sets), and the frame convention.

### Data formats

`detection_events_b64` / `observable_flips_b64` are
`base64.b64encode(np.packbits(bits).tobytes())` over row-major flattened bit arrays
(`[shots][n_detectors]` / `[shots][n_observables]`). Unpack:

```python
import base64, numpy as np
buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
x = np.unpackbits(buf)[: shots * n].reshape(shots, n)
```

Observable flips are relative to the noiseless value (for the |0>-prepared Z memory the
flip IS the raw logical outcome). CNOT results additionally carry per-shot raw
`m_zz`/`m_xx`/`m_z_int` parity bits (packed `[shots]`).

### Handling the raw-data volume (read this before your first experiment)

Each experiment returns **raw per-shot detection events**, so a single result can be
hundreds of KB to several MB of base64. **Do not try to read that base64 into your
reasoning context** — you will exhaust your context window before you have any statistics
(and some harnesses truncate a large tool result to a file stub with a path, which is
expected, not an error). Instead, **process the data with code**: your container has
`python3` with `numpy` (plus `stim` + `pymatching`) on the `PATH`. The estimator only
needs a few **running sufficient statistics**, not the raw shots:

- per-detector mean `⟨x_i⟩` (single-detector rates);
- the two-point matrix `⟨x_i x_j⟩` (correlated edges — `p_ij` from the connected part);
- for the CNOT layouts, per-detector×observable co-occurrence `⟨x_i · obs_k⟩` (the
  detector→observable/parity associations).

The efficient workflow is: request modest `shots` per call (e.g. a few thousand up to
`~20000`), write a small script that unpacks the base64 (from the tool result, or from the
persisted file path if your harness saved it there) and **accumulates** `sum(x)`,
`x.T @ x`, and `x.T @ obs` into running totals, then discards the raw array. Repeat across
calls until your statistics are tight, then build the DEM from the accumulated moments.
You never need more than a single chunk of raw shots in memory at once. Budget both meters
(`shots` and `shots·rounds`, echoed in every result) across the campaign.

## What to produce (four stages)

- **A. Probe** — read the chip map, layouts, schedules, budgets, floors, frame convention.
- **B. Memory characterization** — run single-patch experiments; estimate the d=5 DEM
  from detector data. Two-detector edge rates come from the two-point correlations
  `p_ij = (<x_i x_j> - <x_i><x_j>) / ((1-2<x_i>)(1-2<x_j>))`; boundary (single-detector)
  rates are the RESIDUAL of the parity identity `1-2<x_i> = prod_e (1-2 p_e)` over the
  mechanisms touching detector `i` (raw `<x_i>` can double-count correlated mass). If you
  propose edges beyond the public adjacency, control false discoveries (Bonferroni/FDR over
  candidate pairs) and justify them from data. Self-decode, measure your rates, iterate.
- **C. Surgery characterization** — characterize the CNOT spacetime detector superset with any
  combination of merged-memory and CNOT runs that supports your submitted rates, correlated
  pairs, and detector-observable/parity associations. A mechanism may fire detectors and flip a
  named joint parity; declare that through `flips_observables` according to the public frame
  convention. Self-verify all four logical inputs in both Z and X bases within your budget.
- **D. Submit** — the final DEMs + decoder configs (schema below).

## DEM + decoder submission format

A DEM is a list of mechanisms `{"detectors": [id] or [id1, id2], "p": float in (0, 0.5),
"flips_observables": [names]}`:

- `dem_memory_d5`: local detector ids of the d=5 scoring layout (`memory_d5_scoring.json`,
  `12 * 24 = 288` detectors); every detector needs at least one incident mechanism;
  observable name: `obs_logical` (a bare `"flips_observable": true` is also accepted).
- `dem_cnot_spacetime`: **superset ids** from `cnot_superset.json` (one DEM serves all
  four reporting configurations; mechanisms touching detectors inactive in a
  configuration are dropped for that configuration; every active detector of every
  configuration needs coverage). Observable names: the basis-tagged outputs
  (`obs_z_c_out`, `obs_z_t_out`, `obs_x_c_out`, `obs_x_t_out`, `obs_bell_zz`,
  `obs_bell_xx`) and/or the named parities (`m_zz`, `m_xx`, `m_z_int`, `split_zz_k`,
  `split_xx_k`) — parity flips are folded into outputs by the public frame convention.
- Mechanism count caps: 20x the detector count per DEM. Parallel mechanisms on the same
  detector set are consolidated at replay (probabilities XOR-combined; the highest-p
  component's `flips_observables` wins) — submit one mechanism per detector set.
- Decoder configs: `{"decoder_type": "mwpm" | "belief_matching", "options": {}}`. The
  `options` field is an extension seam; v1 supports only omission or an empty object and rejects
  unknown non-empty options. Every submitted
  mechanism flips <= 2 detectors (graphlike), so correlation-aware matching coincides
  with weighted MWPM here — what matters is which edges, rates, and observable flips you
  put in the DEM, not the decoder label.

## Required final answer

Call `submit_final_answer(answer={...})` with:

| Field | Type | Semantics |
|---|---|---|
| `task_id` | str | `"logical_cnot_decoder_calibration"` |
| `method` | str | characterization strategy, what the merged-layout data revealed, decoder choices, budget split |
| `stage_a_mode` | str | `"device_probe"` |
| `probed_spec` | obj | key facts read (layouts/budgets/floors) |
| `stage_b_mode` | str | `"memory_characterization"` |
| `shots_for_memory_characterization_raw` | int | shots spent on single-patch memory |
| `dem_memory_d5` | list | the d=5 memory DEM (scoring layout) |
| `decoder_config_memory` | obj | `{"decoder_type": ..., "options": {}}` (`options` optional/empty in v1) |
| `self_measured_memory_logical_rate_raw` | float | your own d=5 rate per cycle, in (0, 0.5) |
| `lambda_d3_d5_raw` | float | your measured eps(d=3)/eps(d=5) per cycle, in (0.5, 20); reported, not gated |
| `memory_characterization_note` | str | bulk structure found, with evidence |
| `stage_c_mode` | str | `"surgery_characterization"` |
| `shots_for_surgery_characterization_raw` | int | shots spent on merged + CNOT experiments |
| `dem_cnot_spacetime` | list | the CNOT superset DEM |
| `decoder_config_cnot` | obj | `{"decoder_type": ..., "options": {}}` (`options` optional/empty in v1) |
| `self_measured_cnot_error_raw` | float | your own battery-averaged CNOT error, in (0, 0.5) |
| `surgery_characterization_note` | str | what merged-layout data showed that memory data did not |
| `stage_d_mode` | str | `"submit_decoders"` |

Optional: `dem_memory_d3` (a d=3 DEM behind your lambda; state the patch),
`budget_split_note`, `correlation_evidence`, `truth_table_evidence`,
`f_bell_logical_raw` (your own Bell-witness lower bound, consistency-checked, never
gated).

Before the final submission, qsim evidence must show completed, polled d=5 memory evidence at
the public 12-round scoring protocol and the complete 8-entry CNOT battery (`z`/`x` ×
`00`/`01`/`10`/`11`), also completed and polled, with at least 500 shots accumulated for each
required scientific configuration. Those shots may be split across multiple jobs; only jobs
with an ordered submission, result, and completed poll before the final answer count. Other
memory round counts remain valid characterization experiments but do not satisfy this scoring
evidence minimum. CNOT runs themselves may provide the merged-window
characterization; no dedicated
`run_merged_memory`, Bell, or d=3 job is a hidden requirement. The two reported characterization
shot fields must be positive and sum to at most the public cap. The qsim budget meter is
authoritative; a mismatch with the reported split is flagged, not secretly hard-failed.

## Scoring (pass/fail, by replay on fresh shots)

The verifier reconstructs BOTH DEMs + decoder configs and **replays them on fresh
hidden-noise shots** (independent RNG): ~4e5 memory shots at the fixed scoring layout,
~1e5 fresh shots per battery entry (8 entries: 4 Z-basis + 4 X-basis logical inputs,
frame-corrected two-bit outcome vs the ideal CNOT truth table). **Pass iff the memory
gate AND the CNOT gate (average + per-entry cap) both clear.** It also derives
`Lambda(3->5)` and a logical Bell-fidelity witness lower bound
`F_bell >= (<XX> + <ZZ>)/2` from the bell configurations decoded with YOUR CNOT DEM —
reported in the score report, never gated. Self-reported rates are consistency-checked,
never trusted; a decoder overfit to your own shots fails on fresh ones.
