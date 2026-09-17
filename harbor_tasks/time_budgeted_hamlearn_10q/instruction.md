# Time-budgeted Hamiltonian learning (10-qubit black-box analog device)

You are operating a 10-qubit analog quantum device whose dynamics are governed
by an unknown, time-independent Hamiltonian

```
H = (1/2) sum_P omega_P * P
```

where each `P` is a Pauli string drawn from a **public** 111-term dictionary and
each `omega_P` (in rad/µs) is **hidden**. Your job is to estimate the full
111-length coefficient vector as accurately as possible. Passing requires both
the published worst-coefficient (L∞) and relative full-vector (L2) accuracy gates.

The public hypothesis class contains 1-local terms (`Xi`, `Yi`, `Zi`) and
nearest-neighbor 2-local terms on the line 0–1–2–…–9, with the public spec's
declared upper bound on the number of nonzero coefficients. The dictionary, the
conventions, and all probe constraints are public.

## Surfaces

- **MCP server `qsim`** (`http://qsim:8123/mcp`) — the only surface. Tools:
  - `list_devices()` — list the available public device identifiers.
  - `get_device_spec(device_id)` — the public spec **including the full 111-term
    dictionary** (`terms`: each `{name, pauli, weight}`), the time grid, the
    observable-weight bounds, the budget, and the conventions.
  - `get_lab_notebook(device_id)` — a **stale** prior guess of some coefficients.
    It may be wrong, incomplete, or mislabeled. Use it only as a weak hint.
  - `run_hamiltonian_probe_batch(device_id, rows)` — the experiment primitive
    (below). Returns a `job_id` immediately.
  - `get_job_result(job_id)` — poll until `status` is `"complete"` or `"failed"`.
  - `submit_final_answer(task_id, answer=..., answer_file=..., answer_sha256=...)`
    — submit when done, either inline or from a file (resubmission is allowed
    but capped; the LAST accepted submission is the one scored).

The device id is `hamlearn_10q_chain_v1`; the task id is
`time_budgeted_hamlearn_10q`. The same public materials are also mounted
read-only at `/task_materials/` (`hamiltonian_dictionary.json`,
`public_spec.yaml`, `stale_notebook.yaml`).

The final answer contains two 111-value vectors, so assemble and validate it
programmatically. You may submit the answer dict inline, or write it as UTF-8
JSON to a plain filename directly under the agent-writable `/submission`
directory and call `submit_final_answer` with that `answer_file` filename plus
the SHA-256 of the exact file bytes in `answer_sha256`. Qsim reads the file,
checks the digest, and records the same answer evidence under the same limits as
the inline form. Do not use a subdirectory. The `mcp` Python package is
available if you drive the call from a client script.

## The probe primitive

`run_hamiltonian_probe_batch(device_id, rows)` submits a batch of probes. Each
row is:

```json
{"initial_state": ["z+","z-","x+","x-","y+","y-", ...],  // one label per qubit (length 10)
 "evolve_time_us": 0.10,                                  // on the time grid
 "observable_pauli": "XIIIIIIIII"}                        // length-10 I/X/Y/Z string
```

For each accepted row the device prepares the product state, evolves it under
`U(t) = exp(-i t H)`, measures the Pauli observable, and returns
**raw ±1 outcomes** (a fixed, public number of internal repetitions per probe —
see `num_internal_repetitions`). The device returns shots, never an exact
expectation: you must average and account for shot noise yourself.

Conditional on the exact expectation `mu = <O>_t`, each repetition is an
independent projective outcome with `Pr(+1) = (1 + mu)/2` and
`Pr(-1) = (1 - mu)/2`. The mean of `N` outcomes is unbiased with standard
deviation `sqrt((1 - mu^2)/N)`. Thus the public nominal noise value near
`0.031` is the largest standard deviation, attained near `mu = 0`; it is not a
constant Gaussian noise level for every row.

Result data (`kind="probe_outcome"`) carries, per row, `status`
(`"accepted"`/`"rejected"`), `raw_pauli_outcomes` (list of ±1) and
`num_internal_repetitions` for accepted rows, or a `reject_reason`. It also
reports the cumulative `budget_used_us`, `budget_remaining_us`, and
`accepted_row_count`.

### How completed results are delivered

Large raw-shot batches exceed the inline result-size cap. For such a batch, a
completed `get_job_result` response contains `data: null`, a warning naming the
file, and a machine-readable `metadata.result_artifact` pointer with
`relative_path`, `size_bytes`, `sha256`, and `result_kind`. Qsim writes the
complete result JSON to `/qsim_logs/<relative_path>`, which is mounted read-only
in your container. Read that file with a script and verify its size and SHA-256
before using its populated `data`. Small results remain inline.

Poll every submitted probe job to `"complete"` or `"failed"` before submitting
your final answer. The LAST accepted final submission closes the evidence
record; probe submissions, probe completions, or result polls after it make
the run invalid.

### Working with the data

The raw-shot payloads are large (one batch can be tens of thousands of ±1
values), so plan to process them **programmatically**, not by hand. The `qsim`
server is a standard MCP **streamable-http** endpoint reachable from your
environment, so the practical workflow is to drive `run_hamiltonian_probe_batch`
/ `get_job_result` from a local script and pull the outcomes straight into a
numerical tool (e.g. `numpy`/`scipy`) for averaging and fitting. Averaging a
row's `raw_pauli_outcomes` gives your noisy estimate of `<O>_t`; the public
`nominal_single_probe_noise_1sigma` is its near-zero-expectation upper scale.
However you call it, qsim owns the budget and the evidence — the cumulative
`budget_used_us` it returns is authoritative, so track it rather than your own
local count.

### Constraints (from `get_device_spec`)

- **Time grid**: `evolve_time_us` must lie in `evolve_time_range_us` and be a
  multiple of `evolve_time_resolution_us`. Off-grid or out-of-range times are
  **rejected** (they consume zero budget and return no outcomes — not rounded).
  The current minimum measurable time is `0.010 µs`, and the finite grid
  resolution is `0.002 µs`; these are separate device constraints.
- **Observable weight**: the number of non-identity factors must be within
  `allowed_observable_weight`.
- **Initial state**: each qubit label must be one of `allowed_initial_states`.
- **Batch size**: at most `max_rows_per_batch` rows per call (a larger batch
  fails as a whole).
- **Accepted-row cap**: at most `max_probe_rows` accepted probes over the run.
- **Ingress/submission caps** (`budget` in the public spec — generous
  DoS-hardening limits, far above any sensible protocol): each MCP request
  body ≤ `max_ingress_request_bytes`; the serialized final-answer envelope
  ≤ `max_final_answer_serialized_bytes`, with per-string length ≤
  `max_answer_string_characters` and nesting depth ≤
  `max_final_answer_nesting_depth`; at most `max_final_answer_submissions`
  accepted final submissions (the last one is scored) and
  `max_job_result_polls` result polls and `max_probe_jobs` probe jobs per run.
  Requests over a cap are
  rejected with a normal tool error and change nothing.

### The budget — your scarce resource

The normalized **accepted interrogation-time budget** is capped at
`total_evolution_time_budget_us`. It **accumulates across every batch** in the
run. Each accepted row charges its `evolve_time_us` once; because every row has
the same fixed internal repetition count, this is a constant normalization of
the corresponding physical shot-weighted evolution time. Once exhausted,
further rows are rejected (`budget_exhausted`). You may distribute the budget
across any valid probe states, observables, and evolution times; neither the
time bounds nor this instruction prescribe a preferred allocation. The device
owns and enforces the budget, and the scorer uses its accounting instead of
your self-report.

## Conventions

- **Hamiltonian / time evolution**: `U(t) = exp(-i (t/2) sum_P omega_P P)`.
  (Note the factor of 1/2.) Coefficients are in rad/µs.
- **Term order** (`term_order_convention`):
  `one_local_xyz_then_edge_local_xx_xy_xz_yx_yy_yz_zx_zy_zz` — the 30 one-local
  terms `X0,Y0,Z0,…,X9,Y9,Z9` first, then for each edge `(i,i+1)` the nine
  two-local pairs `XX,XY,XZ,YX,YY,YZ,ZX,ZY,ZZ`. Your answer vector must follow
  this order exactly. `get_device_spec().terms` is already in this order.
- **Pauli strings**: length-10 `I/X/Y/Z`, qubit 0 is the leftmost character.

## Final answer

Call `submit_final_answer("time_budgeted_hamlearn_10q", answer=answer)` when
done, or use the `answer_file` form above for a programmatically assembled
payload.
If you submit more than once (up to `max_final_answer_submissions`), the
last accepted submission is scored. Required
fields:

| Field | Meaning |
|---|---|
| `task_id` | `"time_budgeted_hamlearn_10q"` |
| `hamiltonian_convention` | `"U(t)=exp(-i t/2 sum omega_P P)"` |
| `term_order_convention` | `"one_local_xyz_then_edge_local_xx_xy_xz_yx_yy_yz_zx_zy_zz"` |
| `num_terms` | `111` |
| `omega_vector_raw_rad_per_us` | length-111 estimated coefficients (term order) |
| `omega_vector_raw_rad_per_us_1sigma` | length-111 per-coefficient 1σ (each in `(0, 0.30]`) |
| `activity_summary` | the required two-layer activity record below |

`activity_summary` must have exactly this shape. The zero values below are
placeholders and are not valid activity claims; replace them with your own qsim
evidence:

```json
{
  "schema_version": 2,
  "verifiable": {
    "accepted_probe_rows": 0,
    "total_accepted_evolution_time_us": 0.0,
    "min_accepted_evolve_time_us": 0.0,
    "max_accepted_evolve_time_us": 0.0
  },
  "trajectory": {
    "protocol": "other",
    "fit_model": "other",
    "trajectory_details": "Chronological human-readable account of probe design, execution rounds, analysis, any adaptations or refinement, and anomalies."
  }
}
```

The two layers have different authority:

- **`verifiable` is scored mechanically.** Report qsim's accepted-row count,
  cumulative accepted evolution time, and accepted probe-time range. The
  verifier recomputes all four values from artifact-verified qsim evidence;
  any mismatch fails binary scoring.
- **`trajectory` is for human audit.** `protocol` and `fit_model` use the enums
  below. `trajectory_details` is a nonblank chronological narrative of what
  you tried, whether and how data or diagnostics changed your plan, how you
  fit/refined the estimate, and any failed or anomalous steps. It is preserved
  for review but is not interpreted as evidence and does not affect the numerical score.
  Do not paste raw shot arrays into it; cite aggregate observations instead.

Allowed `protocol` values: `fixed_design`, `adaptive_design`, `mixed_design`,
`other`.

Allowed `fit_model` values: `linear`, `nonlinear`, `probabilistic`, `hybrid`,
`other`. These enums classify the completed trajectory for review; they do not
recommend an experiment design or estimator.

Optional: `omega_vector_refined_rad_per_us` (+ its `_1sigma`) — a refined
estimate (scored **only** if both the refined vector and its σ are present),
`support_terms_detected`, `linf_self_estimate_rad_per_us`,
`notebook_model_rejected`, `notebook_rejection_reason`,
`heldout_probe_error_summary`.

Each coefficient must lie in `[-0.75, 0.75]`.

## Scoring

You pass only if both coefficient-error gates hold:

- worst-case error `L∞ = max_P |ω̂_P − ω*_P| ≤ 0.085 rad/µs`; and
- aggregate error `||ω̂ − ω*||₂ / ||ω*||₂ ≤ 0.12`.

Under the declared convention `H = (1/2) Σ_P ω_P P`, the Pauli strings are
Hilbert-Schmidt orthogonal, so the second quantity is exactly the relative
Frobenius error of the reconstructed Hamiltonian. It prevents many individually
small coefficient errors from accumulating into a poor full Hamiltonian.

You must also stay within the evolution-time budget, submit a structurally
valid answer, poll every probe job to a terminal state, and make the submitted
activity claims match the artifact-verified qsim evidence. The coefficient
checks apply to the vector that is actually scored (the refined vector when
both refined fields are present, otherwise the raw vector).

**Raw-data compatibility — diagnostic.** The verifier computes the exact
finite-time expectation of every executed row under your submitted Hamiltonian,
then compares it with that row's realized raw ±1 outcomes using binomial
deviance. It reports the mean and 90th-percentile row deviances, together with
the historical reference cutoffs `3.0` and `6.0`, for audit only. These
statistics do not enter binary pass: the probe design may be adaptive and the
same outcomes may be used to fit the submitted vector, so they are not an
independent calibrated goodness-of-fit test.

Every probe job must be polled to a terminal state, and only completed,
artifact-verified accepted rows enter this calculation. It neither uses nor
fixes the runtime random seed. Notebook values receive no special treatment:
the submitted vector must satisfy the same coefficient gates.
Tighter worst-coefficient errors earn higher quality tiers
(bronze/silver/gold), but a tier never overrides another binary gate.

Reported per-coefficient 1σ values are retained as method metadata. The score
report includes their median and a single-realization 2σ coverage count for
diagnosis, but neither enters binary pass or the coefficient-error quality
tier. Calibration would require an ensemble of independent attempts or task
instances; it is not inferred from the 111 correlated coefficient errors in
this one fixed instance.

The `activity_summary.verifiable` layer is evidence-bound, not free-form. Its
accepted-row count, cumulative evolution time, and min/max accepted probe times
must match the artifact-verified qsim record within `1e-6 µs`. Its trajectory
layer cannot repair or override a mechanical mismatch.
