# Surrogate-model a hidden 60-qubit bounded-gate circuit under budget

You are operating a black-box quantum device through the MCP tools listed
below. A fixed hidden 60-qubit circuit U(x) sits behind the measurement tool:
it is built from Clifford gates plus **at most 18** single-qubit non-Clifford
rotations driven by a 6-component input vector x in [-pi, pi]^6, followed by
internal readout noise that does not depend on x and does not change during
the run. The circuit itself is the same for every job in this run.

The public response class is disclosed in
`/task_materials/observable_index.json` (schema 6). The circuit contains at
most 18 parameterized single-qubit non-Clifford rotations driven by the six
input angles. For each input angle, every scored correlation C_ij(x) has
**harmonic degree at most 3** (an independently disclosed device property, not
a consequence of the rotation count), so the responses belong to a
finite-dimensional multivariate trigonometric function class. In the standard
complex Fourier expansion

    C_ij(x) = sum_m a_ij,m * exp(i * m dot x),

each integer frequency vector satisfies -3 <= m_k <= 3, and each scored
response contains at most **64 nonzero modes**. Each distinct frequency vector
counts once; conjugate vectors count separately. The active frequency vectors
and coefficients are hidden and may differ across qubit pairs. qsim keeps the
circuit topology, rotation placements and axes, noise parameters, pair supports, exact-zero
status, coefficients, and target values hidden. `observable_index.json`
contains no feature list and no pair-to-support mapping. Choosing the
experiment design, measurement allocation, estimator, and fitted surrogate is
the task.

## Goal

Predict the two-point correlation

    C_ij(x) = < (X_i X_j + Y_i Y_j + Z_i Z_j) / 3 >

for every unordered qubit pair 0 <= i < j < 60 at 40 held-out inputs. The
scored value is the device's **as-measured** expectation, including the
device's own readout errors: do not apply readout or error mitigation to the
reported predictions. Columns follow the exact lexicographic order in
`/task_materials/observable_index.json` (1770 pairs).

The device precommits to the held-out inputs before your first measurement.
`/task_materials/target_challenge.json` gives their count, domain, disclosed
sampling strata, and commitment scheme, but not their coordinates. After you
finish submitting experiments, call
`lock_measurements_and_reveal_challenge`. That call closes measurement
admission for the rest of the run and returns the ordered 40 x 6 input matrix
plus its commitment opening. You may continue polling jobs that qsim admitted
before the lock and may compute for the rest of the wall-clock budget. A
repeated reveal call returns the same opening and sealed counters, marked as a
repeat. No experiment submitted after the lock can consume budget or support
the answer.

The ordered challenge rows are concatenated in the order listed in
`target_challenge.json`: 16 `uniform_torus`, then 8 `near_boundary`, then 8
`diagonal_off_axis`, then four adjacent base/mate pairs (8 rows) from
`sign_related_pairs`. Commitment objects use the UTF-8 bytes produced by
Python 3.11 `json.dumps(value, allow_nan=False, ensure_ascii=False,
separators=(",", ":"), sort_keys=True)`. In the published domain prefixes,
`\0` denotes one NUL byte (`0x00`), not the two characters backslash and zero.

The observable equals (2*SWAP - I)/3, whose spectrum is {-1, +1/3}, so every
true C_ij lies in [-1, 1/3]. The wider scored band below is only a numerical
guard around that physical interval.

## Tools (MCP server `qsim`, `http://qsim:8123/mcp`)

- `get_device_spec(device_id)`: public device spec with the qubit count, input
  dimension, rotation and harmonic bounds, and all budget caps. Device id:
  `boundedgate_60q_v0`.
- `get_lab_notebook(device_id)`: the task's versioned public characterization
  memo.
- `run_basis_shots(device_id, blocks)`: batched measurement job. Each block
  is `{"x": [6 floats in [-pi, pi]], "settings": [{"basis": ..., "shots": n}]}`;
  a basis is either a fixed length-60 string over `x/y/z` (per-qubit
  measurement bases) or `"random_pauli"` (a fresh uniformly random per-qubit
  basis for every shot, returned with the outcomes). Returns a `job_id`
  immediately.
- `get_job_result(job_id)`: poll until `complete`. Results carry raw
  per-shot outcome bitstrings (character i = qubit i; outcome bit b means the
  measured eigenvalue was (-1)^b). Large results are offloaded to
  `/qsim_logs/public_job_results/results/<job_id>.json`; a `public_warnings` entry in
  the result tells you when.
- `lock_measurements_and_reveal_challenge(device_id)`: synchronously and
  irreversibly close new measurement admission, then return the ordered target
  inputs, their precommitment opening, and the sealed shot/job counters. Call
  this after your final experiment submission. Already-admitted jobs may still
  finish and be polled.
- `submit_final_answer(task_id, answer=..., answer_file=..., answer_sha256=...)`:
  submit the complete answer, inline as `answer` or as a file (see the next
  paragraph). You may submit more than once (up to the cap below); if you do,
  the latest logged transport-valid submission is authoritative and replaces
  the prior artifact; the scorer does not fall back to an earlier payload.

The final answer is large (one 40 x 1770 matrix), so do not paste it into a
tool call. Build the payload programmatically with a script and submit it as a
file: write the answer dict as UTF-8 JSON to a file directly inside
`/submission` (an agent-writable directory in your container -- use a plain
filename such as `answer.json`, no subdirectory), then call
`submit_final_answer` with `answer_file` set to that filename and `answer_sha256`
set to the SHA-256 of the exact file bytes. The server reads the file, checks
the hash, and records its content under the same budget and validity rules as an
inline answer, so the payload never has to pass through your context. The `mcp`
Python package is available if you drive the call from an MCP client script.
Passing the answer inline with `answer` is still accepted for smaller payloads.

## Budget (enforced server-side, cumulative over the whole run)

- **315,000 total measured shots** and **40 jobs**; per job at most 64 blocks,
  192 settings total across all blocks (there is no separate per-block limit:
  one block may carry all 192), and 4,000 shots per setting.
- At most **256** accepted public-metadata calls (`list_devices`,
  `get_device_spec`, and `get_lab_notebook` combined), **4,096** accepted
  `get_job_result` polls for jobs from this task, **4** sealed-holdout reveal
  calls, and **4** transport-valid final-answer submissions. Each MCP POST
  body is at most **8 MiB**. The complete
  compact-JSON `FinalAnswer` envelope is at most **4 MiB**, and every string
  or object key inside `answer` is at most 16,384 characters.
- Malformed or over-budget requests are rejected atomically and consume
  nothing. Every completed basis-shot result echoes the authoritative
  `shots_used` / `jobs_used`.
- Wall clock: 1800 s, enforced by the harness. Only a submission that qsim has
  accepted before the clock expires can be scored; you cannot buy more shots
  or time. A run with no accepted submission scores zero regardless of the
  model you hold; if at least one measurement job completed, it is recorded
  as a scientific failure of this task (every gate fails with the reason "no
  prediction was submitted within the wall clock"), not as a harness event.
  You may submit early and replace the submission later.
  Simulating a job takes real compute time that scales with the number of
  distinct inputs it contains. Every completed result's `budget` and the
  reveal receipt echo `elapsed_wall_clock_s`, the device's own clock in
  seconds since it started (it started shortly before your first call), so
  you can read the time the same way you read `shots_used`.

## Required final answer

`submit_final_answer(task_id="time_budgeted_shadow_surrogate_60q", answer={...})`
with these fields (or, for this large answer, the file form above:
`submit_final_answer(task_id="time_budgeted_shadow_surrogate_60q",
answer_file="answer.json", answer_sha256="...")`, where the file holds exactly
that same `answer` object). The prediction and reveal-citation fields are
scored; the rest are recorded for human review and analysis, are optional, and
never affect the verdict.

| field | type | semantics |
|---|---|---|
| `predictions_as_measured` | list[list[float]] | shape [40][1770]; as-measured values, no mitigation; rows follow the reveal receipt order, columns follow `observable_index.json` pair order |
| `challenge_id` | string | copy the identifier from the reveal receipt |
| `target_inputs_sha256` | string | copy the ordered-target digest from the reveal receipt |
| `method` | string | human-readable layer: your design, estimator, and model selection |
| `protocol` | string | informational; free-form, your own name for the measurement protocol |
| `surrogate_model` | string | informational; free-form, your own name for the model class you fitted |
| `shots_used`, `jobs_used` | int | informational; your own count of the device's accounting |
| `validation_rmse` | float | informational; your own held-out validation RMSE |

## Scoring: all gates disclosed, all must pass

Your submission is scored against exact hidden values of C_ij at the target
inputs:

- **G1 accuracy**: overall RMSE over all 40 x 1770 predictions **<= 0.0025**,
  AND the worst single target's RMSE (over its 1770 pairs) **<= 0.0035**.
  An error burst on one target cannot hide in the 40-target average.
- **G2 tail**: the largest absolute error of any single prediction
  **<= 0.04**. A large localized error cannot hide in the 70,800-cell
  averages.
- **G3 validity**: exact prediction shape, predictions within **[-1.05, 0.35]** (a
  numerical guard band around the observable's spectrum {-1, +1/3}), and a
  well-formed experiment lifecycle: poll
  at least one measured job through `get_job_result` to `complete`; lock
  measurement admission and reveal the committed challenge before submitting;
  cite that reveal's `challenge_id` and `target_inputs_sha256`; and make the
  latest logged transport-valid submission the payload you want scored. Only
  jobs admitted before the reveal and evidence available before that
  submission count. The informational fields are never checked against the
  device's counters sealed in the reveal receipt, and unknown extra fields are
  ignored.

There is no partial credit.
