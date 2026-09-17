# Tunable-coupler CZ via a net-zero flux pulse, through a realistic control stack

You operate two transmons (`q1` flux-tunable, `q2` fixed) coupled by a tunable
coupler, via an MCP server. Build a high-fidelity **controlled-Z** using a
**net-zero adiabatic flux pulse** on the `|11> <-> |02>` avoided crossing — and
make it work *through the flux line's distortion*.

You **program DAC samples** for `q1.flux(t)` on the AWG grid (0.4 ns); the flux the
qubit actually sees is your programmed waveform **distorted by the flux line**
(short-time settling). You also set a **constant coupler bias** (it sets both the
effective coupling `J_eff` and the static `ZZ`) and may bracket the flux pulse with
single-qubit microwave rotations (state prep + pre-measure tomography). Readout is
level-resolved per qubit (`0`/`1`/`2`; `2` = leakage).

The lab notebook carries the calibration values and operating notes accepted at
the previous cooldown; the device has been thermally cycled since that entry was
written, and none of the entries have been re-measured on the current cooldown.

## MCP tools

- `list_devices()` / `get_device_spec(device_id)` — device id is `tunable_coupler_cz_v0`.
- `get_lab_notebook(device_id)` — the last recorded calibration + operating notes.
- `run_flux_pulse(device_id, coupler_flux, programmed_flux_q1=[], sample_dt_ns=0.4,
  prep_ops=[], post_ops=[], idle_ns=0.0, measure_qubits=[1,2], shots=4096)` —
  programs a q1.flux DAC waveform at a fixed coupler bias and reads out.
  `prep_ops`/`post_ops` are `{"qubit":1|2,"axis":"x"|"y"|"z","angle_rad":<float>}`.
  Returns a `job_id`; poll `get_job_result`. Useful experiments: `|11>-|02>`
  chevron (prep `|11>`, scan the pulse, watch q2 → level 2); conditional-phase
  Ramsey (prep q1 in `|+>` via `ry(pi/2)`, post `ry(-pi/2)`); static-ZZ (prep
  `|+>|+>`, `idle_ns`); cryoscope (truncated pulse + Ramsey on the quadratic arc);
  CZ verification (computational prep + tomography).
- `get_job_result(job_id)` / `submit_final_answer(task_id, answer)`.

All `run_*` tools are async: submit, then poll `get_job_result`.

## Three stages

**A. Characterize** the device — recover the `q2` anharmonicity and the `|11>-|02>`
crossing, the effective coupling `J` / gap (`2 sqrt(2) J`), and the coupler
`J_eff(Phi_c)` / `zeta(Phi_c)` curves — AND the flux-line short-time **settling**
(a cryoscope: Ramsey vs flux-pulse duration; invert the quadratic sweet-spot arc).

> **Two separate requirements on the settling.** The flux line's unit-step
> response settles as a sum of discrete first-order poles on the 0.4 ns AWG
> grid: `s[n] = 1 - sum_k A_k (1 - alpha_k)^n` with `alpha_k = 0.4/(tau_k + 0.4)`,
> where `n = 1, 2, ...` counts AWG samples after the step edge — the **first
> realized sample after an edge is `s[1]`**, not `s[0]` (`tau_k` in ns, in
> exactly this discrete convention — convert before comparing to any
> continuous-time value). Report the amplitudes `A_k` and time constants
> `tau_k` you recover, as two equal-length lists; each parameter is graded to
> **10% relative** (pole order does not matter). The predistortion residual
> below is a *different* criterion: it is graded on the waveform you actually
> program, so it scores the correction you applied, not the numbers you quoted.
> Budget shots and choose an estimator accordingly.

**B. Design** the intended **net-zero** `q1.flux` trajectory (conditional phase
`pi`, minimal leakage to `|2>`) AND a **predistortion** so the PROGRAMMED DAC
waveform, after the line, realizes the intended flux (mind the DAC bit depth). Park
the coupler **idle** bias where BOTH `J_eff = 0` and `zeta ~ 0`, and pick the
**gate** bias. Report the single-qubit **virtual-Z** corrections.

**C. Verify** on the simulator: conditional phase, leakage, fidelity, the net-zero
robustness advantage under an injected flux offset, the residual idle `ZZ`, and the
predistortion residual.

## Required final answer

```json
{
  "task_id": "tunable_coupler_cz_netzero",
  "method": "<free narrative>",
  "stage_a": {
    "alpha_2_mhz_raw": <float, negative>, "alpha_2_mhz_1sigma": <float>,
    "coupling_j_mhz_raw": <float, |J_eff| AT your stage_b.coupler_gate_flux>,
    "coupling_j_mhz_1sigma": <float>,
    "avoided_crossing_gap_mhz_raw": <float>,
    "crossing_flux_q1_raw": <float, Phi_0>,
    "coupler_decoupling_flux_raw": <float, Phi_0>,
    "idle_zz_khz_raw": <float>, "idle_zz_khz_1sigma": <float>,
    "short_settling_amplitudes_raw": [<float>, ...],
    "short_settling_tau_ns_raw": [<float>, ...],
    "characterization_method": "chevron_coupler_sweep_plus_cryoscope"
  },
  "stage_b": {
    "coupler_idle_flux": <float, Phi_0>, "coupler_gate_flux": <float, Phi_0>,
    "pulse_family": "fast_adiabatic" | "snz" | "net_zero_smooth",
    "intended_flux_samples": [<Phi_0>, ...],
    "programmed_dac_samples": [<Phi_0>, ...],
    "dac_bits": <int <= 16>, "sample_dt_ns": 0.4, "gate_time_ns": <float in [36,100]>,
    "net_zero_satisfied": <bool>,
    "virtual_z_q1_rad": <float>, "virtual_z_q2_rad": <float>,
    "predicted_conditional_phase_rad": <float>, "predicted_leakage_raw": <float>,
    "idle_zz_handling": "coupler_null"
  },
  "stage_c": {
    "measured_conditional_phase_rad_raw": <float>, "measured_conditional_phase_rad_1sigma": <float>,
    "measured_leakage_raw": <float>, "measured_leakage_1sigma": <float>,
    "gate_infidelity_raw": <float>, "gate_infidelity_1sigma": <float>,
    "predistortion_residual_raw": <float>,
    "measured_idle_zz_khz_raw": <float>, "measured_idle_zz_khz_1sigma": <float>,
    "verification_protocol": "conditional_phase_ramsey",
    "leakage_input_set": "four_comp_basis"
  }
}
```

The verifier convolves your `programmed_dac_samples` through the hidden flux line
to the realized flux, then replays the gate on the hidden two-transmon model. It
binds the conditional phase (after YOUR reported virtual-Z), leakage, fidelity,
the net-zero advantage, the idle ZZ (at YOUR coupler idle bias), and the
predistortion residual; the `intended` trajectory is used to check net-zero and
the predistortion residual.

Submission format constraints (violations are rejected without physics scoring):
every number in a **scored** field must be an ordinary finite number (no
`NaN`/`Infinity`, no integers beyond float range);
`programmed_dac_samples` must have exactly the same length as
`intended_flux_samples` (predistortion is length-preserving — if you want a
settle-back tail, include it in both), with at most 300 samples;
`sample_dt_ns` must be `0.4` (the AWG cannot resample; omitting the field is
read as `0.4`, its only legal value); `coupler_idle_flux`,
`coupler_gate_flux`, and the Stage-A `alpha_2_mhz_raw` / `coupling_j_mhz_raw`
fields are required.

**Scored fields:** `stage_a.alpha_2_mhz_raw`, `stage_a.coupling_j_mhz_raw`,
`stage_a.coupler_decoupling_flux_raw`, `stage_a.short_settling_amplitudes_raw`
+ `stage_a.short_settling_tau_ns_raw`, `stage_b.coupler_idle_flux`,
`stage_b.coupler_gate_flux`, `stage_b.intended_flux_samples`,
`stage_b.programmed_dac_samples`, `stage_b.sample_dt_ns`,
`stage_b.virtual_z_q1_rad`, `stage_b.virtual_z_q2_rad`. Every other field —
all of `stage_c`, and the remaining `stage_a`/`stage_b` entries — is **reported
for the record**: it is kept for human review of your method and does not enter
automated scoring. Every graded quantity is recomputed by the replay from the
scored fields above; self-reported `measured_*` values are never used as
grading inputs.

`coupling_j_mhz_raw` is likewise graded at **your own declared
`coupler_gate_flux`**, not at any reference bias: report the effective coupling
you measured *at the bias you run the gate at*. `J_eff` varies with the coupler
bias, so a value measured at a different bias is not the value being graded.

Scoring also requires real execution evidence: at least **15 completed
`run_flux_pulse` jobs** in the experiment log (characterization, design
iteration, and verification all take real experiments). Submissions below this
floor are rejected without scoring.

## Success criteria

Conditional phase within `±0.05 rad` of `pi` (after virtual-Z); leakage `< 1e-2`;
infidelity `< 1.5e-2`; net-zero robustness `>= 3x` better than a unipolar baseline;
idle `|zeta| < 10 kHz`; predistortion residual `< 1e-2`; every reported flux-line
settling parameter (`A_k` and `tau_k`) within `10%`; plus the Stage-A device
values recovered from your own measurements, and the execution-evidence floor
(>= 15 completed `run_flux_pulse` jobs).

## Budget

Wall clock 9000 s.
