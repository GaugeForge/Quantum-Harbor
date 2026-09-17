# Adaptive composite measurement design for H₂O

You operate a quantum measurement device through an MCP server (`qsim`). A fixed
14-qubit Pauli Hamiltonian (H₂O / STO-3G, Jordan–Wigner) and one fixed, unknown
("black-box") prepared state are given. Pauli strings are length 14, leftmost
character acts on q0, rightmost on q13; returned bitstrings use the same order.
The hidden state is fixed, but every benchmark attempt receives fresh random
pilot and production data. Pilot outcomes, production-basis draws, and
production outcomes use independent random streams; no runtime seed is exposed.

The public materials are mounted read-only at `/task_materials`:

- `h2o_sto3g_jw_pauli.csv` — the authoritative Hamiltonian (`term_index`,
  `pauli_q0_to_q13`, `coefficient_hartree`); 1086 terms (one identity + 1085
  non-identity).
- `stale_notebook.json` — a **stale** lab notebook. It contains a usable
  16-component warm-start C-LBCS scheme (`track_a_warm_start`) but its
  statistical advice is wrong; re-derive anything statistical yourself.

## Background

A C-LBCS (composite locally-biased classical shadow) scheme has mixture weights
`r_k` (k = 1…16, `r_k ≥ 0`, Σ r_k = 1) and, for each component k and qubit i, a
local basis distribution `β_i^k = (β_i^k(X), β_i^k(Y), β_i^k(Z))` (each summing to
1). One measurement basis `Q` is drawn by sampling a component `k ∼ r` then each
qubit's axis `∼ β_i^k`. A Hamiltonian term `P_j` is *covered* by `Q` when every
non-identity character of `P_j` equals `Q` at that qubit; its coverage probability
is `h_j = Σ_k r_k Π_{i: P_j[i]≠I} β_i^k(P_j[i])`.

## Tools (qsim MCP)

- `get_observable_spec(device_id)` — conventions, budgets, and the pinned
  estimator formula. (Synchronous.)
- `evaluate_composite_scheme(device_id, mixture_weights, local_basis_probabilities_xyz)`
  — returns the exact **public** state-independent average one-shot variance
  `V_Haar = (2¹⁴/(2¹⁴+1)) Σ_{j: P_j≠I} c_j²/h_j` plus coverage quantiles. `mixture_weights`
  is length 16; `local_basis_probabilities_xyz` is shape `[16][14][3]` with last
  axis order **X, Y, Z**. Synchronous, capped at 40 calls. Returns no hidden-state
  quantities.
- `run_pilot_measurements(device_id, settings)` — run product-Pauli pilot
  settings on the hidden state. `settings` is a list of
  `{"basis": <14-char XYZ string>, "shots": <multiple of 64 in [64,512]>}`.
  Async (returns `{job_id, status}`; poll `get_job_result`). Budget accumulates
  across calls: ≤ 96 settings and ≤ 24576 shots total.
- `execute_locked_composite_scheme(device_id, mixture_weights, local_basis_probabilities_xyz, control_variate_entries)`
  — lock a Track-B scheme + sparse control variates and run production **once**.
  `control_variate_entries` is a list of `{"term_index": <int>, "mean": <float in
  [-1,1]>}`, ≤ 192 entries, no duplicate indices, identity index 0 forbidden.
  Coverage (`h_j ≥ 1e-9` for all terms) is validated at lock time and **fails
  closed** before consuming the single production run; a failed lock counts
  against a small lock-attempt budget. Async; the completed result has 192
  independent bases × 256 shots of raw counts plus a `request_digest` of the
  locked design. qsim first canonicalizes the normalized probability law; that
  one digest-bearing law is used consistently for basis sampling and verifier
  coverage probabilities, without clipping small probabilities.
- `get_job_result(job_id)`, `submit_final_answer(task_id="adaptive_clustered_clbcs_h2o", answer={...})`
  — always available.

## Your task

**Track A.** Submit a 16-component C-LBCS scheme that minimizes the public
`V_Haar`.

**Track B.** You may use up to the available pilot budget to learn about the
hidden state. Then lock a *new* 16-component scheme and up to 192 sparse control
variates, and run production once. Estimate the total energy with the **pinned inverse-coverage
control-variate estimator**

```
E_hat = c_I + (1/(B·R)) Σ_b Σ_r [ Σ_j c_j m_j
        + Σ_{j: P_j ⪯ Q_b} (c_j/h_j) ( μ(P_j, x_br) − m_j ) ],
   μ(P_j, x) = Π_{i: P_j[i]≠I} (−1)^{x[i]},   B = 192, R = 256,
```

In both sums over `j`, `j` ranges over the 1085 non-identity Pauli terms; the
known identity coefficient is the separate constant `c_I`. (`m_j` are your
locked control variates; unlisted means are zero). For setting
`b`, let `z_bar_b` be the mean of the bracketed estimator contribution over its
256 shots, and let `z_bar` be the mean of the 192 `z_bar_b` values. Report the
canonical same-data setting-cluster standard error

```
sigma_cluster = sqrt((1/(B*(B-1))) * Σ_b (z_bar_b - z_bar)^2).
```

The 49152 production shots are **not** 49152 independent basis draws — they are
192 clusters of 256 correlated shots; an ordinary shot-level (IID) standard
error is wrong.

Your reported energy must reproduce the pinned formula on the returned counts
within `1e-4` Hartree. Your reported `uncertainty_hartree_1sigma_cluster` must
reproduce the canonical setting-cluster standard error from those same counts
within `1e-4` Hartree.

## Final answer

Call `submit_final_answer(task_id="adaptive_clustered_clbcs_h2o", answer={...})` where
the `answer` object contains:

```jsonc
{
  "track_a": {
    "mixture_weights": [ /* 16 */ ],
    "local_basis_probabilities_xyz": [ /* [16][14][3], X,Y,Z */ ]
  },
  "track_b": {
    "request_digest": "<the digest returned by execute_locked_composite_scheme>"
  },
  "track_b_result": {
    "energy_raw_hartree": <float>,
    "uncertainty_hartree_1sigma_cluster": <float>
  }
}
```

The outer `submit_final_answer` call already binds the task. qsim's completed
production evidence is authoritative for the Hamiltonian, conventions, locked
Track-B scheme, and control variates, so do not copy those fields into the
answer. You may include a concise `method_summary` or other metadata for later
human audit; all such metadata is optional and unscored.

The required uncertainty number is still the canonical setting-cluster
standard error above. A shot-IID value does not become valid because of a method
label: the verifier recomputes the cluster statistic from the same counts and
requires the numerical readback to match.

## How you are scored

The verifier **recomputes** every scored quantity from your completed qsim
evidence — it does not trust your reported scalars:

- Track A: the exact `V_Haar` of your submitted scheme.
- Track B (primary): the exact clustered private variance of your **locked**
  scheme + control variates against the hidden state.
- The production energy and cluster standard error from your returned raw counts;
  your reported values must match those same-data recomputations. The hidden
  exact energy is not used to reject an otherwise valid random tail draw.

The pass/tier thresholds are hidden, but the comparison is deterministic: the
recomputed `V_Haar` is rounded to **2 decimals** and the recomputed Track-B
`σ_private` to **4 decimals** before being compared (`≤`) to the hidden
thresholds. Both tracks reward lower values; aim to minimize, not to clear a
guessed bar.

The Track-B `request_digest` binds your result to the one locked production
design; the verifier reads and scores that design directly from qsim evidence.
Returning only an energy, omitting the Track-A scheme, failing to complete and
retrieve the locked production run, citing the wrong digest, a mismatched energy
readback, or an under- or over-inflated 1σ uncertainty all fail.

Wall-clock budget: 1800 s.
