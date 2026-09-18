# Conditional-Jeffreys native RF0 experiment

2026-09-04. Isolated, training-free research candidate; original Mass-GS
C32/g45/G256/RF0 remains the production default. No paper, production
dispatch, generation config, merge or route implementation is changed.

## Question and protocol

The preceding [CPU diagnostic](conditional_jeffreys_seed_selection.md)
found a 2.6319% improvement over Mass-GS in a mass-weighted *local*
conditional-output MSE. This experiment tests the actual complete GCA
operator, including the unchanged native merge and global compressed route.

The candidate replaces only the seed-selector callable. It retains RF0,
Uniform16 local probes, the original Uniform256 global normalizer, C32/g45
candidate geometry, a fixed anchor plus three distinct real query rows,
and four bulk cells plus the existing SQ1 singleton. It uses no learned
parameter or dense inference exception. Hellinger-RF0 and original Mass-RF0
are controls; Hellinger-RF10 is a separately labeled historical comparator,
not a change to the candidate's fixed RF0.

Two data scopes must not be pooled or confused:

1. Three canonical step-0/layer-33 captures, original heads 0/10/20/30.
   Every complete output includes all 21 query frames and all 75,600 keys.
   The reference is the original same-call 40-head forced-Flash SDPA output,
   sliced to the saved four heads. RF0 and the other 20 frames are reported
   separately. These captures do not provide full-head timing evidence.
2. Two historical layer-20 holdout calls, 020 and 100, with all 40 heads.
   Stored FP16 Q/K/V are converted to BF16 for every method and the forced
   Flash SDPA reference. Complete-call timing has three warmups followed by
   six alternating-order samples of five calls each. Selector work is
   included; observation/audit replays and compilation are excluded.

All inputs and sources are fingerprinted. Hellinger-RF0 and Mass-RF0 must
reproduce the prior L33 seed-table hashes exactly and prior MSE, per-frame
MSE and per-head MSE within 2e-6 relative plus 1e-12 absolute tolerance.
Observed and unobserved complete outputs must be bitwise equal. Candidate
rows from score-audited selection must match the actual complete call.

The parent checks and monitors one exclusively clean H200 on scai6. A
foreign process or monitor error invalidates timing. No second GPU is used.
Canonical video PSNR, SSIM, LPIPS and VBench are not measured by this
captured-QKV experiment, and no video-quality conclusion follows from it.

## Numerical checks and retained failed attempts

The original score audit compares 600 fixed banks per capture, three
actual native seed prefixes per bank, and all 32 candidate scores, against
FP64 conditional-Jeffreys arithmetic on independently computed GPU FP32
QK logits with TF32 disabled. Its fixed tolerance is 2e-5 absolute plus
1e-4 relative. It separately checks ranking defects, all seed-row
invariants, and the original global normalizer.

- Run `conditional_jeffreys_native_20260904t` stopped at the first capture:
  3 of 57,600 raw scores exceeded tolerance. All 1,800 selection rank checks
  and all 302,400 groups' row invariants passed. No complete output or
  timing was evaluated. All three failures were already-selected candidates
  in head 30, but the prespecified raw-score check did not discard them.
- Run `conditional_jeffreys_native_20260904u` used centered log-softmax and
  the algebraically equivalent cell expression
  `sum_j (p_j - P_cell*w_j) * (log p_j - log w_j)`.
  This reduces subtraction of large nearly equal numbers. It stopped with
  one remaining raw-score failure under the unchanged tolerance; again no
  complete-output or timing claim is supported by this run.

The remaining witness is bank 487 (head 30, query column 20, key frame 10,
key row 22), prefix [22,23], candidate row 22. Independent inspection finds
a minimum ownership margin of 0.210678 bits: this witness is not near an
ownership tie. The nonzero contribution comes from seed 23's cell at probe
columns 25 and 75. A CPU replay alone does not establish the GPU failure's
cause; actual native intermediates are needed.
Both unsuccessful attempts and their original source fingerprints are
retained, not overwritten.

Run `20260904v` encountered a diagnostic-store pointer broadcasting compile
error before producing an audit. The shape-only correction is covered for
J1/J2/J4/J8 by a pointer/mask regression test. Run `20260904w` then recorded
the actual native dot, logits, probabilities, owners and scores for bank487.
Enabling these stores left **all** native rows and audited scores bitwise
unchanged. Replaying the native profiles in FP64 reproduced scores within
2.462e-8. A true FP64 dot product of the original BF16-valued inputs passed
the original score tolerance for all 96 scores in this witness, with maximum
error 1.400e-5; the earlier GPU FP32 reference still failed. This localizes
the residual discrepancy to reference point-product/rounding differences,
not the conditional-Jeffreys cell formula or a changed ownership boundary.

The subsequent audit uses true FP64 QK products for **every** sampled bank,
retains the previous GPU FP32 comparison as a named diagnostic, and preserves
the exact same score/rank tolerances and all 32 candidates, including used
rows. The candidate's inference precision and computation are not changed
by this reference correction. Sampled source vectors and both sets of
reference logits are saved for independent reproduction.

## Complete L33 output results

Run `conditional_jeffreys_native_20260904x` completed all three L33 captures
with the corrected FP64 reference. All score/rank/row gates, both unchanged
baseline replications, and observation-invariance checks passed.

| Selector | All-frame MSE | RF0-frame MSE | Other 20 frames MSE |
| --- | ---: | ---: | ---: |
| Hellinger RF0 | 0.005489318364 | 0.004350076511 | 0.005546280456 |
| Mass RF0 | 0.026362523224 | 0.021992833082 | 0.026581007731 |
| Conditional Jeffreys RF0 | 0.026459249733 | 0.022304207784 | 0.026667001831 |
| Hellinger RF10 control | 0.005586612375 | 0.004988728729 | 0.005616506557 |

Relative to Mass, the candidate increases complete MSE by **0.3669%**.
The changes on individual prompts are +0.0881%, +0.9546%, and +0.1386%:
all three worsen. RF0-frame MSE increases by 1.4158% and the other frames
by 0.3235%. Thus the earlier local conditional-output proxy gain does not
survive this native complete-attention comparison. Failure already occurs
in RF0 itself; temporal transfer alone cannot explain the discrepancy.

These are deterministic same-input comparisons on four selected heads of
one layer and one denoising step, not independent video samples or PSNR.
They do not prove a global quality ranking across all layers or videos.

## Full40-head accuracy and complete-attention time

Run x finished with worker and parent exit 0, no recorded foreign GPU
process, an empty purity log and an independent validation report. Each
method has 60 timed complete calls, equally divided over the two holdouts.
The values below pool equal-size inputs and equal timing counts; SDPA is
measured in this same run on the same H200 with the same BF16 Q/K/V.

| Method | Mean attention (ms) | Speedup vs SDPA | Complete-output MSE |
| --- | ---: | ---: | ---: |
| Flash SDPA reference | 324.6356 | 1.0000x | 0 |
| Hellinger RF0 | 127.5750 | 2.5447x | 0.007158211385 |
| Mass RF0 | 127.5743 | 2.5447x | 0.006405603532 |
| Conditional Jeffreys RF0 | 141.5878 | 2.2928x | 0.006515408419 |
| Hellinger RF10 control | 127.4130 | 2.5479x | 0.007071856711 |

Candidate vs Mass MSE increases by 1.4778% on call020 and 1.8392% on
call100, or **1.7142%** pooled. Both complete calls are slower: 141.2839
versus 127.4522 ms and 141.8917 versus 127.6964 ms. Pooled candidate latency
is **10.9846%** higher than Mass and **10.9839%** higher than Hellinger RF0.
The tiny Hellinger-RF0/Mass timing difference is not evidence that either
is faster. Per-capture six-sample values and dispersions remain in results.

The candidate beats Hellinger in full40 output MSE on these layer-20 inputs,
but does not beat the current Mass default. This does not contradict the
L33 comparison: the inputs/layers/head coverage are different and their MSE
values must not be combined into a single quality headline.

## Decision

Do not promote this conditional-Jeffreys implementation or start its
20-video comparison from these results. It misses the no-slower-than-
Hellinger requirement and worsens complete-output MSE relative to Mass on
every tested input. Keep original Mass-GS C32/RF0 as the default and leave
the paper untouched. PSNR, SSIM, LPIPS and VBench remain **unmeasured** for
this candidate; the goal of PSNR >12 on canonical 4-step/20-video generation
has not been achieved.

The useful result is the negative transfer from a principled local
conditional proxy to the full native operator. Future diagnostics should
separate global block reweighting and native merging effects before adding
further seed-score complexity. The evidence here does not identify which
omitted effect is responsible and does not invalidate the variational
identity under its stated ideal fixed-cell assumptions.

## Reproduction

`analysis/experiment_conditional_jeffreys_native.py` owns the single-GPU
worker, input validation, complete-output comparisons, complete-call
timing and source snapshots. `analysis/verify_conditional_jeffreys_native.py`
implements the independent native-score audit. The new selector lives in
`attention/conditional_jeffreys_seed_triton.py` and is not registered as a
production default. Primary artifacts are kept under
`C:/GMA/experiments/conditional_jeffreys_native_20260904*` and the matching
remote experiment directories. Exploratory table entries belong in the
existing `C:/GMA/tmp.xlsx` only.
