# Fisher-GS recomputed for each query frame and key block

User-authorized accuracy diagnostic, 2026-09-08. Runtime is not an eligibility
gate for this exploration. The user explicitly clarified that every query
frame/key-block pair should recompute its Fisher metric and select seeds.
Production dispatch and the paper repository are unchanged.

## What changes

The existing selector already recomputes Fisher separately for every key
frame, key row, query column and head. RF0 selection observes only the first
query frame and reuses its selected row indices across query frames; merging
still loads the selected real query rows from each current frame.

This diagnostic replaces those shared indices with current-frame indices.
For each query frame, use that frame's actual Q with each key block's U16
probes, recompute rho=softmax(mean candidate logits), and run the same
geometric Fisher-GS. Keep C32/g45, the real matched anchor, four real seeds,
normalized seed-profile ownership, SQ1, conditional merge and global route.
No k-medoids, candidate-count change or extra key probe is introduced.

A 2x2 control also compares centered-logit Euclidean geometry (rho=Uniform16)
with the same GS, once with RF0 reuse and once with current-frame selection.
Every selector uses only current inference Q/K, without V or dense targets.

## Completed experiment

Run: `C:/GMA/experiments/perframe_fisher_cpu_20260908/results.json`.
Computation ran on scai7 CPU with CUDA explicitly disabled and four CPU
threads. All eight H200s were occupied; no GPU job was launched or queued.

Use three original, fingerprint-verified BF16 L33/step0 captures from the
existing four-step Wan14B trajectory. Evaluate original heads 0/10/20/30,
query columns 0/40/75 and query frames 0/10/20: 108 groups, 4860 queries.
Every measured query includes all945 key blocks and all75600 original keys
through a single global compressed attention distribution. This is global
attention on a limited query sample, not a within-block conditional metric.

Selection uses original BF16 Q/K promoted to FP32 for CPU products, followed
by FP64 scale/Fisher/GS. Measurement uses original BF16 Q/K/V promoted to
FP64, full80 ownership, direct normalization of nonempty cells and optimal
entropy-corrected beta over all945x5 compressed components. An independent
FP64 dense calculation on the same full key support is the main reference.

| Geometry and query information | Global reverse KL | Global output MSE vs FP64 dense | Change vs same-geometry RF0 |
| --- | ---: | ---: | ---: |
| Fisher, RF0 reused | 0.0116520656003 | 0.000305900562340 | reference |
| Fisher, recompute each frame | 0.0118083449228 | 0.000307290593040 | +0.4544% MSE |
| Uniform, RF0 reused | 0.0115331196839 | 0.000304692004627 | reference |
| Uniform, recompute each frame | 0.0117444668666 | 0.000294354469214 | -3.3928% MSE |

RF0 itself is exactly unchanged. Restricting to frames10/20 gives Fisher MSE
0.000247272481498 -> 0.000249357527549 (+0.8432%) and mean global reverse KL
0.0110899796224 -> 0.0113243986062 (+2.1138%). Current-frame Fisher changes
99.7105% of later-frame seed sets, so the negligible net gain is not due to
an accidentally reused selector. Of the 72 later-frame query groups, 62
have lower output MSE and 10 have higher MSE; magnitude matters for the mean.

All-frame Fisher output MSE changes by head:

| Original head | RF0 | Recomputed | Relative change |
| --- | ---: | ---: | ---: |
| 0 | 0.0000209212775674 | 0.0000186920741629 | -10.6552% |
| 10 | 0.0000231156317824 | 0.0000214979698253 | -6.9981% |
| 20 | 0.0000150017067131 | 0.0000142355582228 | -5.1071% |
| 30 | 0.00116456363330 | 0.00117473676995 | +0.8736% |

Head30 dominates absolute output error and offsets the improvements in
the other three heads. Prompt-level Fisher MSE changes are mixed as well:
prompt0 improves, while prompts1/2 worsen. Uniform current-frame selection
has 4.21% lower overall MSE than current-frame Fisher in this sample (the
opposite ratio is Fisher 4.39% higher). These are deterministic diagnostic
observations, not statistical or full-video superiority claims.

## Interpretation and validation

Refreshing Q information can help many local groups, but did not produce
a meaningful aggregate accuracy gain here. More current Fisher geometry
does not guarantee a lower hard-partition/global variational gap: GS selects
span directions under a local quadratic surrogate, then hard ownership and
conditional weights are reconstructed. This diagnostic provides no monotone
global optimization theorem and does not establish an accuracy upper bound.

- Two independent selector tests passed: scalar Fisher/uniform GS, logit-shift
  invariance, retained real anchor/distinct rows, and changed current-frame
  selection. RF0 shared/recomputed full outputs are exactly equal in every
  one of the 36 RF0 groups.
- Every evaluated distribution is checked for normalization and the direct
  reverse-KL/log-partition-gap identity. Maximum errors are 5.30e-13 and
  2.39e-13 respectively across all four methods.
- All captures are verified against the existing audited SHA256 receipts;
  archived source bytes remain unchanged. The worker finished successfully.
- No native BF16 candidate execution, video generation, PSNR, SSIM, LPIPS or
  VBench measurement. `mean_attention_ms=null`, `speedup_vs_sdpa=null`.
  The measured 60.78 CPU diagnostic seconds is not attention latency.
- These MSEs are not directly comparable with earlier all-query native
  BF16 screens or sampled within-block conditional-output metrics.
- Results are recorded in the existing `C:/GMA/tmp.xlsx` exploratory sheet.

If applied to every one of the 21 query frames, selector QK/Fisher/GS work
would be approximately21 times RF0 selection, while the merge and core
work per query remains structurally unchanged. This is a selector-work
count, not a measured total-attention slowdown.
