# Fisher query k-medoids without a fixed anchor

Requested 2026-09-08: test Fisher query clustering with all four real seeds
free to move, including direct initialization without an anchor constraint.
The paper repository and production dispatch are unchanged.

## Definition

Use the same C32/g45 real query candidates and Uniform16 probe columns.
For candidate logits z_i, freeze rho = softmax(mean_i z_i). The features are

    x_i = sqrt(rho) * ((z_i - mean(z)) - rho dot (z_i - mean(z))).

Thus ||x_i-x_j||^2 is the shared-reference Fisher quadratic form. Minimize
sum_i min_c ||x_i-x_s[c]||^2 over four distinct real query indices using
alternating nearest-seed assignment and exact within-cluster medoid updates.
The medoid is the cluster member nearest its feature mean. Every seed owns
itself on zero-distance ties, preserving nonempty clusters and distinct rows.
Keep existing seeds on numerical ties. Stop at unchanged seed indices; the
256-round guard raises an error instead of returning truncated results.

The primary direct initialization chooses the real point nearest the global
feature mean, then adds farthest points under nearest-seed Fisher distance.
There is no fixed spatial anchor or random restart. Two paired controls use
the original Fisher-GS initialization with the first seed fixed or free.
This is a deterministic inference-time calculation, with no learned or
calibrated parameters. It is a query-selection reference, not a native kernel.

## Completed CPU diagnostic

Run: `C:/GMA/experiments/fisher_query_kmedoids_noanchor_cpu_20260908`.
Input: the fingerprint-verified `rf0_conditional_jeffreys_banks_20260904s/banks_v2`
cache, three L33/step0 captures, heads 0/10/20/30, 1800 fixed banks and 81000
RF0 target query rows. Select using only 32 candidates and 16 probes; evaluate
all45 queries and all80 keys per sampled bank. All methods use normalized
seed-profile ownership, SQ1 removal and exact-arithmetic entropy-corrected
within-block attention. Cached dense global block masses weight evaluation
only and never enter seed selection.

| Selector | Mean nearest Fisher distance squared | Weighted reverse KL | Weighted conditional output MSE | MSE / GS | Mean rounds | Maximum rounds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fisher-GS | 0.04511870137 | 0.007375569686 | 0.0004387010073 | 1.000 | 0 | 0 |
| Fixed anchor, GS initialization, converged medoids | 0.02526839475 | 0.006730527753 | 0.0003512933845 | 0.801 | 2.575 | 10 |
| All seeds free, GS initialization, converged medoids | 0.02276363468 | 0.01172341373 | 0.006700795590 | 15.274 | 2.607 | 9 |
| No anchor, direct initialization, converged medoids | 0.02257082355 | 0.009368654440 | 0.001746721584 | 3.982 | 2.244 | 8 |

All banks converged and every recorded nearest-medoid objective was
nonincreasing. The direct variant reduces its Fisher coverage loss relative
to GS by 49.97%, while conditional output MSE increases 3.98x. Only 16.61%
of its final seed sets retain the geometrically matched query row.

The increase is concentrated in head30, especially prompt1, query column75,
key frame0/row44, target query row44. Dense attention puts 94.824% mass on this
block. GS selects rows [44,22,26,0]; direct medoids select [3,20,41,0]. The
conditional reverse KL changes from roundoff zero to 0.439895. This one
target's weighted SSE increase is 43.991954, compared with the aggregate net
increase 44.452530. This is diagnostic evidence of an important query missed
by average probe-space coverage, not proof that anchors are universally
necessary or that a global attention error must have the same change.

## Scope and verification

- `tests/test_fisher_query_kmedoids.py`: five tests passed. Independent Fisher
  matrix identity, logit-shift invariance, exhaustive finite cluster-medoid
  checks, objective monotonicity, movement of the first seed, duplicate-feature
  ties, distinct real indices and failure on an insufficient convergence guard.
- The evaluator is the existing full80/SQ1 mathematical diagnostic. This is
  not the historical complete-output BF16 experiment and its MSE must not be
  compared numerically with the historical 0.001257260255 Fisher-GS result.
- No global block reweighting, current-frame-query reuse, native BF16 merge,
  model or video execution was tested. No statistical significance claim.
- `mean_attention_ms = null`; `speedup_vs_sdpa = null`.
  PSNR, SSIM, LPIPS and VBench are unmeasured.
- All eight scai7 H200s were occupied at preflight. No GPU experiment was
  launched or queued. The CPU diagnostic does not establish native latency.
- Results are recorded in `C:/GMA/tmp.xlsx`, the existing exploratory workbook.

The tested unconstrained variants are not promoted. The fixed-anchor local
improvement does not override earlier complete-output negative evidence.

Reproduce with a CPU Python containing NumPy, explicitly disabling CUDA:

    python -m unittest tests.test_fisher_query_kmedoids -v
    python -m analysis.screen_fisher_query_kmedoids_cpu --cache-dir <audited banks_v2> --output-dir <fresh run directory>
