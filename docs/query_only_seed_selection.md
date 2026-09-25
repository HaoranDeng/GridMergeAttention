# Query-Only Seed Selection Exploration

## Scope

This exploration asks whether Hard C4 seeds can be selected without reading
keys. Every tested selector is deterministic, training-free, and receives only
the current query tensor plus fixed row/column indices. Keys are used only by
the offline evaluator to measure the resulting attention-distribution error.
The production candidate remains pure all-layer GMA: no hybrid schedule,
learned router, calibration data, or dense rescue is used.

## Objective

For dense attention distribution `p_q(k)` and the distribution
`p_hat_q(k; S)` induced by a C4 seed set `S`, the evaluation objective is

```text
E_q[ KL(p_q || p_hat_q(.; S)) ].
```

The evaluator includes the tangent/cluster normalizer correction used by the
Hard C4 approximation. It reports exact key-level KL on captured real Q/K,
decomposed into cluster-mass and within-cluster conditional terms. This is an
attention-distribution objective; it deliberately does not use V.

## Selectors Tried

All selectors below were evaluated on real Wan2.1-14B Q/K captures:

- fixed wide rows (retained baseline)
- spatial k-center
- query L2 k-center
- normalized-query L2 k-center
- self-Fisher query k-center
- angular query k-center
- surrogate-Hellinger medoid
- query-row-proxy Hellinger k-center and medoid
- query-row-proxy symmetric-KL medoid
- raw-query QR pivoting
- cosine-query QR pivoting
- query L2 sketches with 16, 32, and 64 dimensions
- one-router-frame temporal sharing from frames 0, 10, and 20

Candidate windows of 5, 6, 7, 9, 11, and 15 rows were screened. Six
candidates was the best stable choice. Larger windows made the no-key proxy
less reliable; at 15 candidates, query L2 was 5.77% worse than fixed wide.

## Real-QKV Results

The full validation covers 40 captured layers, four heads, and three query
frames: 480 cases and 1,728,000 query distributions.

| Selector | Mean key KL | Ratio vs fixed | KL reduction |
| --- | ---: | ---: | ---: |
| Query L2 k-center, D=128 | 0.496247 | 0.91396 | 8.60% |
| Cosine QR | 0.496509 | 0.91445 | 8.56% |
| Row-proxy Hellinger k-center | 0.496723 | 0.91484 | 8.52% |
| Self-Fisher k-center | 0.497793 | 0.91681 | 8.32% |
| Normalized-query L2 k-center | 0.498782 | 0.91863 | 8.14% |
| Query L2 k-center, D=32 | 0.500287 | 0.92141 | 7.86% |
| Fixed wide | 0.542961 | 1.00000 | 0.00% |
| Spatial k-center | 0.554351 | 1.02098 | -2.10% |

Full D=128 query L2 won 391/480 cases. D=32 retained most of the gain and won
376/480 cases. Sharing one router frame across time reduced the gain to only
1.8-3.0%, so the production screen kept per-frame selection.

## Hardware-Aware Candidate

The retained screen candidate is `query_l2_kcenter_c6_d32`:

1. Start with the index-matched query row as a mandatory anchor.
2. Consider a local six-row candidate window.
3. Greedily add the query row farthest from its nearest selected row in a
   32-dimensional query sketch until four rows are selected.
4. Use those real query rows as the ordinary Hard C4 seeds. Merge assignment,
   tangent correction, and second-stage full compressed attention are
   unchanged.

The specialized Triton selector takes about 1.20 ms for the complete
`[1, 21, 45, 80, 40, 128]` query tensor on an H200. The full-D selector takes
about 41.17 ms, so D=32 is the only practical version among the highest-KL
ranked candidates.

## 14B 720p Four-Step Video Screen

The method was run on four matched MovieGen prompts with the StepDistill
checkpoint, seed 0, 1280x720, 81 frames, exactly four denoising steps, and 160
attention calls per video. Quality is measured against the existing matched
Dense Flash reference.

| Method | Videos | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: |
| Fixed Hard C4 | 4 | 9.8183 | 0.2952 | 0.6405 |
| Query-L2 C6-D32 Hard C4 | 4 | 9.7105 | 0.2896 | 0.6395 |
| Delta | 4 | -0.1077 | -0.0056 | -0.0010 |

The dynamic method's steady videos 2-4 measured 123.512 ms/core attention
call, 147.811 ms/full self-attention call, and 37.694 s/video. The retained
20-video fixed Hard C4 baseline measured 110.895 ms, 133.303 ms, and 35.080
s/video on a separate clean H200. This makes the comparison provisional but
shows a clear regression: about 11.4% slower core attention and 7.5% slower
end-to-end. The selector itself is only about 1.20 ms; most of the extra time
comes from the general external-seed table path rather than seed scoring.

## Decision

Do not promote this method and do not spend a 20-video run on it. It is a
useful negative result: a query-only selector can reduce one-call attention KL
on captured Q/K, but that reduction did not improve accumulated video PSNR or
SSIM and the indirect dynamic-seed kernel was slower. The simple fixed-wide
selector remains the hardware-aware Hard C4 choice unless a future no-key
criterion improves matched video quality, not only local attention KL.

Raw evaluator results are stored under:

```text
/local2/denghaoran/GridMergeAttention/experiments/query_only_seed_kl_20260828
```

Production-screen videos, runtime, and quality JSON are stored under:

```text
/local2/denghaoran/GridMergeAttention/experiments/query_l2_kcenter_c6_d32_4step_20260828
```

## Global Query Preprocessing Ablation

A follow-up tested whether the expensive query selection can be computed once
per ``(query frame, column, head)`` from all 45 query rows and then reused over
the 45 target key rows. This is valid computationally because the selector does
not read K. It is still recomputed for every attention call because Q changes
with layer and denoising step.

The variants were evaluated in the requested 2, 1, 3 order:

1. Original variant 2, ``anchor + global C3``: retain the index-matched row and append the first
   three distinct entries from one global D32 farthest-point traversal.
2. Original variant 1, ``global C4``: share the first four global traversal entries across every
   target row.
3. Original variant 3, ``global pool + local C4``: build a global pool of 8 or 12 rows once, retain
   the index-matched row, and choose three pool entries by anchor-conditioned
   farthest-point selection.

The global traversal starts at the real query row nearest the query centroid,
then repeatedly selects the row farthest from its nearest selected row. All
methods are deterministic, training-free, query-only, and use real query rows.

| Selector | Mean key KL | Delta vs fixed | Cases better than fixed | Full-shape selector time |
| --- | ---: | ---: | ---: | ---: |
| Fixed wide | 0.542961 | 0.00% | - | 0 ms |
| Anchor + global C3, D32 | 0.560301 | +3.19% | 264/480 | 7.67 ms |
| Global C4, D32 | 0.652404 | +20.16% | 215/480 | 4.86 ms |
| Global pool 8 + local C4, D32 | 0.612356 | +12.78% | 162/480 | 16.78 ms |
| Global pool 12 + local C4, D32 | 0.603952 | +11.23% | 175/480 | 22.35 ms |

Times cover one complete ``[21, 80, 40, 45, 32]`` selector input on an H200
using vectorized PyTorch. They are diagnostic, not optimized Triton timings.

The anchor-plus-global variant improved the median case and won 264/480 cases,
but had a long failure tail: its mean per-case KL delta was +0.01734, its 90th
percentile delta was +0.14714, and its worst delta was +0.68226. It reduced
mean cluster-mass KL from 0.16604 to 0.16245 while increasing conditional KL
from 0.37692 to 0.39785. The shared global seeds therefore partition some key
rows well but provide poor within-cluster representatives for others.

This ablation separates two claims. Global Q preprocessing is computationally
feasible and can be reused across target rows, but four globally shared query
representatives are not a safe approximation for every key-row block. The
target-row-conditioned local geometry in fixed Hard C4 is an important prior,
not merely duplicated selector work. None of these variants passed the real-QKV
KL gate, so no 14B video generation run was started.

Raw results are stored under:

```text
/local2/denghaoran/GridMergeAttention/experiments/query_global_seed_sharing_20260828
```
