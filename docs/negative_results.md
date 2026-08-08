# Negative Results and Guardrails

This file records experiments that should not be repeated unless there is a
new reason to revisit them. The goal is to keep hard-route / GridMerge C4 work
focused on variants that preserve quality against dense attention.

When an experiment fails, add it here with: result, change tested, evidence,
why not repeat, and the rule it implies. Treat this file as the first place to
check before launching another speed/quality sweep.

## Hard Constraints

- Keep grid-local compression. Do not replace it with global compression or
  share-all compression.
- After token merging, the second stage must run full attention over all
  compressed KV tokens. Do not replace it with sparse routing, sliding-window
  routing, or local-only routing when evaluating the accuracy-preserving path.

## Failed or Rejected Directions

### No-Bias Routing

- Result: bad video quality; rejected by visual inspection.
- Change tested: remove the routing-bias term after KV compression.
- Evidence: no-bias C4 configs such as
  `configs/token_merging_wan_hard_route_c4_fastseed_nobias_fewstep.yaml` and
  `configs/token_merging_wan_hard_route_c4_fastseed_outer_local_r4_nobias_fewstep.yaml`
  generated worse videos than the bias-preserving hard C4 path.
- Why not repeat: the routing bias carries the entropy/log-normalizer
  correction from the first-stage key assignment and merge. Dropping it changes
  the approximation too much, even if it can reduce memory traffic.
- Rule: do not use `*_nobias` merge/routing backends for the main hard C4 path.
  Keep them only as explicit ablations if needed.

### Global or Shared Compression

- Result: bad quality compared with grid-local compression.
- Change tested: compress KV once globally, share one compression across too
  many query groups, or otherwise remove the per-grid compression.
- Why not repeat: different query/key grids need different compressed KV
  summaries. Sharing compression destroys this local adaptation.
- Rule: keep the non-shared grid-local hard-route path for quality-preserving
  experiments.

### Sparse, Sliding-Window, or Local Second-Stage Attention

- Result: rejected for the main method because it changes the requested core
  attention semantics.
- Change tested: route each query only to a local/exact R-window of compressed
  key outers after token merging.
- Why not repeat: this can be faster, but it is no longer full compressed
  attention after merging.
- Rule: use only full compressed attention for the second stage in the main
  method. R-window variants are ablations only.

### Split-Full Fused Kernel

- Result: too slow in the smoke benchmark.
- Change tested: compute per-key-outer partial full-attention stats and reduce
  them afterward to avoid writing all compressed K/V.
- Evidence: `analysis/hard_c4_split_full_smoke_20260807/metrics.json` on AMDHPC
  reported about 1814 ms/call, far slower than the direct-route path.
- Why not repeat directly: it preserves full attention, but the multi-kernel
  partial/reduce implementation has too much overhead.
- Rule: only revisit if the fusion design avoids large partial buffers and
  avoids launching a separate reduction over every group.

### Key-Row-Block Merge Variants

- Result: no speed improvement.
- Change tested: process multiple key rows per Triton program (`kr2`, `kr3`,
  `kr5`) to reduce repeated launch/work overhead.
- Evidence: `analysis/hard_c4_kr_smoke_20260807/metrics.json` showed all tested
  variants slower than the current head-major direct-route path while matching
  its output.
- Rule: do not retry the same key-row blocking unless the memory layout or
  kernel structure changes substantially.

### Token-Major Routing `n8` Tile

- Result: slower than the token-major `n16` routing tile.
- Change tested: use `triton_direct_m32_n8_w1_s1_exp2_tokenmajor` after the
  token-major C4 merge.
- Evidence: `analysis/hard_c4_tokenmajor_n8_n16_20260807/metrics.json` on
  AMDHPC measured about 26.29 ms/call for `n8` versus about 23.57 ms/call for
  `n16`, with the same output error.
- Rule: keep `m32_n16` as the token-major routing tile unless a different
  routing kernel structure is introduced.

## Current Best Direction

The best accepted direction so far is still:

- non-shared grid-local hard-route C4 compression,
- full compressed attention in the second stage,
- local-with-ith seed selection for lower error than random-with-ith,
- inline local seed-row computation to avoid reading the seed table in the C4
  merge kernel,
- token-major intermediate compressed-KV layout, which preserves the same
  full-attention math but makes merge stores more contiguous,
- head-major K/V layout for slightly lower merge overhead,
- direct-route Triton kernels to avoid unnecessary reshape/copy overhead.

This direction improves the old hard C4 implementation substantially, but it
has not yet achieved a clear 2x attention speedup over dense attention while
preserving the desired error level.
