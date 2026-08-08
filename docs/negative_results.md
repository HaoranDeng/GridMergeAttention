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

### Current Fast Path with `j64` Merge Tile

- Result: slower than the current `j32` merge tile.
- Change tested: replace the current non-shared C4 merge tile with
  `triton_c4_direct_route_j64_w2_exp2merge_forcekeys_fp16kvbias_s1_fastdot`
  while keeping token-major output, local-with-ith seed initialization, routing
  bias, and full compressed attention.
- Evidence: `analysis/hard_c4_j64_nohead_20260808/metrics.json` on AMDHPC
  measured about 26.32 ms/call for j64 head-major and 25.84 ms/call for j64
  no-headmajor, versus 24.84 ms/call for current j32 head-major and 24.46
  ms/call for current j32 no-headmajor, with unchanged output error.
- Why not repeat directly: the wider tile covers all 52 key columns in one
  program, but the lower occupancy / higher register pressure outweighs the
  reduced tiling overhead.
- Rule: keep `j32_w2` for the current token-major hard C4 merge path.

### Same-Precision Merge Tile/Warp Sweep

- Result: slower than the current `j32_w2` merge tile.
- Change tested: same-precision hard C4 merge variants `j16_w2`, `j32_w1`,
  and `j32_w4` while keeping local-with-ith seeds, token-major output, full
  compressed attention, and the same output error.
- Evidence: Slurm job `366543` measured about 24.59 ms/call for the current
  `j32_w2` path, versus about 26.58 ms/call for `j16_w2`, 26.60 ms/call for
  `j32_w1`, and 33.27 ms/call for `j32_w4`.
- Why not repeat directly: these variants change launch geometry but do not
  reduce the dominant memory traffic enough to offset poorer occupancy or extra
  scheduling overhead.
- Rule: do not keep or rerun these same tile/warp variants unless the merge
  kernel structure changes substantially.

### Group-Major Inline-Seed Layout

- Result: slower than the current token-major layout with the same output
  error.
- Change tested: keep the same non-shared C4 full-core method and inline
  local-with-ith seed selection, but write/read compressed K/V in group-major
  layout instead of token-major layout.
- Evidence: Slurm job `366560` measured about 24.75 ms/call for
  `hard_c4_full_core_groupmajor_inline_seed`, versus about 22.98 ms/call for
  `hard_c4_full_core_tokenmajor`, with the same rel-L2 vs dense
  (`0.096872`).
- Why not repeat directly: routing reads are more contiguous in group-major
  layout, but the merge-side stores/load scheduling lose more than routing
  gains for the current kernel structure.
- Rule: keep token-major compressed K/V layout for the current hard C4 fast
  path unless the merge/routing fusion strategy changes.

### A2 Query-Outer Pair Kernel

- Result: slower than the current token-major direct-route kernel.
- Change tested: process two query outer groups in one Triton program to reuse
  key/value loads across neighboring query groups.
- Evidence: Slurm job `366561` measured about 31.95 ms/call for
  `hard_c4_full_core_a2_tokenmajor` in a smoke run, versus about 22.98 ms/call
  for the current token-major path in Slurm job `366560`, with similar output
  error.
- Why not repeat directly: the intended key/value reuse is outweighed by
  higher register pressure and lower occupancy in the fused two-outer program.
- Rule: do not keep an A2/query-outer-pair merge kernel unless the kernel is
  redesigned to reduce register pressure and revalidated against the current
  token-major path.

### Whole-Outer `query_outer_chunk=21`

- Result: unsafe for the current no-headmajor token-major hard C4 path.
- Change tested: process all 21 query outer groups in one chunk while keeping
  the same C4 merge and full compressed attention.
- Evidence: Slurm job `366526` hit a Triton illegal memory access while running
  the `chunk=21` case in
  `analysis/hard_c4_nohead_chunk_sweep_20260808`.
- Why not repeat directly: the current direct-route Triton path is not stable
  with the whole outer dimension in one chunk.
- Rule: do not set `query_outer_chunk=21` for the current hard C4 fast path
  unless the routing/merge kernel indexing is changed and revalidated.

### Quantized Intermediate K/V

- Result: rejected for the current optimization direction.
- Change tested: store the compressed hard C4 intermediate K/V in fp8 before
  the full compressed attention stage.
- Evidence: `analysis/hard_c4_fp8_smoke_20260808/metrics.json` showed a speed
  improvement in a smoke run, but it is a quantization method and it increased
  dense-vs-hard output error.
- Why not repeat: the current goal is same-precision memory-bound optimization,
  not quantization. Reducing memory traffic should be attempted without lowering
  K/V precision.
- Rule: do not use fp8/quantized intermediate K/V for the main hard C4 path
  unless quantization is explicitly reopened as a separate experiment.

## Current Best Direction

The best accepted direction so far is still:

- non-shared grid-local hard-route C4 compression,
- full compressed attention in the second stage,
- local-with-ith seed selection for lower error than random-with-ith,
- inline local seed-row computation to avoid reading the seed table in the C4
  merge kernel,
- token-major intermediate compressed-KV layout, which preserves the same
  full-attention math but makes merge stores more contiguous,
- standard K/V block layout for the current token-major `m32_n16` path, avoiding
  the older head-major experiment branch,
- `query_outer_chunk=10`, which was the fastest confirmed safe chunk size in
  `analysis/hard_c4_nohead_chunk_confirm_20260808`,
- direct-route Triton kernels to avoid unnecessary reshape/copy overhead.

This direction improves the old hard C4 implementation substantially, but it
has not yet achieved a clear 2x attention speedup over dense attention while
preserving the desired error level.
