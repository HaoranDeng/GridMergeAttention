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

- Result: slower than the token-major `n16` routing tile, and not competitive
  with the later `n32` default.
- Change tested: use `triton_direct_m32_n8_w1_s1_exp2_tokenmajor` after the
  token-major C4 merge.
- Evidence: `analysis/hard_c4_tokenmajor_n8_n16_20260807/metrics.json` on
  AMDHPC measured about 26.29 ms/call for `n8` versus about 23.57 ms/call for
  `n16`, with the same output error.
- Rule: do not use `n8` for the main token-major routing path.

### Token-Major Routing Larger-M Tiles

- Result: slower than the token-major `m32` routing family.
- Change tested: keep the same C4 merge output and full compressed attention,
  but use token-major direct routing tiles `m64_n16`, `m64_n32`, `m128_n16`,
  `m128_n32`, and `m128_n32_w4`.
- Evidence: Slurm job `366571` measured about 22.90 ms/call for the current
  `m32_n16` full hard C4 path, versus about 24.01 ms/call for `m64_n32`,
  25.22 ms/call for `m64_n16`, 25.51 ms/call for `m128_n32_w4`, 48.62
  ms/call for `m128_n16`, and 50.95 ms/call for `m128_n32`; output error was
  unchanged.
- Why not repeat directly: Q_ROWS is only 30, so larger `BLOCK_M` shapes waste
  lanes/registers in this direct-route layout.
- Rule: keep `BLOCK_M=32` for token-major direct routing unless the routing
  layout changes substantially.

### Grouped `g2` Token-Major Routing

- Result: slower than the ungrouped token-major routing path with identical
  output error.
- Change tested: process two adjacent route groups per Triton routing program
  using `triton_direct_g2_*_tokenmajor`, while keeping the same C4 merge output
  and full compressed attention.
- Evidence: Slurm job `366953` measured dense at about 23.41 ms/call,
  the old `hard_c4_bestlogit_tokenmajor` routing tile `m32_n16` at about
  24.17 ms/call, `m32_n32` at about 23.36 ms/call, `g2_m32_n16` at about
  25.06 ms/call, and `g2_m16_n16` at about 28.28 ms/call. All hard-route
  variants had the same dense-vs-output rel-L2, about 0.099699.
- Why not repeat directly: grouping two route groups reduces program count,
  but doubles the live K/V/Q/accumulator state and loses more to register
  pressure than it gains from launch reduction.
- Rule: do not use `triton_direct_g2_*_tokenmajor` for the main hard C4 route.
  The current routing default is `triton_direct_m32_n32_w1_s1_exp2_tokenmajor`.

### Token-Major Routing `n16` Versus `n32`

- Result: `n32` is the current default because it was slightly faster than
  `n16` in the latest multiprompt real-QKV benchmark with unchanged error.
- Change tested: compare `triton_direct_m32_n16_w1_s1_exp2_tokenmajor` against
  `triton_direct_m32_n32_w1_s1_exp2_tokenmajor` after the same bestlogit C4
  merge.
- Evidence: Slurm job `366953` measured `m32_n16` at about 24.17 ms/call and
  `m32_n32` at about 23.36 ms/call over 180 attention calls, both with
  rel-L2 about 0.099699 versus dense.
- Rule: use `m32_n32_w1_s1_exp2` as the default token-major routing tile until
  a more substantial routing or merge-layout change is validated.

### Generic SDPA Routing Fallback

- Result: slower than direct Triton routing with essentially the same output
  error.
- Change tested: keep the direct C4 merge output, but transpose the compressed
  KV/bias tensors to group-major layout and run the second-stage full
  compressed attention through PyTorch SDPA instead of the direct-route Triton
  kernel.
- Evidence: `analysis/hard_c4_sdpa_routing_20260807/metrics.json` measured
  about 29.83 ms/call for the SDPA routing fallback, versus about 24.50 ms/call
  for direct-route `m32_n32` and about 22.55 ms/call for dense attention. The
  dense-vs-output rel-L2 was effectively unchanged.
- Why not repeat directly: SDPA must consume an additive routing bias and a
  transposed group-major view, so it does not recover enough optimized Flash
  Attention behavior to compensate for the layout/mask overhead.
- Rule: keep the direct-route Triton routing kernel for the main hard C4 path.
  Only revisit SDPA routing if the compressed KV/bias layout changes enough to
  remove the transpose and mask overhead.

### Augmented-Bias SDPA Routing

- Result: compiles and preserves the same dense-vs-hard error, but is much
  slower than the current direct-route Triton path.
- Change tested: represent the routing bias as an extra Q/K channel
  (`q_aug=[q,1]`, `k_aug=[k,-bias/scale]`) and call PyTorch SDPA without an
  additive mask.
- Evidence: corrected real Wan benchmark Slurm job `366946` measured dense at
  about 22.60 ms/call, current `hard_c4_bestlogit_tokenmajor` at about
  23.34 ms/call, and augmented-bias SDPA at about 71.48 ms/call. The rel-L2
  versus dense stayed about 0.155913. A synthetic smoke job `366945` similarly
  measured about 0.204 ms/call for current direct routing versus about
  0.635 ms/call for augmented-bias SDPA.
- Why not repeat directly: avoiding the explicit SDPA mask is not enough. This
  path still materializes augmented K, changes layout, and loses the custom
  token-major direct-routing kernel.
- Rule: do not absorb routing bias into augmented Q/K for the main PyTorch SDPA
  routing path. Keep the bias-preserving direct-route Triton path unless a new
  design avoids both mask overhead and extra augmented-K materialization.

### Seed-Key Approximate Merged K

- Result: faster, but the output error is too high for the accuracy-preserving
  path.
- Change tested: replace the softmax-merged K with seed-query-derived K
  approximations such as `seedkey`, `seedkeymatch`, and `seedkeyproj`, while
  still running full compressed attention after token merging.
- Evidence: corrected Wan benchmark Slurm job `366947` measured dense at about
  22.77 ms/call and current hard C4 at about 23.60 ms/call with rel-L2
  0.155913 versus dense. `seedkey_n32_s1` was faster at about 14.06 ms/call,
  but rel-L2 rose to about 0.604. `seedkeymatch` had rel-L2 about 0.480, and
  `seedkeyproj` had rel-L2 about 0.359.
- Why not repeat directly: the speedup comes from changing the compressed K
  too aggressively, not from a safe memory-traffic reduction.
- Rule: do not replace merged K with seed-query, bias-matched seed-key, or
  scalar-projected seed-key approximations in the main hard C4 path.

### Compact/Merge-Weight Reconstruction Variants

- Result: no speedup in synthetic smoke, with tiny numerical differences for
  K/V reconstruction variants.
- Change tested: avoid materializing some compressed K/V tensors by storing
  merge weights or compact assignments, then reconstructing V or K/V in the
  full compressed-attention routing kernel.
- Evidence: corrected smoke job `366948` measured current bestlogit at about
  0.199 ms/call, versus about 0.236-0.241 ms/call for value-only
  `compactweights`/`mergeweights`. Corrected smoke job `366951` measured
  current bestlogit at about 0.255 ms/call, versus about 0.286 ms/call for
  `compactweightskv` and about 0.423 ms/call for `mergeweightskv`.
- Why not repeat directly: the variants reduce materialized tensors but add
  reconstruction work and less favorable routing memory access, which erased
  the traffic savings in smoke tests.
- Rule: do not rerun the same compact/merge-weight reconstruction variants as
  main candidates unless the design changes enough to avoid source K/V
  reconstruction overhead.

### Token-Major Routing `m16`, `n64`, and Stage Sweeps

- Result: no useful speedup with unchanged error.
- Change tested: keep the same C4 token-major merge output and full compressed
  attention, but use routing variants `m16_n16`, `m16_n32`, `m16_n64`,
  `n64_w1`, `n64_w2`, and extra routing stages `s2`/`s3`.
- Evidence: AMDHPC sweeps
  `hard_c4_n64_routing_sweep_20260808`,
  `hard_c4_m16_routing_sweep_20260808`, and
  `hard_c4_stage_sweep_20260808` measured the best variants around
  22.98-23.42 ms/call, roughly tied with or slower than dense and not better
  than the later `m32_n32_w1_s1` default.
- Why not repeat directly: these settings only retile the second-stage full
  compressed attention and do not reduce the dominant intermediate K/V traffic.
- Rule: do not retry these routing tile/stage settings unless the compressed
  K/V layout or attention kernel structure changes.

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

### Key-Outer Merge Blocking F1-F7

- Result: slower than the non-keyouter token-major merge path with unchanged
  dense-vs-hard error.
- Change tested: process multiple key-outer groups per Triton merge program
  (`f1` through `f7`) while keeping grid-local C4 compression, local-with-ith
  seeds, routing bias, token-major output, and full compressed attention.
- Evidence: AMDHPC sweep
  `hard_c4_f2to7_bf16kv_noplaceholder_smoke_20260808` measured about
  22.79 ms/call for `hard_c4_bestlogit_tokenmajor`, 23.22 ms/call for
  `hard_c4_full_core_tokenmajor`, and 31-34 ms/call for tested keyouter
  variants. `hard_c4_f1_keyouter_smoke_20260808` measured about 22.96 ms/call
  for `f1`, still not a clear improvement over the current path.
- Why not repeat directly: batching key outers increases register pressure and
  store complexity more than it reduces launch or key/value load overhead.
- Rule: do not use `_f1`...`_f7` key-outer blocking for the main hard C4 path
  unless the merge kernel is redesigned.

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

### Fully Fused Streaming Merge + Full Attention

- Result: much slower than the current two-kernel token-major path.
- Change tested: avoid writing compressed K/V/bias to global memory by
  recomputing each C4 merged token inside a single full-attention kernel and
  applying FlashAttention-style online softmax over all compressed tokens.
- Evidence: Slurm job `366569` measured about 526.72 ms/call for
  `hard_c4_fused_full_m32` and about 1008.94 ms/call for
  `hard_c4_fused_full_m16`, versus about 22.73 ms/call for
  `hard_c4_full_core_tokenmajor` in the same smoke benchmark. The error was
  similar, but the runtime was far worse.
- Why not repeat directly: the fused kernel saves global writes/reads of
  compressed K/V/bias, but loses the current merge kernel's reuse across query
  columns and the routing kernel's tensor-core tile over compressed tokens.
- Rule: do not fuse merge and full compressed attention by streaming one
  merged token at a time. Only revisit this direction if the design preserves
  both query-column reuse during merge and route-token tiling during full
  attention.

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

### MI210 Smoke for MI300 Fastdot Backend

- Result: invalid hardware check for the current fast path.
- Change tested: run the MI300-oriented hard C4 direct-route backend
  `triton_c4_direct_route_j32_w2_exp2merge_forcekeys_fp16kvbias_s1_fastdot`
  on `mi2101x` as a provisional smoke test.
- Evidence: Slurm job `366607` failed during Triton compilation because MI210
  allowed dot input precisions were `ieee`, `bf16x3`, and `bf16x6`, while the
  current fastdot path requests `input_precision="tf32"`.
- Why not repeat directly: this is a hardware/compiler capability mismatch, not
  an attention-quality result. It cannot validate speed or correctness for the
  MI300 fastdot backend.
- Rule: benchmark the current fastdot hard C4 path on `mi3001x` for final
  decisions. Use MI210 only with a separate `ieee`/non-fastdot backend.

### Non-Power-of-Two Triton Merge Tiles

- Result: compile-time failure.
- Change tested: exact key/query tile sizes such as `j26`, `j52`, and `l52`
  for the current hard C4 direct-route merge kernel.
- Evidence: Slurm job `366687` wrote
  `analysis/hard_c4_kernel_variant_smoke_20260808/smoke.json`; every `j26`,
  `j52`, or `l52` variant failed with Triton reporting that `tl.arange` ranges
  must be powers of two.
- Why not repeat directly: the current kernel uses `tl.arange(0, BLOCK_J)` and
  `tl.arange(0, BLOCK_L)`, so non-power-of-two block sizes are not legal for
  this implementation.
- Rule: do not submit `j26`, `j52`, or `l52` direct-route backend strings.
  Exact-52 tiling would require a different kernel design that uses
  power-of-two blocks plus masks.

### Packed Token-Major K/V/Bias Layouts

- Result: unsafe or slower on the real Wan hard C4 shape.
- Change tested: packed token-major variants such as `packkv`, `packkvbias`,
  `packkvbiascompact`, `packkbias`, and `packkbiascompact`.
- Evidence: Slurm job `366724` ran real-shape smoke tests with
  `B=1, A=10, F=21, rows=30, columns=52, heads=12, dim=128`.
  `packkv`, `packkvbias`, `packkvbiascompact`, and `packkbias` triggered HIP
  illegal memory accesses. `packkbiascompact` completed, but measured about
  14.94 ms versus about 10.95 ms for the current bestlogit token-major
  baseline in the same smoke.
- Why not repeat directly: toy-shape smoke can pass, but the real Wan layout
  exposes indexing/backend issues. The only safe packed-key-bias compact path
  was still substantially slower and did not reduce total intermediate bytes.
- Rule: do not use packed K/V/bias hard C4 backends for the main path. Focus
  on designs that reduce materialized intermediate data rather than just
  packing the same K/V/bias tensors.

### Precomputed-Logit Materialization

- Result: much slower than both dense attention and the current hard C4
  bestlogit path.
- Change tested: during the hard C4 merge, write precomputed
  `q @ merged_key - routing_bias` logits and then run the second-stage softmax
  over those logits instead of reading materialized merged K.
- Evidence: Slurm jobs `366738` and `366739`. On real Wan QKV, dense attention
  measured 23.34 ms/call, current bestlogit hard C4 measured 24.46 ms/call,
  and precomputed-logit hard C4 measured 161.69 ms/call for fp16 logits and
  165.36 ms/call for fp32 logits. The dense-vs-hard error was also slightly
  worse than the current bestlogit path.
- Why not repeat directly: it reduces intermediate bytes on paper, but the
  precompute step loses the tensor-core-friendly full compressed-attention dot
  product and replaces it with many scalar reductions during merge.
- Rule: do not use precomputed-logit materialization for the main hard C4 path.
  Only revisit it if the logits can be produced with a tensor-core-friendly
  tiling that preserves the second-stage full attention semantics.

### Early Representative-Key K Approximations

- Result: no acceptable speed/error tradeoff.
- Change tested: replace the softmax-merged K with simpler representative-key
  choices such as `repkey`, `top2key`, or `tail4`, while still running full
  compressed attention after token merging.
- Evidence: AMDHPC smoke runs
  `analysis/hard_c4_repkey_smoke_20260807/metrics.json`,
  `analysis/hard_c4_top2key_smoke_20260807/metrics.json`, and
  `analysis/hard_c4_tail4_smoke_20260807/metrics.json`. `repkey` was only
  roughly tied with dense speed and increased dense rel-L2 from about 0.155 to
  0.181 in that smoke; `top2key` and `tail4` were slower than the direct-route
  baseline in the same setup.
- Why not repeat directly: these variants reduce or alter the K merge, but the
  quality loss or extra scheduling cost erased the benefit.
- Rule: do not rerun the exact `repkey`, `top2key`, or `tail4` representative-K
  variants. More constrained K approximations must be separately justified and
  measured against dense error.

### Merge-Side Max-Bias and Stripe Shortcuts

- Result: not acceptable for the main quality-preserving hard C4 path.
- Change tested: replace the entropy/routing-bias calculation with a max-logit
  shortcut (`maxbias`) or replace hard assignment with fixed key stripes
  (`stripe`) while still running full compressed attention in the second stage.
- Evidence: Slurm job `366956` on real Wan QKV measured dense at about
  23.39 ms/call. `hard_c4_bestlogit_maxbias_tokenmajor` was only slightly
  faster at about 23.05 ms/call but worsened rel-L2 from the corrected
  bestlogit value of about 0.09417 to about 0.11392. `stripe` was faster at
  about 17.92 ms/call but worsened rel-L2 to about 0.22431.
- Why not repeat directly: these shortcuts reduce some merge-side work, but
  they change the compression distribution or routing correction enough to
  break the dense-error target.
- Rule: keep the entropy-preserving bestlogit/simplemerge merge for the main
  path. Use `maxbias` or `stripe` only as explicit low-quality ablations.

### Simplemerge `j16` and `j64` Merge Tiles

- Result: no speed improvement over the default `j32` simplemerge tile.
- Change tested: keep corrected bestlogit/simplemerge semantics and only change
  the merge tile from `j32` to `j16` or `j64`.
- Evidence: Slurm job `366961` smoke-tested random hard C4 tensors on MI300X.
  Default `hard_c4_bestlogit_simplemerge_tokenmajor` measured about
  0.1438 ms/call. `j16` measured about 0.1453 ms/call and introduced a tiny
  numerical difference versus the default; `j64` matched output but measured
  about 0.1464 ms/call. Older real-QKV tests had already shown non-simplemerge
  `j16`/`j64` slower than `j32`.
- Why not repeat directly: the simplified kernel does not change the earlier
  tile-size conclusion; `j32` is still the best balance for 52 query columns.
- Rule: keep `j32_w2` for the main simplemerge path.

## Current Best Direction

The best accepted direction so far is still:

- non-shared grid-local hard-route C4 compression,
- full compressed attention in the second stage,
- local-with-ith seed selection for lower error than random-with-ith,
- inline local seed-row computation to avoid reading the seed table in the C4
  merge kernel,
- assigned-logit reuse (`bestlogit`) inside the C4 merge kernel, which keeps
  the same assignment and full compressed-attention semantics while shaving a
  small amount of merge overhead,
- corrected seed-3 best-logit update, which lowered dense rel-L2 from about
  0.09970 to about 0.09417 in Slurm job `366959`,
- specialized `bestlogit_simplemerge`, which matches corrected bestlogit output
  exactly while removing generic merge-kernel branches and improving real-QKV
  attention time from about 23.40 ms/call to about 23.04 ms/call in job
  `366959`,
- token-major intermediate compressed-KV layout, which preserves the same
  full-attention math but makes merge stores more contiguous,
- standard K/V block layout for the current token-major `m32_n32` path, avoiding
  the older head-major experiment branch,
- `query_outer_chunk=10`, which was the fastest confirmed safe chunk size in
  `analysis/hard_c4_nohead_chunk_confirm_20260808`,
- direct-route Triton kernels to avoid unnecessary reshape/copy overhead.

This direction improves the old hard C4 implementation substantially, but it
has not yet achieved a clear 2x attention speedup over dense attention while
preserving the desired error level.
