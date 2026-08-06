# Attention Notes

`core.py` is the dense baseline. It wraps PyTorch scaled dot-product attention
and keeps the same call shape as Wan:

```text
q, k, v: [batch, sequence, head, dim]
out:     [batch, sequence, head, dim]
```

`token_merging.py` is only the GridMerge dispatcher. Each attention variant now
lives in its own file:

- `gridmerge_one.py`: one-token GridMerge / MonarchRT-style compression.
- `hard_route.py`: C-seed hard-route compression.
- `hard_route_group_shared.py`: experimental hard-route compression shared
  inside query-outer groups.
- `hard_route_shared.py`: experimental C-seed hard-route compression shared
  across query outers.
- `hard_route_triton.py`: optional fused Triton KV-compression kernel plus an
  experimental routing kernel for hard-route.
- `select_route.py`: C-seed select-route compression.
- `gridmerge_one_transpose.py`: one-token transpose layout.
- `hard_route_transpose.py`: hard-route transpose layout.
- `select_route_transpose.py`: select-route transpose layout.
- `common.py`: shared reshape, seed sampling, and merge-weight utilities.

For default
480p/81-frame Wan generation, the sequence is rearranged into:

```text
q: [batch, query_outer=21, query_row=30, query_column=52, head, dim]
k: [batch, key_outer=21, key_row=30, key_column=52, head, dim]
v: [batch, key_outer=21, key_row=30, key_column=52, head, dim]
```

In the default `hard_route` / `select_route` layout, each fixed
`(query_outer, key_outer, query_column, key_row)` tile is a
`30 queries x 52 keys` attention block. `hard_route_transpose` and
`select_route_transpose` swap the row/column grouping and use
`52 queries x 30 keys` blocks.

With `num_merge_tokens=1`, GridMerge compresses each key group into one merged
key/value token, then routes the query group to those merged tokens.

```text
merge_weights:   [52]
routing_weights: [30]
tile ~= routing_weights[:, None] * merge_weights[None, :]
```

The only approximation step is:

```text
select merge query -> merge key/value tokens -> route queries -> aggregate values
```

With `num_merge_tokens=C>1`, the two multi-token modes are:

- `hard_route`: sample C seed queries, hard-route each key to its best seed,
  then merge one KV token per seed.
- `hard_route_group_shared`: same hard-route compression, but one compressed KV
  set is shared by each query-outer group. `query_outer_chunk` is the group size.
- `hard_route_shared`: same C-seed hard-route compression, but each key outer
  is compressed once with its matching query outer and then reused for every
  query outer.
- `select_route`: each seed independently compresses the full key group; each
  query then selects the best compressed KV candidate with the
  dot-product-plus-entropy score.

`seed_mode: fast_random_with_ith` keeps the same index-matched seed used by
`random_with_ith`, but chooses the remaining seeds with a lightweight
deterministic hash instead of ranking every candidate query row.

`hard_route_backend` controls the hard-route KV-compression backend:

- `sdpa`: default PyTorch implementation, kept as the reference path.
- `triton`: fused Triton hard-assignment/KV-compression kernel followed by the
  same SDPA routing path.
- `triton_c4`: C=4-only Triton kernel that reuses seed-key dot products between
  hard assignment and KV merge.
- `triton_c4_direct_route_j16`: shared C=4 kernel that writes compressed K/V
  directly in routing layout.
- `triton_c4_direct_route_j32_w2`: route-layout shared C=4 kernel using
  `block_j=32` and `num_warps=2`; fastest current merge backend for the
  default 480p C=4 setup.
- `triton_c4_direct_route_j32_w2_forcekeys`: same route-layout shared C=4
  kernel, but directly assigns the first four key columns to the four seeds.
  This removes the empty-cluster fallback reductions and is an explicit
  low-overhead C=4 experiment.
- `triton_c4_direct_route_j32_w2_forcekeys_exp2merge`: the same force-key
  merge backend, but computes the merge softmax with base-2 exponentials. It
  is kept separate from `triton_c4_direct_route_j32_w2_forcekeys` so the
  original fp32-exp path remains available. This is the fastest measured
  low-error C=4 merge backend on MI300X.
- `triton_c4_direct_route_j32_w2_forcekeys_exp2merge_cached`: same math as the
  backend above, but reuses no-grad intermediate K/V/bias workspace tensors to
  reduce allocator overhead during inference. Outputs should match the
  non-cached backend exactly.
- `triton_c4_direct_route_j32_w2_forcekeys_exp2merge_nobias_cached`: same
  force-key/base-2/cached merge path, but skips routing-bias allocation and
  storage. Use only with a `*_nobias` routing backend; this changes the
  approximation.
- `triton_c4_direct_route_j64_w2_forcekeys_exp2merge_cached` /
  `triton_c4_direct_route_j64_forcekeys_exp2merge_cached`: wider-column merge
  tuning variants. They cover all 52 default query columns in one tile instead
  of two `j32` tiles, trading less repeated K/V loading for a larger per-kernel
  tile.
- `triton_c4_direct_route_j32_w2_forcekeys_exp2merge_maxprob_cached` /
  `triton_c4_direct_route_j32_w2_forcekeys_exp2merge_countbias_cached`:
  approximate routing-bias variants. `maxprob` uses the seed-cluster max-logit
  softmax normalizer as a cheap entropy proxy; `countbias` uses only the routed
  cluster size. Both change the approximation and are kept as explicit tuning
  hooks.
- `triton_c4_direct_route_j32_w2_forcekeys_exp2merge_bf16bias_cached`: same
  exact routing-bias math as the cached force-key/base-2 backend, but stores
  the intermediate routing bias in the K/V dtype to reduce routing-bias memory
  traffic.
- `triton_c4_direct_route_j32_w1_forcekeys_exp2merge_cached`,
  `triton_c4_direct_route_j16_w1_forcekeys_exp2merge_cached`, and
  `triton_c4_direct_route_j16_w2_forcekeys_exp2merge_cached`: merge-kernel
  tile/warp tuning variants for the cached force-key/base-2 path.
- `triton_c4_direct_route_j32_w2_nobias`: aggressive faster variant that skips
  routing-bias computation and storage. This changes the approximation, so it
  is kept as an explicit experiment rather than the default best path.
- `triton_c4_direct_route_j64`: wider-tile version of the route-layout shared
  C=4 kernel; kept as a stable comparison point.

`hard_route_routing_backend` controls the second compressed-attention step:

- `sdpa`: PyTorch SDPA with the routing bias.
- `triton`: fused Triton routing kernel for `q @ K_merged - bias`.
- `triton_m128_n32` / `triton_m256_n32`: faster fused routing kernels for the
  shared C=4 compressed attention path.
- `triton_m128_n32_s1`: `triton_m128_n32` with `num_stages=1`; kept as a
  stable comparison point.
- `triton_grid_m128_n32_s1`: direct grid-query version of `triton_m128_n32_s1`.
  It reads `[B, outer, row, col, head, dim]` query blocks and writes the same
  block layout directly, avoiding the flattened query/output copies. This is
  the fastest current routing backend for the default route-layout C=4 setup.
- `triton_grid_m128_n32_s1_exp2`: same direct grid-query routing backend, but
  computes softmax with base-2 exponentials. On MI300X this preserves the
  measured bf16 output while reducing routing latency. This is the fastest
  measured low-error C=4 routing backend on MI300X.
- `triton_grid_outer_local_r2_m128_n32_s1_exp2` through
  `triton_grid_outer_local_r6_m128_n32_s1_exp2`: experimental faster C=4
  routing backends. They keep the C4 compressed K/V tokens unchanged, but each
  query block routes only to a local union window of key outers. Smaller
  radii are faster and more approximate; larger radii are more conservative.
- `triton_grid_outer_local_r2_m128_n64_s1_exp2` through
  `triton_grid_outer_local_r6_m128_n64_s1_exp2`: same local-window rules, but
  use wider key tiles for routing-kernel tuning.
- `triton_grid_outer_local_r3_m64_n64_s1_exp2` through
  `triton_grid_outer_local_r6_m64_n64_s1_exp2`: smaller query tiles for the
  local-window rule. They reduce the number of key outers read per routing
  tile but launch more tiles.
- `triton_grid_outer_exact_r3_m128_n64_s1_exp2` through
  `triton_grid_outer_exact_r6_m128_n64_s1_exp2`: dynamic-window local routing.
  Each query tile scans only the exact key-outer window implied by the query
  outers it contains, instead of the conservative max-length union window.
  In the 90-call MI300X component profile, using the same C=4 merge backend:

  | backend | attention ms/call | speedup vs dense | rel-L2 vs dense output | output MSE |
  | --- | ---: | ---: | ---: | ---: |
  | exact R3 | 2.55 | 9.16x | 0.282 | 0.0247 |
  | exact R4 | 2.60 | 8.77x | 0.265 | 0.0218 |
  | exact R5 | 2.83 | 8.08x | 0.256 | 0.0204 |
  | exact R6 | 2.83 | 8.01x | 0.249 | 0.0192 |

  R4 is the current faster speed/quality tradeoff; R5 is the more conservative
  balanced point.
- `triton_grid_outer_exact_r5_m128_n128_s1_exp2`,
  `triton_grid_outer_exact_r5_m256_n64_s1_exp2`,
  `triton_grid_outer_exact_r5_m256_n128_s1_exp2`, and the corresponding R6
  `n128`/`m256` variants: tile-size tuning hooks for the exact-window backend.
- `triton_grid_outer_exact_r5_m128_n32_s1_exp2` and
  `triton_grid_outer_exact_r5_m128_n32_w4_s2_exp2`: narrower key-tile tuning
  hooks for the current R5 exact-window backend.
- `triton_grid_outer_exact_r4_m64_n64_s1_exp2` through
  `triton_grid_outer_exact_r6_m64_n64_s1_exp2`: smaller query-tile tuning hooks
  for the exact-window backend.
- `triton_grid_outer_exact_r5_m128_n64_w2_s1_exp2`,
  `triton_grid_outer_exact_r5_m128_n64_w8_s1_exp2`, and
  `triton_grid_outer_exact_r5_m128_n64_w2_s2_exp2`: warp-count tuning hooks
  for the current R5 exact-window backend.
- `triton_grid_outer_exact_r5_m128_n64_w4_s2_exp2` through
  `triton_grid_outer_exact_r5_m128_n64_w4_s4_exp2`: pipeline-stage tuning
  hooks for the current R5 exact-window backend.
- `triton_grid_m128_n32_s1_nobias`: matching aggressive routing backend for the
  no-bias merge path. It skips loading/subtracting routing bias.
- `triton_grid_outer_local_r4_m128_n32_s1_exp2_nobias`: no-bias variant of the
  `r4` local-window routing path.

The easiest experimental hook is `q_init` in `common.merge_query_row` and
`common.merge_query_column`:

- `ith`: key row `i` initializes with query row `i`
- `mean`: every key row initializes with the mean query
- `1st`: every key row initializes with the first query row
