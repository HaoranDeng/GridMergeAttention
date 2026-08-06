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
- `triton_c4_direct_route_j64`: wider-tile version of the route-layout shared
  C=4 kernel; kept as a stable comparison point.

`hard_route_routing_backend` controls the second compressed-attention step:

- `sdpa`: PyTorch SDPA with the routing bias.
- `triton`: fused Triton routing kernel for `q @ K_merged - bias`.
- `triton_m128_n32` / `triton_m256_n32`: faster fused routing kernels for the
  shared C=4 compressed attention path.
- `triton_m128_n32_s1`: `triton_m128_n32` with `num_stages=1`; fastest current
  routing backend for the default 480p C=4 setup.

The easiest experimental hook is `q_init` in `common.merge_query_row` and
`common.merge_query_column`:

- `ith`: key row `i` initializes with query row `i`
- `mean`: every key row initializes with the mean query
- `1st`: every key row initializes with the first query row
