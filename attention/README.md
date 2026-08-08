# Attention Module

This directory keeps one file per attention method so each variant can be read
or reused independently.

## Files

- `core.py`: dense PyTorch scaled dot-product attention wrapper used as the
  baseline.
- `token_merging.py`: dispatcher that reshapes Wan video tokens into grid
  blocks and calls the selected GridMerge variant.
- `common.py`: shared block reshaping, seed selection, merge-query selection,
  and biased SDPA helpers.
- `gridmerge_one.py`: one-token GridMerge / MonarchRT-style compression.
- `gridmerge_one_transpose.py`: one-token transpose layout.
- `hard_route.py`: multi-token hard-route compression.
- `hard_route_transpose.py`: hard-route transpose layout.
- `hard_route_triton.py`: Triton kernels used by the current hard-route path.
- `select_route.py`: multi-token select-route compression.
- `select_route_transpose.py`: select-route transpose layout.

## Shapes

Wan self-attention enters as:

```text
q, k, v: [batch, sequence, head, dim]
```

For default 480p/81-frame Wan inference, token merging rearranges this into:

```text
q: [batch, query_outer=21, query_row=30, query_column=52, head, dim]
k: [batch, key_outer=21, key_row=30, key_column=52, head, dim]
v: [batch, key_outer=21, key_row=30, key_column=52, head, dim]
```

In the default layout, one GridMerge block fixes
`(query_outer, key_outer, query_column, key_row)`, so each block contains:

```text
30 queries x 52 keys
```

The transpose layout swaps row/column grouping:

```text
52 queries x 30 keys
```

## One-Token GridMerge

With `num_merge_tokens=1`, each key group is compressed into one merged key and
one merged value, then all queries attend to the compressed KV tokens.

For one `30q x 52k` block, the rank-1 view is:

```text
merge_weights:   [52]
routing_weights: [30]
tile ~= routing_weights[:, None] * merge_weights[None, :]
```

`q_init` controls which query initializes the key merge:

- `ith`: key row `i` uses query row `i`.
- `mean`: every key row uses the mean query.
- `1st`: every key row uses the first query row.

## Hard Route

With `layout: hard_route` and `num_merge_tokens=C`, each block:

1. Selects `C` seed queries.
2. Assigns every key in the block to its best seed by seed-query/key dot
   product.
3. Softmax-merges one key/value token per seed cluster.
4. Runs full compressed attention from all queries to all compressed KV tokens.

In `hard_route.py`, `hard_route_backend: sdpa` is the readable PyTorch
reference merge. Triton backends implement the same two-stage algorithm but
replace the merge and/or compressed-attention stages with kernels.

The current recommended C4 config keeps the original index-matched query as one
seed:

```yaml
seed_mode: local_with_ith
num_merge_tokens: 4
hard_route_backend: triton_c4_direct_route_j32_w2_exp2merge_forcekeys_fp16kvbias_s1_fastdot_bestlogit_tokenmajor_inline_local_seed
hard_route_routing_backend: triton_direct_m32_n16_w1_s1_exp2_tokenmajor
```

This path is non-shared and grid-local: each
`(query_outer, key_outer, query_column, key_row)` block gets its own compressed
KV tokens. The second step is full compressed attention, not sparse or
windowed attention.

The `bestlogit` suffix reuses the already computed winning seed/key logit when
forming each assigned cluster. It keeps the same assignment and full compressed
attention semantics as the full-core path.

## Select Route

With `layout: select_route` and `num_merge_tokens=C`, each seed independently
compresses the full key group. Each query then selects the best candidate
compressed KV using the dot-product-plus-entropy score.

## Removed Experiments

The shared/group-shared compression paths, global compression paths, no-bias
paths, and local/windowed second-stage routing paths were removed from the code.
They are tracked as rejected directions in `../docs/negative_results.md`.
