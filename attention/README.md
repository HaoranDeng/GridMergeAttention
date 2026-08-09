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
- `hard_route.py`: thin multi-token hard-route entry point. It chooses seed
  queries, then runs either the readable reference path or the optimized C4
  path.
- `hard_route_pytorch.py`: readable PyTorch reference for hard assignment,
  masked softmax, KV merge, and routing-bias construction.
- `hard_route_backend.py`: direct-route backend parsing and optimized routing
  dispatch. This file is backend plumbing, not the algorithmic reference.
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
reference merge. `hard_route_backend: hard_c4` is the current optimized C4
Triton path.

For understanding or changing the algorithm, start from
`hard_route_pytorch.py`. The main helpers are:

- `assign_keys_to_seeds`: hard-assign every key to one seed query.
- `masked_cluster_log_softmax`: run softmax inside each assigned key cluster.
- `hard_route_merge_weights`: combines assignment, masked softmax, and
  entropy routing-bias construction.
- `hard_route_merge_kv_pytorch`: regular `30q x 52k` PyTorch reference merge.
- `hard_route_merge_kv_transpose_pytorch`: transposed `52q x 30k` reference
  merge.

Its core path is intentionally written as:

```text
seed/key logits -> hard assignment -> masked softmax -> merge K/V -> routing bias
```

The Triton code should be treated as an optimization of that reference path,
not as the source of the algorithm.

Then read `hard_route.py` only for the end-to-end control flow:

```text
seed query selection -> KV merge -> full compressed attention
```

The current recommended C4 config keeps the original index-matched query as one
seed:

```yaml
seed_mode: local_with_ith
num_merge_tokens: 4
hard_route_backend: hard_c4
hard_route_routing_backend: triton_direct_m32_n32_w1_s1_exp2_tokenmajor
```

This path is non-shared and grid-local: each
`(query_outer, key_outer, query_column, key_row)` block gets its own compressed
KV tokens. The second step is full compressed attention, not sparse or
windowed attention.

`hard_c4` is now the only supported direct C4 backend name. Old generic
`triton`, `triton_c4`, `triton_c4_j16`, and long experiment-suffix aliases were
removed so new work has only two paths to read: PyTorch `sdpa` for correctness
and `hard_c4` for speed.

For new optimization work, treat the PyTorch file as the readable reference and
the `hard_c4` direct-route backend as the current optimized
baseline.
Older suffixes for representative keys/values, compact or merge-weight
reconstruction, seed-key approximations, key-outer grouping, global/shared
compression, no-bias routing, stripe routing, and grouped/windowed routing are
not accepted by the backend parser. They are tracked only as negative results
in `../docs/negative_results.md` so they are not repeated.

## Select Route

With `layout: select_route` and `num_merge_tokens=C`, each seed independently
compresses the full key group. Each query then selects the best candidate
compressed KV using the dot-product-plus-entropy score.

## Removed Experiments

The shared/group-shared compression paths, global compression paths, no-bias
paths, and local/windowed second-stage routing paths were removed from the code.
Packed K/V/bias intermediates and precomputed-logit materialization were also
removed after they proved unsafe or slower on the real Wan hard C4 shape. These
rejected directions are tracked in `../docs/negative_results.md`.
