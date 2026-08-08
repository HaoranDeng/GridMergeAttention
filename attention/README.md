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
- `hard_route.py`: multi-token hard-route dispatcher and main control flow.
- `hard_route_pytorch.py`: readable PyTorch reference for hard assignment,
  masked softmax, KV merge, and routing-bias construction.
- `hard_route_backend.py`: direct-route backend parsing and Triton wrapper
  functions.
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

For understanding or changing the algorithm, start from
`hard_route_pytorch.py`. The main helpers are:

- `assign_keys_to_seeds`: hard-assign every key to one seed query.
- `masked_cluster_softmax`: softmax only inside the keys assigned to each seed.
- `entropy_routing_bias`: entropy term used as the compressed-token bias.
- `hard_route_merge_weights`: combines assignment, masked softmax, and entropy
  bias construction.
- `hard_route_merge_kv_pytorch`: regular `30q x 52k` PyTorch reference merge.
- `hard_route_merge_kv_transpose_pytorch`: transposed `52q x 30k` reference
  merge.

Its core path is intentionally written as:

```text
seed/key logits -> hard assignment -> masked softmax -> merge K/V -> routing bias
```

The Triton code should be treated as an optimization of that reference path,
not as the source of the algorithm.

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

The `maxbias` suffix is a small routing-bias approximation candidate for the
exact materialized K/V path. It keeps the log-normalizer term but uses
`max(logit) - logsumexp(logit)` instead of
`sum_l alpha_l logit_l - logsumexp(logit)`. This is not the rejected no-bias
path; it still runs full compressed attention with a per-token bias.

The following suffixes are experimental memory-traffic/merge-compute
approximations. They do not change the second stage: after compression, the
model still runs full compressed attention over every compressed token.

- `topkey`: compressed K is the top-logit original key in the seed cluster;
  compressed V still uses the normal softmax merge.
- `topkeyscale`: like `topkey`, but scales the top key so the seed query sees
  the same expected logit as the normal softmax-merged K.
- `topkeyaffine`: like `topkey`, but fits one scalar scale and one logit shift
  against the four seed-query logits before full compressed attention.
- `topkeyaffine_topvalue`: combines the affine top-key approximation with
  top-logit V, skipping both softmax K merge and softmax V merge.
- `topkeyaffine_seedvalue`: combines the affine top-key approximation with the
  forced representative V, also skipping both softmax K merge and softmax V
  merge.
- `indirectkey`: for top-key variants, store only the top-key column index and
  optional scale instead of materializing compressed K; full compressed
  attention then gathers K from the original K tensor.
- `topvalue`: compressed K still uses the normal softmax merge; compressed V is
  the top-logit original value in the seed cluster.
- `top2value`: compressed K still uses the normal softmax merge; compressed V
  is approximated by the two largest value contributors in the seed cluster.
  This keeps a closer V approximation than `topvalue` while avoiding the full
  merged-V tensor.
- `top2seedvalue`: like `top2value`, but keeps the original softmax mass for
  the top two values and assigns the remaining value mass to the representative
  seed value. This is a more conservative V approximation candidate.
- `top3value`: compressed K still uses the normal softmax merge; compressed V
  is approximated by the three largest value contributors in the seed cluster.
- `top3seedvalue`: like `top3value`, but keeps the original softmax mass for
  the top three values and assigns the remaining value mass to the
  representative seed value. This is a more conservative V approximation.
- `topkv`: compressed K and V are both the top-logit original token in the seed
  cluster.
- `seedvalue`: compressed V is the forced representative seed value.
- `indirectkv`: for `topkv`, store only the top-token column index instead of
  materializing compressed K/V; full compressed attention gathers original K/V
  by that index.
- `indirectvalue`: for top-value variants, avoid materializing compressed V;
  full compressed attention gathers original V by the stored top-token column
  index. This can be used with materialized K or indirect top-key K.
- `indirecttop2value`: for `top2value`, store two original value-column
  indices and one mixture weight instead of materializing compressed V. The
  second stage is still full compressed attention over every compressed token.
- `indirecttop2seedvalue`: for `top2seedvalue`, store two original value-column
  indices and two original softmax weights; the routing kernel gathers those
  two values plus the seed value and still runs full compressed attention.
- `indirecttop3value`: for `top3value`, store three original value-column
  indices and two mixture weights; the third weight is implicit. The routing
  kernel gathers three original values and still runs full compressed attention.
- `indirecttop3seedvalue`: for `top3seedvalue`, store three original
  value-column indices and three softmax weights; the remaining mass goes to
  the seed value during full compressed attention.
- `indirectseedvalue`: for seed-value variants, avoid materializing compressed
  V; full compressed attention gathers original V by seed id. This can be used
  with materialized K or indirect top-key K.

These variants are benchmark candidates only. Keep the `bestlogit` path above
as the default until a real-QKV benchmark shows a clear speed gain with
acceptable dense-attention error.

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
