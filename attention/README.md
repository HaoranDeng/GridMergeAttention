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
- `select_route.py`: C-seed select-route compression.
- `gridmerge_one_transpose.py`: one-token transpose layout.
- `hard_route_transpose.py`: hard-route transpose layout.
- `select_route_transpose.py`: select-route transpose layout.
- `global_attention.py`: global seed-token merging without local grid blocks.
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
- `select_route`: each seed independently compresses the full key group; each
  query then selects the best compressed KV candidate with the
  dot-product-plus-entropy score.

The easiest experimental hook is `q_init` in `common.merge_query_row` and
`common.merge_query_column`:

- `ith`: key row `i` initializes with query row `i`
- `mean`: every key row initializes with the mean query
- `1st`: every key row initializes with the first query row
