# Attention Module

This directory now keeps only the maintained dense baseline and hard C4
GridMergeAttention path.

## Files

- `core.py`: dense PyTorch scaled dot-product attention wrapper.
- `token_merging.py`: Wan token reshaping plus hard C4 dispatch.
- `common.py`: block reshaping, local C4 seed selection, and biased SDPA.
- `hard_route.py`: hard C4 end-to-end control flow.
- `hard_route_pytorch.py`: readable PyTorch hard C4 reference.
- `hard_route_backend.py`: constants and dispatch for the optimized hard C4 path.
- `hard_route_triton.py`: Triton C4 merge and full compressed-attention kernels.

## Shapes

Wan self-attention enters as:

```text
q, k, v: [batch, sequence, head, dim]
```

For default 480p/81-frame Wan inference:

```text
q: [batch, query_outer=21, query_row=30, query_column=52, head, dim]
k: [batch, key_outer=21, key_row=30, key_column=52, head, dim]
v: [batch, key_outer=21, key_row=30, key_column=52, head, dim]
```

One hard C4 GridMerge block fixes
`(query_outer, key_outer, query_column, key_row)`, so each block contains:

```text
30 queries x 52 keys
```

## Hard C4

For each `30q x 52k` block:

1. Pick four local-with-ith seed queries. Seed 0 is the index-matched query row;
   the other three are nearby query rows.
2. Assign each of the 52 keys to the seed with the largest seed-query/key dot
   product.
3. Softmax-merge K and V inside each seed cluster.
4. Compute the entropy routing bias.
5. Run full compressed attention from all 30 queries to all compressed KV
   tokens.

The two maintained implementations are:

- `hard_route_backend: sdpa`: readable PyTorch reference.
- `hard_route_backend: hard_c4`: optimized Triton path.

The Triton path preserves the same hard assignment, entropy routing bias, and
full compressed-attention semantics. It does not use global/shared compression,
no-bias routing, or sparse/windowed second-stage attention.
