# Attention Module

The current GCA default is exact-capacity C4, selected by
`configs/token_merging_wan_720p_fisher_c4_exact_capacity_fewstep.yaml`.
Selection uses frame zero and merging uses current-frame queries. Every key
block produces exactly four merged K/V representatives. SQ1, singleton
peeling, raw-K/V side paths, residual tokens, and fifth route entries are
forbidden for C4. C8 analogously means exactly eight merged representatives.
See [the exact-capacity contract](../docs/exact_capacity_contract.md).

The fixed-seed and 480p paths described below remain explicit legacy options;
they are not the new generation default.

This directory keeps only the four maintained attention paths used for current
Wan experiments: SDPA, dense FlashAttention Triton, original MonarchRT, and the
hard-C4 Grid Compressed Attention implementation.

## Files

Kernel filenames encode the maintained model and resolution. Backend labels in
YAML remain stable (`flash_attn_triton`, `monarch_rt`, and `hard_c4`).

| File | Maintained model/config |
|---|---|
| `flash_attn_wan21_1_3b_480p_triton.py` | Wan2.1-T2V-1.3B, 480p, dense FlashAttention reference |
| `monarch_rt_wan21_1_3b_480p_triton.py` | Wan2.1-T2V-1.3B, 480p, one-step MonarchRT |
| `hard_c4_wan21_1_3b_480p_triton.py` | Wan2.1-T2V-1.3B, 480p, hard C4 |
| `hard_c4_wan21_14b_720p_triton.py` | Wan2.1-T2V-14B, 720p, hard C4 |

Supporting files:

- `core.py`: dense PyTorch scaled dot-product attention wrapper.
- `token_merging.py`: Wan token reshaping plus PyTorch/Triton dispatch.
- `hard_route_pytorch.py`: complete readable PyTorch hard C4 reference.

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

1. Pick four contiguous local seed query rows. For key row `k`, use
   `start = clamp(k - 1, 0, rows - 4)` and seed rows
   `[start, start + 1, start + 2, start + 3]`, so the index-matched row stays
   inside the seed window while the seed rows are contiguous in memory.
2. Normalize each seed's 52 seed-key logits over the key dimension, then assign
   each key to the seed that gives that key the largest normalized score.
3. Softmax-merge K and V inside each seed cluster. Empty seed clusters are
   kept as compressed tokens but masked by the routing bias in the second
   attention stage.
4. Compute the entropy routing bias.
5. Run full compressed attention from all 30 queries to all compressed KV
   tokens.

The two maintained implementations are:

- `hard_route_backend: sdpa`: readable PyTorch reference.
- `hard_route_backend: hard_c4`: default two-stage Triton path.

The default Triton merge is tuned for H100/H200-class GPUs and currently uses
about 48 KiB of shared memory per block on Triton 3.7.1/sm90. The four seed-Q
tiles are loaded as two pairs so the compiler can reuse their shared-memory
buffers without changing the four QK dots or hard-C4 arithmetic. Select the
`sdpa` reference on devices that cannot support that per-block footprint.

The Triton path preserves the same key-softmax hard assignment, entropy
routing bias, and full compressed-attention semantics. It is a two-stage
implementation: the first kernel writes merged K/V and routing bias, and the
second kernel reads them back to run full compressed attention with online
softmax. It does not use global/shared compression, no-bias routing, or
sparse/windowed second-stage attention.

The Wan2.1-14B 720p kernel uses a resolution-specific seed pattern. Interior
key row `i` uses `[i - 2, i, i + 1, i + 3]`; boundary rows use the contiguous
window above. This keeps four local seeds and the index-matched query while
covering a slightly wider neighborhood, with no additional seed-key dot
products. The 480p kernel keeps the contiguous pattern.
