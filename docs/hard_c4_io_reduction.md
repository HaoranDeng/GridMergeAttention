# Hard C4 IO Reduction Notes

This note records the current IO model for the default hard C4 GridMerge
kernel on Wan 480p block layout:

```text
Q: [batch, outer=21, row=30, column=52, head=12, dim=128]
K/V: same shape
C = 4 compressed tokens per 52-key row
dtype = bf16
```

## Current Two-Stage Kernel

The current Triton kernel has two stages:

1. Merge: compute four compressed K/V tokens and routing bias for each
   `30-query x 52-key` grid block.
2. Route: run full softmax attention over all compressed tokens.

The logical HBM traffic estimate is:

| Part | GB | Fraction |
| --- | ---: | ---: |
| seed-query loads | 8.454 | 18.0% |
| key loads | 2.113 | 4.5% |
| value loads | 2.113 | 4.5% |
| intermediate K/V/bias stores | 17.039 | 36.3% |
| route query loads | 0.101 | 0.2% |
| intermediate K/V/bias loads | 17.039 | 36.3% |
| output stores | 0.101 | 0.2% |
| **total** | **46.961** | **100%** |

The intermediate K/V/bias tensor accounts for about `72.6%` of this logical
traffic, counting both store and reload.

For reference, the dense FlashAttention-style logical traffic at
`FLASH_BLOCK_M=128` is about `51.728` GB.

## What Full Fusion Can Save

A fused kernel can only avoid the intermediate IO if it consumes the compressed
tokens immediately. However, the compressed K/V for a query-column tile is
shared by all 30 query rows in the block. If a fused program covers fewer than
30 query rows, it recomputes merge work for each query-row tile.

Estimated logical traffic:

| Fused tile | GB | vs current hard C4 | vs FlashAttention |
| --- | ---: | ---: | ---: |
| `I=30, J=52` | 12.882 | 0.274x | 0.249x |
| `I=30, J=16` | 25.562 | 0.544x | 0.494x |
| `I=30, J=8` | 38.243 | 0.814x | 0.739x |
| `I=16, J=16` | 50.923 | 1.084x | 0.984x |
| `I=16, J=8` | 76.284 | 1.625x | 1.475x |
| `I=8, J=16` | 101.645 | 2.165x | 1.965x |
| `I=8, J=8` | 152.367 | 3.245x | 2.946x |

The useful region is therefore narrow:

- Covering all `30` query rows is important. Otherwise merge recomputation
  removes the IO benefit.
- `J=52` gives the best IO reduction, but the logical live state is about
  `1302` KiB, too large for a practical single Triton program.
- `I=30, J=16` reduces logical traffic by about `45.6%`, but its logical live
  state is about `419` KiB.
- `I=30, J=8` is closer to a feasible live state at about `222` KiB, but only
  reduces logical traffic by about `18.6%`.

## H200 Stage Profile

Short synthetic H200 profile for the current hard C4 kernel:

```text
experiment: hard_c4_io_recheck_20260813
GPU: NVIDIA H200
shape: Q [1, 10, 30, 52, 12, 128], K/V [1, 21, 30, 52, 12, 128]
merge_ms: 5.247
route_ms: 1.970
full_ms: 7.245
merge_pct_of_full: 72.4%
route_pct_of_full: 27.2%
```

This means the next speed path should not merely remove the intermediate
tensor. It also needs to keep the merge stage from being recomputed across
query-row tiles.
