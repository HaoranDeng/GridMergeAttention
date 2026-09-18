# Hard C4 IO Model

This note estimates logical HBM traffic for one Wan 480p self-attention call.
It is meant as a kernel-design sanity check, not a measured profiler result.
The model ignores L2 cache reuse, masked-lane implementation details, and
compiler scheduling.

Default Wan 480p shape:

```text
outer = 21 frames
rows  = 30
cols  = 52
heads = 12
dim   = 128
C     = 4
dtype = bf16
```

The maintained hard-C4 implementation has two stages:

1. Merge stage: for each `30-query x 52-key` grid block, load four local seed
   query rows, load one key row and one value row, then write four compressed
   K/V tokens and one routing-bias value per compressed token.
2. Route stage: read the compressed K/V plus routing bias and run full
   compressed attention.

Default estimate from:

```bash
python analysis/estimate_attention_io.py
```

| Path / Part | Logical traffic | 128-d bf16 vector units |
| --- | ---: | ---: |
| FlashAttention total, block M=128 | 51.728 GB | 202.064 M |
| flash q load | 0.101 GB | 0.393 M |
| flash k load | 25.764 GB | 100.639 M |
| flash v load | 25.764 GB | 100.639 M |
| flash output store | 0.101 GB | 0.393 M |
| Hard C4 total | 46.961 GB | 183.440 M |
| hard merge seed-q load | 8.454 GB | 33.022 M |
| hard merge K load | 2.113 GB | 8.256 M |
| hard merge V load | 2.113 GB | 8.256 M |
| hard intermediate store | 17.039 GB | 66.560 M |
| hard route Q load | 0.101 GB | 0.393 M |
| hard intermediate load | 17.039 GB | 66.560 M |
| hard output store | 0.101 GB | 0.393 M |

Key ratios:

```text
hard / flash logical traffic                 = 0.908x
hard intermediate store+load fraction        = 72.6%
hard without intermediate IO / flash traffic = 0.249x
```

So eliminating intermediate K/V/bias traffic is the largest remaining IO target:
it removes at most about `72.6%` of hard-C4's logical HBM traffic and would give
an optimistic `~4.0x` IO reduction relative to the current FlashAttention-style
logical traffic.

That is only an upper bound. A fully fused kernel must still preserve the
second-stage full compressed attention. If it computes one query-column tile at
a time, it may lose the current merge stage's sharing across all 52 query
columns and reload K/V much more often. A useful fused design therefore needs
to keep enough query-column and query-row work in SRAM to consume merged K/V
before writing it out, without making the accumulator tile too large.

The query-column tile is the key tradeoff. A fused design that consumes
compressed K/V immediately but only covers a small number of query columns must
reload the original K/V rows once per query-column tile:

| Fused query-column tile | Logical traffic | Versus FlashAttention |
| ---: | ---: | ---: |
| 1 | 228.45 GB | 4.416x |
| 4 | 63.60 GB | 1.230x |
| 8 | 38.24 GB | 0.739x |
| 16 | 25.56 GB | 0.494x |
| 32 | 17.11 GB | 0.331x |
| 52 | 12.88 GB | 0.249x |

This means a useful fused kernel probably needs a reasonably large column tile,
roughly `J_TILE >= 16`, otherwise the saved intermediate traffic is replaced by
extra original K/V loads. The hard part is keeping that many query columns and
enough query rows live without making the FP32 accumulator tile too large.

Approximate per-program live tensor footprint from the same script:

| I_TILE | J_TILE | Approx. live tensors |
| ---: | ---: | ---: |
| 8 | 8 | 89.0 KiB |
| 8 | 16 | 152.0 KiB |
| 8 | 32 | 278.0 KiB |
| 16 | 8 | 137.5 KiB |
| 16 | 16 | 249.0 KiB |
| 16 | 32 | 472.0 KiB |
| 30 | 8 | 222.4 KiB |
| 30 | 16 | 418.8 KiB |
| 30 | 32 | 811.5 KiB |

This is why the naive fused direction is awkward. The IO model wants a large
`J_TILE`, but the live-state model wants small `I_TILE * J_TILE`. The old
streaming fused prototypes were slow because they picked the small-live-state
side and lost the tensor-core-friendly merge/route tiles. A promising new
fusion would need to avoid both extremes, for example by fusing only part of
the pipeline or by keeping the route accumulator small while still sharing K/V
across many query columns.
