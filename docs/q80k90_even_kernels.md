# Q80K90 even-capacity native kernels (C4, C6, C8, C10)

`analysis/q80k90_even_kernel.py` with `analysis/q80k90_even_{layout,select,merge,route}.cu`
is the Q80K90 operator of `kernels/experimental/q80k90_c4_c12` (v4) rebuilt on
the design of the Q90K80 even-capacity kernels (`analysis/q90k80_even_*`,
`docs/q90k80_kernels.md`). It is the same mathematical operator: the v4
candidate table, the RF0 Fisher selector whose first two seeds are the row /
column-pair intersections, hard ownership by the per-representative
log-softmax, softmax-weighted merged K/V with the entropy bias, exactly C
representatives per (query group, key block), zero K/V and -inf bias for an
empty representative, no singleton, raw-token, residual or dense route.

## Geometry and layouts

The 21 x 45 x 80 latent video is cut per frame into query groups of one row
(80 consecutive tokens, 45 per frame, 945 in all) and key blocks of two
adjacent columns (90 keys, 40 per frame, 840 in all); key j of block
(frame, pair) is the token at row j % 45, column 2*pair + j // 45. This is the
transpose of the Q90K80 cut (query group = two columns, key block = one row),
so the port swaps the two roles:

| | Q90K80 even | Q80K90 even |
|---|---|---|
| query group / key block | 90 / 80 tokens | 80 / 90 tokens |
| groups x blocks per head | 840 x 945 | 945 x 840 |
| merge score GEMM | m64n80k16 (80 keys) | m64n96k16 (90 keys + 6 masked) |
| route GEMMs | m64n96k16 (90 queries + 6 pad) | m64n80k16 (80 queries, no pad) |
| Q, K, V access | permuted copies (fused layout kernel 1.8 ms) | in place, one fragment-ordered Q copy (0.4 ms) |
| route output | inverse layout kernel | written in place |

A query group is 80 consecutive tokens of the caller's `[1, N, H, 128]`
tensors, so the selector gathers its candidate queries and probe keys, the
merge gathers its key block and the route reads Q and writes the output in
that layout directly. Workspaces: seeds `[H, 45, 840, C]` (as v4), merged K/V
`[hc, 840, 945, C, 128]` (key-block major, so a compact merge tile is one
contiguous run of rows and leaves through two TMA stores), bias
`[hc, 945, 840, C]` (group major, identical to v4). head_chunk 8 (default)
keeps 2 x C x 1.63 GB of merged K/V (C8 26 GB, C10 32.6 GB).

## Kernels

- select: frame loop over the 21 key frames with the 32 candidate queries of a
  row in shared memory, FP32 greedy Fisher residual loop, REDUX argmax (port of
  the Q90K80 selector). One CTA holds one column pair and four query rows.
- merge: compact 64-row tiles of GPT groups (C4 16, C6 10, C8 8, C10 6), next
  tile's seeds and fragment-ordered queries prefetched, fused 32-bit ownership
  argmax, stmatrix staging to a 128B-swizzled buffer and TMA stores, 3 CTAs/SM,
  three (C4) or four tile slices per (key block, head).
- route: one CTA per (query group, head), producer warpgroup with a 5-D TMA
  box per 64-key chunk (piecewise boxes for C6/C10), two consumer warpgroups,
  S^T = K Q^T and O^T += V^T P^T, lazy-rescaling softmax (FA4 style, tau 8).

## Usage

```python
from analysis.q80k90_even_kernel import Q80K90Even
op = Q80K90Even(8)            # 4, 6, 8 or 10; head_chunk=8
out = op(q, k, v)             # CUDA BF16 [1, 75600, 40, 128], token order 21 x 45 x 80
```

Set `TORCH_EXTENSIONS_DIR` to a private writable directory first. Each call
returns a new token-major output tensor; the workspace is reused, so use one
instance per concurrent stream. Hopper sm_90a, CUTLASS headers from
`flashinfer/data/cutlass/include`.

Q, K and V are read through their token and head strides, so both a
contiguous token-major `[1, N, H, 128]` tensor and the head-major view an
attention processor holds (`x.transpose(1, 2)` of a contiguous `[1, H, N, 128]`)
are consumed without a copy (`input_strides`); anything else is made
contiguous first. The output is always token-major, so a processor that does
`out.transpose(1, 2).flatten(2, 3)` gets a free view. On head-major inputs this
removes three 774 MB copies per call: C4 53.36 -> 52.01 ms (-1.34, 22/24
rounds), C8 100.56 -> 99.13 ms (-1.43, 21/24), outputs bitwise identical to
the copying path for both layouts.

## Validation (synthetic QKV, head_chunk 8)

Against the v4 Triton path (FP32 selector as in production) on the same QKV:

| Capacity | Output rel-L2 | Seeds changed | Bias rel-L2 (last chunk) | Merged K rel-L2 | Empty-slot mismatches |
|---|---:|---:|---:|---:|---:|
| C4 | 3.09e-3 | 0 | 1.1e-4 | 7.4e-4 | 0 |
| C6 | 3.16e-3 | 1.1e-7 | 1.8e-4 | 9.1e-4 | 0 |
| C8 | 3.18e-3 | 1.7e-7 | 2.5e-4 | 1.0e-3 | 0 |
| C10 | 3.23e-3 | 2.6e-7 | 3.3e-4 | 1.2e-3 | 0 |

The output difference is the Q90K80 lazy-route tier (3.0-3.2e-3 there). The
native FP32 seeds are closer to the FP64 selector than v4's Triton FP32 seeds
(C8: 2.5e-7 vs 4.1e-7 changed). Consecutive calls are bitwise identical.

## Synthetic timing (random N(0,1) QKV, head_chunk 8, 20 calls, interleaved with v4)

Final build (merge `MERGE_SUM_WAYS=2`), scai6 GPU 4, 2026-09-25:

| Capacity | layout | select | merge | route | Full call mean / median (ms) | v4 same run (ms) | Change |
|---|---:|---:|---:|---:|---:|---:|---:|
| C4 | 0.40 | 3.74 | 27.7 | 17.3 | 50.4 / 50.1 | 64.3 | -22 % |
| C6 | 0.40 | 5.13 | 46.9 | 28.4 | 80.3 / 80.0 | 110.2 | -27 % |
| C8 | 0.40 | 6.72 | 53.6 | 36.9 | 95.7 / 96.0 | 124.1 | -23 % |
| C10 | 0.40 | 8.19 | 81.7 | 48.6 | 137.0 / 137.0 | 229.3 | -40 % |

On scai7 GPU 3 the same day the first build measured C8 95.8 ms against the
Q90K80 even-capacity C8 at 93.4 / 91.3 ms (select 6.10, merge 48.4, route 37.9,
fused layout 1.82).
The Q80K90 merge carries the larger key block (90 keys, padded to 96 for the
WGMMA N granularity, against 80 exact keys: about 20 % more scalar work per
(query group, key block) row) and the route the smaller one (840 x C instead
of 945 x C keys per query and no padded query column). The C8 route reads the
130 GB of merged K/V per call in 33-37 ms, 10-25 % above the DRAM time at
4.4 TB/s; the merge is latency bound at three warps per scheduler as in
Q90K80.

## Real QKV (96-case protocol)

Fresh frozen StepDistill dense trajectories (3 prompts x 4 steps), layers 0, 5,
..., 35, all 40 heads; per case and capacity the new operator, the production
v4 operator, the Q90K80 even-capacity operator of the same capacity and dense
Flash SDPA are timed three times each on the same QKV in rotating order
(`analysis/q80k90_even_real40.py --with-q90`, scai6 GPU 4, 2026-09-25,
`docs/q80k90_even_real40.json`; build with `MERGE_SUM_WAYS=2`, before the C4
three-slice default and the strided inputs, which do not change the numbers
of contiguous inputs beyond the -0.77 ms C4 merge gain):

| Capacity | New (ms) | v4 (ms) | Q90K80 even (ms) | Dense (ms) | New vs dense | New - v4 | New - Q90K80 | rel-L2 new vs v4 (mean / max) | rel-L2 to dense (new = v4) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C4 | 48.51 | 65.63 | 50.68 | 326.07 | 6.72x | -17.12 (96/96) | -2.17 (96/96 faster) | 1.4e-3 / 1.8e-3 | 0.207 |
| C6 | 80.20 | 112.22 | 75.60 | 327.46 | 4.08x | -32.01 (96/96) | +4.60 | 1.4e-3 / 1.7e-3 | 0.178 |
| C8 | 95.45 | 124.79 | 93.82 | 329.80 | 3.46x | -29.34 (96/96) | +1.63 (10/96 faster) | 1.5e-3 / 1.8e-3 | 0.157 |
| C10 | 137.57 | 228.59 | 130.03 | 329.01 | 2.39x | -91.02 (96/96) | +7.54 | 1.4e-3 / 1.7e-3 | 0.140 |

The rel-L2 to bf16 dense is identical to v4's to four digits for every
capacity; the new operator is faster than dense and than v4 in every case.

## End-to-end video timing

I2V (Wan2.1-I2V-14B-720P, the frozen I2V snapshot's unchanged generate.py with
only the adapter's kernel factory replaced, `analysis/q80k90_even_i2v_timing.py`;
720x1280, 81 frames, 50 FlowUniPC steps, CFG 5, prompts 000-004 of the
sparkler_v1 default inputs, base seed 0 + index; C8, 4000 attention calls per
video; scai7 GPU 7, 2026-09-25):

| Video | Batch 1 (first build) s | Batch 2 (final C8) s | v4 C8 reference s |
|---|---:|---:|---:|
| 000 | 1071.1 (cold Triton cache) | 787.3 | 1010.1 (cold start) |
| 001 | 782.7 | 781.2 | 931.8 |
| 002 | 786.5 | 783.5 | 932.6 |
| 003 | 786.7 | 784.6 | 929.7 |
| 004 | 784.5 | 782.8 | 928.9 |
| Mean | 842.3 (785.1 without 000) | **783.9** | 946.6 (930.8 without 000) |
| Attention per call | 98.7 ms | 98.3 ms | 137-156 ms |

The v4 reference is the earlier production run (scai6 GPU 4, same prompts);
batch 1's first video paid a cold Triton cache in this account (VAE padding
autotune); batch 2 ran warm. Non-attention time is about 290-296 s per video
in all runs.

T2V teaser (Wan2.1-T2V-14B, samples 000/015, seed 0, scai6 GPU 4, the same GPU,
fastops and protocol as the paper's full-attention measurement; worker in the
paper repository's `figure_sources/gca_teaser/measurements/gca_c4_even_20260925/`):
C4 attention 218.93 s + other 335.43 s = 554.37 s per video (54.73 ms per call),
against full attention 1343.12 + 334.33 = 1677.45 s: 3.03x end-to-end and 6.13x
attention. The first build, which copied the processor's head-major Q/K/V
before every call and used four C4 merge slices, measured 562.82 s (56.71 ms
per call, 2.98x).

## Sustained load: why the video pipeline sees a higher per-call time

Inside video generation the H200 sits at its 700 W power cap and the SM
clock drops from 1980 MHz to about 1750 MHz (I2V) or 1600 MHz (the
diffusers T2V pipeline, whose non-attention kernels draw more power). The
merge and the selector are clock bound and slow down in proportion; the
route is DRAM bound and does not (`analysis/q80k90_even_i2v_diag.py`, 800
in-pipeline calls, then the same QKV re-timed in the same process):

| I2V C8 call (ms) | select | merge | route | total | SM clock |
|---|---:|---:|---:|---:|---:|
| first call, cold GPU | 6.6 | 52.4 | 31.5 | 90.9 | 1980 MHz |
| in the pipeline, steady | 8.4 | 59.5 | 32.2 | 100.5 | 1747 MHz mean |
| same QKV isolated right after | 8.0 | 58.0 | 32.0 | 98.4 | 1706 MHz |
| isolated after an L2 flush | 8.5 | 59.5 | 32.1 | 100.6 | |
| isolated behind a power load | 9.2 | 61.1 | 32.4 | 102.8 | 1597 MHz |

The adapter adds 0.01 ms (no copies, no fallback); the pipeline's only
intrinsic cost is a cold L2 (about 2 ms). The 96-case protocol above
measured 95.5 ms because the workspace release/allocation between cases
lets the GPU cool. In sustained mode (`--sustained 3`: operators kept alive,
three untimed calls before every timed one) the same protocol gives C8
99.66 ms (v4 124.3, Q90K80 even 97.4, dense 327.4; 3.29x), which is the
in-pipeline number. Per-call figures for video generation should be quoted
from this mode. T2V C4 behaves the same (48.5 -> 54.9 ms at 1594-1615 MHz).

Tried against this and closed: psi-permuted K/V tiles with direct
register-to-global stores (`MERGE_PSI_K/V`; V direct is bitwise identical
and kept as the default: -0.95 ms in the benchmark state, neutral under a
power load; K direct as well -1.2 / +0.8 ms, numerics tier). Under the cap
time tracks energy, and the merge's energy is its WGMMA work and the 130 GB
of merged K/V it writes, neither of which a store-path change reduces.

## A/B (interleaved on one QKV, `analysis/q80k90_even_ab.py`)

Adopted: three merge tile slices per (key block, head) for C4 (20 tiles per
CTA instead of 15: merge -0.77 ms, 18/20 rounds; bitwise identical), four for
C6/C8/C10 (three or two slices: C6 +0.46/+1.73, C10 +0.61/+1.43 ms; C8 three
slices neutral). head_chunk 10 instead of 8 is neutral (C4 +0.01 ms).

Adopted: `MERGE_SUM_WAYS=2` (two partial sums in the 24-term row reductions):
merge C8 -0.94 ms (19/24 rounds faster), C10 -0.63 (16/16), C4 -0.35 (17/20),
C6 -0.15 (9/16, neutral); merged K rel-L2 1.8e-4 against the serial order.

Measured and not adopted (C8 unless noted): `MERGE_SUM_WAYS=4` (+1.6 ms,
spills), `MERGE_GATHER_LATE=0` (+5 ms, spills 80 B), `MERGE_SPLIT_ISSUE=0`
(neutral over 24 rounds), tile slices other than four per (key block, head)
(15/20/60 tiles per CTA +1.5..2.5 ms, 40 neutral), `MERGE_PACKED=0` (C6 +10,
C10 +35 ms) and `=2` (C6 +0.6, C10 +2.2 ms), and for the selector
`SEL_MINBLOCKS=4/6/7` (+0.24/+0.12/+0.38 ms: the selector is shared-memory
limited to five 4-warp CTAs per SM, so higher bounds only add spills), three
warps per CTA (+0.23 ms) and five warps per CTA (no idle warp, 4 CTAs/SM,
+0.12 ms). The selector's last CTA per column pair has three idle warps; that
costs nothing measurable. Route: the lazy-rescaling threshold 12 or 16 instead
of 8 (+0.15 / +0.17 ms). Merge: the tile slice as the fastest grid index, so
the four slices of a (key block, head) run together and share their K/V in L2
(+3.6 ms, 1/24 rounds faster).
