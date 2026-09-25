# Q90K80 RF0 kernels

Inference-only query-group/key-merge attention with exact capacities 12, 14
and 16. These are the source bytes used by the validated H200 experiments.
Each Python module loads its matching CUDA translation unit from the same directory.

## Operator

- Q groups contain two H45 chunks, K groups contain W80 keys.
- RF0 seed indices are reused across frames within each attention call.
- Both query/key intersections are retained, followed by Fisher selection from
  the balanced 32-candidate pool and Uniform16 probes (0, 5, ..., 75).
- Exactly C merged representatives are stored, even when tensor-core tiles
  use internal padding. Empty clusters use zero KV and negative-infinity bias.
- Every query attends over the full compressed route. No dense, raw-token,
  singleton or residual attention branch is included. No training is required.

## Environment and invocation

Tested with Linux, NVIDIA H200, PyTorch 2.8.0+cu128, Triton 3.7.1 and CUDA 13
nvcc. The native backend targets Hopper sm_90a and needs the CUTLASS headers
bundled in `flashinfer/data/cutlass/include` beside the PyTorch installation.
The Python modules also retain the earlier Triton merge implementation, but
the reported fast configuration explicitly selects the native CUDA backend.

Choose a writable extension cache before invoking the module (the frozen
source otherwise defaults to the experiment server cache):

```sh
export TORCH_EXTENSIONS_DIR=/tmp/q90k80_extensions
export TORCH_CUDA_ARCH_LIST=9.0a
export MAX_JOBS=2
```

```python
from analysis.q90k80c12_kernel import Q90K80C12

operator = Q90K80C12(
    merge_backend="cuda", native_tiles=105,
    route_m=128, route_n=64, head_chunk=4,
)
# q, k, v: CUDA BF16 [1, 75600, 40, 128], token-major.
# Token order corresponds to 21 x 45 x 80, head dimension 128.
output = operator(q, k, v)
```

For C14/C16 use `Q90K80C14`/`Q90K80C16` from the corresponding module with
the same configuration. Instances cache scratch tensors and should not be
shared across simultaneous calls. The validated full40-head shape requires
substantial scratch memory; these modules are specialized experimental kernels,
not a general arbitrary-shape attention API.

The full video pipeline used process-start
`LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` to avoid an observed
SentencePiece/native-extension C++ runtime conflict on the experiment servers.

## Validation

The numerical comparison uses 384 cases: three prompts, four steps, eight
layers (0, 5, 10, 15, 20, 25, 30, 35) and heads 0, 10, 20, 30. The target is
explicit FP32 sampled dense attention. See `q90k80_validation.json` for exact
source hashes, implementation/reference errors and invariants.

The complete-call benchmark includes layouts, RF0 selector, merge and the
full compressed route. Each method was compared with dense Flash SDPA on its
own exclusive H200, with identical QKV, shape, dtype and timing procedure:
96 real40-head QKV groups, three alternating repetitions per group.

| Method | Mean L2 to dense | Full call (ms) | Paired dense (ms) | Speedup |
|---|---:|---:|---:|---:|
| Q90K80C12 | 0.082846976 | 243.544 | 325.350 | 1.336x |
| Q90K80C14 | 0.075700125 | 267.900 | 333.411 | 1.245x |
| Q90K80C16 | 0.070054025 | 256.961 | 329.188 | 1.281x |

These timings use StepDistill QKV and do not establish native 50-step video
end-to-end speedups. Twenty-video quality evaluation is still in progress.

## 2026-09-21 Q90K80C14 kernel optimization

`Q90K80C14(merge_backend='cuda')` now runs three native CUDA kernels
(`q90k80c14_select.cu`, `q90k80c14_merge.cu`, `q90k80c14_route.cu`, all linked
against libcuda for TMA descriptors). The operator definition is unchanged:
same RF0 candidate table, same fp64 Fisher selection, same exact-14 merge and
full compressed route.

- Selector: one warp per query group in native CUDA. The greedy Fisher loop
  runs in FP32 (approved 2026-09-21; the C4 selector precedent) with a
  REDUX-based argmax and balanced register sums; on random QKV 1.4e-6 of the
  seed entries differ from the fp64 Triton selector. The previous fp64 native
  selector (seeds bitwise identical, 19.8 ms) is in the git history.
- Merge: the validated WGMMA kernel plus `AXIS_FAST_MATH` (approximate lg2/rcp),
  TMA tensor stores for mk/mv from a 128B-swizzled staging buffer filled with
  stmatrix, next-tile query prefetch, and a 3 CTA/SM launch bound (no
  register spills).
  `native_tiles` default 105 -> 35. The C8-style fused 32-bit ownership
  argmax is on by default (merge -4.6 ms): keys whose two best
  representatives score within 16 ulp of each other may change owner (about
  1e-3 of keys on random QKV; final-output rel-L2 to the old build moves from
  2.49e-3 to 2.55e-3). `merge_fused_argmax=False` restores the exact argmax.
- Route: transposed formulation, S^T = K_tile . Q^T with WGMMA m64n96k16 (96
  query rows instead of 128), column softmax through shared memory, P^T fed to
  the PV WGMMA from shared memory, two consumer warpgroups on alternate key
  chunks (private two-stage TMA rings each) merged in-CTA, plus a producer
  warpgroup (setmaxnreg 232/232/40).
- Triton fallbacks keep working; the Triton route defaults moved to
  `route_m=128, route_warps=8, route_stages=3` (fastest measured Triton config).

Synthetic benchmark (random Gaussian QKV, [1, 75600, 40, 128] BF16, one H200,
30-40 full calls, mean): 283.4 ms -> **172.5 ms** (176 ms with
`merge_fused_argmax=False`). Stage times: select 31.4 -> 10.7, merge
129.9 -> 89.5 (94.2 exact argmax), route 118.9 -> 68.9 ms. Final output vs the
pre-optimization build: rel-L2 2.5e-3 (bf16 rounding of the route accumulation
order; fast-math alone is 6.5e-4), max-abs 3.8e-6. Consecutive pipeline calls
are bitwise identical and hundreds of back-to-back merge/route launches ran
without fault.

Not done: the 384-case FP32-dense / StepDistill protocol above was not rerun,
so the C14 rows of `q90k80_validation.json` still describe the earlier build.
Measured floors for this shape: the mk/mv intermediate is 227.6 GB per call,
about 52 ms of DRAM time to write and again to read; both kernels are now
compute-bound above that (merge ~80 ms compute-only, route ~59 ms), so the
140 ms target was not reached.

## 2026-09-23 Q90K80 even capacities C4, C6, C8, C10

`analysis/q90k80_even_kernel.py` (`Q90K80Even(cap)`, aliases `Q90K80C4` ..
`Q90K80C10`) drives one native kernel family parameterized by `AXIS_CAP`:
`q90k80_even_select.cu`, `q90k80_even_merge.cu`, `q90k80_even_route.cu`,
`q90k80_even_layout.cu`. The operator is the same RF0 two-intersection Fisher
GMA as C12/C14/C16 with exactly CAP representatives per (query group, key
block); no singleton, raw or residual route. The Triton kernels in the same
module implement the identical operator and layouts and serve as the reference.

Design (all derived from the 2026-09-21 C14 kernels):

- Merge: a 64-row WGMMA tile holds `GPT = 32 / (CAP/2)` query groups of CAP
  rows (C4 16, C6 10, C8 8, C10 6), so C10 needs 140 tiles per (kb, head)
  instead of the 210 a 16-row-per-group tile would need. Lane pair p =
  warp*8 + lane/4 owns rows (p, half 0/1); group = p / (CAP/2). The ownership
  argmax reduces the fused 32-bit key over the group's lane pairs (xor
  butterfly for C4/C8, index shuffles for C6/C10, plus a shared-memory
  exchange between the two warps of a group that crosses a warp boundary).
  Seeds for the next tile are loaded one tile ahead; the next tile's query
  gather is issued after the ownership keys so it overlaps the exchange.
- mk/mv layout `[head][kb][group][CAP][128]` (group = frame*40 + col): a tile
  is one contiguous run of GPT*CAP rows, stored with two TMA boxes; bias stays
  group-major `[head][group][kb][CAP]`.
- Route: the C14 transposed kernel with a 5-D TMA box
  `{64 d, CAP, 1 group, 64/CAP kb, 1 head}` per 64-key chunk. For C6/C10
  (64 % CAP != 0) a chunk is still 64 consecutive keys, fetched as a partial
  head key block, a run of whole key blocks and a partial tail block with
  exact-height boxes (the 60-key alternative costs 6 % of the route).
- Selector: the C14 FP32 REDUX selector with CAP slots, restructured as a
  frame loop: RF0 seeds and the 32 candidate queries depend on (head, group,
  key row) only, so one CTA keeps its four groups' candidate queries in shared
  memory and loops over the 21 frames, loading only the 16 probe keys per
  frame (prefetched one frame ahead). Seeds are bitwise those of the per-kg
  selector; 5 CTAs/SM instead of 8. C4 4.8 -> 3.2 ms, C6 5.4 -> 4.6,
  C8 6.6 -> 6.0, C10 7.9 -> 7.5 (the greedy loop, not the prologue, dominates
  at larger capacities and it is occupancy bound).
- Layout: one fused kernel produces qp, the WGMMA-packed q, kp and vp from
  q/k/v (bitwise identical to the torch permutes plus the pack kernel), and one
  produces the token-major output; 4.6 ms -> 1.8 ms per call.

Validation (random Gaussian QKV, [1, 75600, 40, 128] BF16, one H200): native
output against the Triton reference path rel-L2 2.46e-3 (C4), 2.59e-3 (C6),
2.56e-3 (C8), 2.60e-3 (C10), max-abs 3.8e-6, the same tier as the C14 native
build (bf16 rounding of the route accumulation order, fast-math lg2/rcp and
the fused near-tie argmax). Seeds differ from the fp64 Triton selector on
0 / 2.2e-7 / 5.0e-7 / 6.6e-7 of entries; empty-representative flags agree
except 1.8e-6 of C10 entries (near ties). Consecutive calls are bitwise
identical. The 384-case StepDistill FP32-dense protocol of the table above was
not rerun for these capacities.

Synthetic full-call timings (same QKV as above, head_chunk 8, 30 calls, scai7
GPU 4, SW power cap active at 686 W so clocks moved between 1635 and 1980 MHz):

| Capacity | Estimate before implementation | Full call mean / median (ms) | select | merge | route | layout |
|---|---:|---:|---:|---:|---:|---:|
| C4 | 53 | 52.2 / 51.4 | 3.2 | 25.8 | 20.1 | 1.8 |
| C6 | 77 | 78.2 / 78.0 | 4.6 | 41.3 | 31.9 | 1.8 |
| C8 | 96 | 98.0 / 97.3 | 6.0 | 48.9 | 41.6 | 1.8 |
| C10 | 122 | 129.9 / 129.8 | 7.5 | 72.2 | 50.2 | 1.8 |

(Before the frame-loop selector: 53.3 / 78.9 / 98.1 / 130.3 ms mean; the
real-QKV table below was measured with the per-kg selector.)

Later the same day: four merge tile slices per (kb, head) instead of six
(merge 25.8/41.3/48.9/72.2 -> 25.3/40.5/48.3/69.4 ms in isolation) and the
operator reusing its layout buffers across calls; full call medians
50.7 / 77.1 / 96.1 / 130.5 ms (means are 1-3 ms higher because of the power
cap; per-call spread is about 20 %). The user's follow-up targets were
50 / 70 / 90 / 110 ms. The DRAM floor of the merged K/V round trip alone
(written once by the merge, read once by the route, 2 x cap x 16.3 GB per
call at 4.4 TB/s) is 30 / 44 / 59 / 74 ms; with select and layout that
leaves at most 15 / 20 / 24 / 27 ms for all merge and route compute, which
neither kernel approaches (merge is scalar-issue bound at about 38 % tensor
utilization, the route is shared-memory bound about 30 % above its DRAM
time). Architectures that would remove the round trip were re-derived and
rejected: a fused merge+route per group needs the route to attend cap-wide
key tiles (n16 WGMMA, 2.7x the tensor time) and either reload every group's
queries per key block or keep 64x more rows than the register file holds;
keeping the intermediate L2 resident needs a route unit of at most one
group per head, far below the 132 CTAs the route needs in flight.

The C14 kernels measured on the same GPU the same day: merge 90.4, route 69.6,
select 10.7 ms (172.5 ms shipped figure reproduced). Per merge tile the new
kernel spends 7.5k CTA cycles against C14's 6.9k, but all 64 rows are real
(C14 pads 8 of 64), so per useful row it is 4 % cheaper. C10 pays about 6 %
of its merge for the five-lane-pair reductions and the cross-warp exchange.

Measured and closed: 4 CTAs/SM at 128 registers (+4 %), one tile slice per
CTA (+10 %, tail), a double staging buffer (neutral to worse), 2-D store
descriptors (neutral), the C14 frame-minor tile order (neutral), packing the
straddling groups per warp pair to break the barrier chain (neutral: the
exchange is instruction-throughput bound, not barrier bound; removing it
entirely would give C10 about 6 %), DRAM write cost of the merge (about 13 %,
the same share as C14), the selector with candidate queries in registers
(64 registers, 4 CTAs/SM: slower than the per-kg selector for C8/C10), a
four-vectors-per-thread layout kernel (1.87 vs 1.82 ms), two frames per
selector iteration with interleaved greedy loops (slower: 6.8/8.5 ms for
C8/C10 at 4 CTAs/SM), xor-butterfly reductions instead of REDUX in the
selector (slower), rsqrtf for the basis norm (-0.2 ms, 2.5e-7 of seeds
change; not adopted). A further merge round against the same targets, all
measured on C8 and all closed: reusing the softmax exponentials as bf16 in
shared memory instead of a second ex2 per (row, key) (merge 48.0 -> 51.2 ms
and rel-L2 2.6e-3 -> 5.3e-3), pre-multiplying the merge's packed queries by
scale*log2e in the layout kernel to drop one FMUL per (row, key) (48.0 ->
48.05, rel-L2 2.65e-3), FFMA normalization (48.03), and four CTAs/SM through
an 8 KB two-round staging buffer (the half staging alone costs 2 ms at
3 CTAs; at 4 CTAs the kernel spills 72-80 B and measures 53.7-55.6 ms; the
query prefetch is worth 2 ms at 3 CTAs). Removing instructions from the
scalar phase does not move the merge, so it is latency bound at three warps
per scheduler, and every route to more warps costs registers it does not
have. Route: a query-split variant (four consumer warpgroups of 24 query
columns each, all consuming every 64-key chunk from one shared 4-stage ring,
four threads per query in the softmax, 96 registers, no cross-warpgroup
merge) ran correctly (rel-L2 to the reference 1.46e-3) but at 55.9 ms
against 40.0 ms for C8: with N=24 every WGMMA reads a 2 KB A operand for a
quarter of the work, so the K and V tiles cross the shared-memory port four
times per chunk and the tensor pipe becomes operand bound. (Lesson recorded:
setmaxnreg draws from a per-CTA pool, so consumers can only request what the
producer releases; an unbalanced request deadlocks the CTA.) Register caps
of 160/152/144 on the two-warpgroup route spill 192 B and run at 64 ms, so
three warpgroups holding all 96 columns are out of reach. The C10 residual
against the estimate (130 vs 122 ms) is the merge: the five-lane-pair
ownership reduction (four index shuffles per column instead of two) and the
cross-warp exchange of groups that straddle warps cost about 8 ms, and no
64-row tiling of 10-row groups avoids them.

Real-QKV protocol (96 cases: 3 prompts x 4 StepDistill steps x layers 0, 5,
..., 35, 40 real heads, dense Flash SDPA timed on the same QKV in alternating
order, three repetitions): see `analysis/q90k80_even_real40.py` and the table
below.

| Capacity | Cases | Full call (ms) | Paired dense (ms) | Speedup | Mean rel-L2 to bf16 dense (40 heads) | Faster than dense in every case |
|---|---:|---:|---:|---:|---:|---|
| C4 | 96 | 54.297 | 327.463 | 6.031x | 0.1913 | yes |
| C6 | 96 | 83.616 | 325.383 | 3.891x | 0.1692 | yes |
| C8 | 96 | 104.109 | 325.180 | 3.123x | 0.1535 | yes |
| C10 | 96 | 133.767 | 326.533 | 2.441x | 0.1412 | yes |

Run: scai7 GPU 4, NVIDIA H200, head_chunk 8, complete=True, LD_PRELOAD libstdc++ as in the video pipeline. Results: `/local2/denghaoran/GridMergeAttention/q90k80_even_real40_20260923/real40.json`. The relative L2 here is against the bf16 Flash SDPA output over all 40 heads, not the FP32 sampled-dense 4-head metric of the C12/C14/C16 table, so the two columns are not comparable. For reference the Q80K90 family measured 61.4 / 108.9 / 121.3 / 224.7 ms (C4 / C6 / C8 / C10) under the same protocol on 2026-09-22.

### Lazy route softmax (2026-09-25, default)

`ROUTE_LAZY=1` (now the default in `analysis/q90k80_even_route.cu`) replaces
the route's column softmax. Before, every 64-key chunk wrote the 64x96 FP32
logits to shared memory (24 KB), synchronized, and let 96 threads each reduce
one query column serially (64 loads, max, 64 ex2, sum) before writing P^T. Now
each thread keeps a per-query offset m for its 24 query columns and forms
p = ex2(t - m) directly in the S^T accumulator layout, with per-thread partial
column sums; P^T leaves through six `stmatrix.trans` per thread into a
per-warpgroup double buffer in the old logits area. m is stale (FA4-style lazy
rescaling): only when some logit in the warpgroup exceeds m + 8 (base 2), and
always on the first chunk, is the exact chunk maximum exchanged through a
4 x 96 table and O and the partial sums rescaled. p <= 2^8 keeps FP32 sums and
BF16 P far from overflow; the result is the same softmax with a different
stabilizing offset. The O^T accumulator has the same column mapping as S^T, so
the rescale is thread-local. The logits round trip and one of the two
per-chunk warpgroup barriers of the old slow path are gone.

Accuracy. Against the Triton reference the rel-L2 rises from 2.46-2.60e-3 to
2.97-3.21e-3, but that reference rounds P after an exact running maximum like
the old kernel, so their rounding errors are correlated.
`analysis/q90k80_even_route_check.py` runs both routes on the same merged K/V
and compares them with an FP64 softmax over 64 random groups x 8 heads:

| Capacity | Old route vs FP64 | Lazy route vs FP64 | Old vs lazy |
|---|---:|---:|---:|
| C4 | 2.305e-3 | 2.259e-3 | 2.84e-3 |
| C6 | 2.384e-3 | 2.331e-3 | 3.05e-3 |
| C8 | 2.374e-3 | 2.312e-3 | 3.05e-3 |
| C10 | 2.384e-3 | 2.352e-3 | 3.12e-3 |

With the threshold at 0 (rescale on every increase) the lazy route matches the
old one to 1.8e-5 (C8), which checks the code path. Consecutive calls stay
bitwise identical; the consumer warpgroups spill 8 B.

Synthetic (scai7 GPU 7, head_chunk 8, 30 calls; the ABBA A/B before the
default switch gave route 20.1/30.9/40.2/49.1 -> 18.3/29.6/37.0/47.6 ms):

| Capacity | select | merge | route | layout | Full call mean / median (ms) |
|---|---:|---:|---:|---:|---:|
| C4 | 3.25 | 25.1 | 18.3 | 1.82 | 49.7 / 48.8 |
| C6 | 4.63 | 40.8 | 29.5 | 1.82 | 75.8 / 75.1 |
| C8 | 6.04 | 48.2 | 35.6 | 1.83 | 93.3 / 92.1 |
| C10 | 7.50 | 69.7 | 47.4 | 1.82 | 127.5 / 126.5 |

Real QKV, same 96-case protocol, with the old and the lazy route timed in the
same run on the same QKV (the three methods rotate order per repetition;
`--ab-route-flags=-DROUTE_LAZY=1`; `docs/q90k80_even_real40_lazy.json`):

| Capacity | Old route (ms) | Lazy route (ms) | Difference | Cases faster | Paired dense (ms) | Speedup old -> lazy | Mean rel-L2 to bf16 dense (both) |
|---|---:|---:|---:|---:|---:|---:|---:|
| C4 | 53.11 | 48.35 | -4.76 | 96/96 | 323.56 | 6.09x -> 6.69x | 0.1913 |
| C6 | 78.69 | 75.01 | -3.68 | 96/96 | 324.71 | 4.13x -> 4.33x | 0.1692 |
| C8 | 99.80 | 93.29 | -6.51 | 96/96 | 324.15 | 3.25x -> 3.47x | 0.1535 |
| C10 | 131.77 | 130.22 | -1.56 | 83/96 | 322.97 | 2.45x -> 2.48x | 0.1412 |

Largest per-case rel-L2 between the two routes: 1.7e-3. Run: scai7 GPU 7,
2026-09-25, `/local2/denghaoran/GridMergeAttention/q90k80_even_real40_lazy_20260925/`.
The old-route column of this run is 1-5 ms faster than the 2026-09-23 table
above (different GPU and the merge/selector changes since).

Closed in the same round (C8, merge in isolation 48.1-48.2 ms unless noted):
packed 4-bit owner labels (`MERGE_PACKED_LABELS`, neutral, same 167
registers), one ex2 per key column instead of per (row, column)
(`MERGE_ONE_EX2`, bitwise identical, spills 48 B, 5.47 vs 4.65 ms per head
chunk), the K/V merge GEMMs as four pipelined m64n64 passes on two
accumulators (`MERGE_SPLIT_N`, spills 64 B, 5.70 vs 4.65 ms; without the spill
in the profiling build, neutral), 2- and 4-way partial sums in
reduce80/dot80 (`MERGE_SUM_WAYS`, 49.3-49.5 vs 49.3 ms on another GPU;
neutral). Phase profile of the C8 merge per tile (CTA cycles): score wait 576,
scalar block 3757, merge GEMM waits 672 + 587, staging and TMA 848 + 988.
