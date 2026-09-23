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
- Selector: the C14 FP32 REDUX selector with CAP slots.
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
| C4 | 53 | 53.3 / 52.5 | 4.8 | 25.8 | 20.1 | 1.8 |
| C6 | 77 | 78.9 / 78.1 | 5.4 | 41.2 | 32.0 | 1.8 |
| C8 | 96 | 98.1 / 96.7 | 6.6 | 49.0 | 41.2 | 1.8 |
| C10 | 122 | 130.3 / 129.7 | 7.9 | 72.1 | 50.2 | 1.8 |

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
the same share as C14).

Real-QKV protocol (96 cases: 3 prompts x 4 StepDistill steps x layers 0, 5,
..., 35, 40 real heads, dense Flash SDPA timed on the same QKV in alternating
order, three repetitions): see `analysis/q90k80_even_real40.py` and the table
below.

(Run in progress at the time of this commit; the table follows in the next commit.)
