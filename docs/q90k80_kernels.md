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

- Selector: one warp per query group in native CUDA, fp64 greedy loop with the
  reference formulas. Seeds are bitwise identical to the Triton selector on
  the synthetic check below.
- Merge: the validated WGMMA kernel plus `AXIS_FAST_MATH` (approximate lg2/rcp),
  TMA tensor stores for mk/mv from a 128B-swizzled staging buffer, next-tile
  query prefetch, and a 3 CTA/SM launch bound (no register spills).
  `native_tiles` default 105 -> 35.
- Route: transposed formulation, S^T = K_tile . Q^T with WGMMA m64n96k16 (96
  query rows instead of 128), column softmax through shared memory, P^T fed to
  the PV WGMMA from shared memory, two consumer warpgroups on alternate key
  chunks (private two-stage TMA rings each) merged in-CTA, plus a producer
  warpgroup (setmaxnreg 232/232/40).
- Triton fallbacks keep working; the Triton route defaults moved to
  `route_m=128, route_warps=8, route_stages=3` (fastest measured Triton config).

Synthetic benchmark (random Gaussian QKV, [1, 75600, 40, 128] BF16, one H200,
40 alternating full calls, mean): 283.4 ms -> 192.0 ms. Stage times:
select 31.4 -> 19.9, merge 129.9 -> 97.4, route 118.9 -> 68.9 ms.
Final output vs the pre-optimization build: rel-L2 2.5e-3 (bf16 rounding of
the route accumulation order; fast-math alone is 6.5e-4), max-abs 3.8e-6,
seeds identical. 100 consecutive pipeline calls are bitwise identical and 300
back-to-back merge/route launches ran without fault.

Not done: the 384-case FP32-dense / StepDistill protocol above was not rerun,
so the C14 rows of `q90k80_validation.json` still describe the earlier build.
Measured floors for this shape: the mk/mv intermediate is 227.6 GB per call,
about 52 ms of DRAM time to write and again to read; both kernels are now
compute-bound above that (merge ~80 ms compute-only, route ~59 ms), so the
140 ms target was not reached.
