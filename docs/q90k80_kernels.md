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
